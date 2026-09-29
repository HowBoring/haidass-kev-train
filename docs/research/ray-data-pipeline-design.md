# Ray Core semantics for the data-generation pipeline (design research)

Research date: 2026-09-29. Source: official Ray docs (`docs.ray.io/en/latest`), static reading
only; no runtime measured, no code written, no endpoints contacted. Companion reading of the
current implementation is in `docs/research/data-pipeline-concurrency.md`.

Scope: verify that **Ray Core** (tasks/actors/objects) fits the proposed pipeline, and pin down
which properties Ray **guarantees** versus which are **proposed design mechanisms** we must build
ourselves. Markers below: **[G]** = Ray guarantee (documented), **[P]** = proposed design
mechanism (our responsibility, not provided by Ray).

## 1. Process isolation vs. thread races

- **[G]** Each actor runs in its own Python process; methods on the same actor execute serially
  (sync actor) and methods on different actors run in parallel. Class variables are per-process,
  not shared across actor instances.
  [Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- **[G]** Neither threaded actors nor async actors bypass the GIL; true CPU parallelism requires
  separate processes (or GIL-releasing native libs).
  [AsyncIO / Concurrency for Actors](https://docs.ray.io/en/latest/ray-core/actors/async_api.html)
- Implication for us: process-per-actor isolation removes the current per-HTTP-attempt daemon
  threads and lock-guarded `in_flight` counters (`generation.py:74-99`). It does **not** make
  "shared mutable business state" races disappear by magic — it makes them impossible by
  construction **only if** all shared business state lives behind one owner actor whose methods
  are short and serial. **[P]** Put ledger state in exactly one synchronous coordinator actor;
  keep HTTP actors stateless with respect to business state.
- **[P] / caveat:** isolation of *business state* is a design property of our layout, not a Ray
  feature. Ray does not prevent a worker from spawning its own threads or touching shared files.

## 2. Resource semantics: logical CPUs are admission control, not hard limits

- **[G]** Ray resources are **logical**; `num_cpus` is used for admission control only. Ray does
  not pin CPUs, does not prevent a `num_cpus=1` task from using multiple physical CPUs, and does
  not limit physical memory even when `memory=` is requested ("It is your responsibility to make
  sure tasks or actors use no more resources than specified").
  [Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- **[G]** Actor default is 1 CPU for *scheduling* and **0 CPU for running** — an unbounded number
  of default actors can pile onto one node. Docs recommend always setting `num_cpus` explicitly.
  [Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- **[G]** The sum of logical requirements of concurrently executing tasks/actors on a node
  cannot exceed that node's logical total — this is the documented lever to bound concurrency and
  avoid OOM.
  [Pattern: limit running tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html)
- **[G]** Ray sets `OMP_NUM_THREADS=<num_cpus>` (1 if unset) to avoid thread oversubscription.
- Implications:
  - **[P]** Sync CPU actors (tokenize/parse/check) get `num_cpus=1`; per-node tokenize parallelism
    is bounded by logical CPU count. This bounds *admission*, not real usage: a pathological
    tokenizer path can still oversubscribe a core. Sizing is our job; no speedup is promised.
  - **[P]** Async HTTP actors get fractional `num_cpus` (e.g. 0.1–0.25) since they are I/O-bound,
    but their real concurrency is bounded by `max_concurrency` (§3), not by CPU resources.

## 3. Async actor reentrancy (HTTP I/O only)

- **[G]** An async actor runs all methods on **one thread, one event loop**; coroutines are
  multiplexed at `await` points. `max_concurrency` caps in-flight coroutines (default **1000**).
  Blocking `ray.get`/`ray.wait` inside an async actor method is not allowed (it would freeze the
  loop). Any `async def` method makes the whole actor an AsyncActor.
  [AsyncIO / Concurrency for Actors](https://docs.ray.io/en/latest/ray-core/actors/async_api.html)
- Implications:
  - **[P]** HTTP actors: `async def` + `httpx.AsyncClient` (already in the environment; promote
    to a locked dep at implementation time), `max_concurrency` set to the per-replica request
    budget. Never put CPU-heavy parsing/encoding in these actors — it would block the loop and
    serialize every outstanding request.
  - **[P]** Interleaving hazard: between two `await`s another coroutine of the *same* actor can
    run. Any per-request mutable scratch state must be local to the coroutine, not on `self`.
  - **[P]** CPU validation of a returned response must be submitted as a separate call to a CPU
    actor, not awaited inside the HTTP actor; a worker does not hold a CPU core while waiting on
    the LLM.

## 4. Fault tolerance and retries: no exactly-once external HTTP

- **[G]** Actor tasks are **at-most-once by default** (`max_task_retries=0`): `ray.get` may raise
  `RayActorError` *even though the task executed* (actor died right after). With
  `max_task_retries>0` actor tasks become **at-least-once**: "Retried methods may execute twice."
  `max_restarts` (default 0) re-runs the constructor but **does not restore application state**;
  app-level checkpointing is our responsibility.
  [Actor Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html)
- **[G]** Plain tasks default to `max_retries=3` on worker death; `retry_exceptions=True` retries
  on user exceptions only if the function is idempotent.
  [Task Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html)
- **[G]** Non-detached actors **fate-share with their creator** (owner dies → actor dies, no
  automatic recovery even with `max_restarts`). Detached actors survive creator death but only
  until cluster destruction.
  [Actor Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html)
- **[G]** `ActorUnavailableError` on `ray.get` means "no guarantee on whether the task executed";
  side effects "may or may not be observable".
- Implication: **there is no exactly-once guarantee for external HTTP calls.** A retried or
  ambiguous request may have reached the endpoint and produced a billable/committed generation.
  - **[P]** The durable ledger, not Ray's retry flags, is the dedup/replay authority. HTTP actor
    methods keep `max_task_retries=0` and `retry_exceptions=False`; retry *policy* (respecting the
    existing 100 accepted / 2000 attempts / 4 h budgets and the current error taxonomy in
    `generation.py`) is decided by the coordinator/driver against ledger state.
  - **[P]** Attempt lifecycle: coordinator grants a permit (transactional ledger row, **counted
    before dispatch**) → HTTP actor attempts → outcome committed back transactionally and
    idempotently (§7 fencing). Accounting is **per granted permit, not per actual dispatched
    request**: a crash between permit grant and HTTP send conservatively leaves the permit
    consumed/UNKNOWN.
  - **[P]** A local timeout leaves the permit **UNKNOWN**: it must not be silently released,
    auto-retried as "free", or auto-expired on local timeout alone — the remote side cannot be
    reliably cancelled and may still complete. UNKNOWN resolves only on a backend-terminal
    outcome or a verified hard deadline; until then the credit stays held and the affected
    endpoint is paused.

## 5. Cancellation

- **[G]** `ray.cancel(ref)` sends a `KeyboardInterrupt` to the worker mid-execution;
  `force=True` force-exits the worker. Cancelled tasks are **not** automatically retried.
  [Task Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html)
- **[G]** Actor-task cancellation only allows `force=False` (`force=True` raises `ValueError`;
  kill the actor with `ray.kill` instead). For a running **sync/threaded** actor method,
  cancellation only sets a flag the method must check cooperatively; for an **async** actor
  method, Ray cancels its `asyncio.Task` (so an `await`ed HTTP request gets a real
  `asyncio.CancelledError`). `recursive=True` (default) also cancels descendants but does not
  force-kill child actor tasks.
  [ray.cancel API](https://docs.ray.io/en/latest/ray-core/api/doc/ray.cancel.html)
- Implications:
  - **[P]** Local cancellation of an in-flight async HTTP attempt is best-effort: cancelling the
    asyncio task closes the *client* side (httpx aborts the connection), which is strictly better
    than today's timed-out daemon thread that keeps running and consuming endpoint capacity
    (`generation.py:92-95`). But **remote cancellation is not guaranteed** — the endpoint may
    still finish the generation server-side; the ledger keeps the attempt UNKNOWN until a
    backend-terminal outcome or verified hard deadline (§4), never a local timeout alone.
    Do not promise endpoint-side revocation.

## 6. Object refs, object store, backpressure

- **[G]** Remote objects are immutable, per-node object store, distributed reference counting:
  any live `ObjectRef` (local var, argument of a pending task, nested in another object) **pins**
  the object; top-level task args pin until the task completes. If the object store fills, objects
  spill to disk (default 30% of memory reserved for the store; store memory is not a schedulable
  logical resource).
  [Objects](https://docs.ray.io/en/latest/ray-core/objects.html),
  [Memory Management](https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html),
  [Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- **[G]** Unbounded submission of pending tasks grows the pending queue and can OOM; the
  documented remedy is a `ray.wait` loop bounding in-flight refs. Bounding *concurrent* execution
  is done via resource requirements, not `ray.wait`.
  [Pattern: limit pending tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html),
  [Pattern: limit running tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html)
- **[G]** Don't return `ray.put()` refs from tasks / let refs outlive their owner; the driver
  should own long-lived refs.
  [Fault tolerance overview](https://docs.ray.io/en/latest/ray-core/fault-tolerance.html)
- Implications:
  - **[P]** Bounded driver dispatch: the driver (orchestrator, separate from the ledger actor)
    keeps at most N in-flight stage refs and uses `ray.wait` for flow control — this replaces
    `bulk.py`'s thread-per-lane executor without building a queue framework.
  - **[P]** Work units are **fixed, content-identified Parquet row-group spans** (file path +
    row-group index + content hash of the shard), computed once by the driver. The driver
    schedules **span descriptors only** (tiny top-level args); CPU actors read/decode the bounded
    row batch locally and run prepare/validate there. This replaces today's modulo-row striping
    (which re-reads and re-decodes every batch of a shared file per lane) and offset-based resume
    rescans (`ufw.py:297-335`, `bulk.py:199-200`) — Ray is not repartitioning sources; the spans
    are ours, and resume is span-completion lookup in the ledger, not history replay.
  - **[P]** Do not closure-capture large refs in long-lived actors (pins memory for the job's
    lifetime); return values directly so the driver owns refs (fault-tolerance guidance above).

## 7. Failure boundary: coordinator, driver, head node

- **[G]** The GCS is **not fault tolerant by default** (in-memory); if it fails, the whole
  cluster fails. FT requires external Redis (officially supported only via KubeRay for Ray Serve;
  otherwise "at your own risk") or the alpha embedded RocksDB backend — itself single-writer on
  one persistent volume.
  [GCS Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/gcs.html)
- **[G]** During GCS recovery no actor/resource management works; a driver is not a fault-tolerant
  entity either — if the driver process dies, non-detached actors it created die with it (§4).
- Implications — **explicitly no seamless failover/HA:**
  - **[P]** The coordinator is a **single-owner, synchronous actor** with short transactional
    methods (permit/commit/heartbeat) over a **local SQLite ledger on one durable host**.
    Use `NodeAffinitySchedulingStrategy(node_id=..., soft=False)` for this local-file ledger:
    if the designated host is unavailable, fail closed rather than create another database.
    On restart, verify the durable ledger's run identity before accepting work.
    [Scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html).
    SQLite is single-writer — exactly what a single owner actor provides; this is a deliberate
    boundary, not a gap to engineer around. Workers are genuinely multi-node; only the ledger is
    single-host.
  - **[P]** Distinguish the **driver** (bounded orchestration run loop, `ray.wait` dispatch) from
    the **ledger actor** (short sync methods). The blocking run loop must not live *on* the
    ledger actor, or it cannot serve permit/commit calls (sync actor methods are serial).
  - **[P]** Coordinator/host failure = **restart-from-ledger**, not transparent recovery: a new
    driver re-opens the same SQLite file, **first reconciles** (safe pause — re-derive UNKNOWN
    permits against the ledger/backend per §4, **no silent fresh retries**), then resumes.
    Complete/partial admission ordering is ledger-defined and independent of completion speed;
    the 4 h time budget and external generation remain nondeterministic, as today.
  - **[P]** **Fencing:** every driver incarnation gets a monotonically increasing epoch from the
    ledger; all commits carry `(epoch, permit_id)` and are **idempotent transactions** — a stale
    epoch or duplicate commit is rejected/ignored. This closes the zombie-driver window that
    Ray's ownership model (§4) cannot.
  - **[P]** GCS/cluster loss: with default in-memory GCS, treat cluster loss as run loss; the
    durable artifacts (frozen suites, SQLite ledger) on the pinned host are the restart point.

## 8. Ray Core vs Ray Data / Ray Serve (short verdict)

- **Ray Core** fits: the pipeline is a stateful, per-request workflow with external HTTP side
  effects, fine-grained attempt accounting (100/2000/4h budgets), candidate/source validation
  semantics, and a transaction ledger — exactly actor/task primitives.
- **Ray Data** is a streaming batch loader/preprocessor (read → map_batches → consume); its
  execution model targets stateless record transforms, not per-attempt ledgered HTTP generation
  with ambiguous-outcome accounting. Not chosen. [Ray Data](https://docs.ray.io/en/latest/data/data.html)
- **Ray Serve** is an online serving layer (ingress, replicas, request routing); this pipeline
  is an offline batch producer with its own client-side budget semantics. Not chosen.
- No new framework dependencies beyond `ray` itself; `httpx`/`pyarrow` already present would be
  promoted to explicit locked deps at implementation time only.

## 9. Compact architecture recommendation (design only, no speedup claims)

1. **Driver process (bounded orchestration, multi-node):** computes the fixed,
   content-identified row-group **span plan** once, writes a **versioned plan manifest**
   (plan version + per-span shard identity/hash); dispatches span descriptors to stage calls;
   `ray.wait` bounds in-flight refs. Owns no business state. Existing raw validation, gold
   handling, `source_ref` identity and group→split rules are retained unchanged — only the
   partitioning mechanism changes.
2. **Sync CPU actors** (`num_cpus=1`, `max_restarts=0`, `max_task_retries=0`): tokenizer loaded
   once per actor in `__init__`; these actors read/decode their span's row batch data-locally.
   Schedule only on nodes with verified access to the pinned source snapshot and tokenizer
   assets; soft locality preferences are safe only when fallback nodes can access the same
   data. Actors run CPU parsing, candidate validation, and generator prompt/template
   tokenization. Model-based screening remains an HTTP stage. The HTTP actor never tokenizes.
3. **Async HTTP actors** (`num_cpus≈0.1`, `max_concurrency=K`): I/O only; each coroutine performs
   exactly one ledger-permitted attempt on a fully prepared request; timeouts surface as UNKNOWN
   to the coordinator; client connection aborted via asyncio cancellation (best-effort remote
   effect, §5).
4. **Coordinator actor** (sync, 1 CPU, single instance, hard node affinity with startup
   file-identity check, §7): sole owner of business state; short transactional methods against local SQLite
   (single writer); grant permit → idempotent fenced commit; enforces 100/2000/4h budgets,
   dedup, group→split integrity; UNKNOWN permits held per §4, never silently recycled.
5. **Aggregation** verifies actual scanned shard-row spans against the frozen plan and checks
   non-overlap, record membership, and completeness of required work. Unread planned rows are
   not reported as scanned. The old `line % stripes` lane check is replaced, not retained.
6. **Failure model:** worker-actor crash → driver decides against ledger; driver crash → safe
   pause + reconcile before resubmission (§7); coordinator/host crash → restart-from-ledger on
   the same host; cluster/GCS loss → run loss, resume from durable artifacts. No exactly-once
   HTTP, no transparent failover, no claimed throughput improvement — the intended gain is
   multi-core/multi-node CPU utilization (bounded by logical admission control) and elimination
   of thread-per-attempt races, subject to measurement at implementation time.
7. **Migration boundary:** existing frozen suites remain loadable for training as-is; the new
   runner must **not** silently resume old in-flight lane chains (offset/stripe state is
   meaningless under spans), and no exact reproduction of the old accepted set is claimed —
   span plan and budget timing change traversal.

Skipped: placement groups, custom resources, GCS-FT (Redis/RocksDB). Hard node affinity keeps
the single ledger on its durable host; verified source access controls CPU-worker placement.
This design has distributed workers but no seamless control-plane failover. Add a replicated
ledger and an appropriate Ray head recovery deployment only if HA becomes a requirement.
