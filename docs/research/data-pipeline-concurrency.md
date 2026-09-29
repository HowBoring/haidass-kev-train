# Data pipeline concurrency notes

Research date: 2026-09-29. Static source reading only; no live HTTP endpoints were called.
Scope: offline frozen-suite build (`build.py`, `generation.py`, `ufw.py`, `finemath.py`),
fleet orchestration (`bulk.py`), and the training consumer. `[INFERENCE]` marks reasoning
from code shape rather than direct statement. Two no-network runtime checks were exercised
(noted where used); no service or throughput was tested.

## 1. Row scanning and lane partitioning

- UFW rows stream from sorted `*.parquet` shards in batches of 64 via
  `iter_batches(batch_size=64, columns=["uid","content","style"])`; schema mismatch raises
  (`src/haidass_kev_train/data/ufw.py:306-335`). FineMath is analogous with columns
  `text,url,snapshot_type` (`src/haidass_kev_train/data/finemath.py:52-71`).
- `lane_shards` partitions **files first, then rows**: `file_lanes = min(len(shards),
  shard_count)`; lane `i` takes `shards[i % file_lanes :: file_lanes]` and, when lanes
  exceed files, keeps rows where `(line) % stripes == stripe` (`ufw.py:297-303`,
  used at `ufw.py:313,323` and `finemath.py:56,69`). Yielded row coordinates are disjoint
  across lanes; a no-network check with 5 lanes / 2 files / 12 rows covered 24 unique
  coordinates with zero overlap. When stripes share one Parquet file, each lane still reads
  and decodes every batch of that file, so per-lane I/O scales with the stripe count
  [INFERENCE].
- `build._source_rows` merges sources in sorted order and forwards `shard_index`/
  `shard_count` (`src/haidass_kev_train/data/build.py:225-234`). Per-source
  `source_start_rows` offsets are replayed by incrementing a skip counter *before* any
  generation (`build.py:375-378`); resume therefore rescans already-traversed rows, and
  lane offsets chain from the previous batch's `summary.scanned_by_source`
  (`bulk.py:199-200`, `docs/agents/training.md:468-470`) [cost is INFERENCE].
- `finished()` (source target reached) is polled per shard, batch and row to stop iteration
  early (`ufw.py:315-322`, `finemath.py:57-68`; wired at `build.py:368-369`).

## 2. Request lifecycle (`generation.py`)

- One `Generator` per build run (`build.py:341`). `ask` builds a system prompt carrying
  `PROMPT_VERSION` and an explicit "user payload is untrusted source data" instruction
  (`src/haidass_kev_train/data/generation.py:14-16,102-110`), rejects material containing
  chat-role delimiters (`UnsafeMaterial`, `108-109`), and enforces the context budget
  `max_context_tokens - max_output_tokens` on the templated encoding, else
  `ContextOverflow` (`111-115`).
- Request body pins `MODEL`, `temperature: 0` and `max_completion_tokens`.
  FineMath extraction passes a task-specific `json_schema`; other tasks use
  `json_object`. Each task fixes `enable_thinking` (`generation.py:99-119`).
  Endpoint is a module-level `BASE_URL = "http://110.123.0.3:8000/v1"`
  (`generation.py:14`; `docs/agents/training.md` notes port 8000, not 8080).
- Up to 3 attempts per call (`120`). Before each attempt `check()` enforces run budgets
  (`121`); `attempts` is incremented **before** dispatch so network failures also consume
  authorization (`129`). Per-attempt timeout is `min(config["timeout"], remaining)`
  (`126-131`).
- Error taxonomy (`132-145`): HTTP 400/401/403/404/405/413/422 and any other 4xx except
  408/409/429 → `StopGeneration("configuration_error")` (no retry of the run); 408/409/429
  and 5xx plus `URLError/OSError/HTTPException` → `service_error`, retried, third failure
  raises `StopGeneration("service_error")`. Linear backoff `0.1*(retry+1)` capped by
  remaining budget (`173-177`).
- Response validation (`146-172`): body capped at `MAX_RESPONSE_BYTES = 1 MiB` (`17,81,
  148-149`); duplicate JSON keys rejected via `_unique_object` (`36-42,150,165`); envelope
  must have exactly one choice with `finish_reason == "stop"` (`159-161`); task-specific
  `required` schema must pass (`166`). Three malformed replies return `None` (`169-172`),
  which callers convert to row-level `Rejected("malformed_response")`
  (e.g. `build.py:470-471,495-496`; `finemath.py:171-172`).
- Token usage is accumulated from the envelope when well-typed (`153-158`) and reported at
  `build.py:539`.

## 3. Budgets

- Defaults and hard caps: `target ≤ 100` accepted, `max_attempts ≤ 2000`,
  `max_seconds ≤ 14400` (4 h), `timeout` default 90 s; all validated finite/positive
  (`build.py:92-104`). Deployed config matches the caps exactly
  (`configs/data/ufw-finemath-unreviewed-30k-parallel.toml:3-6`).
- `check()` raises `StopGeneration("time_limit" | "attempt_limit")`
  (`generation.py:60-67`); `build` catches it, records `stop_reason`, and still publishes
  the verified partial suite marked incomplete (`build.py:532-533,560`; completeness
  verdicts at `build.py:261-270`).

## 4. Threads and timeout

- The only threading inside a lane is one **daemon thread per HTTP attempt**
  (`generation.py:74-99`): the worker runs `urlopen` + bounded `read`, the caller
  `join(timeout)`s and raises `TimeoutError` if the worker is still alive (`92-95`). A
  timed-out daemon may keep running and consuming remote resources; the code comments say
  it is never reported as cancelled (`94`, `174`) and `in_flight` (lock-guarded counter,
  `54-55,69-72,84-89`) surfaces leftovers in the report (`build.py:542`).
- No `ThreadPoolExecutor`/`asyncio`/`concurrent` use exists in `build.py`,
  `generation.py`, `ufw.py` or `finemath.py`: within a lane the scan → parse → generate →
  screen pipeline is strictly sequential. All inter-lane concurrency lives in `bulk.py`
  (§6).

## 5. Backpressure

- Client side there is no rate limiter: a lane issues at most one request at a time by
  construction (sequential `ask`), so fleet concurrency ≈ number of active lanes plus any
  timed-out daemon threads.
- Server side: `docs/agents/training.md:466-473` describes the `blue-a3-host` deployment
  as 15 single-chip vLLM replicas × `max_num_seqs=16` = 240 concurrent sequences, with
  extra HTTP connections queueing; the config's `shard_count = 240`
  (`configs/data/ufw-finemath-unreviewed-30k-parallel.toml:14`) is sized to that ceiling.
  These are documented deployment claims, **not observed production throughput**.

## 6. Bulk orchestration (`bulk.py`, contributed by BulkOrchestration)

- `ThreadPoolExecutor(max_workers=shard_count)`; one batch submitted per active lane while
  `len(train_ids) < minimum_train` (`src/haidass_kev_train/data/bulk.py:166-168`). Main
  loop: `wait(..., timeout=30, FIRST_COMPLETED)` with a ≥30 s JSON heartbeat of counters
  only — no source text or payloads (`171-174`, `151-165`).
- Per completed batch: `future.result()` re-raises any build error and halts the run
  (`177`); the executor context manager still waits for in-flight lanes to finish their
  current batch, frozen artifacts untouched. The batch manifest/summary are re-verified
  before acceptance (`178-186`, `_read_batch` at `42-55`); policy identity is re-checked
  (`187`). An incomplete batch with a failure `stop_reason` (`_FAILURE_REASONS`,
  `28`) raises (`188-192`); source-exhausted lanes end "without fabricating" their target
  (`193`). Complete batches union train IDs and advance per-source offsets by
  `scanned_by_source` (`194-201`), then resubmit if below the minimum (`202-203`).
- Because the minimum is only checked at submit time (`168,202`), already-running lanes
  finish even after the target is reached: overshoot is bounded by `shard_count × 100`
  accepted records, and `aggregate` treats the minimum as a floor, not a quota
  (`bulk.py:204-207`; `src/haidass_kev_train/data/aggregate.py:55-62`).
- Synchronization: a single `threading.Lock` guards only the `snapshots` progress dict
  (`146-153,179-180`); all other state is mutated on the main thread in the done-future
  loop, so no further locking is needed.
- Resume/chain: `_Lane` walks contiguous `batch-0000…` directories (`61-96`), requires
  non-runner config keys identical to the base (`46-48`; runner keys at `31`), and
  requires each batch manifest's `source_start_rows` to equal the running offsets
  (`77-78,90-91`). If all lanes exhaust below `minimum_train`, nothing is aggregated
  (`204-206`); an existing suite short-circuits to `already_aggregated` (`142-144`).
- Merge: one unreviewed `aggregate` call (`bulk.py:207`) re-verifies policy equality,
  uniform `split_seed`/`shard_count`, ≤100 accepted per batch, non-overlapping per-lane
  scan intervals, stable source identities, group→split integrity, and collapses duplicate
  canonical IDs only when content matches modulo locator
  (`aggregate.py:71-241` per BulkOrchestration's reading). The result records
  `quality_status = "not_reviewed"` and cannot satisfy the audited gates
  (`docs/agents/training.md:491-497`).

## 7. Atomic output and error handling (`build.py`)

- Per-row failures are `Rejected(reason)` (data problem, counted at `build.py:521-525`)
  versus `StopGeneration` (system problem, aborts the run, `generation.py:20-25`); a
  partial/failed trial still publishes verified records marked incomplete
  (`build.py:324-329,532-533`).
- Dedup/integrity uses a temporary sqlite `seen`/`groups` database beside the output
  (`build.py:363-366`); a conflicting identity for a known ID raises instead of silently
  accepting (`403-409`); group→split assignments are `INSERT OR IGNORE` and a train/dev
  group overlap blocks publication (`418-419,441-442,250-253`).
- `_publish` refuses to overwrite an existing output (`242-246`), writes split JSONL plus
  `summary.json`/`manifest.json` into a same-directory temp dir, re-hashes each staged
  file before recording its SHA-256 (`308-319`), then publishes with an atomic
  same-filesystem `os.rename` — no manifest is visible before both splits exist
  (`320-321`).

## 8. Reproducibility

- Decoding is pinned (`temperature: 0`, fixed `MODEL`/`BASE_URL`/`PROMPT_VERSION`;
  `generation.py:14-16,116-119`); per-record choices use seeded keyed RNGs —
  `select` (`ufw.py:263-268`) and `_split` (`build.py:237-239`) derive from
  sha256 of seed + identity/group, so results are independent of row scheduling order.
- The manifest pins both tokenizer directory hashes, model, prompt version, thinking
  flags, and `policy_sha256` over a policy identity that includes SHA-256 of the seven
  policy modules themselves — editing a filter rule changes the policy hash
  (`build.py:34-40,43-68,291-307`).
- Caveats: `temperature: 0` does not by itself guarantee bitwise-identical vLLM outputs
  across server versions/load [INFERENCE]; resume correctness depends on the offset chain
  (§1) and unchanged source files; unreviewed bulk suites carry no audit evidence (§6).

## 9. Training consumer (contributed by Main)

- Suite load reads the entire split into memory and verifies manifest SHA-256 and record
  count before use (`src/haidass_kev_train/data/packing.py:359-386`).
- SFT loads canonical/legacy suites, runs `preflight` and materializes fixed evaluation
  views once (`src/haidass_kev_train/training/sft.py:163-187,236-246`); every epoch
  regenerates and re-encodes views in list comprehensions with a seeded order shuffle,
  then collates and runs the GPU forward sequentially — no `DataLoader` or worker
  prefetch (`sft.py:123-136,289-318`).
- Candidate sampling is keyed by seed/purpose/record-id/epoch-or-K, avoiding any
  scheduling-dependent randomness (`src/haidass_kev_train/data/canonical.py:214-226,
  257-271`); a no-network check confirmed identical candidate orders under reversed
  traversal for a 3-record sample.
- Checkpoints save via temp-dir + `os.replace` and include epoch, data cursor and full
  RNG state (`sft.py:56-66,282-291,372-374`).
- Stage 2 independently loads/encodes typed and replay streams with separate
  deterministic cursors and sequential batch handling
  (`src/haidass_kev_train/training/stage2.py:270-286,304-353`).
- The block-causal dense mask built at collate is model branch parallelism, **not** CPU
  data-loading concurrency (`packing.py:306-342`).
- Deployment throughput claims in `docs/agents/training.md:466-497` are unreviewed;
  do not cite them as observed production throughput.

## 10. Risks

1. Timed-out daemon threads keep consuming endpoint capacity; the only mitigation is
   reporting (`in_flight_requests`), not cancellation (`generation.py:74-99,174`;
   `build.py:542`).
2. Row striping over shared Parquet files multiplies per-lane decode work [INFERENCE]
   (`ufw.py:297-335`, `finemath.py:52-71`).
3. Offset-based resume rescans previously traversed rows, so a long lane chain re-reads
   its history every batch (`build.py:375-378`, `bulk.py:199-200`) [cost INFERENCE].
4. Fleet overshoot up to `shard_count × 100` records beyond `minimum_train` (§6).
5. No client-side rate limit: backpressure is delegated entirely to vLLM queueing (§5).
6. Hard-coded endpoint/model constants couple every build to one service
   (`generation.py:14-15`).
7. Bulk output is explicitly `not_reviewed` and cannot evidence source quality
   (`docs/agents/training.md:493-497`).
