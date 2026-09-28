# Training system

Read this before changing data rendering, attention masks, model construction, trainable parameters, losses, augmentation, calibration, or evaluation. The full design source is `docs/chatgpt/Haidass-Kev-Train-Analysis.md`: architecture and data are roughly lines 14–1500; RLCD is roughly lines 2354–3250; the training plan and experiment matrix are roughly lines 3254–3679.

Before designing another decision-v7 SFT scheduler/budget run, typed-decisions Stage-2 objective comparison, or Kev ID/OOD evaluation, read the [completed experiment record](../experiments/2026-09-23-decision-v7-stage2.md) for controls, observed tradeoffs, and entry gates.

## Architecture contract

A training item contains one shared state and typed questions: `noul`, `choice`, or ordinal `score`. Pack the state and all question branches into one forward pass. The block-causal mask lets every branch attend to the state while isolating branches from each other.

Render five logical structural markers by inserting explicitly mapped existing token IDs:

| Logical marker | Existing tokenizer token | ID |
|---|---|---|
| `<|kev_state|>` | `<|object_ref_start|>` | 6 |
| `<|kev_question|>` | `<|object_ref_end|>` | 7 |
| `<|kev_option|>` | `<|box_start|>` | 8 |
| `<|kev_option_end|>` | `<|box_end|>` | 9 |
| `<|kev_decide|>` | `<|quad_start|>` | 10 |

The pinned tokenizer does **not** contain the literal Kev spellings as single tokens.
The user selected reuse of existing special-token IDs rather than vocabulary expansion.
These five visual/reference delimiters are repurposed for this text-only Decision Model;
do not use chat/vision rendering or allow their literal spellings inside input text.
Insert mapped IDs directly and persist both logical names and tokenizer spellings in artifacts.

The primary model is the Haidass Qwen3 backbone with its language-model head removed, LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, and a pointer head mapping hidden size 576 to dimension 256. `training_mode="full"` instead trains the complete backbone and pointer head without PEFT. Both modes expose `DecisionModel.pointer_head` and score options from the `<|kev_decide|>` and `<|kev_option_end|>` representations with scaled dot products; neither generates text.

Use PyTorch SDPA with an additive `[B, 1, L, L]` block-causal mask. Phase 1 uses this path exclusively because flash-attn, xformers, bitsandbytes, DeepSpeed, and FSDP do not provide the required arbitrary branch mask. Run forward and backward under BF16 autocast while keeping FP32 master weights for LoRA, the pointer head, and special-token embeddings; keep the model's stored parameters out of permanent BF16 casting.

## Decision Model interface

`PackedDecisionBatch` carries `input_ids [B,L]`, reset-aware `position_ids [B,L]`, BF16 additive `attention_bias [B,1,L,L]`, `decide_positions [B,Q]`, `option_end_positions [B,Q,K]`, `question_mask [B,Q]`, `option_mask [B,Q,K]`, optional FP32 `target_probs [B,Q,K]`, and CPU metadata (`record_id`, `question_name`, `question_type`, `src`, `group_id`, `variant`). Hard labels become one-hot `target_probs`; soft targets remain intact.

State positions increase normally. Every question branch restarts at the state length. Question branches remain isolated from one another, but options inside a question stay in one causal sequence: later options may attend to earlier options and keep sequential positions. Do not enable Kev's `option_isolation`/shared-option-position mode for the primary run; its exact permutation invariance produced no gain at 0.6B and reduced 4B transfer accuracy to 0.729, a significant 5.8-point loss. A branch attends to the full state and its own causal history; state tokens never attend to branches. The model supports the backbone limit of 4096 tokens. Kev v4/v6 packing retains `max_packed=2048` and rejects overflow instead of truncating.

`DecisionModel.forward(batch)` returns only BF16 option logits shaped `[B,Q,K]`. Invalid options of valid questions are `-inf`; invalid question rows are zero and excluded by `question_mask`. `option_distribution(logits, option_mask, temperature=1)` casts logits to FP32 before masking, temperature scaling, and softmax.

Objectives remain outside the model. CE, proper losses, and RLCD return per-question values `[B,Q]` plus a valid mask; the shared reducer averages all valid questions equally. The model never selects an objective or returns a scalar loss.

## Construction order

Preserve this order for the LoRA arm; the full arm keeps the directly attached backbone/head trainable instead of applying PEFT:

1. Resolve the five mapped tokenizer spellings above, assert each encodes as its expected single existing ID, and keep the logical Kev names separate from tokenizer text.
2. Keep tokenizer length and the input embedding matrix unchanged; do not add tokens or resize embeddings.
3. Load `AutoModelForCausalLM` with explicit FP32 weights and retain `model.model` as the Qwen3 backbone, removing the language-model head from the Decision Model.
4. Attach the FP32 PointerHead to the backbone before PEFT wrapping.
5. Apply PEFT LoRA with rank 16, alpha 32, dropout 0.05, the seven projection targets, the five reused IDs in `trainable_token_indices`, and `modules_to_save=["pointer_head"]`.
6. Put LoRA parameters, PointerHead parameters, and the five reused token rows in the optimizer.

Completion criterion: tokenizer length and embedding shape remain unchanged; every trainable group receives gradients and an optimizer update; save/reload preserves token IDs and decision outputs. In LoRA mode, non-selected embedding rows remain unchanged and adapted rows are restored exactly. Rebuild and attach the PointerHead before `PeftModel.from_pretrained`; otherwise PEFT can silently drop its saved weights.

## Artifact and checkpoint contract

A Decision Model Artifact contains `decision_model.json` and mode-specific FP32 weights. Existing LoRA v1 artifacts remain PEFT-native adapter directories; full artifacts declare `training_mode="full"` and store the backbone and pointer head in `model.safetensors`. The manifest pins the base revision, tokenizer, five marker mappings, architecture, and dtype. Loading rejects mismatched manifests, incomplete states, or non-FP32 weights. Evaluation loads freeze every parameter; training loads restore the mode's trainable set. Artifact hashes cover the relevant mode-specific files.

A Training Checkpoint additionally stores optimizer, scheduler, scaler, global step, data cursor, and Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states. Resume requires an exact resource and training-configuration match so baseline branches can start from the same SFT state.

Temperature fitting produces a separate Calibration Artifact bound to the Decision Model Artifact hash and calibration-split identity. It never mutates the model artifact.

## Training contract

Stage 1 is supervised decision training with cross-entropy over the pointer distribution. Preserve full soft targets when present; hard labels become one-hot targets.

Stage 2 inherits the complete Stage 1 checkpoint and optimizes supervised CE plus Laya-style RLCD and Stage 1 replay. RLCD perturbs logits with Gaussian noise and uses a group-mean-baseline score-function estimator; it is not token rollout or PPO.

Fit temperature only after training on an independent calibration split. Report raw and temperature-scaled probabilities separately.

Compare objectives from the same SFT checkpoint with the same data, trainable parameters, replay policy, and update budget:

- A: SFT checkpoint only.
- B: continued CE.
- C: CE plus directly differentiated proper-scoring loss.
- D: CE plus RLCD.

## Data invariants

- Permute `target` with its options. A changed option order with a stale target is invalid.
- Adding or removing options requires a newly defined target distribution.
- Do not apply arbitrary option permutations to ordinal `score` questions.
- Keep all questions and augmentations from one case in the same split.
- Use an independent calibration split; training data cannot also calibrate temperature.
- Report accuracy, NLL/Brier, ordinal RPS/MAE, option-reordering stability, and Stage 1 capability retention.
- State, instructions, and criterion descriptions may be structured JSON. Render nested fields
  faithfully rather than assuming every description or instruction is a string.

## Resources

Pinned Hub revisions and local destinations are authoritative in `configs/resources.toml`. Download them with `scripts/download_resources.sh`.

The next-round frozen suite is `data/raw/kev-suites/v7/decision-v7/`; `v4/transfer-v4/` remains evaluation-only. The earlier v4/v6 suites remain available for existing experiments. Verify JSONL files against each suite's `manifest.json`. Soft-label workflow data is under `data/raw/typed-decisions/`.

## Running Stage 1

Use `uv run --no-sync python -m haidass_kev_train.training.sft --config configs/training/decision-v7.toml --output artifacts/checkpoints/decision-v7`.
The executable `haidass-kev-train` invokes the same trainer. The full-backbone arm uses
`decision-v7-full.toml`; the no-augmentation control uses `decision-v7-unaugmented.toml`.
These presets hold the v7 source selection fixed. Typed-decisions is not mixed into Stage 1.

Online augmentation is keyed by seed, epoch, and record ID, not global RNG or traversal order.
Choice permutations preserve keyed/list soft targets. Added distractors receive zero target
mass; removed mass transfers to explicit none. None-present/absent siblings retain group
identity. Ineligible binary choices skip correct-none replacement; ordinal score order is
unchanged. All switches off plus `shuffle=false` returns an unchanged baseline.
Epochs rebuild augmented encodings; the fixed update budget is not a promise of an exact
number of raw-data passes. Question-equal CE normalization spans the whole accumulation group.

`scripts/check_adaptation.py` checks real CUDA gradients, frozen rows, branch isolation and
artifact round trips. BF16 GEMM rounding changes with packed shape: test same-shape sibling
perturbations for exact isolation and use FP32 for packed-versus-separate equivalence.
`scripts/profile_adaptation.py` measures actual median/maximum packed lengths; it does not
claim throughput or memory bounds for unseen lengths. `scripts/check_resume.py` compares
interrupted and uninterrupted pilot runs, including optimizer, scheduler, RNG and data cursor.

The pilot config is `configs/training/pilot.toml`. `--stop-after N` pauses after N additional
updates without changing the training budget. Resume with the same config/output and
`--resume <checkpoint-directory>`. Checkpoint writes are atomic; existing checkpoint directories
are never overwritten. Resume rejects resource, configuration and runtime-policy changes.
BF16 needs no loss scaler; its checkpoint entry is explicitly `None`. CUDA determinism and the
cuBLAS workspace policy are fixed for reproducible resume, with FP32 master parameters.

`learning_rate` controls the backbone/adapters; `head_learning_rate` controls the pointer head.
Logs record each group's effective LR and pre-clip gradient norm. Changing a budget or LR
requires a new run, never a modified resume configuration. To plan a finite comparison:

`scheduler = "cosine"` preserves the warmup-plus-cosine schedule; `warmup_steps` must remain
below the update budget. `scheduler = "onecycle"` uses PyTorch `OneCycleLR` with the backbone
and head learning rates as `max_lr`, `total_steps = max_steps`, and `pct_start = 0.1`; its
documented defaults supply cosine annealing and momentum cycling. Set `warmup_steps = 0`
because OneCycle owns its warmup. Scheduler state is checkpointed and restored on resume.
`decision-v7-onecycle.toml` is the scheduler-only v7 comparison preset.


```bash
uv run --no-sync python -m haidass_kev_train.training.experiments \
  --config configs/training/decision-v7.toml --output artifacts/comparisons/v7 \
  --budgets 2190 4380 --learning-rates 0.00005 --modes lora full --seeds 42 --plan-only
```

Omit `--plan-only` to execute the explicit arms serially. The comparison refuses existing
outputs and budgets incompatible with the configured warmup. Each arm records its resolved
configuration, provenance, and selected development checkpoint; no test-based promotion occurs.

### Canonical dynamic-candidate input (`canonical_choice_v1`)

A Frozen Decision Suite of Canonical Decision Records enters the same trainer with
`data_format = "canonical_choice_v1"`; its absence keeps the Kev record path unchanged.
Train and development JSONL store canonical records (`source`/`state`/`question`/`gold` plus
exactly five distinct `distractors`, `_meta` carrying stable `id`, `group_id`, matching
`source`, a `source_ref` Source Trace (shard path, row locator, raw-text SHA-256, and
zero-based half-open question/answer spans), and the Candidate Validation Path in
`validation`; see `haidass_kev_train.data.canonical`), never fixed position labels. The
loader reuses manifest SHA-256/count verification and checks train/development group
integrity. Each epoch materializes one Decision View per canonical (K=2..6 from
`[canonical] k_probabilities`, default 0.10/0.20/0.30/0.25/0.15; K−1 distractors without
replacement, gold retained, full shuffle; keyed by seed/epoch/record id, order- and
worker-independent). Development and the Training Probe evaluate five fixed views per
canonical, one per K with `<id>/k<K>` view ids and `clean` semantics, independent of epoch
and training RNG. Canonical mode performs no legacy augmentation: `shuffle` must be `false`
and every augmentation probability 0, and an explicitly conflicting setting is an error.
All six candidates are preflighted through the pinned tokenizer and encoder at the
configured `max_packed` (start new canonical recipes at 1024; old Kev 2048 presets are
unchanged); overflow or structural-marker collisions reject the record, never truncate.
Canonical content, the suite manifest, and the sampling configuration join the resume
identity: changed data or policy fails resume instead of starting a different experiment.

Canonical diagnostics add `report.canonical`: raw equal-case/equal-K accuracy/NLL,
per-source and per-K metrics, distinct view/canonical/group counts, the 29% random
baseline, and content-aligned permutation flips for at most 200 fixed cases
(two orders for K=2, three otherwise). `initialization.json` binds the initial
development report to the resume identity; `metrics.jsonl` records each source's
comparison against that initialization and distinguishes the NLL-selected best
checkpoint from the final checkpoint. Neither training probes nor permutations
select checkpoints.

```bash
uv run --no-sync python -m haidass_kev_train.training.sft --config <canonical-config.toml> --output <output>
```

### Offline UFW and FineMath builder (tickets #23–26)

The public library entry is `haidass_kev_train.data.build.build(config: dict, output: Path | str) -> dict`.
It streams original UFW `ultrafineweb_{en,zh}_l3/qa/*.parquet` and FineMath-4+
`*.parquet` (original `text`/`url`/`snapshot_type` rows) locally; it publishes
`train.jsonl`, `development.jsonl`, `summary.json`, then `manifest.json` with
one atomic directory rename. Load canonical records with `load_canonical_suite(path, split)`;
never point UFW at `qa_cleaned/` or `multi_style/`.

Example bounded configuration (paths must exist; choose an unused output directory):

```toml
seed = 42
split_seed = 42
target = 3
max_attempts = 12
max_seconds = 180
timeout = 30
tokenizer_path = "models/base/haidass1.5-143m"
generator_tokenizer_path = "/mnt/models/MODELS/Qwen3.8-27B"
max_packed = 1024
max_answer_tokens = 32
max_source_tokens = 8192
max_context_tokens = 32768
max_output_tokens = 512

[sources]
ufw-en = "/mnt/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/qa"
ufw-zh = "/mnt/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3/qa"
finemath = "/mnt/data/finemath-4plus/finemath-4plus"

[source_targets]
ufw-en = 1
ufw-zh = 1
finemath = 1

```

```bash
uv run --no-sync python -m haidass_kev_train.data.build \
  --config <build.toml> --output data/processed/ufw-finemath-trial
```

This command **does call** `http://110.123.0.3:8000/v1/chat/completions` with exact model
`qwen3.8-27b`; obtain authorization before running it, even with small bounds. Optional
`api_key_env = "YOUR_ENV_VAR_NAME"` refers to an environment variable, never a key in the
config/artifact. HTTP is not confidential transport; only the current row's bounded
material and screening candidates are sent, not the complete corpus. The first
real-source call requires a separately authorized minimal non-sensitive service probe
for final JSON, task-specific thinking, usage and truncation; API metadata does not verify it.
UFW location, cleanup, construction and screening explicitly disable thinking;
FineMath assisted extraction and candidate construction explicitly enable it.
The Haidass tokenizer checks all six training candidates;
the separately pinned Qwen tokenizer and its rendered chat template measure each
generator request against the context limit **including** `max_output_tokens`
reserved for the answer; no source is truncated. Their local file
hashes are frozen in the manifest. The Qwen path shown above is the local default;
set it explicitly elsewhere, with its tokenizer and chat template present. Default
ceilings are 100 accepted, 2000 requests **including failures/retries**, and four hours;
CLI smaller bounds cannot increase those trial ceilings. HTTP bodies are bounded to
1 MiB and each attempt has a total wall deadline. A timed-out daemon request may
continue remotely; `in_flight_requests` reports still-running local attempts.

UFW bounds the raw original content before parsing or assisted dispatch and
recovers original short-answer and original-choice questions deterministically
first. Ambiguous structure can request `ufw_locate`; assisted character spans,
raw text, field attribution and source hash are checked before selecting one
cheap eligible QA per row by seed and stable identity. The selected QA cannot fall back
to a different QA on rejection. Unambiguous old option letters map to the original
option text, with `mcq` and original `option_spans` retained in Source Trace;
only presentation count/list wording can be removed. When necessary,
`ufw_cleanup` can request a presentation-only rewrite, checked against allowed
substitutions. Both assisted calls consume the same finite request/time budget as
candidate generation and screening. `assisted_location`,
`answer_mapping_ambiguous`, and `presentation_conversion` denote distinct
rejections. Screening remains model judgement, not proof or human certification.

FineMath extracts at most one labelled existing problem and final answer per webpage;
original character spans, raw-text hash, URL and snapshot identify and trace it.
A nonempty pre-question prefix must be retained as original-text givens or
rejected; worked solutions, including unlabelled derivations after the
question, do not enter model-visible state or question. Source gold is
preserved, not regenerated or mathematically proved.

The bounded programmatic checker accepts exact signed integers, finite decimals,
scientific notation, fractions, percentages, short case-sensitive identifier
expressions with parentheses, arithmetic, bounded integer powers and square
roots, and explicit finite solution sets (`x=±2`, `{-2,2}`). `x=...` is
interpreted as an answer value, never a request to solve for missing gold.
Finite sets remain whole answers: one root is not a substitute for both.
An explicit finite set among scalar candidates (in either direction) is an
answer-type mismatch rejected before adjudication, even when values differ.
Fixed-ratio units are mm/cm/m/km, mg/g/kg, ms/s/min/h and mL/L, including
squared/cubed length and finite compound dimensions such as m/s and km/h.
The implementation bounds answer length, parser depth/nodes, exponents and
calculation; arbitrary executable code and unrestricted symbolic parsing are
not allowed. Exact known equivalence among any of all 15 candidate pairs,
dimension mismatch, hostile/out-of-scope syntax and prohibited conversions
reject without an LLM call. Expressions needing unstated domain assumptions,
undefined division or square-root branch claims are not silently equated or
declared distinct. Absolute temperature, exchange rates, month lengths,
missing figures and necessary conditions remain excluded.

If any pair is undecided but remains inside the admitted scope, a *single*
independent `finemath_adjudicate` task sends original problem conditions,
source gold and all six candidate texts to the same pinned model with
thinking enabled. It does not send construction self-assessment. Strict final
JSON must explicitly approve answer-type validity and non-equivalence for
each of the 15 pairs; rejection, uncertainty, equivalent/unknown pair, or
invalid output after at most two same-task retries rejects the sample.
Service/configuration failure stops the trial under the shared attempt/time
budget, not as bad source data. Records needing this path bear
`finemath_llm_adjudicated`, rather than `finemath_programmatic`. The
summary records `validation_paths` (accepted by path), `unknown_cases`,
`program_rejected` (FineMath hard/structural rejections excluding malformed
generation), `llm_accepted`, `llm_rejected`, rejection reasons and all request
attempts. Manifest `build.thinking` fixes the adjudication task policy and
`build.prompt_version` fixes prompt identity.
Neither same-model independent adjudication nor controlled HTTP verification
is a mathematical proof or source quality certification: correlated errors
remain possible and the first admitted batch needs source-grounded human
review before scaling.

The suite reports rejection/unknown/program-relation counts, attempts, source and
split distributions, and source-URL groups assigned before filtering. Grouping
preserves URL query parameters and puts snapshots of a page in one split; absent or
blank URL/snapshot identity falls back to the original raw-text group. Missing
development or target coverage stays `complete=false`, never repaired by moving
groups. This is not a semantic proof of source gold or a human data quality gate.
Neither controlled HTTP checks nor incomplete suites demonstrate real service
behavior, source quality or readiness for scaling.
`manifest.json` verifies file SHA/count even for explicitly incomplete trials.

### First-trial Data Quality Gate (ticket #28)

Only an authorized operator starts a real trial. Before sending real source rows, obtain
permission for the smallest **non-sensitive** request to the pinned endpoint/model and
observe strict final JSON, `chat_template_kwargs.enable_thinking` both off and on,
reasoning separated from final content, usage, truncation/error behavior and finite
timeout. A metadata GET or controlled-HTTP test is not that probe. The builder is
invoked with the TOML example above, changing `target = 100` and source targets to
`ufw-en = 30`, `ufw-zh = 30`, `finemath = 40`; default limits remain at most 2000
attempts/four hours. This document does not authorize making those requests.

The builder's `summary.json` is **machine trial status**, not a human quality result.
The exact SHA-256 of `manifest.json` identifies this audited batch; the verified
`manifest.build.policy_sha256` identifies the generation strategy independently of
batch-specific files, seed and authorized quantity. A subsequent larger dataset may
reuse a passing policy identity only if its strategy really matches; its file hashes
and batch manifest hash must differ. Never edit a failing batch or replace bad rows
to turn it into a passing batch.

```bash
uv run --no-sync python -m haidass_kev_train.data.quality prepare \
  --suite data/processed/ufw-finemath-trial --output artifacts/quality/first-review
# Inspect every line of first-review/review.jsonl against the original local Parquet;
# independently write first-review/assessments.jsonl, one assessment for each case.
uv run --no-sync python -m haidass_kev_train.data.quality report \
  --suite data/processed/ufw-finemath-trial \
  --review artifacts/quality/first-review/review.jsonl \
  --assessments artifacts/quality/first-review/assessments.jsonl \
  --output artifacts/quality/first-review/gate.json
```

`prepare_review(suite, output)` and `quality_gate(suite, review, assessments, output)`
are equivalent Python entry points. Preparation verifies both frozen splits,
group integrity, each source Parquet row, original uid or URL/snapshot, stable ID,
raw-text SHA-256 and character spans; `review.jsonl` includes the original row
text, question/answer/givens/old-option spans, canonical question/gold/five
distractors, validation path, stable `case_id`, policy/batch identity and
`trace_sha256`. Do not generate assessments from the builder or model.
Each operator-written JSONL assessment has exactly:

```json
{"case_id":"<review.jsonl case_id>","batch_manifest_sha256":"<review.jsonl batch_manifest_sha256>","policy_sha256":"<review.jsonl policy_sha256>","trace_sha256":"<review.jsonl trace_sha256>","reviewer":"<human identity>","source_verified":true,"serious_error":false,"category":null,"reason":"","paths":[]}
```

Review source fidelity, unique gold, all five distractors and their pairwise
equivalence, answer type/format shortcuts, ambiguity, MCQ conversion and
FineMath solution leakage. Add `\"symbolic\"` and/or `\"unit\"` to `paths` **only
after** personally inspecting those FineMath cases; source and validation paths
are taken from verified records. Set `serious_error=true`, a nonempty `reason`
and `category` to one of `wrong_gold`, `correct_distractor`, `ambiguity`,
`changed_question`, `leakage`, `shortcut`, `source_mismatch`, `other` for serious
findings. A source not independently verified must have a serious verdict.
Record a merely cosmetic note in `reason` with null `category`; escalate when
meaning or candidate quality changes.

The gate report records `status` (`pass`, `fail`, `incomplete`), both identities,
`accepted`, `audited`, `required_audited=100`, `severe_count`, examples with
case locators, per-source/path coverage, `unverified_paths` and evidence paths.
No assessments, fewer than 100 accepted or missing reviews are **incomplete**
unless a serious finding already makes the audited batch fail. Only 100
individually reviewed cases covering **ufw-en, ufw-zh and finemath**, with zero
serious errors and a completed build, can pass; a completed 100-case batch
missing any of those sources is incomplete. Synthetic assessments in tests
prove status logic only, not a real human audit. Missing symbolic, unit or
LLM-path cases stay unverified, not silently counted; unlike missing sources,
these absent paths do not independently block a zero-serious gate.
Keep review/assessments/reports private and access-controlled: Parquet rows may
contain sensitive third-party text; do not upload them or credentials to issue
comments, external review services or model inputs. No real backend call,
100-case generation, human audit, overfit check or spending has been performed
or authorized by these instructions.

### Audited 128-case overfit and bounded scaling decision (ticket #29)

These are **offline preparation and read-only evidence commands**, never a
generation, GPU training, human-review, or spending authorization. First
obtain a real, independently written 100-case `quality_gate` pass against the
original Parquet and operator assessments as above. Obtain separate permission
to construct a later completed suite with enough distinct train canonicals,
using the identical `manifest.build.policy_sha256` generation strategy; its
batch manifest SHA may differ and its records need not be disjoint from the
audited batch. Freeze its train/development files, seed, source groups,
tokenizer, optimizer configuration, evaluation and finite optimizer-update
budget **before** observing training results. Do not move development records,
repeat train records, resume with a changed budget, search seeds, or extend a
failed run automatically.

```bash
uv run --no-sync python -m haidass_kev_train.training.overfit \
  --suite <later-completed-builder-suite> \
  --quality <audited-quality-report.json> \
  --audited-suite <original-100-case-suite> \
  --review <review.jsonl> --assessments <human-assessments.jsonl> \
  --base-config <reviewed-canonical-sft.toml> \
  --output <new-overfit-recipe-directory>
```

The base config must be an actual public SFT TOML with a pinned local
`base_path`, tokenizer and finite `max_steps` (at most 500), not a placeholder
path; `eval_interval = checkpoint_interval <= max_steps` ensures every probe
event has its own saved checkpoint. This command verifies the audited evidence
against the original source, same policy, frozen suite and tokenizer preflight,
then writes `suite/` with **exactly 128 distinct original train canonicals
across all three sources**, unmodified parent development, `config.toml` for
the existing SFT, and provenance `plan.json`. `probe_groups=128` covers all
train groups, yielding 640 fixed five-K probe views (not 640 independent
cases); training still samples candidates anew each epoch. The recipe retains
the supplied seed, optimizer and update budget, and does not train.

Only after explicit GPU authorization, run the existing public trainer with
`--config <recipe>/config.toml --output <new-overfit-run>`. The existing
four-update `configs/training/pilot.toml` is only an engineering smoke, not
overfit evidence. A real check requires the same saved optimizer-update
checkpoint at or below 500 where **every K=2..6 accuracy is ≥0.95 and raw
natural-log, equal-case/equal-K NLL is ≤0.15**. Missing probe rows, a smaller
probe, nonfinite gradients, identity mismatch or unfinished updates do not
grant the next stage. Diagnose labels, mapping, mask, gradients and data
without changing a running budget.

After separately authorizing an actual 1k–5k train-canonical Data Pilot,
freeze its completed builder suite, config (explicit positive `max_steps`,
seed, source selection and `development_selection = \"clean\"`), fixed
development, initial checkpoint and evaluation policy before training. The
approximately 30k full target is another separately authorized finite run;
these sizes are not automatic actions. Run the same public SFT entry point,
retaining each run's `metrics.jsonl`, `initialization.json`, `best.json`,
`config.toml`, selected `step-*` and final checkpoint. Then consume them:

```bash
uv run --no-sync python -m haidass_kev_train.evaluation.gates \
  --quality <audited-quality-report.json> \
  --audited-suite <original-100-case-suite> \
  --review <review.jsonl> --assessments <human-assessments.jsonl> \
  --overfit-suite <recipe>/suite --overfit-config <recipe>/config.toml \
  --overfit-run <actual-overfit-run> \
  --pilot-suite <actual-pilot-suite> --pilot-config <frozen-pilot.toml> \
  --pilot-run <actual-pilot-run> --output <new-decision-report.json>
```

`--full-suite`, `--full-config` and `--full-run` optionally report an actual
~30k full run. Missing stages may be omitted and remain `incomplete`; output
path must be new. Each gate has `pass`, `fail` or `incomplete` with evidence
locations. Inconsistent hashes/identity or observed threshold failures
block. The consumer regenerates the first quality verdict from original
Parquet plus human assessment files; an isolated `\"status\":\"pass\"` JSON is
not audit evidence. It verifies policy equality, derived subset provenance,
training-log update steps, stored checkpoints, frozen SFT initialization
identity and canonical diagnostic denominators. Evaluate with the same
software/runtime/resource pins as training because the initialization
identity includes PyTorch/CUDA versions and `configs/resources.toml`.
Training artifacts are trusted local files, not tamper-proof attestations.

Pilot-to-full recommendation additionally needs all quality/engineering/
overfit prerequisites, all three development sources with at least **50
independent Source Groups each**, and the existing development
**source-macro-NLL-selected best checkpoint** (not the final checkpoint
unless identical) at raw T=1 five-K equal-case accuracy ≥0.34 *per source*,
with NLL below the same persisted initialization on the identical
development. Inadequate coverage is `incomplete`, not an invitation to move
or reuse cases. Inspect the reported source/K buckets and deterministic
candidate-content-aligned permutation flips (two orders at K=2, three
otherwise, at most 200 distinct cases); no hard flip cutoff exists and stable
incorrect predictions are still incorrect. A `recommend_full` true indicates
only development learning signal and a reason to request human review and
separate resources; neither a successful pilot nor a full run establishes
calibration, statistical significance, OOD generalization, or SFT Complete.
No human audit, authorized real BF16 CUDA overfit/reload/resume, pilot or full
run has been performed by publishing this recipe.



### Optional W&B tracking

Set `WANDB_ENTITY` to the stable team name. Pass `--wandb-project <project>` to
the Decision SFT, Stage 2, or experiment-comparison command to enable W&B;
`--wandb-name <name>` optionally names an individual SFT/Stage 2 run. A standalone
run defaults to its output-directory name; comparison arms use their distinct
arm names and share a comparison group. Without `--wandb-project`, no W&B
credentials or network are needed. Use `WANDB_API_KEY` or an existing W&B login
for online uploads.

W&B mirrors aggregate training metrics and evaluation reports, plus pre-clip
histograms of trainable backbone (LoRA or full) and pointer-head gradients only
at scheduled development evaluations. It does not upload samples, per-record
predictions, model weights, checkpoint files, or captured console output.
`metrics.jsonl` remains authoritative. The run ID is stored in `wandb-run.json`
under the training output directory and reused when resuming with the same
output, entity, and project. W&B upload errors do not stop local training.

## Evaluation Lane

At each evaluation interval, score a fixed, group-preserving probe drawn only from training
and the complete development split. Probe IDs are logged; selection is independent of input
record order. Diagnostics preserve model mode and Python/NumPy/torch RNG state, report raw
temperature-1 all/clean aggregates, task metrics, policy-pair behavior, and key-aligned Choice
reordering. Pair rates include denominators and incomplete counts; no pairs means null rates.

`development_selection` explicitly chooses `clean` or `all`; the v7 presets select the lowest
clean macro-NLL. Probe metrics never select checkpoints. `metrics.jsonl` stores `train_probe`
and `development` reports, and `best.json` points to the selected complete checkpoint.
Pausing off cadence does not introduce an extra selection evaluation.

After selecting the checkpoint, invoke `python -m haidass_kev_train.evaluation.run` with
`--artifact`, `--suite`, `--split`, `--calibration-out`, `--out`, and optionally
`--reorder-stability`. Use the same frozen suite as the training configuration, with transfer-v4 reported separately by the official comparison runner.
Fit temperature only on the manifest-verified `calibration` split; bind it to the portable
artifact hash and split identity. Raw and calibrated reports remain separate.
Option reversal preserves choice targets and excludes ordinal score questions.
Reports include accuracy, NLL/Brier, ordinal RPS and expected-index MAE, source/type breakdowns,
and choice reversal stability. Nonfinite valid logits and invalid targets are hard errors.

CPU preparation may overlap training. GPU evaluation is serialized with training on the
single 5090. Do not evaluate the locked test before checkpoint selection, or interpret a
four-update pilot's scores as SFT Complete or as evidence for RLCD entry.

## Running Stage 2

Prepare the pinned aggregate typed-decisions TRAIN file with
`uv run --no-sync python -m haidass_kev_train.training.typed_decisions --output data/processed/typed-decisions-stage2`.
The workflow-stratified, case-disjoint split is 960 training, 120 development,
and 120 calibration cases; all five questions remain together. Preparation checks
packed lengths and rejects overflow. It never reads the official test file.

Run `python -m haidass_kev_train.training.stage2 --config configs/training/stage2-b-ce.toml --parent <stage1-checkpoint> --output <empty-output>`.
Use `stage2-c-proper.toml` and `stage2-d-rlcd.toml` for the other controlled arms.
All arms retain current-data CE and separately question-normalized Stage 1 replay.
Only the additional objective differs: none, directly differentiated proper loss,
or Gaussian logits-space RLCD. The reward combines clipped log score, spherical
score and ordinal-only normalized RPS. Hard and full soft targets are supported.

`--parent` inherits Stage 1 model weights, Adam moments/step counters and global
RNG, but uses fresh Stage 2 optimizer hyperparameters, scheduler and data streams.
`--resume <stage2-checkpoint>` instead restores the exact Stage 2 optimizer,
scheduler, both data cursors, global RNG and dedicated RL-noise RNG. Parent hashes
and the transition policy are recorded. LoRA/full mode is inherited, not converted.

RLCD samples and advantages are detached; the live centered logits parameterize
the Gaussian score-function density in the valid-option K-1 subspace. The
self-inclusive group baseline has a (G-1)/G gradient scale with normalization off;
the default advantage normalization is per microbatch over eligible entries.
Padding and singleton questions contribute no RL term. A separate noise generator
keeps the data/dropout RNG stream aligned across arms.

`best.json` selects solely on raw typed-development NLL. Stage 1 retention and OOD
reports are diagnostics, not selection criteria. Fit temperature separately on
calibration, report raw metrics too, and distinguish teacher agreement from
real-world probability calibration. A technical RLCD experiment does not imply
the SFT policy-capability entry gate has passed.

`scripts/evaluate_official.py` evaluates development only by default.
`--include-locked-test` is an explicit final-promotion opt-in, never for exploration.
The focused real-CUDA handoff/resume check is
`uv run --no-sync python tests/test_stage2.py Stage2CudaIntegrationTests.test_real_handoff_singleton_logging_and_exact_resume`.
