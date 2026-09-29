# FineMath extraction: constrained decoding and Pydantic

Research date: 2026-09-29. Official documentation reading (vLLM, Qwen, Pydantic) plus
parent-run live probes against the deployed endpoint (§8). `[INFERENCE]` marks
reasoning beyond a direct source statement.

## 1. Problem framing

- Initially every generator task sent `response_format: {"type":"json_object"}`.
  Now `finemath_extract` sends task-specific `json_schema`: a strict Pydantic
  `_Located` model for labelled offsets or `_Quoted` for unlabelled answer-text
  evidence (`generation.py:102-121`, `finemath.py:16-36,195-260`). Other
  tasks retain `json_object`. The older answer-text experiment had three
  malformed retries without raw replies, so the failure stage is unknown.
- Two failure classes must stay separate:
  1. **Syntax/shape failures** — not valid JSON, or JSON not matching the required
     object shape. Addressable server-side by constrained decoding.
  2. **Factual/grounding failures** — valid JSON of the right shape whose span values
     do not point at real problem/answer text in the source. **No decoding mechanism
     fixes this**; `_checked` in `finemath.py:82-138` remains the enforcement layer,
     and the extraction task is already source-offsets-only (`finemath.py:141-142`).

## 2. vLLM options for schema-constrained chat responses

### 2.1 `response_format` with `json_schema` (OpenAI-compatible)

- Current vLLM docs show chat completions accepting
  `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`
  with a schema taken directly from `Model.model_json_schema()`
  (https://docs.vllm.ai/en/latest/features/structured_outputs/, "Online Serving"
  section). The vLLM structured-outputs *example* page additionally shows
  `"strict": True` inside the `json_schema` object
  (https://docs.vllm.ai/en/stable/examples/features/structured_outputs/).
- **Version caveat:** older vLLM did not document this form. The v0.6.1-era docs
  describe `response_format` as supporting only `json_object`/`text`, with
  schema-constrained output done via the legacy `guided_json` parameter
  (https://docs.vllm.ai/_/downloads/en/v0.6.1/pdf/; guided-era page at
  https://docs.vllm.ai/en/v0.6.5/usage/structured_outputs.html). The deployed server
  reports version 0.23.0 and accepted the `json_schema` + `strict` form in probes
  (§8), so this caveat is resolved in favor of the new API.

### 2.2 `structured_outputs` extra parameter (vLLM-native)

- The legacy `guided_json` / `guided_regex` / `guided_grammar` / `guided_choice` /
  `guided_whitespace_pattern` / `structural_tag` fields were **removed in vLLM
  v0.12.0** and replaced by `structured_outputs`, sent as an extra body parameter,
  e.g. `extra_body={"structured_outputs": {"json": schema}}`
  (https://docs.vllm.ai/en/latest/features/structured_outputs/, deprecation warning).
- This implies a direction split: **old servers** understand `guided_json` but maybe
  not `json_schema` response format; **new servers (≥ v0.12.0)** reject `guided_*`
  fields. The portable choice, if supported, is the OpenAI `response_format` form.

### 2.3 Constraining backend

- Structured outputs are on by default in the OpenAI-compatible server; the backend is
  `auto` (xgrammar or guidance/llguidance), selectable via
  `--structured-outputs-config.backend`
  (https://docs.vllm.ai/en/latest/features/structured_outputs/).
- Regex dialect and JSON Schema keyword coverage depend on the backend [INFERENCE from
  the backend-dependent regex note in the same page]; keep the schema to plain
  `type`/`properties`/`required`/`additionalProperties` rather than exotic keywords.

### 2.4 Reasoning models combined with structured output

- vLLM documents structured output with reasoning models; reasoning text is separated
  into the `reasoning` field (previously `reasoning_content`) and the JSON constraint
  applies to `content` (https://docs.vllm.ai/en/latest/features/structured_outputs/,
  "Reasoning Outputs"; https://docs.vllm.ai/en/latest/features/reasoning_outputs/).
- The Qwen3 series uses parser name `qwen3` and is listed with structured output
  support `json`, `regex`; its reasoning is enabled by default and disabled via
  `chat_template_kwargs: {"enable_thinking": False}`
  (https://docs.vllm.ai/en/latest/features/reasoning_outputs/, supported-models table
  and note).
- **Caveat (v0.11.2+):** with reasoning enabled, structured outputs can become
  disabled if reasoning content is not parsed into the separate `reasoning` field;
  vLLM documents `--structured-outputs-config.enable_in_reasoning=True` to force both
  together (https://docs.vllm.ai/en/latest/features/structured_outputs/, note after
  the reasoning example). On the deployed server, probes returned schema-constrained
  `content` with `enable_thinking=True` (§8), so the constraint holds in thinking
  mode there; whether the server sets `enable_in_reasoning` is not observable from
  the client [INFERENCE that it is unnecessary given the observed behavior].

### 2.5 Thinking budget and truncation

- vLLM exposes a per-request `thinking_token_budget` sampling parameter plus
  server-side `--reasoning-config` boundary strings; when the budget is hit, vLLM
  forces the reasoning end token. Without it, reasoning is bounded only by normal
  generation limits (https://docs.vllm.ai/en/latest/features/reasoning_outputs/,
  "Thinking Budget Control"). `max_completion_tokens` therefore has to cover
  reasoning *and* the final JSON; exhaustion yields a truncated reply rather than
  `finish_reason == "stop"` [INFERENCE], which the generator already rejects
  (`generation.py:159-161`). Server support for `thinking_token_budget` is
  version-dependent; not in the current request shape.

## 3. Qwen thinking-mode controls

- `enable_thinking` is a chat-template argument (hard switch) for hybrid Qwen3
  models; `/think` and `/no_think` prompt tags are the soft switch
  (https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html).
- Qwen's own vLLM deployment notes caution that `chat_template_kwargs` is not part of
  the standard OpenAI API and framework handling may differ
  (https://qwen.readthedocs.io/en/v3.0/deployment/vllm.html).
- The switch applies only to hybrid Qwen3 models; `Qwen3-*-Instruct-2507` is
  non-thinking-only and `Qwen3-*-Thinking-2507` thinking-only
  (https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html). The served
  checkpoint `qwen3.8-27b` must be verified as hybrid before assuming both modes work.
- Qwen recommends different sampling settings per mode (thinking: temperature 0.6,
  top-p 0.95, top-k 20; non-thinking: temperature 0.7, top-p 0.8, top-k 20)
  (same quickstart page). The pipeline pins `temperature: 0` for reproducibility
  (`generation.py:116`) — a deliberate deviation, noted for the record.

## 4. Pydantic: what it can and cannot do here

- `model_json_schema()` returns a jsonable dict compliant with JSON Schema Draft
  2020-12 / OpenAPI 3.1 (https://docs.pydantic.dev/latest/concepts/json_schema/).
  This is exactly the dict vLLM's docs feed into `response_format` (§2.1), so the
  model definition is the single source of truth for both request schema and local
  validation. Local pyproject already pins `pydantic>=2.0`; no new dependency.
- `model_validate_json` parses and validates in one step (jiter-backed since v2.5.0),
  honors `ConfigDict(strict=True)`, and raises `ValidationError` on type/shape
  mismatch (https://docs.pydantic.dev/latest/concepts/json/). Partial-JSON salvage via
  `pydantic_core.from_json(..., allow_partial=True)` exists but requires v2.7.0+ and
  defaults on all fields to be reliable (same page); it rescues truncated replies, it
  does not prevent them.
- **Semantic limits:** Pydantic rejects bad replies after generation; schema
  decoding constrains their format. Neither proves that a returned span points
  to the intended question/answer. `finemath._checked` still validates original
  source offsets, answer labels, figure dependencies and solution leakage
  (`finemath.py:95-151`); no schema can certify mathematical truth.
- The generator retains its duplicate-key JSON guard before Pydantic validates
  the parsed object (`generation.py:36-42,152-169`). In a local check,
  `model_validate_json` by itself accepted `{"complete":false,"complete":true,...}`
  and kept `true`; strict mode did not reject that ambiguity.

## 5. Concrete request shape for the answer-text pilot (probed)

```json
{
  "model": "qwen3.8-27b",
  "messages": [
    {"role": "system", "content": "Extract the question and its existing final answer, quote evidence exactly. Return only JSON."},
    {"role": "user", "content": "Question: What is 2 + 2? Solution: 2 + 2 = 4. Final answer: 4."}
  ],
  "temperature": 0,
  "max_completion_tokens": 512,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "source_extract",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "complete": {"type": "boolean"},
          "question_text": {"type": ["string", "null"]},
          "answer_text": {"type": ["string", "null"]},
          "answer_evidence": {"type": ["string", "null"]},
          "reason": {"type": "string"}
        },
        "required": ["complete", "question_text", "answer_text", "answer_evidence", "reason"],
        "additionalProperties": false
      }
    }
  },
  "chat_template_kwargs": {"enable_thinking": true}
}
```

The five-field answer-text pilot schema is now also used for original FineMath
pages without `Answer:` labels, but only after a cheap printed-answer cue
(`\boxed` or `(final) answer is`) and with exact source substring checks
(`finemath.py:195-223`). The labelled path separately uses required nullable
question/answer spans and optional givens (`finemath.py:16-24`).
`guided_json` is not needed on this server.

## 6. Verdict

- **Yes for syntax/shape:** schema-constrained decoding on the deployed server
  addresses not-JSON, wrong-keys and wrong-types replies. Truncation, HTTP
  errors, incomplete final replies, and runtime failures remain possible.
  Pydantic can generate the request schema and strictly validate the response;
  it is already a dependency. The prior three malformed retries cannot be
  classified from the saved experiment records alone (§8).
- **No for grounding:** constrained decoding enforces the *shape* of JSON, not
  source fidelity. The first disjoint paired A/B (§9) saw no format failures in
  either arm and the same unsafe accepts in both. Grounding checks remain
  required, and this small box-enriched sample cannot measure corpus yield.

## 7. Probe checklist (for the live endpoint)

1. Server vLLM version / feature negotiation: does `response_format` json_schema
   return 200 with constrained content, or a 400 configuration error?
2. Is `strict: true` honored or silently ignored (unsupported keyword rejected vs.
   accepted-but-unenforced)?
3. With `enable_thinking=True`: is reasoning separated (`reasoning`/`reasoning_content`
   field) and is `content` still schema-constrained — or does the server need
   `--structured-outputs-config.enable_in_reasoning=True`?
4. Truncation behavior at the configured `max_completion_tokens` with thinking on
   (finish_reason observed), and whether `thinking_token_budget` is accepted.
5. Confirm the checkpoint is a hybrid Qwen3 (both thinking modes) per §3.

## 8. Live probe results (Main, 2026-09-29)

- POST `/v1/chat/completions` with
  `response_format={"type":"json_schema","json_schema":{"name":"source_extract","strict":true,"schema":<5-field required object, additionalProperties:false>}}`
  returned HTTP 200 with `finish_reason=stop` and schema-matching JSON under
  both thinking and non-thinking mode. This confirms support for this `response_format`
  shape and a structured final `content` on the tested requests. Acceptance of
  `strict: true` does not independently prove enforcement of the flag; the
  reasoning field was not recorded. Probe items 4–5 remain open.
- On a previously problematic FineMath shard0 row125 (thinking on), one
  constrained request returned `{"complete": false, ..., "reason": "non_scalar"}`
  in 32 completion tokens. This is a successful request on that source, not an
  A/B estimate of the schema's effect on the prior retries.
- A synthetic forced-enum request returned the enum marker even when the user
  message asked for `{}` — the constraint holds against prompt pressure.
- **Do not overread:** a handful of successful probes is not evidence of a zero
  failure rate, and the raw replies behind the three prior `malformed_response`
  retries were never logged, so the cause of those failures (shape vs. truncation vs.
  other) cannot be attributed retroactively.

- `GET /version` reports `{"version": "0.23.0"}` (as returned by the server; not
  independently corroborated). The working `response_format` form is sufficient;
  there is no reason to add legacy `guided_json` on this deployment. The observed
  response does not establish which server-side reasoning switches are configured.
- Pydantic v2 `model_json_schema()` output (nullable fields as `anyOf`, `required`,
  `additionalProperties: false`) worked as the request schema with thinking on —
  confirming the single-source-of-truth pattern of §4 end to end.
- **Grounding counter-example:** on the synthetic source
  `Question: What is 3 + 1? Answer: 4.` the constrained reply was schema-valid yet
  fabricated `answer_evidence = "The answer is directly stated in the quoted answer
  as '4.'"` — a string that is not a source substring. This is the sharpest observed
  illustration that schema-constrained decoding guarantees shape, never grounding:
  span/evidence checks like `_checked` (`finemath.py:82-138`) remain load-bearing.

## 9. Implemented path and disjoint source A/B (2026-09-29)

- `Generator.ask` accepts an optional task schema. The initial FineMath
  labelled offset path used `_Located.model_json_schema()` once at import,
  plus `model_validate()` after duplicate-key-safe JSON parsing. A later
  unlabelled source-answer path uses `_Quoted` with the same guards.
  `finish_reason == "stop"` and `_checked` remain enforced. A real service
  call returned a well-formed but incorrect offset pair for a synthetic
  labelled question; `_checked` rejected it as `invalid_source_location`.
  Structured output does not repair misplaced offsets.
- Independently of the 13 prior tuning pages (shards 0–3), selected the first
  five raw pages in each of shards 4–7 at row ≥1000 with `\boxed` and
  ≤5500 characters: 20 pages, deliberately box-enriched, not a uniform
  FineMath sample. Compared the same optimized five-field locator prompt,
  model `qwen3.8-27b`, `temperature=0`, thinking on, 1024 output tokens,
  up to three attempts per page, using `json_object` versus Pydantic-generated
  `json_schema`; arm order alternated by page.

  | Arm | Pages | Requests | Exhausted/invalid format | Claimed answers | Invalid exact quotes | Other wrong accepts |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | JSON object | 20 | 20 | 0 | 14 | 5 | 2 |
  | JSON schema | 20 | 20 | 0 | 14 | 5 | 2 |

  Each arm used 24,727 tokens and made the same accept/refuse decisions.
  The five invalid quotes included rewritten LaTex formatting, a paraphrased
  question, and an option letter. Of the nine quote-valid accepts, one relied
  on a diagram (`shard7:1001`) and one source had missing question terms
  (`shard5:1038`), leaving seven plausible source-grounded numeric candidates
  per arm, **not mathematically verified or human-audited**. In this set,
  unsafe accepts were 7/14 claimed answers in *both* arms, not a statistical
  estimate of the full corpus; quoted-source validation rejected the five
  literal mismatches but not the two semantic gaps.

## 10. Quick-training policy and bounded source yield (2026-09-29)

- A deterministic random-row profile sampled 500 rows each from FineMath
  shards 0, 1, 16, 31, 32, 47, 48 and 63: 4,000 / 6,699,493 rows.
  It found 63 line-anchored answer labels, 23 boxed expressions (none with
  those labels), and 197 occurrences of `answer is`. These are *candidate
  cues*, not verified numeric question/answer pairs.
- The new unlabelled path keeps the optimized thinking-on prompt and
  Pydantic-generated five-field schema; it anchors unique exact original
  question, answer-evidence and answer-text substrings. Existing figure,
  prohibited-conversion, solution-leakage, answer-type and distractor checks
  remain. It does not guarantee that a source question contains every
  necessary condition; quality assessment is explicitly postponed.
- A bounded real build on shard 4 after row 1,000 scanned 1,374 source rows
  and used 80 Qwen requests in 98.578 seconds. It accepted one math
  candidate with raw spans (row 1,658, answer `312`), stopped on the
  attempt cap, and produced an incomplete batch — **not** an eligible input
  to the 30k merge. The record is under
  `data/processed/finemath-unreviewed-sampling-shard4-after1000/`.
  One data point cannot establish corpus-wide acceptance.
- At that local 1/80 request yield, 40 math records per 100-record batch
  would require ~3,200 math requests before other sources and breach the
  hard 2,000-request limit. The quick-training config instead requests
  47 UFW-en / 48 UFW-zh / 5 FineMath, uses 64 non-striped math shard lanes
  and 1,024 completion tokens. This trades math coverage for obtaining
  an unreviewed initial training suite; it is **not** a quality pass.

- A full three-source lane-0 batch using that config subsequently completed
  100/100 records (5 FineMath, 47 UFW-en, 48 UFW-zh) from 2,029
  scanned rows. It used 382 Qwen requests and 711.184 seconds; the
  frozen train/development split is 98/2. All five accepted math records
  passed the programmatic candidate check; this is **not** independent
  mathematical or human review.

- The final unreviewed suite contains 35,619 distinct train and 1,881
  development records from 375 complete batches: train FineMath / UFW-en /
  UFW-zh = 1,783 / 16,720 / 17,116; development = 92 / 905 / 884.
  Four additional frozen batches with empty development were excluded.
  The merged source records passed manifest checksum, canonical parsing
  and disjoint train/development Source Group checks. Human/source-math
  accuracy review remains outstanding.
- Aggregation initially stopped while reading a checksum-valid batch with
  U+0085 inside a JSON string: `str.splitlines()` split a physical JSONL
  record. The shared loader now separates rows only on literal LF; the
  original 96-record affected train split then loaded unchanged. The
  existing frozen batches were resumed and merged, not regenerated.
