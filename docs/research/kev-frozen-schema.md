# Frozen Kev Schema → Our Decision Record

- **Ticket:** [`Map the frozen Kev schema to our decision record`](https://github.com/HowBoring/haidass-kev-train/issues/5)
- **Evidence:** local frozen suites at `data/raw/kev-suites/v4/decision-v4/` and `data/raw/kev-suites/v6/decision-v6/` (manifests + all 31,188 JSONL lines parsed programmatically)
- **Date:** 2026-09-22
- **Planned contract compared against:** `docs/chatgpt/Haidass-Kev-Train-Analysis.md` (Kev pointer-head input format; `question_loss()` soft-target / ordinal-RPS discussion in section 七 / 落地细节 1–2)

All counts below were computed by parsing every line of every split; the numbers match the `files` blocks of both manifests exactly, which cross-validates both the manifests and this survey.

## 1. Suite inventory (manifest evidence)

`manifest.json` in both suites pins `version: 3`, base/dataset Hub revisions, `parent_files` SHA-256s, and per-file `{sha256, records, questions}`.

| suite | split | records | questions | sha256 (prefix) |
|---|---|---|---|---|
| v4 | train | 10,896 | 13,896 | `cb55b79e…` |
| v4 | calibration | 728 | 908 | `15c88976…` |
| v4 | development | 1,204 | 1,468 | `8d5765d7…` |
| v4 | test | 1,176 | 1,440 | `cd7d129a…` |
| v6 | train | 13,896 | 16,896 | `7c28800d…` |
| v6 | calibration | 908 | 1,088 | `72ba2454…` |
| v6 | development | 1,204 | 1,468 | `8d5765d7…` (identical bytes to v4) |
| v6 | test | 1,176 | 1,440 | `cd7d129a…` (identical bytes to v4) |

Verified by byte comparison: **v6 `development.jsonl` and `test.jsonl` are byte-identical to v4's** (`sha256sum` both `8d5765d7…` / `cd7d129a…`). Line counts on disk match manifest records exactly for all 8 files.

**v6 = v4 + 3 knowledge MCQ sources.** Splitting by `id` prefix: v6 train adds `arc`, `csqa`, `openbookqa` (1,000 records each) and drops 5 two-sibling `contrastive/*` pairs (10 records) that v4 train had; v6 calibration adds 60 records per MCQ source (180). Development/test are unchanged. This matches `manifest.protocol.inherited_eval = "evals/v4/decision-v4"` in the v6 manifest.

## 2. Record shape

Every line is one JSON object with exactly three top-level keys — verified constant across all 31,188 lines:

```json
{
  "state":     <string | object | list-of-message-objects>,
  "questions": { "<question_name>": {Question}, ... },
  "_meta":     { ...provenance... }
}
```

### 2.1 Shared state

`state` is the shared evidence block for all questions of the record. Three representation kinds coexist:

| kind | shape | share of records | example source |
|---|---|---|---|
| plain string | `"What 's the origin of the word \` news ' ?"` | ~62% of records | public classification rows (raw row text) |
| dict (1–2 keys) | `{"ticket": {"channel": "email", "body": …}}`, `{"document": …}`, `{"question": …}`, `{"case": …, "policy": …}` | ~34% | rendered public rows / MCQ / synthetic policy |
| list of chat messages | `[{"role": "customer", "content": …}]` (always exactly 1 message, role always `customer`) | ~4.7% | public rows in "chat-log" render |

Per-source state shapes (combined v4+v6 counts, across all splits):

- public sources (`agnews`, `amazon`, `banking77`, `boolq`, `dbpedia14`, `imdb`, `mnli`, `sst5`, `trec`, `yelp`): each mixes **all four** shapes (`str`, `list`, `{"document"}`, `{"ticket"}`) — e.g. `trec`: 1,742 str / 194 list / 384 document / 264 ticket.
- MCQ sources (`arc`, `csqa`, `openbookqa`, v6 only): only `{"question": …}` (1,060 each).
- synthetic sources (`compositional`, `contrastive`, `legacy_policy`): only `{"policy": …, "case": …}` (1,280 / 392 / 1,216 respectively).

**Loader implication:** the state renderer must be polymorphic over 4 surface forms (string, message list, `{document}`, `{ticket}`, `{question}`, `{case, policy}`) and preserve whatever text it renders — there is no single canonical serialization. `_meta.text_sha256` does **not** equal `sha256` of any obvious canonical JSON/UTF-8 encoding of the local `state` (tested raw UTF-8, default/sorted/ASCII-safe JSON variants — no match); it is a provenance hash of upstream text, so use it only for identity/provenance bookkeeping, not for validating local state bytes.

### 2.2 Questions map

`questions` is a dict keyed by a **semantic question name** (not a generic index). Full inventory of `(name, type, src)` triples across both suites:

| name | type | src | total questions (each suite) |
|---|---|---|---|
| `intent` | choice (77-way +none) | banking77 | 1,292 |
| `relation` | choice (3-way) | mnli | 1,292 |
| `answer_type` | choice (6-way) | trec | 1,292 |
| `category` | choice (14-way) | dbpedia14 | 1,292 |
| `topic` | choice (4-way) | agnews | 1,292 |
| `answer` | choice 4–5-way | arc / csqa / openbookqa (v6) | 1,060 each |
| `decision` | choice 2–4-way | composition_* (8 shapes) | 80 each |
| `decision` | choice 3-way | contrastive_quantity_limit / contrastive_spend_threshold | 192 / 228 |
| `decision` | noul | contrastive_age_eligibility / contrastive_return_window | 192 each |
| `answer` | noul | boolq | 1,220 |
| `positive` | noul | imdb | 1,220 |
| `recommend` | noul | yelp_yn | 1,220 |
| `is_scitech`/`is_sports`/`is_business`/`is_world` | noul | agnews_yn | 654/618/601/615 |
| `rating` | score (5) | yelp | 1,220 |
| `sentiment` | score (5) | sst5 | 1,220 |
| `stars` | score (5) | amazon | 1,220 |

The name is per-record unique and stable per source (e.g. every boolq question is always `answer`); our decision record should keep it as the question identity (it becomes the packed branch label).

### 2.3 Question object

Exactly two key-shapes exist (counts across v4 train: 11,079 with `criteria`, 2,817 without):

- `{type, instructions, criteria, label, src}` — for `choice` and `score`
- `{type, instructions, label, src}` — for `noul` (no criteria in 3,617 cases)

**BUT:** `noul` questions *sometimes* carry a `criteria` dict too — always exactly `{"true": <str>, "false": <str>}` (5,830 occurrences across the two suites; e.g. imdb `positive`: `{"true": "Clearly positive overall", "false": "Negative or mixed"}`). So `criteria` presence is not type-deterministic.

| field | choice | score | noul |
|---|---|---|---|
| `type` | `"choice"` (5,672 in v4 train) | `"score"` (3,000) | `"noul"` (5,224) |
| `instructions` | always str | always str | always str |
| `criteria` | **dict** `label_key → str or null` | **list** of exactly 5 strs, ordinal | absent, or dict `{"true": str, "false": str}` |
| `label` | str, always a key of `criteria` (0 violations in 14,344 choice questions across both suites) | int in {0,1,2,3,4} — **index into `criteria`, not a 1-based rating** | bool |
| `src` | fine-grained source id | same | same |

`src` is finer than `_meta.source`: `_meta.source` groups into `banking77` / `compositional` / `contrastive` / `legacy_policy` / public names, while `src` distinguishes e.g. `composition_disjunction` vs `composition_nested_or`, `contrastive_spend_threshold` vs `contrastive_age_eligibility`, and `agnews` vs `agnews_yn`. A record may mix public and synthetic `src` values under one `_meta.source`.

### 2.4 Hard targets only; ordinal semantics

- **There is no `target` / soft-distribution field anywhere in the frozen data.** Every question carries a hard `label` (str key / int index / bool). The planned soft-target CE path must therefore synthesize one-hot targets from `label` at load time; a `target` key in our own decision record is an extension, not a frozen-schema field.
- **Ordinal constraint lives in list order:** `score.criteria` is a 5-element list; `label` is a 0-based index into it. All 3,660 score questions across both suites have exactly 5 criteria and labels distributed {0: 649, 1: 810, 2: 702, 3: 838, 4: 661}. The criteria strings encode the level semantics (`"1 star: terrible experience"` … `"5 stars: excellent"` for yelp; bare `"very negative"` … for sst5; `"1 star: very negative"` … for amazon — three different renderings, see §5). Never permute score options; `choice` permutation augmentation must not be applied to `score` or to `noul` (`true`/`false` order in its criteria dict is semantic).

## 3. Provenance (`_meta`)

Keys observed (presence varies by provenance class):

| key | always? | notes |
|---|---|---|
| `id`, `group_id`, `source`, `variant`, `text_sha256` | always (all 31,188 records) | `id` unique within suite (0 cross-split `id` overlap); `group_id` groups variant/sibling records |
| `row`, `row_sha256`, `repo`, `revision`, `split` | public sources only (e.g. v4 train: 10,448 of 10,896) | `repo`/`revision` = Hub dataset + pinned commit (matches `manifest.dataset_revisions`); `split` = upstream dataset split (mostly `train`; mnli variants use `validation_matched`; synthetic rows use `split: "generated"`) |
| `pair_id`, `sibling` | synthetic + variant records | `sibling` ∈ {`a`,`b`}; contrastive pairs are minimal pairs (same policy, one changed fact, opposite label) |
| `pair_kind` | synthetic records | `relevant` / `irrelevant` |
| `family`, `family_id`, `render_style`, `certificate` | `compositional` records | `render_style` ∈ {0,1} (train/calib/dev); `certificate` = executable ground truth: `{tree, atoms[{kind,fields,threshold}], facts, order, deciding_field, label}` with atom `kind` ∈ {`eq`,`ge`,`le`,`lt`,`match`,`flag`} |
| `none_key` | variant records only | always the literal `"none_of_these"` (264 records per suite) |

No `_meta` key is a valid "always present" assumption except the first five. Cross-split leakage check: `group_id` sets are disjoint across train/calibration/development/test (0 overlap for every pair, verified in v4) — loader must group by `group_id` (or `pair_id`) if it ever splits or samples, not by record.

## 4. Variants and augmentation records

`variant` ∈ {`clean`, `none_present`, `none_absent`, `permuted`}. Non-clean records exist **only** for `agnews`, `banking77`, `dbpedia14`, `mnli`, `trec` (24 each per variant) and `contrastive` (12 each) — 132 non-clean records per suite, all in development+test (except contrastive's, test only), always sharing the `group_id` of their clean counterpart from the upstream eval row (e.g. `banking77/test/914` → `banking77/test/914/none_present` / `-none_absent` / `-permuted`).

Semantics verified on `banking77/test/914` (77-way `intent`):

- **`none_present`**: `criteria` has 78 keys — the 77 real labels **plus** `none_of_these: "None of these options describes the answer"`; gold label is the real label.
- **`none_absent`**: same question, gold label **is** `none_of_these` (the row genuinely matches none), criteria has the 77 real keys + `none_of_these`.
- **`permuted`**: same criteria set, option presentation order varies — criteria dict insertion order differs; our record must not rely on criteria JSON key order.

So the "none of the above" option is a first-class label value (`none_of_these`) that the renderer must include as an ordinary option and the head must score like any other. `noul` and `score` never get variants.

## 5. Choice criteria value conventions (mismatches to flag for the renderer)

`criteria` values are option descriptions, but their conventions differ per source:

- `banking77`, `mnli`, `agnews` variants, and `composition_*` use human strings, but **many values are `null`** — `null` means "no description; key itself is the display text". Null-valued criteria by src (per suite): banking77 1,292, dbpedia14 1,292, agnews 943, mnli 838. The mnli example in §2 of the raw peek shows `{"entailment": {...}, "neutral": null, "contradiction": str}` — values may even be **nested objects** in one observed case (v4 train row: mnli `entailment` value = `{"what": "The hypothesis follows from the premise"}`).
- `arc`/`csqa`/`openbookqa` use generic `opt_1..opt_4/5` keys with the option text as value.
- `dbpedia14`/`trec` use semantic keys with null/str values.
- `score` criteria are lists (see §2.4), with **three different string formats** (yelp vs sst5 vs amazon).

Choice criteria sizes observed (per suite): 2, 3, 4, 5, 6, 7, 14, 15, 77, 78 keys — i.e. K ranges from 2 to 78. The renderer and pointer head must handle arbitrary K with heterogeneous key styles and null/nested values; never assume `opt_N` keys.

## 6. Record↔question multiplicity

Records carry 1, 2, or 3 questions. The multi-question records are fixed composites (v4 train: 1,000 each of 2-q and 3-q):

- **2-question**: `{"rating": score(yelp), "recommend": noul(yelp_yn)}` — same record, two types.
- **3-question**: `{"topic": choice(agnews), "is_scitech": noul, "is_business": noul}` (also `is_sports`/`is_world` variants) — a choice plus two yn probes on one article.

The packed forward pass must support mixed types and heterogeneous K per record, with per-question loss heads consuming one shared state and isolated option branches. Q-counts per split (v4): train {1: 8,896, 2: 1,000, 3: 1,000}; development {1: 1,032, 2: 80, 3: 92}; test {1: 1,004, 2: 80, 3: 92}; calibration {1: 608, 2: 60, 3: 60}. v6 same except train {1: 11,896, 2: 1,000, 3: 1,000}.

## 7. Split semantics

| split | role (from manifest `protocol`) | notes |
|---|---|---|
| train | `public_train_pool` sampling + exactly one synthetic arm | `arm_selection: "train_sources selects public sources plus exactly one synthetic arm"`; `synthetic_records_per_arm: 448` (the `composition_*` 56×8 = 448 + `contrastive_*` 112×4 = 448) |
| calibration | shared; stratified by family and group | temperature-scaling / calibration data — must stay disjoint from training samples (per plan §六.4) |
| development | macro development NLL = primary metric | contains the none-variant/permuted pairs |
| test | locked; byte-identical across v4/v6 | `legacy_test: "Inherited v2 locked test bytes retained without inspecting examples"` |

Context budgets in `manifest.context`: `max_state 384, max_branch 1024, max_packed 2048, truncate: false` — the renderer must pack (state + question + options) branches within these limits, and because `truncate: false`, overflow is a hard error, not a silent truncation.

## 8. Comparison to the planned contract (`Haidass-Kev-Train-Analysis.md`)

The planned interface (`x = (S, Q, {O_i})`, `z = f_θ(x) ∈ R^K`, pointer head over options, CE with optional soft target, RPS for ordinal) matches the frozen schema with these concrete deltas:

1. **Soft target:** planned `target` field does not exist in the frozen data (§2.4). Loader: derive one-hot from `label`; keep `target` as an optional extension field for our own data. The analysis doc's warning that Kev's `question_loss()` short-circuits soft-CE before the ordinal-RPS branch applies to our implementation too — implement `soft-CE` and `ordinal RPS` as one unified loss path, not an early-return if/else.
2. **Options:** the analysis doc's example shows options rendered inline from criteria. Frozen reality: criteria **values** may be `null` (§5) — when null, render the **key** as the option text; there is no separate options field. `none_of_these` participates as a normal option (§4).
3. **Ordinal:** `score.label` is a 0-based list index (§2.4) — convert to RPS with `K = len(criteria) = 5` always; never shuffle score options.
4. **Type↔criteria coupling:** `criteria` presence does not imply `choice` (noul sometimes has `{"true","false"}` criteria) (§2.3). Branch on `type`, not on criteria presence.
5. **State polymorphism:** 4 state forms (§2.1) — the analysis doc's single `<state>` block example (§ "3. 关键点") assumes stringified state; our renderer needs per-form serialization rules.
6. **Question identity:** keep `questions`' semantic key names as branch identity (§2.2) — they are stable per source and disambiguate multi-question records.
7. **Grouping:** any resampling/eval aggregation must group by `group_id` (variants/siblings share it; it never crosses splits) (§3, §4).
8. **Certificate:** `compositional` records carry an executable `certificate` — we can verify labels programmatically (tree/atoms/facts) if we ever regenerate or sanity-check the synthetic arm (§3).

## 9. Edge cases the renderer/loader must preserve

1. State can be a JSON list of chat messages (role always `customer`, always length 1) — must not be coerced to string by `str()`.
2. `criteria` value may be `null` (render the key), a string, or (one observed mnli case) a nested object `{"what": …}` — normalize defensively.
3. `none_of_these` is a label value; do not filter it from criteria.
4. `score` labels are 0-based indices into the criteria list; criteria order is semantic (ordinal).
5. `permuted` variant relies on criteria dict order — never assume insertion order; look up labels by key.
6. Variant/contrastive records share `group_id`/`pair_id` with clean counterparts — keep them out of training if their clean sibling is in eval, and vice versa; grouping keys are the only safe join.
7. `_meta.text_sha256` is provenance-only (§2.1); do not use it to validate re-serialized state.
8. mnli variant rows carry upstream `split: "validation_matched"` while their suite file is development/test — `_meta.split` ≠ suite split; suite split is the filename only.
9. Mixed `src` values under one `_meta.source` (e.g. `source: compositional` with `src: composition_disjunction`) — metrics per-arm should use `src`, suite-level grouping uses `source`.
10. 77-way banking77 means option enumeration up to K=78 with `none_of_these`; packing must fit `max_packed 2048` with `truncate: false` — assert, don't truncate.
11. `context.truncate: false` in the manifest means the frozen data fits the budgets; our tokenizer/renderer must reproduce the budgets or fail loudly.

## 10. Recommendation for the Decision Model interface ticket

Adopt this internal decision record (one-to-one with frozen JSONL, plus extensions marked +):

```text
DecisionRecord:
  state:      State                      # str | {document|ticket|question} | {case, policy} | [message]
  questions:  list[TypedQuestion]        # preserves semantic name, order, per-branch K
  meta:       Provenance                 # id, group_id, source, variant, text_sha256, optional row/repo/pair/family/certificate
TypedQuestion:
  name:        str                       # semantic key from frozen data
  type:        "choice" | "score" | "noul"
  instructions: str
  options:     list[Option]              # from criteria: {key, description?}; score = ordered list; noul = [true, false] when criteria present
  label:       str | int | bool          # hard target; + optional target: list[float] for our own soft data
  src:         str
```

Key decisions for the interface ticket: (a) treat `noul` as a 2-way choice with fixed option order at the head level; (b) convert everything to a unified `(options, soft_target)` view where soft targets are one-hot for frozen data; (c) implement loss as unified soft-CE + optional ordinal-RPS term (no early return); (d) carry `group_id` through to sampling and metrics.
