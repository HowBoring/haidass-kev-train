# Additional Jev-like training datasets

Research date: 2026-09-23. Scope: additional public typed-decision data beyond the pinned Kev decision-v7 and LocalLLaMA/typed-decisions. This is a documentation/schema review, not a row-level quality audit or evidence of training improvement. Counts and quality claims below are publisher-reported. No candidate training mix was changed; no test payloads were downloaded.

## Recommended investigation order

| Candidate | Published size | Label evidence | Recommendation |
|---|---|---|---|
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | release-v2 redistributable: 79,116 train rows; separate control configs | Controlled synthetic/reference labels; manifests, generators, grouped splits and audits | First investigate text rule/contrastive controls, not the whole game/control mixture |
| [altslate/certo-decisions-v2](https://huggingface.co/datasets/altslate/certo-decisions-v2) | v2.1 total 970,797; 620,000 program_verified and 60,000 exact_posterior across project splits | Executed rules and known probability worlds, plus separately identified public-task gold | First investigate program_verified/exact_posterior TRAIN slices; independently check generators |
| [n4ze3m/typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | 6,682 train cases / 23,319 questions; 732 validation cases | Same DeepSeek teacher generates and labels; average of three distributions | Secondary soft-target breadth experiment, not trusted truth/calibration ground truth |
| [kaivoss/system-one-270m-data](https://huggingface.co/datasets/kaivoss/system-one-270m-data) | 7,537 states / 25,002 questions | gpt-oss-20b permuted-read ensemble | Secondary soft-target experiment; needs state-level independent train/dev/calibration splits and rendered-prompt conversion |

## Specific qualifications

Open-Jev publishes twelve configs with train/calibration/validation/test/OOD separation. The release-v2 config is contained in browser-drone-expansion: summing their sizes double-counts data. `metadata_json` and `record_json` may contain privileged labels and must not be rendered as model input. Distribution, independent binary, categorical and ordinal targets require different conversion semantics. Original generated records are CC0; source code MIT; the card explicitly retains unverified licensing of some upstream TypeSafe question descriptions. Check THIRD_PARTY_NOTICES and selected source-specific provenance, rather than treating the top-level license tag as blanket permission. Several new control corpora have not been trained or evaluated on models; data audits are not downstream performance evidence.

Certo's MIT applies to assembly/schema; upstream sources retain their own licenses. The large mixture contains MultiNLI/DBpedia/BoolQ (already represented in SFT), and GoEmotions is adjacent to our held-out emotion task. Select original programmatic slices, not the entire mixture. Exact posterior is only ground truth under the declared synthetic world, not real-world calibration. Independent-binary targets must become separate noul questions, not one softmax.

Typed-decisions-synth is MIT and case-split, but explicitly says no human checked labels. Self-agreement filtering removes difficult ambiguous cases; correct options occur first about half the time. Permute choice options with targets, never ordinal levels. Excluding four named benchmark workflows is a publisher claim, not proof of semantic independence.

System-one-270m-data declares Apache-2.0 and open-weight gpt-oss-20b teacher generation. 84.6% of questions are classed clear; no human validation. The teacher's uncertainty is not verified event probability. Split on state_id, never row; the card does not establish an existing independent calibration partition.

## Other leads, not approved

- [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3): scout found a large Jev-teacher distillation corpus and repackaged Open-Jev stream. Needs primary-source row/provenance audit, deduplication and upstream service-terms review. Dataset license alone does not settle teacher-output rights.
- [vagmi/jevlite_dataset](https://huggingface.co/datasets/vagmi/jevlite_dataset): scout found state-grouped soft decisions with CC-BY-SA licensing. Review actual obligations; SA is not automatically a ban on training or redistribution.
- [XinranSong/nanojev-blackjack-finite-v2](https://huggingface.co/datasets/XinranSong/nanojev-blackjack-finite-v2): potential exact-distribution diagnostic, not a substitute for general NLP performance.

## Admission gates

1. Pin a revision and checksum; review source-specific licenses and applicable teacher service terms.
2. Inspect TRAIN only initially; preserve original held-out groups, and keep all variants of a state/family together.
3. Audit target/option alignment, normalization, ordinal ordering, length, position bias, duplicates and label leakage. Replay generators on sampled training cases where available.
4. Exclude existing evaluation-only source families. Reusing train-source tasks is not automatically forbidden, but must be tracked/deduplicated. Do not inspect locked test labels to tune filtering.
5. If no calibration split exists, derive one from TRAIN at group level before training. Absence of a published calibration split is not disqualifying.
6. Hard one-hot labels are valid inputs to CE, proper scoring and RLCD; they simply do not provide empirical soft uncertainty targets. Do not claim they make proper scoring mathematically invalid.
7. Use a separate, explicitly named augmentation experiment after the frozen SFT comparison. Hold update budget and retention evaluation fixed; test one source addition at a time. No arbitrary universal mixture percentage is established by current evidence.

## Primary sources read by Main

- https://huggingface.co/datasets/ZefanCai/Open-Jev/raw/main/README.md
- https://huggingface.co/datasets/altslate/certo-decisions-v2/raw/main/README.md
- https://huggingface.co/datasets/n4ze3m/typed-decisions-synth/raw/main/README.md
- https://huggingface.co/datasets/kaivoss/system-one-270m-data/raw/main/README.md

Discovery used public clone listings and a read-only agent; recommendations above were narrowed and corrected against these primary cards. No claim that this list exhausts the ecosystem or that any candidate is already accepted as high quality.
