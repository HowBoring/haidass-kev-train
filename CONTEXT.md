# Haidass Typed Decision Training

This context defines the construction of source-grounded decision data, the staged conversion of a pretrained Haidass backbone into a Decision Model, and the evidence gates for data scaling and training progression.

## Language

### Decision model and training lifecycle

**Decision Model**:
A model that consumes shared state and typed questions and returns a probability distribution over each question's options rather than generating answer text.
_Avoid_: Classifier, text generator


**Training Checkpoint**:
The complete resumable state of one Decision SFT or post-training run, including its optimization and progression state.
_Avoid_: Model artifact, export

**Decision Model Artifact**:
The portable trained Decision Model state required for deterministic evaluation when combined with its pinned base resource, excluding optimization and run-progression state.
_Avoid_: Training checkpoint, full training state

**Calibration Artifact**:
An independently fitted probability-calibration state bound to one Decision Model Artifact and one calibration split identity.
_Avoid_: Calibrated model, training checkpoint

**Model Adaptation Gate**:
The readiness boundary where the adapted backbone, structural-token mapping, decision readout, and trainable parameter groups have demonstrated a valid optimization step and checkpoint round trip.
_Avoid_: Model surgery complete, setup done

**Special Token Reuse**:
Mapping Kev structural markers onto existing Haidass tokenizer entries while preserving the pretrained vocabulary size and token IDs.
_Avoid_: Vocabulary extension, token addition

**Decision SFT**:
Supervised training that fits the Decision Model's option distributions from hard or soft targets.
_Avoid_: Instruction tuning, generative SFT

**SFT Runnable**:
The pilot state where Decision SFT can train stably, resume from a checkpoint, and invoke development evaluation.
_Avoid_: Training complete, smoke test passed

**SFT Complete**:
The state where the fixed-budget Decision SFT run has selected a checkpoint on development evidence and evaluated it with independent calibration and locked test data.
_Avoid_: Run finished, final model

**Evaluation Lane**:
The workstream that prepares reproducible benchmark inputs, metrics, and reports concurrently with Decision SFT without competing for the training GPU.
_Avoid_: Benchmark deployment, validation code

**RLCD Post-training**:
A post-SFT stage that optimizes perturbed option distributions with a relative policy-gradient objective while retaining supervised constraints.
_Avoid_: RLHF, GRPO, rollout training

**RLCD Entry Gate**:
The evidence boundary requiring a stable SFT checkpoint and comparable continued-CE and direct proper-loss baselines before RLCD experiments begin.
_Avoid_: Start RL, Stage 2 ready

### Decision data and evidence

**Source Record**:
An original source entry from which a decision case may be extracted, potentially containing several questions. It is distinct from the derived Canonical Decision Record and may share a Source Group with other entries or snapshots.
_Avoid_: Training example, independent document

**Source Trace**:
The evidence linking a decision case to its original material, including the source question, answer, and any permitted presentation conversion. It enables review without claiming complete data lineage or proving the source answer correct.
_Avoid_: Correctness certificate, provenance platform

**Decision State**:
The permitted context shared by a case's questions, excluding appended source QA annotations and worked solutions. Ordinary evidence or givens from which the answer can be inferred are not themselves Answer Leakage.
_Avoid_: Entire source record, solution text

**Source-grounded Answer**:
The answer supported by the original source and preserved as the unique correct choice for a decision case. A complete finite solution set can be one answer; a generator's replacement solution is not a source-grounded answer.
_Avoid_: Generated answer, arbitrary answer position

**Answer Leakage**:
Unintended exposure of answer annotations, worked solutions, or correctness markers in the model-visible problem. Legitimate source evidence, necessary givens, and the correct candidate among unmarked alternatives are not leakage merely because they contain the answer's text.
_Avoid_: Answer substring match, ordinary supporting evidence

**QA Presentation Conversion**:
A conversion of an existing question and answer that removes dependencies on the original option labels, count, or layout while preserving the question's meaning and the source answer. It is not the creation of a new question or the synthesis of a replacement answer.
_Avoid_: QA regeneration, answer synthesis

**LLM Equivalence Adjudication**:
A model judgment of whether candidate answers are equivalent under a problem's conditions when programmatic checks cannot decide. Acceptance on this basis retains uncertainty and is not a mathematical proof or a programmatically verified result.
_Avoid_: Verified equivalence, proven correctness

**Generation Trial**:
A bounded attempt to produce canonical decision records for human inspection before scaling data construction. Its accepted-record target is distinct from generator-call consumption and may remain unmet when a resource limit is reached.
_Avoid_: Training pilot, full dataset build

**Canonical Decision Record**:
A decision-training case containing the permitted context, question, one source-grounded correct answer, and a pool of incorrect alternatives, without a fixed presented candidate set or answer position.
_Avoid_: Materialized request, fixed-label example

**Decision View**:
A presentation of one Canonical Decision Record with a chosen candidate subset and order, and supervision aligned to that presentation. Multiple views remain observations of the same underlying case, not independent examples.
_Avoid_: New source case, independent sample

**Source Group**:
Source records linked by a known document identity and kept together when separating training from development data. Grouping expresses known shared origin, not a claim of complete duplicate detection.
_Avoid_: Source row, deduplicated corpus

**Candidate Pool**:
The source-grounded answer and accepted distractors belonging to a Canonical Decision Record before a particular Decision View is chosen. Pool membership does not specify which candidates will be presented or their order.
_Avoid_: Presented options, fixed classification labels

**Distractor**:
A plausible answer that is incorrect under the case's conditions and is not equivalent to its source-grounded answer or another distractor. A different spelling or unit alone does not make an answer a distractor.
_Avoid_: Any other string, equivalent answer

**Candidate Set**:
The candidates selected from a Candidate Pool for a Decision View, independent of their presentation order. Reordering a fixed Candidate Set is distinct from changing its membership.
_Avoid_: Candidate pool, permutation

**Dynamic Candidate Sampling**:
Varying the candidate count, subset, and order across training presentations while retaining the same source-grounded answer. Different presentations remain views of one case rather than newly acquired supervision.
_Avoid_: New question generation, fixed-label classification

**Fixed Evaluation View**:
A Decision View whose candidate membership and order remain fixed across evaluations. It supports comparable development and Training Probe measurements without following training-time candidate changes.
_Avoid_: Epoch augmentation, independent case

**Frozen Decision Suite**:
An identified collection of accepted Canonical Decision Records with fixed source-group assignments to training and development. Runtime Decision Views are presentations of that collection, not changes to the underlying supervision.
_Avoid_: Live generation stream, model artifact

**Programmatic Equivalence Check**:
A bounded comparison of candidate meanings under the problem's conditions that distinguishes established equivalence, established difference, and an undecided result. An undecided result is not evidence of difference, and comparing candidates does not independently prove the source answer correct.
_Avoid_: String deduplication, universal mathematical proof

**Candidate Validation Path**:
The recorded basis on which a case's candidate checks were accepted, distinguishing programmatic checks from acceptance requiring LLM Equivalence Adjudication. Any reliance on adjudication preserves that uncertainty for the case as a whole.
_Avoid_: Quality score, proof of correctness

**Generation Attempt**:
One inference attempt used to construct or adjudicate decision data, including an attempted retry or a request that fails. Its consumption is distinct from the number of accepted Canonical Decision Records.
_Avoid_: Accepted record, completed source case

**Training Probe**:
A fixed, source-group-preserving selection of training cases evaluated through Fixed Evaluation Views to diagnose fitting. It is neither held-out development evidence nor the basis for selecting the production checkpoint.
_Avoid_: Validation split, checkpoint-selection set

**Overfit Check**:
A bounded experiment that checks whether the real training path can fit a small fixed set of training cases while preserving the intended candidate variation. Passing demonstrates fitting and integration evidence, not generalization.
_Avoid_: Engineering smoke, development success

**Permutation Flip**:
A change in the predicted candidate's meaning when only the order of a fixed Candidate Set changes. A position-index change alone is not a flip, and consistently selecting the wrong candidate can still produce no flips.
_Avoid_: Candidate-index change, accuracy

**Data Pilot**:
A small-scale data and training experiment used to assess development learning evidence before proposing a larger run. It is distinct from a Generation Trial, an Overfit Check, and SFT Complete.
_Avoid_: Four-step smoke, completed training

**Data Quality Gate**:
The human-review boundary for accepting the correctness and usability of a generated decision-data batch. Passing it is evidence about the reviewed batch, not proof that the entire source or future output is error-free.
_Avoid_: Generator approval, corpus certification

**Data Scaling Gate**:
The evidence boundary for proceeding from a small data experiment to a larger one after data quality, training integration, and development learning checks pass. It is not SFT Complete, an OOD claim, or authorization for additional resource use.
_Avoid_: Final model acceptance, automatic expansion
