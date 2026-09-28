# Haidass Typed Decision Training

This context defines the staged conversion of a pretrained Haidass language-model backbone into a typed decision model and the evidence gates between training stages.

## Language

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
