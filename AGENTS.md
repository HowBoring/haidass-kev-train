# Repository agent guide

This repository trains `DALabCommunity/Haidass1.5-143M` as a Kev-style typed decision model on one RTX 5090: shared-state block-causal encoding, LoRA, and a pointer readout produce option probabilities instead of generated text.

## Start here

- Use Python 3.12 with `uv`; run CUDA work in the CUDA 13.2 Docker environment.
- Before changing data, model, training, loss, or evaluation behavior, read `docs/agents/training.md` and the relevant section of `docs/chatgpt/Haidass-Kev-Train-Analysis.md`.
- Before changing dependencies, Docker, installation, or runtime commands, read `docs/agents/development.md`.
- For Hugging Face authentication, downloads, or uploads, use `.agents/skills/hf-cli/SKILL.md`. Pinned repositories and revisions live in `configs/resources.toml`.

## Agent workflows

- GitHub Issues are the issue and specification tracker. Read `docs/agents/issue-tracker.md` before issue operations.
- Triage uses the canonical label mapping in `docs/agents/triage-labels.md`.
- Domain documentation is single-context. Follow `docs/agents/domain.md` before reading or writing `CONTEXT.md` or ADRs.
- Proactively offload deterministic, non-decision-making tasks—such as environment setup, dependency installation, dataset and model downloading and processing, and specific code implementation—to task agents, thereby allowing the primary agent to focus on the big picture.
- The primary agent should provide guidance or take over tasks that sub-agents are unable to complete.

## Documentation policy

Keep this file minimal through progressive disclosure, following [A Complete Guide To AGENTS.md](https://www.aihero.dev/a-complete-guide-to-agents-md). Put task-specific guidance in `docs/agents/` and add a trigger-focused pointer here.
