# Repository Guidelines

## Project Overview

Reproduce [Kev](https://github.com/jaredpalmer/kev)-style decision-model training on **`DALabCommunity/Haidass1.5-143M`** (Qwen3 architecture, raw pretrained base, 30 layers, hidden 576, 64k vocab, BF16, tied embeddings), on a single RTX 5090 (Blackwell, sm_120) via Docker.

Kev is not a text generator: it is a **frozen-ish backbone + LoRA + pointer-style readout head**. A shared `<state>` prefix and multiple typed questions (`noul` yes/no, `choice` K-way, `score` ordinal) are packed into ONE forward pass with a block-causal mask (state visible to all branches; branches isolated from each other). No autoregressive decoding — the `<decide>` hidden state and each option's `</opt>` hidden state are projected into a shared 256-d pointer space, scored by scaled dot-product, softmaxed into an option distribution.

**Current state: early scaffold.** The only source file is the uv stub `src/haidass_kev_train/__init__.py` (`main()` prints hello-world). No data pipeline, model, training, or eval code exists yet. The authoritative design spec is `docs/chatgpt/Haidass-Kev-Train-Analysis.md` (3679-line ChatGPT analysis, Chinese) — read relevant sections before implementing anything.

## Architecture & Data Flow

Planned (per the reference doc — not yet implemented):

```text
jaredpalmer/kev-suites (HF, frozen JSONL, SHA-256 verified)
        │
        ▼
(state, question, options, label/target) records
        │  render with 5 NEW special tokens:
        │  <|kev_state|> <|kev_question|> <|kev_option|> <|kev_option_end|> <|kev_decide|>
        ▼
packed sequences + block-causal attention mask (PyTorch SDPA, additive [B,1,L,L] mask)
        ▼
Haidass1.5-143M backbone (lm_head stripped; backbone = model.model)
  + LoRA r=16 (alpha=32, dropout=0.05, targets q/k/v/o/gate/up/down_proj)
  + 5 trainable special-token embeddings (PEFT trainable_token_indices — do NOT unfreeze the whole table)
  + PointerHead(576 → 256)  (~295k params)
        ▼
option distribution → CE loss (soft-label CE when `target` present)
```

Training is staged: **Stage 1** Kev-style supervised SFT (CE on option distribution); **Stage 2** CE + Laya-style RLCD (Gaussian-perturbed logits, group-mean-baseline REINFORCE, proper-scoring reward) + data replay; then **temperature calibration on an independent held-out split**.

Critical construction order (getting this wrong yields "runs but special embeddings never train"):
`tokenizer.add_special_tokens` → `model.resize_token_embeddings(len(tokenizer))` (64000→64005) → strip lm_head → PEFT LoRA → attach PointerHead. All three param groups (LoRA, PointerHead, 5 special embeddings) must be in the optimizer.

Planned experiment matrix (same checkpoint, same budget): A = SFT only, B = +CE, C = +direct proper loss (no sampling), D = CE + RLCD. Also planned: LoRA-faithful variant vs. full fine-tune variant (143M fits easily in 32 GB).

## Key Directories

| Path | Purpose |
|---|---|
| `src/haidass_kev_train/` | Package source (uv src-layout). Currently only a stub `__init__.py`. |
| `data/` | Empty scaffold dir; intended for downloaded suites/datasets. |
| `docker/` | `Dockerfile` — CUDA 13.2 devel toolchain image (see below). |
| `docs/chatgpt/` | `Haidass-Kev-Train-Analysis.md` — the design spec; consult it before non-trivial work. |
| `.agents/skills/hf-cli/` | Skill for the `hf` CLI — use it for all Hugging Face Hub operations (downloading `jaredpalmer/kev-suites`, `DALabCommunity/Haidass1.5-143M`, `LocalLLaMA/typed-decisions`, uploads, auth). |

## Development Commands

Package manager is **uv** (pinned 0.12.17 in Docker; `uv_build` backend). No lockfile exists yet.

```bash
uv sync                      # install locked deps (editable project); inside the container
                             # UV_PROJECT_ENVIRONMENT=/opt/venv → venv lives at /opt/venv,
                             # NOT .venv (deliberately outside the /workspace bind mount)
uv add <pkg>                 # add dependency (first add generates uv.lock)
uv run haidass-kev-train     # run console entry point (currently the stub)
docker build -f docker/Dockerfile -t haidass-kev-train .
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
    -v "$PWD:/workspace" \
    -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
    haidass-kev-train
```

Notes:
- The Dockerfile provisions only the toolchain (CUDA 13.2 devel, uv, Python 3.12); it does NOT `uv sync`.
- **Network constraint (this host):** per-connection throughput to PyPI mirrors and download.pytorch.org collapses to ~100 KB/s after a fast initial burst, so plain `uv sync` stalls on large wheels. Large wheels (torch, triton, nvidia-cudnn/cublas/cufft/cusolver/cusparse/cusparselt/nccl) are prefetched with `aria2c -x16 -s16` burst-cycling into `/workspace/wheels/` and installed via `uv pip install --python /opt/venv/bin/python --no-deps /workspace/wheels/*.whl`; everything else installs with `uv sync --no-install-package torch --no-install-package triton --no-install-package nvidia-cublas --no-install-package nvidia-cudnn-cu13 --no-install-package nvidia-cufft --no-install-package nvidia-cusolver --no-install-package nvidia-cusparse --no-install-package nvidia-cusparselt-cu13 --no-install-package nvidia-nccl-cu13`. Use `uv run --no-sync ...` (plain `uv run`/`uv sync` would try to re-fetch torch from the index).
- After container start, verify `torch.cuda.get_device_capability() == (12, 0)` (verified: RTX 5090 D, torch 2.14.0+cu132).

## Code Conventions & Common Patterns

Almost no code exists, so conventions are thin — follow what is established:

- **Python 3.12+**, uv src-layout, console script via `[project.scripts]` → `haidass_kev_train:main`.
- Type hints on public functions (the stub uses `def main() -> None:`).
- Environment/tooling pins live in `docker/Dockerfile`; Python deps belong in `pyproject.toml` (never `requirements.txt`).
- Attention backend is **PyTorch SDPA with a custom block-causal mask** — explicitly NO flash-attn / xformers / bitsandbytes / deepspeed / FSDP in phase 1 (flash-attn cannot express arbitrary branch masks).
- Precision policy: BF16 autocast for forward/backward; FP32 master weights for LoRA, PointerHead, and special embeddings. Never permanently cast the model to BF16.
- Data handling: when permuting options in augmentation, permute `target` in lockstep; never alter `label` while keeping a stale soft-label `target`; `score`-type questions get no arbitrary permutation augmentation.
- Calibration: fit temperature on a dedicated split, never on training items; report raw AND temperature-scaled probabilities.

## Important Files

| File | Role |
|---|---|
| `pyproject.toml` | uv project def, console entry point, build backend, full dep set (torch pinned to cu132 via explicit index). |
| `.python-version` | Pins Python 3.12. |
| `docker/Dockerfile` | `nvidia/cuda:13.2.0-devel-ubuntu24.04`, uv 0.12.17, `TORCH_CUDA_ARCH_LIST="12.0"`, `HF_HOME=/workspace/.cache/huggingface`, `HF_HUB_ENABLE_HF_TRANSFER=1`. |
| `src/haidass_kev_train/__init__.py` | Stub `main()`; entry point target. |
| `docs/chatgpt/Haidass-Kev-Train-Analysis.md` | Design spec: Kev architecture (~L14–1500), env/Docker decision (~L1500–2350), Laya/RLCD analysis (~L2354–3250), training plan + experiment matrix (~L3254–3679). |
| `.agents/skills/hf-cli/SKILL.md` | `hf` CLI reference — use for every HF Hub interaction. |

## Runtime/Tooling Preferences

- **Python ≥ 3.12**, managed by **uv** — never pip/poetry; keep `uv.lock` committed once generated.
- **Docker** for training: CUDA 13.2, target RTX 5090 / Blackwell (`sm_120`). Keep the image toolchain-only unless deliberately changing that decision.
- **Hugging Face Hub** is the data/model source of truth; use the `hf` CLI (see `.agents/skills/hf-cli`). Prefer frozen, checksummed suites (`jaredpalmer/kev-suites` — verify `manifest.json` SHA-256) over re-sampling the 13 public source datasets. Suite layout in that repo is `v6/decision-v6/` and `v4/decision-v4/` (NOT `evals/v6/...` as the reference doc describes — the hub repo has no top-level `evals/`). Additional soft-label data: `LocalLLaMA/typed-decisions` (parquet, per-workflow subsets).
- HF cache is `HF_HOME=/workspace/.cache/huggingface` in the container (host-side: `$HOME/.cache/huggingface`, bind-mounted in the `docker run` command above).

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default canonical triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repository uses a single-context domain-doc layout. See `docs/agents/domain.md`.
