# Development workflow

Read this before changing dependencies, Docker, installation, resource downloads, or runtime commands.

## Runtime contract

- Python is 3.12 or newer and managed with `uv`.
- Dependencies belong in `pyproject.toml`; keep `uv.lock` committed.
- Training runs in the CUDA 13.2 Docker image targeting RTX 5090 / Blackwell `sm_120`.
- The Docker image provides the toolchain only. Install project dependencies after starting the container.
- `UV_PROJECT_ENVIRONMENT=/opt/venv`; the environment stays outside the `/workspace` bind mount.

## Container commands

```bash
docker build -f docker/Dockerfile -t haidass-kev-train .
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
  haidass-kev-train
```

This host throttles large single-connection package downloads. Inside the container, install prefetched large wheels first:

```bash
uv pip install --python /opt/venv/bin/python --no-deps /workspace/wheels/*.whl
```

Then install the remaining locked dependencies without fetching those packages again:

```bash
uv sync \
  --no-install-package torch \
  --no-install-package triton \
  --no-install-package nvidia-cublas \
  --no-install-package nvidia-cudnn-cu13 \
  --no-install-package nvidia-cufft \
  --no-install-package nvidia-cusolver \
  --no-install-package nvidia-cusparse \
  --no-install-package nvidia-cusparselt-cu13 \
  --no-install-package nvidia-nccl-cu13
```

Use `uv run --no-sync ...` afterward so `uv` does not try to fetch the excluded wheels again.

## Smoke commands

```bash
uv run --no-sync haidass-kev-train
uv run --no-sync python -c 'import torch; print(torch.cuda.get_device_capability())'
```

CUDA setup is complete when the second command prints `(12, 0)`.

Download the pinned base model and public legacy datasets with:

```bash
scripts/download_resources.sh
```

For constrained internet access, set `HF_ENDPOINT` before invoking the script. Hugging Face operations must follow `.agents/skills/hf-cli/SKILL.md`.

The private unreviewed UFW–FineMath canonical suite is not downloaded by
this public-resource script; use the pinned `hf download` command in
`docs/agents/training.md` after authenticating with organization access.

## Dependency changes

Use `uv add <package>` for Python dependencies. Keep environment and toolchain pins in `docker/Dockerfile`; do not add a `requirements.txt`.
