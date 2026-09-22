FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System tools
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv
#
# Pin uv for reproducibility.
# Python itself will also be installed/managed by uv.
# ---------------------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:0.12.17 \
    /uv /uvx /bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Install a uv-managed CPython.
RUN uv python install 3.12

# Make the project environment the default Python environment
ENV PATH="/opt/venv/bin:/opt/uv/python/bin:$PATH"

# ---------------------------------------------------------------------------
# CUDA / PyTorch build configuration
# ---------------------------------------------------------------------------

# RTX 5090 = Blackwell / sm_120.
# This primarily matters when compiling CUDA extensions.
ENV TORCH_CUDA_ARCH_LIST="12.0"

# Useful for source-built CUDA extensions.
ENV CUDA_HOME=/usr/local/cuda

# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

ENV HF_HOME=/workspace/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

CMD ["/bin/bash"]
