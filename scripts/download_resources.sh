#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

workers="${HF_DOWNLOAD_WORKERS:-8}"

hf download DALabCommunity/Haidass1.5-143M \
  --revision d8a00d4943971088e6f0f4e08fb317dd5ba33ed1 \
  --local-dir models/base/haidass1.5-143m \
  --max-workers "$workers"

hf download jaredpalmer/kev-suites \
  --type dataset \
  --revision a957287d1c502a4e2e3b9d9d1325c2c6f27f181c \
  --include "v4/decision-v4/*" \
  --include "v6/decision-v6/*" \
  --local-dir data/raw/kev-suites \
  --max-workers "$workers"

hf download LocalLLaMA/typed-decisions \
  --type dataset \
  --revision ea9306458d6e9563628369a3d1e72e362fb381d2 \
  --local-dir data/raw/typed-decisions \
  --max-workers "$workers"
