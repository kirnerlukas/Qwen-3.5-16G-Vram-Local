#!/usr/bin/env bash
# Download Qwen3.5-35B-A3B-UD-IQ4_NL from unsloth HuggingFace repo.
#
# UD-IQ4_NL (17.8 GB) is the recommended upgrade from Q3_K_S for tool calling:
#   - Uses IQ4_NL (better accuracy than Q4_K_S) on attention + embedding layers
#   - Lower precision on MoE expert layers (less important for JSON formatting)
#   - Fits 16 GB VRAM at ngl=32 (32/41 layers on GPU, ~9 layers on 28-core CPU)
#   - Reliable JSON tool call output for pi-mono TypeBox schemas
set -euo pipefail

cd /home/uniberg/Qwen-3.5-16G-Vram-Local

DEST=./models/unsloth-gguf
MODEL=Qwen3.5-35B-A3B-UD-IQ4_NL.gguf
REPO=unsloth/Qwen3.5-35B-A3B-GGUF

if [[ -f "$DEST/$MODEL" ]]; then
  echo "Already downloaded: $DEST/$MODEL"
  ls -lh "$DEST/$MODEL"
  exit 0
fi

echo "Downloading $MODEL (~17.8 GB) from $REPO ..."

.venv/bin/python - <<PYEOF
from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id="$REPO",
    filename="$MODEL",
    local_dir="$DEST",
    local_dir_use_symlinks=False,
    cache_dir=".hf-cache/hub",
)
print(f"Saved to: {path}")
PYEOF

ls -lh "$DEST/$MODEL"
echo "Done. Run scripts/tune_35b_q4_ngl.sh next to find the best --ngl value."
