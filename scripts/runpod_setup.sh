#!/bin/bash
# RunPod setup and experiment runner for molly-evolve
# Usage: bash scripts/runpod_setup.sh [--quicktest]
#
# This script:
#   1. Installs dependencies
#   2. Clones/updates the repo
#   3. Runs the full comparison experiment
#   4. Uploads results to HuggingFace Hub

set -e

REPO_URL="https://github.com/mathornton01/molly-evolve.git"
WORK_DIR="/workspace/molly-evolve"
RESULTS_DIR="/workspace/results"
MODEL="meta-llama/Llama-2-7b-hf"
HF_TOKEN="${HF_TOKEN:-}"

# Early validation
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "  LLaMA-2 is a gated model — set your HuggingFace token:"
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

echo "============================================"
echo "  Molly-Evolve RunPod Setup"
echo "============================================"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# 1. Install dependencies
echo ">>> Installing dependencies..."
pip install -q torch transformers datasets peft bitsandbytes scipy accelerate huggingface_hub sentencepiece pytest 2>&1 | tail -5
echo "  Done."

# 2. Clone or update repo
if [ -d "$WORK_DIR" ]; then
    echo ">>> Updating existing repo..."
    cd "$WORK_DIR"
    git pull origin main
else
    echo ">>> Cloning repo..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# Install the package (skip C++ extension build — pure Python is sufficient)
MOLLY_NO_EXTENSIONS=1 pip install -e . 2>&1 | tail -3

# 3. Quick sanity check
echo ""
echo ">>> Running quick sanity tests..."
python -m pytest tests/test_stability.py -x -q --tb=short 2>&1 | tail -5
echo ""

# 4. Determine extra args
EXTRA_ARGS=""
if [ "$1" = "--quicktest" ]; then
    EXTRA_ARGS="--quicktest --device cpu --epochs 1 --n-train 10 --n-eval 5"
    echo ">>> QUICKTEST MODE (CPU, minimal data)"
fi

# 5. Run experiment
echo ">>> Starting experiment..."
echo "  Model:   $MODEL"
echo "  Output:  $RESULTS_DIR"
echo "  Time:    $(date)"
echo ""

python experiments/full_comparison.py \
    --model "$MODEL" \
    --output "$RESULTS_DIR" \
    --epochs 3 \
    --n-train 200 \
    --n-eval 50 \
    --max-length 256 \
    --gc-threshold 0.80 \
    --gc-alpha 0.3 \
    --gc-max-repair-pct 0.03 \
    --gc-batch-size 5 \
    $EXTRA_ARGS

echo ""
echo ">>> Experiment complete: $(date)"
echo ">>> Results in: $RESULTS_DIR"

# 6. List results
echo ""
echo ">>> Results files:"
find "$RESULTS_DIR" -name "*.json" -type f 2>/dev/null | head -20

# 7. Upload to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo ">>> Uploading models to HuggingFace..."
    LATEST_RUN=$(ls -1dt "$RESULTS_DIR"/run_* | head -1)

    if [ -d "$LATEST_RUN/gene-conv/model" ]; then
        python -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_folder(
    folder_path='$LATEST_RUN/gene-conv/model',
    repo_id='mathornton/molly-gc-llama2-7b',
    repo_type='model',
    create_pr=False,
)
print('  Uploaded gene-conv model')
" 2>&1 || echo "  WARNING: gene-conv upload failed"
    fi

    if [ -d "$LATEST_RUN/lora/model" ]; then
        python -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_folder(
    folder_path='$LATEST_RUN/lora/model',
    repo_id='mathornton/molly-lora-llama2-7b',
    repo_type='model',
    create_pr=False,
)
print('  Uploaded LoRA model')
" 2>&1 || echo "  WARNING: LoRA upload failed"
    fi
fi

echo ""
echo "============================================"
echo "  All done! $(date)"
echo "============================================"
