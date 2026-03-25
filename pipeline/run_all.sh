#!/bin/bash
# =============================================================
# Master script: End-to-end pipeline for Show-o2 evaluation
#
# GPU allocation:
#   GPU 0: Qwen3-VL-8B eval server (vLLM)
#   GPU 1,2,3: Show-o2 1.5B image generation (3 shards)
#
# Hardware: 4x L20 48G
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs outputs/generated outputs/analysis

# =============================================
# Step 0: Extract DALLE3 prompts
# =============================================
echo "=========================================="
echo "Step 0: Extracting DALLE3 prompts from BLIP3o-60k"
echo "=========================================="

if [ ! -f dalle3.txt ]; then
    python3 step0_extract_dalle3_prompts.py --output dalle3.txt
    echo "Extracted prompts to dalle3.txt"
else
    echo "dalle3.txt already exists, skipping extraction"
fi

PROMPT_COUNT=$(wc -l < dalle3.txt)
echo "Total prompts: $PROMPT_COUNT"

# =============================================
# Step 1: Clone Show-o2 and setup environment
# =============================================
echo ""
echo "=========================================="
echo "Step 1: Setting up Show-o2 environment"
echo "=========================================="

if [ ! -d "Show-o" ]; then
    echo "Cloning Show-o repository..."
    git clone https://github.com/showlab/Show-o.git
    cd Show-o
    bash build_env.sh || {
        echo "build_env.sh failed, installing manually..."
        pip install torch torchvision torchaudio
        pip install transformers accelerate omegaconf wandb
        pip install diffusers einops timm
    }
    cd "$SCRIPT_DIR"
fi

# Download Wan2.1 VAE if not present
if [ ! -f "Show-o/show-o2/Wan2.1_VAE.pth" ]; then
    echo "Downloading Wan2.1 VAE weights..."
    cd Show-o/show-o2
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Wan-AI/Wan2.1-T2V-14B',
    filename='Wan2.1_VAE.pth',
    local_dir='.',
)
"
    cd "$SCRIPT_DIR"
fi

# =============================================
# Step 2: Start Qwen3-VL eval server (GPU 0)
# =============================================
echo ""
echo "=========================================="
echo "Step 2: Starting Qwen3-VL eval server on GPU 0"
echo "=========================================="

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Eval server already running on port 8000"
else
    echo "Starting eval server in background..."
    bash start_eval_server.sh &
    EVAL_SERVER_PID=$!
    echo "Eval server PID: $EVAL_SERVER_PID"

    echo "Waiting for eval server to be ready..."
    MAX_WAIT=300
    WAITED=0
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Eval server did not start within ${MAX_WAIT}s"
            exit 1
        fi
        echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
    done
    echo "Eval server is ready!"
fi

# =============================================
# Step 3: Generate images with Show-o2 (GPU 1,2,3)
# =============================================
echo ""
echo "=========================================="
echo "Step 3: Generating images with Show-o2 on GPUs 1,2,3"
echo "=========================================="

if [ -f outputs/generated/generation_results.json ]; then
    echo "Generation results already exist, skipping generation"
else
    python3 generate_images.py \
        --mode parallel \
        --gpu-ids 1 2 3 \
        --prompts dalle3.txt \
        --output-dir outputs/generated \
        --batch-size 2 \
        --config Show-o/show-o2/configs/showo2_1.5b_demo_512x512.yaml \
        --showo-repo Show-o/show-o2 \
        2>&1 | tee logs/generation.log

    echo "Image generation complete!"
fi

GEN_COUNT=$(python3 -c "
import json
with open('outputs/generated/generation_results.json') as f:
    r = json.load(f)
print(sum(1 for x in r if x['status'] == 'success'))
")
echo "Successfully generated: $GEN_COUNT images"

# =============================================
# Step 4: Evaluate images with Qwen3-VL
# =============================================
echo ""
echo "=========================================="
echo "Step 4: Evaluating images with Qwen3-VL"
echo "=========================================="

python3 evaluate_images.py \
    --results outputs/generated/generation_results.json \
    --output outputs/eval_results.json \
    --api-url http://localhost:8000/v1 \
    --model-name qwen3-vl-eval \
    --workers 4 \
    --resume \
    2>&1 | tee logs/evaluation.log

echo "Evaluation complete!"

# =============================================
# Step 5: Analyze results
# =============================================
echo ""
echo "=========================================="
echo "Step 5: Analyzing failure patterns"
echo "=========================================="

python3 analyze_results.py \
    --eval-results outputs/eval_results.json \
    --output-dir outputs/analysis \
    2>&1 | tee logs/analysis.log

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
echo ""
echo "Key output files:"
echo "  - dalle3.txt                             : Prompts from BLIP3o-60k"
echo "  - outputs/generated/                      : Generated images"
echo "  - outputs/generated/generation_results.json: Generation metadata"
echo "  - outputs/eval_results.json               : Per-image evaluation results"
echo "  - outputs/analysis/dalle3_failures.txt    : Failed prompts (Show-o2 errors)"
echo "  - outputs/analysis/failure_analysis.json  : Detailed failure analysis"
echo ""
