#!/bin/bash
# Environment setup for Show-o2 evaluation pipeline
#
# Expected location: Show-o/show-o2/pipeline/
# Run this ONCE before running the pipeline.

set -e

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
SHOWO2_ROOT="$(dirname "$PIPELINE_DIR")"

cd "$PIPELINE_DIR"

echo "============================================="
echo "Setting up environment"
echo "  Pipeline dir: $PIPELINE_DIR"
echo "  Show-o2 root: $SHOWO2_ROOT"
echo "============================================="

# --- 1. Install Python packages ---
echo ""
echo "[1/4] Installing Python packages..."

pip install --upgrade pip

pip install vllm
pip install transformers accelerate datasets huggingface_hub
pip install openai
pip install omegaconf wandb einops timm
pip install qwen-vl-utils==0.0.14
pip install Pillow numpy tqdm

# --- 2. Show-o2 dependencies ---
echo ""
echo "[2/4] Installing Show-o2 dependencies..."

if [ -f "$SHOWO2_ROOT/../build_env.sh" ]; then
    echo "Running Show-o build_env.sh..."
    cd "$SHOWO2_ROOT/.."
    bash build_env.sh || echo "build_env.sh had issues, continuing"
    cd "$PIPELINE_DIR"
elif [ -f "$SHOWO2_ROOT/../requirements.txt" ]; then
    pip install -r "$SHOWO2_ROOT/../requirements.txt"
fi

# --- 3. Download Wan2.1 VAE ---
echo ""
echo "[3/4] Downloading Wan2.1 VAE model..."

if [ ! -f "$SHOWO2_ROOT/Wan2.1_VAE.pth" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Wan-AI/Wan2.1-T2V-14B',
    filename='Wan2.1_VAE.pth',
    local_dir='$SHOWO2_ROOT',
)
print(f'Downloaded to: {path}')
"
else
    echo "VAE model already exists at $SHOWO2_ROOT/Wan2.1_VAE.pth"
fi

# --- 4. Pre-download models ---
echo ""
echo "[4/4] Pre-downloading model weights..."

python3 -c "
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

print('Downloading Qwen3-VL-8B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True)

print('Downloading Qwen2.5-1.5B-Instruct (Show-o2 LLM backbone)...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

print('Downloading SigLIP...')
snapshot_download('google/siglip-so400m-patch14-384')

print('Downloading show-o2-1.5B-HQ...')
snapshot_download('showlab/show-o2-1.5B-HQ')

print('Done pre-downloading!')
" || echo "Some pre-downloads may have failed; they will be fetched on first run"

# --- 5. Extract prompts ---
echo ""
echo "[5/5] Extracting prompts from BLIP3o-60k..."

if [ ! -f "$PIPELINE_DIR/dalle3.txt" ]; then
    python3 step0_extract_dalle3_prompts.py --output dalle3.txt
else
    echo "dalle3.txt already exists"
fi

echo ""
echo "============================================="
echo "Setup complete!"
echo ""
echo "To run the full pipeline:"
echo "  cd $PIPELINE_DIR"
echo "  bash run_all.sh"
echo ""
echo "Or step by step:"
echo "  1. bash start_eval_server.sh &       # GPU 0: Qwen3-VL"
echo "  2. python generate_images_simple.py   # GPUs 1,2,3: Show-o2"
echo "  3. python evaluate_images.py          # Evaluate"
echo "  4. python analyze_results.py          # Analyze failures"
echo "============================================="
