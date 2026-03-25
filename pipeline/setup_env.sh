#!/bin/bash
# Environment setup for 4x L20 48G pipeline
#
# Run this ONCE before running the pipeline.
# It installs all dependencies for:
#   1. Qwen3-VL eval server (vLLM)
#   2. Show-o2 1.5B image generation
#   3. BLIP3o-60k dataset extraction
#   4. Evaluation & analysis scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================="
echo "Setting up environment for Show-o2 + Qwen3-VL pipeline"
echo "============================================="

# --- 1. Install Python packages ---
echo ""
echo "[1/4] Installing Python packages..."

pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install vllm

pip install transformers accelerate datasets huggingface_hub
pip install openai
pip install omegaconf wandb einops timm
pip install qwen-vl-utils==0.0.14

pip install Pillow numpy tqdm

# --- 2. Clone Show-o2 ---
echo ""
echo "[2/4] Cloning Show-o repository..."

if [ ! -d "Show-o" ]; then
    git clone https://github.com/showlab/Show-o.git
    cd Show-o
    if [ -f "build_env.sh" ]; then
        bash build_env.sh || echo "build_env.sh had issues, continuing with manual install"
    fi
    cd "$SCRIPT_DIR"
else
    echo "Show-o already cloned"
fi

# --- 3. Download VAE model ---
echo ""
echo "[3/4] Downloading Wan2.1 VAE model..."

VAE_PATH="Show-o/show-o2/Wan2.1_VAE.pth"
if [ ! -f "$VAE_PATH" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Wan-AI/Wan2.1-T2V-14B',
    filename='Wan2.1_VAE.pth',
    local_dir='Show-o/show-o2',
)
print(f'Downloaded to: {path}')
"
else
    echo "VAE model already exists"
fi

# --- 4. Pre-download models ---
echo ""
echo "[4/4] Pre-downloading model weights (this may take a while)..."

python3 -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading Qwen3-VL-8B-Instruct tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True)

print('Downloading show-o2-1.5B...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

print('Downloading SigLIP...')
# This will be used by Show-o2 internally
from huggingface_hub import snapshot_download
snapshot_download('google/siglip-so400m-patch14-384')

print('Pre-downloading show-o2-1.5B-HQ weights...')
snapshot_download('showlab/show-o2-1.5B-HQ')

print('Done pre-downloading!')
" || echo "Some model pre-downloads may have failed, they will be downloaded on first run"

# --- 5. Extract prompts ---
echo ""
echo "[5/5] Extracting prompts from BLIP3o-60k..."

if [ ! -f "dalle3.txt" ]; then
    python3 step0_extract_dalle3_prompts.py --output dalle3.txt
else
    echo "dalle3.txt already exists"
fi

echo ""
echo "============================================="
echo "Setup complete!"
echo ""
echo "Verify GPUs:"
echo "  nvidia-smi"
echo ""
echo "To run the full pipeline:"
echo "  cd $SCRIPT_DIR"
echo "  bash run_all.sh"
echo ""
echo "Or run steps individually:"
echo "  1. bash start_eval_server.sh &       # GPU 0: Qwen3-VL"
echo "  2. python generate_images_simple.py   # GPUs 1,2,3: Show-o2"
echo "  3. python evaluate_images.py          # Evaluate with Qwen3-VL"
echo "  4. python analyze_results.py          # Analyze failures"
echo "============================================="
