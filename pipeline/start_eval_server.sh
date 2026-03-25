#!/bin/bash
# Step 1: Start Qwen3-VL evaluation server on GPU 0 via vLLM
#
# Uses Qwen3-VL-8B-Instruct (the closest VL model to "Qwen3-9B" with vision)
# Serves an OpenAI-compatible API on port 8000
#
# Prerequisites:
#   pip install vllm qwen-vl-utils==0.0.14

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
PORT=8000

echo "============================================="
echo "Starting Qwen3-VL-8B-Instruct eval server"
echo "  GPU: 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Port: $PORT"
echo "  Model: $MODEL_NAME"
echo "============================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt "video=0" \
    --trust-remote-code \
    --served-model-name "qwen3-vl-eval" \
    --chat-template-content-format "openai" \
    2>&1 | tee logs/eval_server.log
