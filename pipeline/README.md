# Show-o2 DPG Evaluation Pipeline

Evaluate Show-o2 1.5B image generation quality on BLIP3o-60k DALLE3 prompts, using Qwen3-VL as an automated judge.

## Architecture

```
GPU 0  →  Qwen3-VL-8B (vLLM server, evaluator)
GPU 1  →  Show-o2 1.5B (image gen, shard 0)
GPU 2  →  Show-o2 1.5B (image gen, shard 1)
GPU 3  →  Show-o2 1.5B (image gen, shard 2)
```

**Hardware**: 4x NVIDIA L20 48GB

## Quick Start

```bash
cd pipeline

# 1. Setup environment (one-time)
bash setup_env.sh

# 2. Run full pipeline
bash run_all.sh
```

## Step-by-Step

```bash
# Step 0: Extract DALLE3 prompts from BLIP3o-60k
python step0_extract_dalle3_prompts.py --output dalle3.txt

# Step 1: Start eval server (GPU 0, runs in background)
bash start_eval_server.sh &

# Step 2: Generate images (GPUs 1,2,3 in parallel)
python generate_images_simple.py --gpu-ids 1 2 3 --prompts dalle3.txt

# Step 3: Evaluate with Qwen3-VL
python evaluate_images.py \
    --results outputs/generated/generation_results.json \
    --output outputs/eval_results.json

# Step 4: Analyze failures
python analyze_results.py \
    --eval-results outputs/eval_results.json \
    --output-dir outputs/analysis
```

## Output Files

| File | Description |
|------|-------------|
| `dalle3.txt` | Extracted prompts from BLIP3o-60k |
| `outputs/generated/*.png` | Generated images |
| `outputs/generated/generation_results.json` | Generation metadata |
| `outputs/eval_results.json` | Per-image eval results (correct/incorrect + reason) |
| `outputs/analysis/dalle3_failures.txt` | Failed prompts list |
| `outputs/analysis/failure_analysis.json` | Categorized failure analysis |

## Evaluation Criteria

The Qwen3-VL judge checks:
- Object classes match the prompt
- Correct count of each object
- Colors are correct
- Spatial positions are correct
- Image is photorealistic (not cartoon/sketch)
- Single coherent scene (no collages)

## Failure Categories

The analysis script categorizes failures into:
- `wrong_count` — incorrect number of objects
- `wrong_color` — incorrect colors
- `wrong_position` — spatial arrangement errors
- `wrong_size` — scale/proportion issues
- `missing_object` — objects missing entirely
- `not_photorealistic` — cartoon/sketch style
- `collage_style` — multi-panel/grid output
- `wrong_object` — wrong object type
- `quality_issue` — blurry/distorted
- `text_rendering` — text in image is wrong
