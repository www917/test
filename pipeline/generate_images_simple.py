"""
Step 2 (Simple): Generate images using Show-o2's native inference_t2i.py

Directly calls Show-o2's built-in inference_t2i.py per shard.
Expected location: Show-o/show-o2/pipeline/
inference_t2i.py lives at ../inference_t2i.py (the show-o2 root)

Usage:
    # Launch 3 GPUs in parallel:
    python generate_images_simple.py --gpu-ids 1 2 3 --prompts dalle3.txt

    # Single GPU:
    CUDA_VISIBLE_DEVICES=1 python generate_images_simple.py --gpu-ids 1 --prompts dalle3.txt
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOWO2_ROOT = os.path.dirname(PIPELINE_DIR)  # ../  (show-o2 root)


def split_prompts(prompts_file: str, num_shards: int, output_dir: str) -> list[str]:
    """Split prompts file into N shards."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    shard_files = []
    shard_size = (len(prompts) + num_shards - 1) // num_shards

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, len(prompts))
        shard_prompts = prompts[start:end]

        shard_file = os.path.join(output_dir, f"prompts_shard{shard_id}.txt")
        with open(shard_file, "w", encoding="utf-8") as f:
            for p in shard_prompts:
                f.write(p + "\n")

        shard_files.append(os.path.abspath(shard_file))
        print(f"Shard {shard_id}: {len(shard_prompts)} prompts ({start}-{end})")

    return shard_files


def run_showo2_native(gpu_id, shard_id, prompts_file, config, output_dir):
    """Run Show-o2's native inference_t2i.py on a specific GPU.
    
    cwd is set to SHOWO2_ROOT so inference_t2i.py can find models/, training/ etc.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_MODE"] = "disabled"

    shard_output = os.path.abspath(os.path.join(output_dir, f"shard_{shard_id}"))
    os.makedirs(shard_output, exist_ok=True)

    cmd = [
        sys.executable, "inference_t2i.py",
        f"config={config}",
        f"validation_prompts_file={prompts_file}",
        f"experiment.output_dir={shard_output}",
        "batch_size=2",
        "guidance_scale=7.5",
        "num_inference_steps=50",
        "mode=t2i",
    ]

    print(f"[Shard {shard_id}] GPU {gpu_id}: Running in {SHOWO2_ROOT}")
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=SHOWO2_ROOT, env=env)
    return proc


def collect_results(prompts_file, output_dir, num_shards):
    """Collect all generated images and create a results JSON."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    final_dir = os.path.join(output_dir, "all_images")
    os.makedirs(final_dir, exist_ok=True)

    results = []

    for shard_id in range(num_shards):
        shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
        if not os.path.exists(shard_dir):
            continue

        images = sorted(
            [f for f in os.listdir(shard_dir) if f.endswith((".png", ".jpg", ".jpeg"))],
        )

        shard_size = (len(all_prompts) + num_shards - 1) // num_shards
        start_idx = shard_id * shard_size

        for i, img_file in enumerate(images):
            global_idx = start_idx + i
            if global_idx >= len(all_prompts):
                break

            new_name = f"{global_idx:05d}.png"
            src = os.path.join(shard_dir, img_file)
            dst = os.path.join(final_dir, new_name)
            shutil.copy2(src, dst)

            results.append({
                "index": global_idx,
                "prompt": all_prompts[global_idx] if global_idx < len(all_prompts) else "",
                "image_path": os.path.abspath(dst),
                "status": "success",
            })

    for idx in range(len(all_prompts)):
        if not any(r["index"] == idx for r in results):
            results.append({
                "index": idx,
                "prompt": all_prompts[idx],
                "image_path": None,
                "status": "missing",
            })

    results.sort(key=lambda x: x["index"])

    results_file = os.path.join(output_dir, "generation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in results if r["status"] == "success")
    print(f"\nCollected {success}/{len(all_prompts)} images -> {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--prompts", type=str, default="dalle3.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/generated")
    parser.add_argument("--config", type=str,
                        default="configs/showo2_1.5b_demo_512x512.yaml",
                        help="Config path relative to show-o2 root")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_shards = len(args.gpu_ids)
    prompts_abs = os.path.abspath(args.prompts)
    shard_files = split_prompts(prompts_abs, num_shards, args.output_dir)

    processes = []
    for shard_id, gpu_id in enumerate(args.gpu_ids):
        proc = run_showo2_native(
            gpu_id, shard_id, shard_files[shard_id],
            args.config, args.output_dir,
        )
        processes.append(proc)

    print(f"\nWaiting for {len(processes)} processes to complete...")
    for proc in processes:
        proc.wait()

    print("All processes finished. Collecting results...")
    collect_results(prompts_abs, args.output_dir, num_shards)


if __name__ == "__main__":
    main()
