"""
Step 2: Generate images using Show-o2 1.5B on GPUs 1,2,3

Reads dalle3.txt, splits prompts across 3 GPUs, generates images with Show-o2.
Each GPU runs independently on its assigned shard.

Usage:
    # Run all 3 GPUs in parallel (recommended):
    python generate_images.py --gpu-ids 1 2 3 --prompts dalle3.txt --output-dir outputs/generated

    # Or run a single GPU shard:
    python generate_images.py --gpu-ids 1 --shard-id 0 --num-shards 3 --prompts dalle3.txt
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path


def run_single_shard(args):
    """Generate images for a single shard on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch
    from PIL import Image
    import numpy as np

    sys.path.insert(0, args.showo_repo)

    from omegaconf import OmegaConf
    from models import Showo2, Wan21VAE
    from training.prompting_utils import UniversalPrompting
    from transformers import AutoTokenizer
    from transport import create_transport

    device = torch.device("cuda:0")

    with open(args.prompts, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    shard_size = (len(all_prompts) + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = min(start_idx + shard_size, len(all_prompts))
    prompts = all_prompts[start_idx:end_idx]

    print(f"[Shard {args.shard_id}] GPU {args.gpu_id}: Processing prompts {start_idx}-{end_idx} "
          f"({len(prompts)} prompts)")

    config = OmegaConf.load(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, padding_side="left"
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
    )

    print(f"[Shard {args.shard_id}] Loading VAE model...")
    vae_model = Wan21VAE(config.model.vae_model.pretrained_model_path)
    vae_model = vae_model.to(device).eval()
    vae_model.requires_grad_(False)

    print(f"[Shard {args.shard_id}] Loading Show-o2 model...")
    model = Showo2.from_pretrained(config.model.showo.pretrained_model_path)
    model = model.to(device, dtype=torch.bfloat16).eval()

    transport_config = OmegaConf.to_container(config.transport, resolve=True)
    transport = create_transport(**{
        k: v for k, v in transport_config.items()
        if k not in ["sampling_method", "guidance_scale", "num_inference_steps"]
    })

    os.makedirs(args.output_dir, exist_ok=True)

    batch_size = args.batch_size
    guidance_scale = config.transport.get("guidance_scale", 7.5)
    num_steps = config.transport.get("num_inference_steps", 50)

    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_indices = list(range(start_idx + i, start_idx + i + len(batch_prompts)))

        print(f"[Shard {args.shard_id}] Generating batch {i // batch_size + 1}, "
              f"prompts {batch_indices[0]}-{batch_indices[-1]}")

        try:
            with torch.no_grad():
                input_ids, _ = uni_prompting(
                    (batch_prompts, None), "t2i_gen"
                )
                input_ids = input_ids.to(device)

                if guidance_scale > 0:
                    uncond_input_ids, _ = uni_prompting(
                        ([""] * len(batch_prompts), None), "t2i_gen"
                    )
                    uncond_input_ids = uncond_input_ids.to(device)
                else:
                    uncond_input_ids = None

                latent_h = config.model.showo.image_latent_height
                latent_w = config.model.showo.image_latent_width
                patch_size = config.model.showo.patch_size
                latent_dim = config.model.showo.image_latent_dim

                z = torch.randn(
                    len(batch_prompts),
                    latent_dim,
                    latent_h,
                    latent_w,
                    device=device,
                    dtype=torch.bfloat16,
                )

                sampler = transport.get_sampler()
                samples = sampler(
                    model.flow_forward,
                    z,
                    model_kwargs=dict(
                        input_ids=input_ids,
                        uncond_input_ids=uncond_input_ids,
                        guidance_scale=guidance_scale,
                    ),
                    num_steps=num_steps,
                )

                images = vae_model.decode(samples)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images = (images * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            for j, (img_array, prompt, global_idx) in enumerate(
                zip(images, batch_prompts, batch_indices)
            ):
                pil_image = Image.fromarray(img_array)
                filename = f"{global_idx:05d}.png"
                filepath = os.path.join(args.output_dir, filename)
                pil_image.save(filepath)

                results.append({
                    "index": global_idx,
                    "prompt": prompt,
                    "image_path": filepath,
                    "status": "success",
                })

        except Exception as e:
            print(f"[Shard {args.shard_id}] Error generating batch at index {i}: {e}")
            import traceback
            traceback.print_exc()
            for j, (prompt, global_idx) in enumerate(
                zip(batch_prompts, batch_indices)
            ):
                results.append({
                    "index": global_idx,
                    "prompt": prompt,
                    "image_path": None,
                    "status": f"error: {str(e)}",
                })

    results_file = os.path.join(
        args.output_dir, f"generation_results_shard{args.shard_id}.json"
    )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Shard {args.shard_id}] Done! Generated {len([r for r in results if r['status'] == 'success'])} "
          f"images. Results saved to {results_file}")


def launch_parallel(args):
    """Launch multiple GPU processes in parallel."""
    gpu_ids = args.gpu_ids
    num_shards = len(gpu_ids)

    processes = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable, __file__,
            "--mode", "single",
            "--gpu-id", str(gpu_id),
            "--shard-id", str(shard_id),
            "--num-shards", str(num_shards),
            "--prompts", args.prompts,
            "--output-dir", args.output_dir,
            "--batch-size", str(args.batch_size),
            "--config", args.config,
            "--showo-repo", args.showo_repo,
        ]
        print(f"Launching shard {shard_id} on GPU {gpu_id}...")
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    for proc in processes:
        proc.wait()

    print("\nAll shards complete. Merging results...")
    all_results = []
    for shard_id in range(num_shards):
        results_file = os.path.join(
            args.output_dir, f"generation_results_shard{shard_id}.json"
        )
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                all_results.extend(json.load(f))

    all_results.sort(key=lambda x: x["index"])
    merged_file = os.path.join(args.output_dir, "generation_results.json")
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in all_results if r["status"] == "success")
    print(f"Merged {len(all_results)} results ({success} successful) → {merged_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate images with Show-o2 1.5B")
    parser.add_argument("--mode", choices=["parallel", "single"], default="parallel")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[1, 2, 3],
                        help="GPU IDs for generation (default: 1 2 3)")
    parser.add_argument("--gpu-id", type=int, default=1,
                        help="Single GPU ID (for single mode)")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--prompts", type=str, default="dalle3.txt",
                        help="Path to prompts file (one per line)")
    parser.add_argument("--output-dir", type=str, default="outputs/generated",
                        help="Directory to save generated images")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per GPU (L20 48G can handle 2-4 at 512x512)")
    parser.add_argument("--config", type=str,
                        default="Show-o/show-o2/configs/showo2_1.5b_demo_512x512.yaml",
                        help="Show-o2 config file")
    parser.add_argument("--showo-repo", type=str, default="Show-o/show-o2",
                        help="Path to Show-o2 source code")
    args = parser.parse_args()

    if args.mode == "parallel":
        launch_parallel(args)
    else:
        run_single_shard(args)


if __name__ == "__main__":
    main()
