#!/usr/bin/env python3
"""Test dalle3.txt prompts with show-o2-1.5B: generate twice, collect double-failures.

Phase 1: For each prompt, generate an image twice with different random seeds.
          Score each with the lite teacher prompt. If both fail (hard_ok=False),
          record it as a hard case.
Phase 2: Build a 4:1 (pass:fail) mixed dataset and run a test.

Expected layout (GRPO/ inside show-o2/):
    Show-o/
      show-o2/
        models/
        transport/
        GRPO/
          test_dalle3_double_gen.py   ← this file
          eval_compare.py
          configs/showo2_1.5b_grpo.yaml
          dalle3_double_gen/          ← output dir (auto-created)

Usage:
    cd Show-o/show-o2

    # Phase 1 – single GPU
    CUDA_VISIBLE_DEVICES=0 python GRPO/test_dalle3_double_gen.py --phase find --gpu-shard 0 --num-shards 1

    # Phase 1 – multi-GPU (run each in a separate terminal)
    CUDA_VISIBLE_DEVICES=0 python GRPO/test_dalle3_double_gen.py --phase find --gpu-shard 0 --num-shards 4
    CUDA_VISIBLE_DEVICES=1 python GRPO/test_dalle3_double_gen.py --phase find --gpu-shard 1 --num-shards 4
    CUDA_VISIBLE_DEVICES=2 python GRPO/test_dalle3_double_gen.py --phase find --gpu-shard 2 --num-shards 4
    CUDA_VISIBLE_DEVICES=3 python GRPO/test_dalle3_double_gen.py --phase find --gpu-shard 3 --num-shards 4

    # Phase 2 – merge shards and build 4:1 mixed dataset
    python GRPO/test_dalle3_double_gen.py --phase merge

    # Phase 3 – test on 4:1 mixed dataset
    CUDA_VISIBLE_DEVICES=0 python GRPO/test_dalle3_double_gen.py --phase test
"""
from __future__ import annotations

import multiprocessing
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import argparse
import json
import logging
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib import request
import base64

import torch
from PIL import Image
from omegaconf import OmegaConf

# ─── Path setup ───────────────────────────────────────────────────────────
# GRPO_DIR = Show-o/show-o2/GRPO/
# ROOT     = Show-o/show-o2/          (parent of GRPO_DIR)
GRPO_DIR = Path(__file__).resolve().parent
ROOT = GRPO_DIR.parent  # show-o2 root (contains models/, transport/, etc.)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import WanVAE, omni_attn_mask_naive
from models.misc import prepare_gen_input
from transport import Sampler, create_transport
from utils import denorm

sys.path.insert(0, str(GRPO_DIR))
from eval_compare import load_model, build_hyper

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────
# CONFIG_PATH is relative to cwd (show-o2 root), same as original
CONFIG_PATH = "GRPO/configs/showo2_1.5b_grpo.yaml"

# ★ 改成你自己的 prompts 路径
PROMPT_TXT = "/home/jiajiangcan/hdd_jbw/jiajiangcan/dataset_unilip/data/blip3o-60k/geneval_train.txt"

OUTPUT_DIR = GRPO_DIR / "dalle3_double_gen"

SYSTEM_PROMPT = (
    "You are an expert image evaluator.\n\n"
    "Your task is to determine whether the given image faithfully satisfies "
    "the visual instruction and the expectation checklist.\n\n"
    "Follow these rules strictly:\n"
    "1. The image must match **all** expectations, including:\n"
    "   - Object classes\n"
    "   - Counts of each object\n"
    "   - Colors of each object\n"
    "   - Spatial position within the image (e.g., \"above\", \"below\", based on real pixel position)\n"
    "   - Size and relative scale of objects\n"
    "2. The image must appear as a **natural, coherent, photo-like single image**.\n"
    "   - Do NOT allow collage-style or multi-panel images. Only one consistent, coherent scene is acceptable.\n"
    "3. Be very strict and conservative in your judgment.\n\n"
    "Return your result as a JSON object using this format:\n"
    "{\n"
    '  "correct": 1 if the image fully satisfies all expectations, else 0,\n'
    '  "reason": "You may explain in detail what is missing or incorrect"\n'
    "}"
)


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last <= first:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[first : last + 1])


def normalize_result(parsed: dict) -> dict:
    correct = int(parsed.get("correct", 0))
    reason = str(parsed.get("reason", "")).strip()
    return {
        "correct": correct,
        "reason": reason,
    }


def teacher_score(
    image: Image.Image,
    prompt: str,
    api_base: str,
    model_name: str,
    api_key: str = "EMPTY",
    max_retries: int = 3,
) -> dict:
    """Score an image using the strict evaluator prompt."""
    image_url = image_to_data_url(image)
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Visual instruction: {prompt}",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            req = request.Request(
                f"{api_base.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"]
            parsed = normalize_result(extract_json_object(raw))
            return parsed
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 + attempt)

    logger.warning("Teacher scoring failed for '%.60s': %s", prompt, last_exc)
    return {
        "correct": 0,
        "reason": str(last_exc),
        "error": True,
    }


def _unwrap(model):
    m = getattr(model, "module", model)
    if hasattr(m, "base_model"):
        return m.base_model.model
    return m


@torch.no_grad()
def generate_image(
    model, vae_model, sampler, tokenizer, showo_token_ids, hyper, config,
    prompt: str, device: torch.device, weight_type: torch.dtype,
) -> Image.Image:
    text_tokens, text_tokens_null, mod_pos, mod_pos_null = prepare_gen_input(
        [prompt], tokenizer, hyper.num_t2i_image_tokens,
        hyper.bos_id, hyper.eos_id, hyper.boi_id, hyper.eoi_id,
        hyper.pad_id, hyper.img_pad_id, hyper.max_text_len, device,
    )
    z = torch.randn(
        (1, hyper.image_latent_dim,
         hyper.latent_height * hyper.patch_size,
         hyper.latent_width * hyper.patch_size),
        device=device, dtype=weight_type,
    )
    gs = float(config.transport.guidance_scale)
    if gs > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([text_tokens, text_tokens_null], dim=0)
        mod_pos = torch.cat([mod_pos, mod_pos_null], dim=0)
    block_mask = omni_attn_mask_naive(
        text_tokens.size(0), text_tokens.size(1), mod_pos, device,
    ).to(weight_type)
    model_kwargs = dict(
        text_tokens=text_tokens, attention_mask=block_mask,
        modality_positions=mod_pos, output_hidden_states=True,
        max_seq_len=text_tokens.size(1), guidance_scale=gs,
    )
    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol, rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
    if gs > 0:
        samples = torch.chunk(samples, 2)[0]
    decoded = vae_model.batch_decode(samples.detach().unsqueeze(2)).squeeze(2)
    image = denorm(decoded)[0]
    return Image.fromarray(image)


# ─── Phase 1: find failure prompts (single generation) ────────────────────

def run_find(args):
    device = torch.device("cuda:0")
    weight_type = torch.bfloat16
    config = OmegaConf.load(CONFIG_PATH)

    vae_model = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type, device=device,
    )
    model = load_model(config, device, checkpoint_dir=None)
    gen_model = _unwrap(model)
    tokenizer, showo_token_ids, hyper = build_hyper(config)

    transport_obj = create_transport(
        config.transport.path_type, config.transport.prediction,
        config.transport.loss_weight, config.transport.train_eps,
        config.transport.sample_eps, config.transport.snr_type,
    )
    sampler = Sampler(transport_obj)
    api_base = config.teacher.api_base
    model_name = config.teacher.model_name

    all_prompts = Path(PROMPT_TXT).read_text(encoding="utf-8").splitlines()
    all_prompts = [p.strip() for p in all_prompts if p.strip()]
    shard_prompts = all_prompts[args.gpu_shard :: args.num_shards]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fail_path = OUTPUT_DIR / f"fail_shard{args.gpu_shard}.jsonl"
    pass_path = OUTPUT_DIR / f"pass_shard{args.gpu_shard}.jsonl"
    progress_path = OUTPUT_DIR / f"progress_shard{args.gpu_shard}.json"

    done_ids = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        done_ids = set(progress.get("done_ids", []))
        logger.info("Resuming shard %d: %d already done", args.gpu_shard, len(done_ids))

    logger.info(
        "Shard %d/%d: %d prompts (total %d)",
        args.gpu_shard, args.num_shards, len(shard_prompts), len(all_prompts),
    )

    n_fail = 0
    n_pass = 0
    n_processed = 0

    with open(fail_path, "a", encoding="utf-8") as f_fail, \
         open(pass_path, "a", encoding="utf-8") as f_pass:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = local_idx * args.num_shards + args.gpu_shard
            pid = f"dalle3_{global_idx:05d}"

            if pid in done_ids:
                continue

            t0 = time.time()

            try:
                img = generate_image(
                    gen_model, vae_model, sampler, tokenizer, showo_token_ids,
                    hyper, config, prompt, device, weight_type,
                )
                score = teacher_score(img, prompt, api_base, model_name)
            except Exception as exc:
                logger.error("[shard%d] gen error for %s: %s", args.gpu_shard, pid, exc)
                score = {
                    "correct": 0,
                    "reason": str(exc),
                    "error": True,
                }

            is_ok = score.get("correct", 0) == 1

            entry = {
                "id": pid,
                "prompt": prompt,
                "correct": score["correct"],
                "reason": score.get("reason", ""),
            }
            if not is_ok:
                f_fail.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_fail.flush()
                n_fail += 1
            else:
                f_pass.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_pass.flush()
                n_pass += 1

            done_ids.add(pid)
            n_processed += 1
            elapsed = time.time() - t0

            if n_processed % 10 == 0:
                progress_path.write_text(
                    json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False),
                    encoding="utf-8",
                )

            status_str = "PASS" if is_ok else "FAIL"
            logger.info(
                "[shard%d] %4d/%d  %s  %.1fs  fail=%d pass=%d  %.50s",
                args.gpu_shard, n_processed, len(shard_prompts),
                status_str, elapsed, n_fail, n_pass, prompt,
            )

    progress_path.write_text(
        json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Shard %d done: %d processed, %d fail, %d pass",
        args.gpu_shard, n_processed, n_fail, n_pass,
    )


# ─── Phase 2: merge shards and build 4:1 mixed dataset ──────────────────

def run_merge(args):
    fail_records = []
    pass_records = []

    for path in sorted(OUTPUT_DIR.glob("fail_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fail_records.append(json.loads(line))
        logger.info("Loaded %s (cumulative fail: %d)", path.name, len(fail_records))

    for path in sorted(OUTPUT_DIR.glob("pass_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pass_records.append(json.loads(line))
        logger.info("Loaded %s (cumulative pass: %d)", path.name, len(pass_records))

    seen = set()
    unique_fail = []
    for r in fail_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique_fail.append(r)

    seen_pass = set()
    unique_pass = []
    for r in pass_records:
        if r["id"] not in seen_pass:
            seen_pass.add(r["id"])
            unique_pass.append(r)

    merged_fail_path = OUTPUT_DIR / "dalle3_fail.json"
    with open(merged_fail_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Prompts from dalle3.txt that failed show-o2-1.5B generation (hard_ok=False)",
                "total_prompts_scanned": len(unique_fail) + len(unique_pass),
                "fail_count": len(unique_fail),
                "pass_count": len(unique_pass),
                "fail_rate": len(unique_fail) / max(len(unique_fail) + len(unique_pass), 1),
                "prompts": unique_fail,
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Merged fail JSON saved: %s (%d entries)", merged_fail_path, len(unique_fail))

    n_fail = len(unique_fail)
    n_pass_needed = n_fail * args.ratio
    if n_pass_needed > len(unique_pass):
        logger.warning(
            "Not enough pass prompts for %d:1 ratio: need %d, have %d. Using all pass prompts.",
            args.ratio, n_pass_needed, len(unique_pass),
        )
        n_pass_needed = len(unique_pass)

    random.seed(42)
    sampled_pass = random.sample(unique_pass, n_pass_needed)

    mixed = []
    for r in unique_fail:
        mixed.append({"id": r["id"], "prompt": r["prompt"], "label": "hard"})
    for r in sampled_pass:
        mixed.append({"id": r["id"], "prompt": r["prompt"], "label": "easy"})
    random.shuffle(mixed)

    mixed_path = OUTPUT_DIR / "dalle3_mixed_4to1.json"
    with open(mixed_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": f"Mixed dataset: {args.ratio}:1 (pass:fail) ratio for testing",
                "ratio": f"{args.ratio}:1",
                "hard_count": n_fail,
                "easy_count": n_pass_needed,
                "total": len(mixed),
                "prompts": mixed,
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info(
        "Mixed %d:1 dataset saved: %s (%d hard + %d easy = %d total)",
        args.ratio, mixed_path, n_fail, n_pass_needed, len(mixed),
    )


# ─── Phase 3: test on 4:1 mixed dataset ─────────────────────────────────

def run_test(args):
    mixed_path = OUTPUT_DIR / "dalle3_mixed_4to1.json"
    if not mixed_path.exists():
        logger.error("Mixed dataset not found: %s. Run --phase merge first.", mixed_path)
        return

    data = json.loads(mixed_path.read_text(encoding="utf-8"))
    prompts = data["prompts"]

    device = torch.device("cuda:0")
    weight_type = torch.bfloat16
    config = OmegaConf.load(CONFIG_PATH)

    vae_model = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type, device=device,
    )
    model = load_model(config, device, checkpoint_dir=None)
    gen_model = _unwrap(model)
    tokenizer, showo_token_ids, hyper = build_hyper(config)

    transport_obj = create_transport(
        config.transport.path_type, config.transport.prediction,
        config.transport.loss_weight, config.transport.train_eps,
        config.transport.sample_eps, config.transport.snr_type,
    )
    sampler = Sampler(transport_obj)
    api_base = config.teacher.api_base
    model_name = config.teacher.model_name

    results = []
    test_output = OUTPUT_DIR / "test_results_4to1.jsonl"

    start = args.start
    end = min(args.end, len(prompts)) if args.end else len(prompts)
    test_prompts = prompts[start:end]

    logger.info("Testing %d prompts (indices %d-%d) from mixed dataset", len(test_prompts), start, end)

    with open(test_output, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(test_prompts, start=start):
            prompt = rec["prompt"]
            label = rec["label"]
            pid = rec["id"]

            t0 = time.time()
            try:
                img = generate_image(
                    gen_model, vae_model, sampler, tokenizer, showo_token_ids,
                    hyper, config, prompt, device, weight_type,
                )
                score = teacher_score(img, prompt, api_base, model_name)
            except Exception as exc:
                logger.error("Test gen error for %s: %s", pid, exc)
                score = {
                    "correct": 0,
                    "reason": str(exc),
                    "error": True,
                }
            elapsed = time.time() - t0

            entry = {
                "id": pid,
                "prompt": prompt,
                "label": label,
                "correct": score["correct"],
                "reason": score.get("reason", ""),
                "elapsed_s": round(elapsed, 2),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(entry)

            status = "OK" if score["correct"] == 1 else "FAIL"
            logger.info(
                "[test] %3d/%d  %s  %s  %.1fs  %.50s",
                i + 1 - start, len(test_prompts), label.upper(), status,
                elapsed, prompt,
            )

    hard_results = [r for r in results if r["label"] == "hard"]
    easy_results = [r for r in results if r["label"] == "easy"]

    def _stats(items, name):
        if not items:
            return
        ok = sum(1 for r in items if r["correct"] == 1)
        logger.info(
            "  %s: %d items, pass=%d (%.1f%%)",
            name, len(items), ok, ok / len(items) * 100,
        )

    logger.info("=== Test Summary ===")
    _stats(results, "ALL")
    _stats(hard_results, "HARD")
    _stats(easy_results, "EASY")
    logger.info("Results saved to %s", test_output)


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dalle3.txt with show-o2-1.5B double generation")
    parser.add_argument("--phase", choices=["find", "merge", "test"], required=True)
    parser.add_argument("--gpu-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ratio", type=int, default=4, help="Pass:Fail ratio for mixed dataset (default 4:1)")
    parser.add_argument("--start", type=int, default=0, help="Start index for test phase")
    parser.add_argument("--end", type=int, default=None, help="End index for test phase")
    args = parser.parse_args()

    if args.phase == "find":
        run_find(args)
    elif args.phase == "merge":
        run_merge(args)
    elif args.phase == "test":
        run_test(args)
