"""
Step 3: Evaluate generated images using Qwen3-VL via vLLM OpenAI-compatible API

Sends each generated image + its prompt to Qwen3-VL for strict evaluation.
The model judges whether the image faithfully matches the prompt.

Usage:
    python evaluate_images.py \
        --results outputs/generated/generation_results.json \
        --output outputs/eval_results.json \
        --api-url http://localhost:8000/v1

Prerequisites:
    - Qwen3-VL eval server must be running (see start_eval_server.sh)
    - pip install openai
"""

import os
import json
import base64
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = """You are an expert image evaluator.

Your task is to determine whether the given image faithfully satisfies the visual instruction and the expectation checklist.

Follow these rules strictly:
1. The image must match **all** expectations, including:
   - Object classes
   - Counts of each object
   - Colors of each object
   - Spatial position within the image (e.g., "above", "below", based on real pixel position)
   - Size and relative scale of objects
2. The image must appear as a **natural, coherent, photo-like single image**.
   - Do NOT allow stylized images (e.g., cartoons, sketches, anime).
   - Do NOT allow collage-style or multi-panel images. Only one consistent, realistic scene is acceptable.
3. Be very strict and conservative in your judgment. 

Return your result as a JSON object using this format:
{
  "correct": 1 if the image fully satisfies all expectations, else 0,
  "reason": "You may explain in detail what is missing or incorrect"
}
"""


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def evaluate_single(client, model_name, prompt: str, image_path: str, index: int,
                    max_retries: int = 3) -> dict:
    """Evaluate a single image against its prompt."""
    if not os.path.exists(image_path):
        return {
            "index": index,
            "prompt": prompt,
            "image_path": image_path,
            "correct": 0,
            "reason": "Image file not found",
            "eval_status": "error",
        }

    img_b64 = encode_image_base64(image_path)
    ext = Path(image_path).suffix.lstrip(".")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    user_message = (
        f"**Visual Instruction (Prompt):** {prompt}\n\n"
        f"Please evaluate whether the image below faithfully satisfies the above instruction. "
        f"Check all object classes, counts, colors, spatial positions, sizes, and overall coherence."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img_b64}"
                    },
                },
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )

            raw_text = response.choices[0].message.content.strip()

            json_str = raw_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            if "{" in json_str:
                start = json_str.index("{")
                end = json_str.rindex("}") + 1
                json_str = json_str[start:end]

            result = json.loads(json_str)
            correct = int(result.get("correct", 0))
            reason = result.get("reason", "")

            return {
                "index": index,
                "prompt": prompt,
                "image_path": image_path,
                "correct": correct,
                "reason": reason,
                "raw_response": raw_text,
                "eval_status": "success",
            }

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {
                "index": index,
                "prompt": prompt,
                "image_path": image_path,
                "correct": 0,
                "reason": f"Failed to parse JSON from model response: {raw_text[:200]}",
                "raw_response": raw_text,
                "eval_status": "parse_error",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [Retry {attempt+1}] Error evaluating index {index}: {e}, waiting {wait}s")
                time.sleep(wait)
                continue
            return {
                "index": index,
                "prompt": prompt,
                "image_path": image_path,
                "correct": 0,
                "reason": f"API error: {str(e)}",
                "eval_status": "api_error",
            }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images with Qwen3-VL")
    parser.add_argument("--results", type=str, default="outputs/generated/generation_results.json",
                        help="Path to generation results JSON")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--model-name", type=str, default="qwen3-vl-eval",
                        help="Model name as served by vLLM")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel evaluation workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing eval results (skip already evaluated)")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.api_url, api_key="dummy")

    with open(args.results, "r", encoding="utf-8") as f:
        gen_results = json.load(f)

    successful = [r for r in gen_results if r.get("status") == "success" and r.get("image_path")]
    print(f"Total generation results: {len(gen_results)}")
    print(f"Successfully generated images: {len(successful)}")

    already_evaluated = set()
    existing_results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing_results = json.load(f)
        already_evaluated = {r["index"] for r in existing_results}
        print(f"Resuming: {len(already_evaluated)} already evaluated")

    to_evaluate = [r for r in successful if r["index"] not in already_evaluated]
    print(f"To evaluate: {len(to_evaluate)}")

    eval_results = list(existing_results)
    completed = 0
    total = len(to_evaluate)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for r in to_evaluate:
            future = executor.submit(
                evaluate_single,
                client, args.model_name,
                r["prompt"], r["image_path"], r["index"],
            )
            futures[future] = r

        for future in as_completed(futures):
            result = future.result()
            eval_results.append(result)
            completed += 1

            status = "PASS" if result["correct"] == 1 else "FAIL"
            print(f"[{completed}/{total}] idx={result['index']} {status} | {result['prompt'][:60]}...")

            if completed % 50 == 0:
                eval_results.sort(key=lambda x: x["index"])
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(eval_results, f, ensure_ascii=False, indent=2)
                print(f"  (checkpoint saved: {completed}/{total})")

    eval_results.sort(key=lambda x: x["index"])
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    total_eval = len(eval_results)
    correct = sum(1 for r in eval_results if r.get("correct") == 1)
    incorrect = sum(1 for r in eval_results if r.get("correct") == 0)
    errors = sum(1 for r in eval_results if r.get("eval_status") != "success")

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"  Total evaluated: {total_eval}")
    print(f"  Correct (PASS):  {correct} ({100*correct/max(total_eval,1):.1f}%)")
    print(f"  Incorrect (FAIL): {incorrect} ({100*incorrect/max(total_eval,1):.1f}%)")
    print(f"  Errors:          {errors}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
