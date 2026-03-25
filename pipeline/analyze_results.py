"""
Step 4: Analyze evaluation results to identify Show-o2 failure patterns

This script reads the eval results and produces:
1. Summary statistics
2. Failed prompts list (dalle3_failures.txt)
3. Failure reason categorization
4. Prompt complexity analysis (which types of prompts Show-o2 struggles with)

Usage:
    python analyze_results.py \
        --eval-results outputs/eval_results.json \
        --output-dir outputs/analysis
"""

import os
import re
import json
import argparse
from collections import Counter, defaultdict


def categorize_failure(reason: str) -> list[str]:
    """Categorize a failure reason into one or more error types."""
    reason_lower = reason.lower()
    categories = []

    if any(w in reason_lower for w in ["count", "number", "too many", "too few", "missing object", "only one", "only two"]):
        categories.append("wrong_count")

    if any(w in reason_lower for w in ["color", "red", "blue", "green", "yellow", "white", "black", "brown", "pink", "purple", "orange"]):
        categories.append("wrong_color")

    if any(w in reason_lower for w in ["position", "spatial", "left", "right", "above", "below", "behind", "front", "top", "bottom", "center"]):
        categories.append("wrong_position")

    if any(w in reason_lower for w in ["size", "scale", "large", "small", "big", "tiny", "proportion"]):
        categories.append("wrong_size")

    if any(w in reason_lower for w in ["missing", "absent", "not present", "no ", "lacks", "without"]):
        categories.append("missing_object")

    if any(w in reason_lower for w in ["cartoon", "anime", "sketch", "stylized", "illustration", "drawing", "painting"]):
        categories.append("not_photorealistic")

    if any(w in reason_lower for w in ["collage", "multi-panel", "grid", "multiple images", "split"]):
        categories.append("collage_style")

    if any(w in reason_lower for w in ["wrong object", "incorrect object", "different object", "not a "]):
        categories.append("wrong_object")

    if any(w in reason_lower for w in ["blurry", "distorted", "artifact", "incoherent", "unnatural", "deformed"]):
        categories.append("quality_issue")

    if any(w in reason_lower for w in ["text", "word", "letter", "writing", "sign"]):
        categories.append("text_rendering")

    if not categories:
        categories.append("other")

    return categories


def analyze_prompt_complexity(prompt: str) -> dict:
    """Analyze the complexity of a prompt."""
    words = prompt.split()
    objects_mentioned = len(re.findall(r'\b(a|an|the)\s+\w+', prompt.lower()))

    has_color = bool(re.search(
        r'\b(red|blue|green|yellow|white|black|brown|pink|purple|orange|golden|silver|gray|grey)\b',
        prompt.lower()
    ))
    has_spatial = bool(re.search(
        r'\b(left|right|above|below|behind|front|top|bottom|center|next to|beside|between|on top of|under|near)\b',
        prompt.lower()
    ))
    has_count = bool(re.search(
        r'\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\b',
        prompt.lower()
    ))
    has_size = bool(re.search(
        r'\b(large|small|big|tiny|huge|enormous|little|tall|short)\b',
        prompt.lower()
    ))

    and_count = prompt.lower().count(" and ")

    return {
        "word_count": len(words),
        "object_count": objects_mentioned,
        "has_color": has_color,
        "has_spatial": has_spatial,
        "has_count": has_count,
        "has_size": has_size,
        "conjunction_count": and_count,
        "complexity_score": (
            len(words) / 10
            + objects_mentioned * 2
            + int(has_color) + int(has_spatial) * 2
            + int(has_count) + int(has_size)
            + and_count
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Show-o2 evaluation results")
    parser.add_argument("--eval-results", type=str, default="outputs/eval_results.json")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.eval_results, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    passed = [r for r in results if r.get("correct") == 1]
    failed = [r for r in results if r.get("correct") == 0]

    print("=" * 70)
    print("SHOW-O2 EVALUATION ANALYSIS")
    print("=" * 70)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total evaluated:  {total}")
    print(f"  Passed (correct): {len(passed)} ({100*len(passed)/max(total,1):.1f}%)")
    print(f"  Failed:           {len(failed)} ({100*len(failed)/max(total,1):.1f}%)")

    # --- Failure reason categorization ---
    failure_categories = Counter()
    category_examples = defaultdict(list)

    for r in failed:
        reason = r.get("reason", "")
        cats = categorize_failure(reason)
        for cat in cats:
            failure_categories[cat] += 1
            if len(category_examples[cat]) < 3:
                category_examples[cat].append({
                    "prompt": r["prompt"],
                    "reason": reason[:200],
                })

    print(f"\n📋 Failure Categories (a failure can belong to multiple categories):")
    for cat, count in failure_categories.most_common():
        pct = 100 * count / max(len(failed), 1)
        print(f"  {cat:25s}: {count:4d} ({pct:5.1f}%)")

    # --- Prompt complexity analysis ---
    pass_complexities = [analyze_prompt_complexity(r["prompt"]) for r in passed]
    fail_complexities = [analyze_prompt_complexity(r["prompt"]) for r in failed]

    print(f"\n📐 Prompt Complexity Analysis:")
    for metric in ["word_count", "object_count", "complexity_score", "conjunction_count"]:
        pass_vals = [c[metric] for c in pass_complexities] if pass_complexities else [0]
        fail_vals = [c[metric] for c in fail_complexities] if fail_complexities else [0]
        pass_avg = sum(pass_vals) / max(len(pass_vals), 1)
        fail_avg = sum(fail_vals) / max(len(fail_vals), 1)
        print(f"  {metric:25s}: PASS avg={pass_avg:.2f} | FAIL avg={fail_avg:.2f}")

    bool_metrics = ["has_color", "has_spatial", "has_count", "has_size"]
    print(f"\n  Attribute presence in failed prompts:")
    for metric in bool_metrics:
        fail_pct = 100 * sum(1 for c in fail_complexities if c[metric]) / max(len(fail_complexities), 1)
        pass_pct = 100 * sum(1 for c in pass_complexities if c[metric]) / max(len(pass_complexities), 1)
        print(f"  {metric:25s}: PASS={pass_pct:5.1f}% | FAIL={fail_pct:5.1f}%")

    # --- Save failed prompts ---
    failures_txt = os.path.join(args.output_dir, "dalle3_failures.txt")
    with open(failures_txt, "w", encoding="utf-8") as f:
        for r in failed:
            f.write(r["prompt"] + "\n")
    print(f"\n💾 Failed prompts saved to: {failures_txt}")

    # --- Save detailed failure analysis ---
    failure_analysis = {
        "summary": {
            "total": total,
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(100 * len(passed) / max(total, 1), 2),
        },
        "failure_categories": dict(failure_categories.most_common()),
        "category_examples": dict(category_examples),
        "failed_prompts": [
            {
                "index": r["index"],
                "prompt": r["prompt"],
                "reason": r.get("reason", ""),
                "categories": categorize_failure(r.get("reason", "")),
                "complexity": analyze_prompt_complexity(r["prompt"]),
            }
            for r in failed
        ],
    }

    analysis_json = os.path.join(args.output_dir, "failure_analysis.json")
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump(failure_analysis, f, ensure_ascii=False, indent=2)
    print(f"💾 Detailed failure analysis saved to: {analysis_json}")

    # --- Top hardest prompt patterns ---
    print(f"\n🔥 Top 20 Failed Prompts (sorted by complexity):")
    sorted_failures = sorted(
        zip(failed, fail_complexities),
        key=lambda x: x[1]["complexity_score"],
        reverse=True,
    )
    for i, (r, c) in enumerate(sorted_failures[:20]):
        print(f"  {i+1:2d}. [score={c['complexity_score']:.1f}] {r['prompt'][:80]}")
        print(f"      Reason: {r.get('reason', 'N/A')[:80]}")

    # --- Word-count buckets ---
    print(f"\n📊 Pass rate by prompt length:")
    buckets = defaultdict(lambda: {"pass": 0, "fail": 0})
    for r in results:
        wc = len(r["prompt"].split())
        if wc <= 10:
            bucket = "1-10 words"
        elif wc <= 20:
            bucket = "11-20 words"
        elif wc <= 30:
            bucket = "21-30 words"
        else:
            bucket = "31+ words"

        if r.get("correct") == 1:
            buckets[bucket]["pass"] += 1
        else:
            buckets[bucket]["fail"] += 1

    for bucket in ["1-10 words", "11-20 words", "21-30 words", "31+ words"]:
        if bucket in buckets:
            p = buckets[bucket]["pass"]
            fl = buckets[bucket]["fail"]
            total_b = p + fl
            rate = 100 * p / max(total_b, 1)
            print(f"  {bucket:15s}: {p}/{total_b} = {rate:.1f}% pass")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
