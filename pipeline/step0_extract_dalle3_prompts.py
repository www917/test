"""
Step 0: Extract DALLE3-category prompts from BLIP3o-60k dataset → dalle3.txt

BLIP3o-60k is a webdataset with 7,103 samples across 6 categories.
This script downloads the dataset and filters out the DALLE3 subset.

Since the dataset is stored as webdataset (tar), we use HuggingFace datasets
to load and filter. The "text" column contains prompts.

The DALLE3 prompts in BLIP3o-60k are typically more complex, descriptive prompts
(longer than simple "a photo of X" style prompts from other categories).
"""

import os
import json
import argparse


def extract_prompts(output_file="dalle3.txt", cache_dir=None):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    print("Downloading BLIP3o-60k dataset...")
    dataset_path = snapshot_download(
        repo_id="BLIP3o/BLIP3o-60k",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    import glob
    tar_files = glob.glob(os.path.join(dataset_path, "*.tar"))

    if not tar_files:
        for root, dirs, files in os.walk(dataset_path):
            tar_files.extend(
                os.path.join(root, f) for f in files if f.endswith(".tar")
            )

    print(f"Found {len(tar_files)} tar files")

    print("Loading dataset...")
    ds = load_dataset(
        "webdataset",
        data_files=tar_files,
        split="train",
        cache_dir=cache_dir,
    )

    print(f"Total samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    if "json" in ds.column_names:
        all_prompts = []
        dalle3_prompts = []
        for sample in ds:
            meta = json.loads(sample["json"]) if isinstance(sample["json"], str) else sample["json"]
            prompt = meta.get("text", meta.get("prompt", meta.get("caption", "")))
            category = meta.get("category", meta.get("source", ""))
            all_prompts.append((prompt, category))
            if "dalle3" in str(category).lower() or "dall-e" in str(category).lower():
                dalle3_prompts.append(prompt)

        if dalle3_prompts:
            print(f"Found {len(dalle3_prompts)} DALLE3 prompts via category field")
            with open(output_file, "w", encoding="utf-8") as f:
                for p in dalle3_prompts:
                    f.write(p.strip() + "\n")
            return

        categories = set(cat for _, cat in all_prompts)
        print(f"Available categories: {categories}")

    if "txt" in ds.column_names:
        all_prompts = []
        for sample in ds:
            text = sample["txt"]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            all_prompts.append(text.strip())
    elif "text" in ds.column_names:
        all_prompts = []
        for sample in ds:
            text = sample["text"]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            all_prompts.append(text.strip())
    else:
        print(f"Available columns: {ds.column_names}")
        print("Trying to extract text from first sample...")
        print(ds[0])
        raise ValueError("Cannot determine text column. Check dataset structure above.")

    simple_pattern_prompts = []
    complex_prompts = []

    for p in all_prompts:
        is_simple = (
            p.lower().startswith("a photo of")
            and len(p.split()) <= 15
        )
        if is_simple:
            simple_pattern_prompts.append(p)
        else:
            complex_prompts.append(p)

    print(f"\nPrompt distribution:")
    print(f"  Simple 'a photo of ...' prompts: {len(simple_pattern_prompts)}")
    print(f"  Complex/descriptive prompts: {len(complex_prompts)}")
    print(f"\nSample simple prompts: {simple_pattern_prompts[:3]}")
    print(f"Sample complex prompts: {complex_prompts[:3]}")

    print(f"\nWriting ALL {len(all_prompts)} prompts to {output_file}")
    print("(You can manually filter for DALLE3-specific ones if category info is available)")

    with open(output_file, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(p + "\n")

    print(f"Done! Wrote {len(all_prompts)} prompts to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract DALLE3 prompts from BLIP3o-60k")
    parser.add_argument("--output", type=str, default="dalle3.txt")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    extract_prompts(args.output, args.cache_dir)


if __name__ == "__main__":
    main()
