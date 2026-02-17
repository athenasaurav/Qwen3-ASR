"""
Prepare a mixed multi-dataset training set for Qwen3-ASR fine-tuning.

Reads a recipe JSON file that specifies multiple HuggingFace datasets,
their column mappings, language labels, and sampling limits. Downloads
each dataset, extracts audio as WAV, and produces a single combined
train.jsonl + eval.jsonl with a manifest of per-dataset/per-language stats.

Usage:
    python finetuning/prepare_mixed_dataset.py \
      --recipe ./finetuning/recipes/arabic_mixed_v1.json

Recipe JSON format: see finetuning/recipes/ for examples.

Output JSONL format (one JSON per line):
    {"audio": "/abs/path/to/000000.wav", "text": "language Arabic<asr_text>actual text"}
"""

import argparse
import json
import os
import random
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm


def process_dataset_entry(
    item,
    text_column: str,
    audio_column: str,
    language: str | None,
    dialect_column: str | None,
    dialect_map: dict | None,
    wav_path: str,
    target_sr: int,
) -> dict | None:
    """Process a single dataset item: extract audio, build JSONL entry.

    Returns a dict {"audio": ..., "text": ...} or None if the item should be skipped.
    """
    # Get text
    text = item.get(text_column, "")
    if not text or not text.strip():
        return None
    text = text.strip()

    # Determine language label
    if dialect_column and dialect_map and dialect_column in item:
        raw_dialect = str(item[dialect_column]).strip()
        lang_name = dialect_map.get(raw_dialect, raw_dialect)
    elif language:
        lang_name = language
    else:
        lang_name = "None"

    # Format target text
    target = f"language {lang_name}<asr_text>{text}"

    # Extract and save audio
    audio_data = item[audio_column]
    audio_array = np.array(audio_data["array"], dtype=np.float32)
    source_sr = audio_data["sampling_rate"]

    if source_sr != target_sr:
        audio_array = librosa.resample(
            audio_array, orig_sr=source_sr, target_sr=target_sr
        )

    sf.write(wav_path, audio_array, target_sr)

    return {
        "audio": os.path.abspath(wav_path),
        "text": target,
        "language": lang_name,
    }


def process_single_dataset(ds_config: dict, output_dir: str, target_sr: int) -> list[dict]:
    """Process one dataset entry from the recipe. Returns list of JSONL entries."""
    name = ds_config["name"]
    hf_path = ds_config["hf_path"]
    text_column = ds_config["text_column"]
    audio_column = ds_config.get("audio_column", "audio")
    splits = ds_config.get("splits", ["train"])
    language = ds_config.get("language")
    dialect_column = ds_config.get("dialect_column")
    dialect_map = ds_config.get("dialect_map")
    max_samples = ds_config.get("max_samples")
    hf_config = ds_config.get("hf_config")
    filter_rules = ds_config.get("filter")

    # For FLEURS-style datasets with multiple configs
    configs = ds_config.get("configs")
    config_to_language = ds_config.get("config_to_language")

    all_entries = []

    if configs and config_to_language:
        # Multi-config dataset (e.g., FLEURS with per-language configs)
        for cfg in configs:
            cfg_language = config_to_language.get(cfg, "None")
            wavs_dir = os.path.join(output_dir, "wavs", name, cfg)
            os.makedirs(wavs_dir, exist_ok=True)

            print(f"  Loading {hf_path} config={cfg} ...")
            try:
                ds = load_dataset(hf_path, cfg)
            except Exception as e:
                print(f"  WARNING: Failed to load {hf_path}/{cfg}: {e}")
                continue

            for split in splits:
                if split not in ds:
                    print(f"  WARNING: Split '{split}' not in {hf_path}/{cfg}")
                    continue

                split_data = ds[split]
                print(f"  Processing {name}/{cfg}/{split} ({len(split_data)} samples)")

                for i, item in enumerate(tqdm(split_data, desc=f"  {name}/{cfg}/{split}", leave=False)):
                    wav_path = os.path.join(wavs_dir, f"{split}_{i:06d}.wav")
                    entry = process_dataset_entry(
                        item=item,
                        text_column=text_column,
                        audio_column=audio_column,
                        language=cfg_language,
                        dialect_column=None,
                        dialect_map=None,
                        wav_path=wav_path,
                        target_sr=target_sr,
                    )
                    if entry:
                        entry["dataset"] = name
                        entry["config"] = cfg
                        all_entries.append(entry)
    else:
        # Standard single-config dataset
        wavs_dir = os.path.join(output_dir, "wavs", name)
        os.makedirs(wavs_dir, exist_ok=True)

        print(f"  Loading {hf_path} ...")
        ds = load_dataset(hf_path, name=hf_config)

        for split in splits:
            if split not in ds:
                print(f"  WARNING: Split '{split}' not in {hf_path}")
                continue

            split_data = ds[split]

            # Apply filter if specified
            if filter_rules:
                for col, val in filter_rules.items():
                    before = len(split_data)
                    split_data = split_data.filter(lambda x, c=col, v=val: x[c] == v)
                    print(f"  Filtered {col}=={val}: {before} -> {len(split_data)} samples")

            # Apply max_samples limit
            if max_samples and len(split_data) > max_samples:
                indices = random.sample(range(len(split_data)), max_samples)
                split_data = split_data.select(sorted(indices))

            print(f"  Processing {name}/{split} ({len(split_data)} samples)")

            for i, item in enumerate(tqdm(split_data, desc=f"  {name}/{split}", leave=False)):
                wav_path = os.path.join(wavs_dir, f"{split}_{i:06d}.wav")
                entry = process_dataset_entry(
                    item=item,
                    text_column=text_column,
                    audio_column=audio_column,
                    language=language,
                    dialect_column=dialect_column,
                    dialect_map=dialect_map,
                    wav_path=wav_path,
                    target_sr=target_sr,
                )
                if entry:
                    entry["dataset"] = name
                    all_entries.append(entry)

    return all_entries


def build_manifest(entries: list[dict]) -> dict:
    """Build a manifest with per-dataset and per-language statistics."""
    stats = {
        "total_samples": len(entries),
        "per_dataset": defaultdict(int),
        "per_language": defaultdict(int),
        "per_dataset_language": defaultdict(lambda: defaultdict(int)),
    }

    for entry in entries:
        ds = entry.get("dataset", "unknown")
        lang = entry.get("language", "unknown")
        stats["per_dataset"][ds] += 1
        stats["per_language"][lang] += 1
        stats["per_dataset_language"][ds][lang] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    return {
        "total_samples": stats["total_samples"],
        "per_dataset": dict(stats["per_dataset"]),
        "per_language": dict(stats["per_language"]),
        "per_dataset_language": {k: dict(v) for k, v in stats["per_dataset_language"].items()},
    }


def write_jsonl(entries: list[dict], path: str):
    """Write entries to a JSONL file (only audio + text fields)."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            row = {"audio": entry["audio"], "text": entry["text"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Prepare mixed multi-dataset training data from a recipe file."
    )
    p.add_argument(
        "--recipe", type=str, required=True,
        help="Path to recipe JSON file (e.g., finetuning/recipes/arabic_mixed_v1.json)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling and sampling (default: 42)"
    )
    args = p.parse_args()

    # Load .env for HF token
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    load_dotenv(os.path.join(project_root, ".env"))

    # Read recipe
    with open(args.recipe, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)

    recipe_name = recipe.get("name", "unnamed")
    output_dir = recipe.get("output_dir", f"./finetuning/data/{recipe_name}")
    target_sr = recipe.get("target_sr", 16000)
    datasets_config = recipe.get("datasets", [])

    os.makedirs(output_dir, exist_ok=True)

    print(f"Recipe: {recipe_name}")
    print(f"Description: {recipe.get('description', '')}")
    print(f"Output dir: {output_dir}")
    print(f"Target SR: {target_sr}")
    print(f"Datasets: {len(datasets_config)}")
    print()

    # Process each dataset
    all_train_entries = []
    all_eval_entries = []

    for i, ds_config in enumerate(datasets_config):
        ds_name = ds_config["name"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(datasets_config)}] Dataset: {ds_name}")
        print(f"{'='*60}")

        entries = process_single_dataset(ds_config, output_dir, target_sr)

        # Split: if dataset has validation split, use it for eval; otherwise take 5% for eval
        splits = ds_config.get("splits", ["train"])
        if "validation" in splits or "dev" in splits:
            # All entries from train go to train, validation/dev go to eval
            train_entries = [e for e in entries if "validation" not in e.get("config", "")]
            eval_entries = [e for e in entries if "validation" in e.get("config", "")]
            # If no entries ended up in eval (split logic was different), just use all as train
            if not eval_entries:
                train_entries = entries
        else:
            train_entries = entries
            eval_entries = []

        all_train_entries.extend(train_entries)
        all_eval_entries.extend(eval_entries)

        print(f"  -> {len(train_entries)} train, {len(eval_entries)} eval entries")

    # If no eval data from validation splits, sample 2% from train for eval
    if not all_eval_entries and all_train_entries:
        eval_size = max(100, int(len(all_train_entries) * 0.02))
        random.shuffle(all_train_entries)
        all_eval_entries = all_train_entries[:eval_size]
        all_train_entries = all_train_entries[eval_size:]
        print(f"\nNo validation splits found. Sampled {eval_size} eval entries from train.")

    # Shuffle training data
    random.shuffle(all_train_entries)

    # Write JSONL files
    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    print(f"\nWriting {len(all_train_entries)} train entries to {train_path}")
    write_jsonl(all_train_entries, train_path)

    print(f"Writing {len(all_eval_entries)} eval entries to {eval_path}")
    write_jsonl(all_eval_entries, eval_path)

    # Build and save manifest
    manifest = build_manifest(all_train_entries)
    manifest["eval_samples"] = len(all_eval_entries)
    manifest["recipe"] = recipe_name

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Recipe: {recipe_name}")
    print(f"Total train samples: {manifest['total_samples']}")
    print(f"Total eval samples: {manifest['eval_samples']}")
    print(f"\nPer-dataset breakdown:")
    for ds, count in sorted(manifest["per_dataset"].items(), key=lambda x: -x[1]):
        print(f"  {ds}: {count}")
    print(f"\nPer-language breakdown:")
    for lang, count in sorted(manifest["per_language"].items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

    # Disk usage
    total_size = 0
    for dirpath, _, filenames in os.walk(output_dir):
        for fn in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, fn))
    print(f"\nTotal disk usage: {total_size / (1024 * 1024):.1f} MB")

    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {eval_path}")
    print(f"  {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
