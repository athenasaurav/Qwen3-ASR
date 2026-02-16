"""
Convert a HuggingFace ASR dataset into the JSONL + WAV format
required by qwen3_asr_sft.py.

Usage example (sawtarabi):
    python finetuning/prepare_hf_dataset.py \
      --hf_path ArabicSpeech/sawtarabi \
      --text_column text_not_diacritized \
      --audio_column audio \
      --dialect_column dialect \
      --dialect_map "MSA=Arabic,EGY=Arabic,CS_EGY_ENG=Arabic,English=English" \
      --splits train,validation \
      --output_dir ./finetuning/data/sawtarabi

Output JSONL format (one JSON per line):
    {"audio": "/abs/path/to/000000.wav", "text": "language Arabic<asr_text>actual text"}
"""

import argparse
import json
import os
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm


def parse_dialect_map(map_str: str) -> dict:
    """Parse 'KEY1=VAL1,KEY2=VAL2' into {KEY1: VAL1, KEY2: VAL2}."""
    if not map_str:
        return {}
    mapping = {}
    for pair in map_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(f"Invalid dialect_map entry (missing '='): '{pair}'")
        key, val = pair.split("=", 1)
        mapping[key.strip()] = val.strip()
    return mapping


def process_split(
    dataset,
    split_name: str,
    text_column: str,
    audio_column: str,
    dialect_column: str | None,
    dialect_map: dict,
    output_dir: str,
    target_sr: int,
):
    """Process a single dataset split: extract WAVs and write JSONL."""
    wavs_dir = os.path.join(output_dir, "wavs", split_name)
    os.makedirs(wavs_dir, exist_ok=True)

    # Use "eval.jsonl" for validation split (matches qwen3_asr_sft.py convention)
    jsonl_name = "eval.jsonl" if split_name == "validation" else f"{split_name}.jsonl"
    jsonl_path = os.path.join(output_dir, jsonl_name)

    lang_counts = defaultdict(int)
    skipped = 0
    written = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            # Get text
            text = item.get(text_column, "")
            if not text or not text.strip():
                skipped += 1
                continue

            text = text.strip()

            # Get language label
            if dialect_column and dialect_column in item:
                raw_dialect = str(item[dialect_column]).strip()
                lang_name = dialect_map.get(raw_dialect, raw_dialect)
            else:
                lang_name = "None"

            # Format target text
            target = f"language {lang_name}<asr_text>{text}"

            # Extract and save audio
            audio_data = item[audio_column]
            audio_array = np.array(audio_data["array"], dtype=np.float32)
            source_sr = audio_data["sampling_rate"]

            # Resample if needed
            if source_sr != target_sr:
                audio_array = librosa.resample(
                    audio_array, orig_sr=source_sr, target_sr=target_sr
                )

            wav_filename = f"{i:06d}.wav"
            wav_path = os.path.join(wavs_dir, wav_filename)
            sf.write(wav_path, audio_array, target_sr)

            # Write JSONL entry with absolute path
            entry = {
                "audio": os.path.abspath(wav_path),
                "text": target,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            lang_counts[lang_name] += 1
            written += 1

    return written, skipped, lang_counts, jsonl_path


def main():
    p = argparse.ArgumentParser(
        description="Convert a HuggingFace ASR dataset to JSONL + WAV for fine-tuning."
    )
    p.add_argument("--hf_path", type=str, required=True,
                    help="HuggingFace dataset path (e.g. ArabicSpeech/sawtarabi)")
    p.add_argument("--hf_config", type=str, default=None,
                    help="HuggingFace dataset config/subset name")
    p.add_argument("--text_column", type=str, required=True,
                    help="Column containing ground truth text")
    p.add_argument("--audio_column", type=str, default="audio",
                    help="Column containing audio data")
    p.add_argument("--dialect_column", type=str, default=None,
                    help="Column with dialect/language labels per sample")
    p.add_argument("--dialect_map", type=str, default="",
                    help="Comma-separated KEY=VALUE mapping of dialect labels to "
                         "Qwen3-ASR language names (e.g. 'MSA=Arabic,EGY=Arabic,English=English')")
    p.add_argument("--splits", type=str, default="train",
                    help="Comma-separated split names to process (e.g. 'train,validation')")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for JSONL files and WAV audio")
    p.add_argument("--sr", type=int, default=16000,
                    help="Target audio sampling rate (default: 16000)")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Max samples per split (for testing)")

    args = p.parse_args()

    # Load .env for HF token
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    load_dotenv(os.path.join(project_root, ".env"))

    dialect_map = parse_dialect_map(args.dialect_map)
    splits = [s.strip() for s in args.splits.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {args.hf_path}")
    print(f"Config: {args.hf_config or '(default)'}")
    print(f"Text column: {args.text_column}")
    print(f"Audio column: {args.audio_column}")
    print(f"Dialect column: {args.dialect_column or '(none)'}")
    print(f"Dialect map: {dialect_map or '(none — will use language None)'}")
    print(f"Splits: {splits}")
    print(f"Target SR: {args.sr}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Load dataset
    print("Loading dataset from HuggingFace...")
    ds = load_dataset(args.hf_path, name=args.hf_config)

    for split in splits:
        if split not in ds:
            print(f"WARNING: Split '{split}' not found in dataset. Available: {list(ds.keys())}")
            continue

        split_data = ds[split]
        if args.max_samples:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))

        print(f"\n--- Processing split: {split} ({len(split_data)} samples) ---")

        written, skipped, lang_counts, jsonl_path = process_split(
            dataset=split_data,
            split_name=split,
            text_column=args.text_column,
            audio_column=args.audio_column,
            dialect_column=args.dialect_column,
            dialect_map=dialect_map,
            output_dir=args.output_dir,
            target_sr=args.sr,
        )

        print(f"\nSplit '{split}' complete:")
        print(f"  Written: {written} samples")
        print(f"  Skipped (empty text): {skipped}")
        print(f"  JSONL: {jsonl_path}")
        print(f"  Language distribution:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            print(f"    {lang}: {count}")

    # Print total disk usage
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(args.output_dir):
        for fn in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, fn))
    print(f"\nTotal disk usage: {total_size / (1024 * 1024):.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
