"""
Qwen3-ASR Evaluation Script.

Evaluates a Qwen3-ASR model (base, full-finetune, or LoRA/QLoRA adapter)
on any HuggingFace ASR dataset. Reads dataset configuration from .md files,
loads credentials from .env, and auto-selects CER for CJK languages.

Usage examples:

  # Base model on LibriSpeech
  python evaluate_model.py \\
    --model_path Qwen/Qwen3-ASR-1.7B \\
    --dataset_name librispeech \\
    --language en

  # LoRA adapter on Common Voice Chinese
  python evaluate_model.py \\
    --model_path Qwen/Qwen3-ASR-1.7B \\
    --adapter_path ./finetuning/runs/my-lora-checkpoint \\
    --dataset_name common_voice \\
    --language zh-CN

  # Quick test with 10 samples
  python evaluate_model.py \\
    --model_path Qwen/Qwen3-ASR-1.7B \\
    --dataset_name fleurs \\
    --language en_us \\
    --max_samples 10
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from jiwer import cer, wer
from tqdm import tqdm

# Add this script's directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import (
    is_cjk_language,
    load_dataset_config,
    resolve_language_name,
)
from text_normalizer import normalize_text


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a Qwen3-ASR model on a HuggingFace dataset."
    )

    # Model
    p.add_argument(
        "--model_path", type=str, required=True,
        help="HF model ID or local path to base model / full-finetune checkpoint.",
    )
    p.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to LoRA/QLoRA adapter directory. If provided, loads base model "
             "from --model_path and applies this adapter.",
    )

    # Dataset
    p.add_argument(
        "--dataset_name", type=str, required=True,
        help="Dataset name matching a .md file in evaluation/datasets/ "
             "(e.g. librispeech, common_voice, fleurs).",
    )
    p.add_argument(
        "--language", type=str, required=True,
        help="Language code for dataset loading (e.g. en, zh-CN, ja, en_us).",
    )
    p.add_argument(
        "--split", type=str, default=None,
        help="Override the default dataset split from config "
             "(e.g. test.other for LibriSpeech).",
    )

    # Output
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory. Default: auto-generated under evaluation/results/.",
    )

    # Inference
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for model loading (default: cuda:0).",
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to N samples (for quick testing).",
    )

    # Credentials
    p.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated datasets. Default: reads from .env HF_TOKEN.",
    )

    return p.parse_args()


def load_model(model_path, adapter_path, device, dtype_str):
    """
    Load a Qwen3-ASR model with optional LoRA adapter.

    Three modes:
    1. Base model or full-finetune checkpoint: just model_path
    2. LoRA/QLoRA adapter: model_path (base) + adapter_path (adapter dir)
       -> loads base, applies adapter, merges, wraps in Qwen3ASRModel
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]

    from qwen_asr import Qwen3ASRModel

    if adapter_path is None:
        # Base model or full-finetune checkpoint
        return Qwen3ASRModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )

    # LoRA/QLoRA adapter: load base + adapter, merge, wrap
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    print(f"  Loading base model: {model_path}")
    base_model = AutoModel.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device
    )

    print(f"  Loading adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("  Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)

    return Qwen3ASRModel(
        backend="transformers",
        model=merged_model,
        processor=processor,
        max_inference_batch_size=32,
    )


def generate_report(metadata, num_samples, num_errors):
    """Generate a human-readable markdown evaluation report."""
    score = metadata["score"]
    metric = metadata["metric"]
    score_str = f"{score:.4f}" if score is not None else "N/A"

    adapter_str = metadata["adapter_path"] or "None (base / full-finetune)"

    return f"""# Evaluation Report

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | `{metadata['model_path']}` |
| Adapter | `{adapter_str}` |
| Dataset | `{metadata['dataset_hf_path']}` |
| Language | `{metadata['language']}` ({metadata['language_canonical']}) |
| Split | `{metadata['split']}` |
| Samples | {num_samples} |
| Errors | {num_errors} |

## Results
| Metric | Score |
|--------|-------|
| {metric} | **{score_str}** |

## Environment
| Detail | Value |
|--------|-------|
| GPU | {metadata['gpu_name']} |
| Dtype | {metadata['dtype']} |
| Device | {metadata['device']} |
| Platform | {metadata['platform']} |
| Timestamp | {metadata['timestamp']} |
"""


def main():
    # Load .env from project root
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    load_dotenv(os.path.join(project_root, ".env"))

    args = parse_args()

    # Resolve HF token: CLI arg > .env > environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Load dataset config from .md file
    dataset_config = load_dataset_config(args.dataset_name)
    print(f"Dataset config: {dataset_config.hf_path} "
          f"(text_column={dataset_config.text_column}, "
          f"audio_column={dataset_config.audio_column})")

    # Resolve language name for Qwen3-ASR
    lang_canonical = resolve_language_name(args.language)

    # Auto-select metric: CER for CJK, WER for others
    use_cer = is_cjk_language(args.language)
    metric_name = "CER" if use_cer else "WER"
    print(f"Language: {args.language} -> {lang_canonical} (metric: {metric_name})")

    # Check if gated dataset needs token
    if dataset_config.is_gated and not hf_token:
        print(
            f"WARNING: Dataset '{dataset_config.hf_path}' is gated and requires "
            f"an HF token. Set HF_TOKEN in .env or pass --hf_token."
        )

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    if args.adapter_path:
        print(f"  with LoRA adapter from {args.adapter_path}")
    model = load_model(args.model_path, args.adapter_path, args.device, args.dtype)
    print("Model loaded.\n")

    # Load dataset
    split = args.split or dataset_config.split
    print(f"Loading dataset {dataset_config.hf_path} [{args.language}] split={split}...")

    ds_kwargs = {"split": split}
    if hf_token:
        ds_kwargs["token"] = hf_token

    # Some datasets use language as config name, others don't
    # LibriSpeech has no language config; Common Voice and FLEURS do
    try:
        dataset = load_dataset(dataset_config.hf_path, args.language, **ds_kwargs)
    except Exception:
        # Fallback: try loading without language as config name
        dataset = load_dataset(dataset_config.hf_path, **ds_kwargs)

    if args.max_samples:
        n = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(n))

    print(f"Dataset loaded: {len(dataset)} samples\n")

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = os.path.basename(args.model_path.rstrip("/"))
    if args.adapter_path:
        adapter_short = os.path.basename(args.adapter_path.rstrip("/"))
        model_short = f"{model_short}_adapter-{adapter_short}"

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            project_root, "evaluation", "results", model_short,
            f"{args.dataset_name}_{args.language}_{timestamp}",
        )
    os.makedirs(out_dir, exist_ok=True)

    # Run evaluation
    text_col = dataset_config.text_column
    audio_col = dataset_config.audio_column

    results = []
    ground_truths = []
    predictions = []
    errors = []
    start_time = time.time()

    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating ({metric_name})")):
        try:
            audio_data = item[audio_col]
            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]
            ground_truth = item[text_col]

            # Transcribe
            transcription = model.transcribe(
                audio=(audio_array, sampling_rate),
                language=lang_canonical,
            )
            prediction = transcription[0].text

            # Normalize for metric computation
            gt_norm = normalize_text(ground_truth, args.language)
            pred_norm = normalize_text(prediction, args.language)

            ground_truths.append(gt_norm)
            predictions.append(pred_norm)

            results.append({
                "id": i,
                "ground_truth": ground_truth,
                "ground_truth_normalized": gt_norm,
                "prediction": prediction,
                "prediction_normalized": pred_norm,
            })
        except Exception as e:
            errors.append({"id": i, "error": str(e)})

    elapsed = time.time() - start_time

    # Compute metrics
    score = None
    if ground_truths and predictions:
        if use_cer:
            score = cer(ground_truths, predictions)
        else:
            score = wer(ground_truths, predictions)

    # Build metadata
    gpu_name = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    metadata = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "dataset_name": args.dataset_name,
        "dataset_hf_path": dataset_config.hf_path,
        "language": args.language,
        "language_canonical": lang_canonical,
        "split": split,
        "num_samples": len(results),
        "num_errors": len(errors),
        "metric": metric_name,
        "score": score,
        "elapsed_seconds": round(elapsed, 2),
        "dtype": args.dtype,
        "device": args.device,
        "timestamp": timestamp,
        "gpu_name": gpu_name,
        "platform": platform.platform(),
    }

    # Save results
    with open(os.path.join(out_dir, "transcriptions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if errors:
        with open(os.path.join(out_dir, "errors.json"), "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    # Generate markdown report
    report = generate_report(metadata, len(results), len(errors))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    score_str = f"{score:.4f}" if score is not None else "N/A"
    print(f"\nEvaluation complete.")
    print(f"  {metric_name}: {score_str}")
    print(f"  Samples: {len(results)} | Errors: {len(errors)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Results: {out_dir}")


if __name__ == "__main__":
    main()
