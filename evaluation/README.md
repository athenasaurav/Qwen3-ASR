# Evaluation Guide

This directory contains everything needed to evaluate Qwen3-ASR models on HuggingFace ASR datasets.

## Quick Start

```bash
# Evaluate base model on LibriSpeech (English, non-gated)
python evaluation/scripts/evaluate_model.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --dataset_name librispeech \
  --language en

# Quick test with 10 samples
python evaluation/scripts/evaluate_model.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --dataset_name librispeech \
  --language en \
  --max_samples 10
```

## Supported Datasets

| Dataset | Config Name | Gated | Languages | Text Column |
|---------|-------------|-------|-----------|-------------|
| LibriSpeech | `librispeech` | No | `en` | `text` |
| Common Voice | `common_voice` | Yes | `en`, `zh-CN`, `de`, `fr`, `ja`, `ko`, ... | `sentence` |
| FLEURS | `fleurs` | No | `en_us`, `zh_cn`, `de_de`, `fr_fr`, `ja_jp`, ... | `transcription` |

To add a new dataset, create a `.md` file in `evaluation/datasets/` following `TEMPLATE.md`.

## Evaluating Different Model Types

### Base Model
```bash
python evaluation/scripts/evaluate_model.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --dataset_name fleurs \
  --language en_us
```

### Full Fine-tune Checkpoint
```bash
python evaluation/scripts/evaluate_model.py \
  --model_path ./finetuning/runs/my-sft-run/checkpoint-200 \
  --dataset_name librispeech \
  --language en
```

### LoRA / QLoRA Adapter
```bash
python evaluation/scripts/evaluate_model.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --adapter_path ./finetuning/runs/my-lora-run/checkpoint-200 \
  --dataset_name common_voice \
  --language zh-CN
```

## Gated Datasets (e.g., Common Voice)

Some datasets require a HuggingFace token. Set it in the project root `.env` file:

```
HF_TOKEN=hf_your_token_here
```

The evaluation script loads this automatically via `python-dotenv`. You can also pass `--hf_token` directly.

## Metrics

- **WER** (Word Error Rate) — Used for non-CJK languages (English, German, French, etc.)
- **CER** (Character Error Rate) — Auto-selected for CJK languages (Chinese, Japanese, Korean, Cantonese)

The script determines the metric automatically based on the `--language` code.

## Output Format

Results are saved to `evaluation/results/{model_name}/{dataset}_{language}_{timestamp}/`:

```
├── transcriptions.json   # Per-sample: ground truth, prediction (raw + normalized)
├── summary.json           # Metadata + WER/CER score + timing + hardware info
├── report.md              # Human-readable markdown report
└── errors.json            # Failed samples (only if errors occurred)
```

## CLI Reference

```
python evaluation/scripts/evaluate_model.py \
  --model_path MODEL_PATH       # Required: HF ID or local path
  --dataset_name DATASET_NAME   # Required: matches .md file in datasets/
  --language LANGUAGE            # Required: language code
  [--adapter_path ADAPTER_PATH] # Optional: LoRA/QLoRA adapter directory
  [--split SPLIT]               # Optional: override dataset split
  [--output_dir OUTPUT_DIR]     # Optional: custom output location
  [--device DEVICE]             # Optional: cuda device (default: cuda:0)
  [--dtype DTYPE]               # Optional: bfloat16/float16/float32
  [--max_samples N]             # Optional: limit samples for testing
  [--hf_token TOKEN]            # Optional: HF token (default: from .env)
```
