# CLAUDE.md for Evaluation

This file provides specific instructions for Claude when working within the `/evaluation` directory. It inherits all rules from the root `CLAUDE.md`.

## WHAT: Tech Stack & Structure

- **Purpose**: This directory contains all resources for evaluating the performance of Qwen3-ASR models (base, full-finetune, LoRA, and QLoRA).
- **Key Libraries**:
  - `jiwer` — WER (Word Error Rate) and CER (Character Error Rate) computation
  - `python-dotenv` — Loading HF tokens from `.env` for gated datasets
  - `tqdm` — Progress bars during evaluation
  - `peft` — Loading LoRA/QLoRA adapters for finetuned model evaluation
- **Metrics**: WER for non-CJK languages, CER auto-selected for CJK languages (Chinese, Japanese, Korean, Cantonese).

### Directory Structure

- `/evaluation/datasets/`: Markdown files describing each dataset. Each `.md` file **must** contain a `json:config` fenced code block that the evaluation script parses for `hf_path`, `text_column`, `audio_column`, `split`, `sampling_rate`, `is_gated`, and `languages`.
- `/evaluation/results/`: Stores evaluation outputs organized as `{model_name}/{dataset}_{language}_{timestamp}/`.
- `/evaluation/scripts/`: Python scripts for running evaluations:
  - `evaluate_model.py` — Main evaluation script
  - `dataset_config.py` — Parses `json:config` blocks from dataset `.md` files
  - `text_normalizer.py` — Language-aware text normalization for metric computation

## WHY: Architectural Decisions

Evaluation is a critical part of the ASR model development lifecycle. Dataset `.md` files serve as both human-readable documentation and machine-readable configuration (via the `json:config` block), so adding a new dataset only requires creating one file — no script changes needed. The evaluation script auto-selects CER for CJK languages where word segmentation is unreliable.

## HOW: Commands & Workflow Rules

### Workflow

1.  **Select a model, dataset, and language**: The user will specify these via the `/evaluate` command.
2.  **Parse the dataset config**: The script reads the `json:config` block from the corresponding `.md` file to get column names, HF path, and gating info.
3.  **Load credentials**: HF token loaded from `.env` automatically (or passed via `--hf_token`).
4.  **Run evaluation**: Execute `evaluate_model.py` with appropriate flags. For LoRA/QLoRA models, pass `--adapter_path`.
5.  **Save results**: Output saved to `evaluation/results/{model}/{dataset}_{lang}_{timestamp}/` containing `transcriptions.json`, `summary.json`, `report.md`, and optionally `errors.json`.
6.  **Analyze**: Use the `results-analyzer` agent to compare results across runs.

### Hard Rules

1.  **ALWAYS** use the `/evaluate` command to initiate an evaluation run.
2.  **ALWAYS** read the `json:config` block from the dataset `.md` file — never hardcode column names.
3.  **NEVER** write evaluation results directly to the root of `/evaluation/results`. Always create a subdirectory.
4.  **ALWAYS** install required libraries before running: `pip install jiwer python-dotenv tqdm`.
5.  **ALWAYS** use CER for CJK languages (zh, ja, ko, yue) — the script does this automatically.
