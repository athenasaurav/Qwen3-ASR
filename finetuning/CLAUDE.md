# CLAUDE.md for Finetuning

This file provides specific instructions for Claude when working within the `/finetuning` directory. It inherits all rules from the root `CLAUDE.md`.

## WHAT: Tech Stack & Structure

- **Purpose**: This directory contains all resources for fine-tuning Qwen3-ASR models.
- **Key Script**: `finetuning/qwen3_asr_sft.py` — A unified script supporting three modes:
  - **SFT** (`--mode sft`): Full parameter fine-tuning (default)
  - **LoRA** (`--mode lora`): Low-Rank Adaptation via the `peft` library
  - **QLoRA** (`--mode qlora`): Quantized LoRA (4-bit/8-bit) via `peft` + `bitsandbytes`
- **Key Libraries**:
  - `peft` — Required for LoRA/QLoRA modes
  - `bitsandbytes` — Required for QLoRA quantization
  - `python-dotenv` — Loading credentials from `.env`

### Directory Structure

- `/finetuning/configs/`: Configuration files for fine-tuning runs. Examples provided:
  - `example_finetune_config.json` — SFT (full fine-tuning)
  - `example_lora_config.json` — LoRA fine-tuning
  - `example_qlora_config.json` — QLoRA fine-tuning
- `/finetuning/runs/`: Stores the output of fine-tuning runs, including checkpoints and logs.

## WHY: Architectural Decisions

A single unified script handles all three fine-tuning modes to avoid code duplication. The DataCollator, Trainer, audio processing pipeline, and checkpoint management are shared across modes. LoRA/QLoRA modes use the `peft` library and save only adapter weights, making checkpoints much smaller than full fine-tuning.

For QLoRA, the base model is loaded in 4-bit quantized form to drastically reduce VRAM requirements, enabling fine-tuning of 1.7B models on consumer GPUs.

## HOW: Commands & Workflow Rules

### Workflow

1.  **Create a config file**: Create a JSON config in `/finetuning/configs/` specifying `mode`, model path, dataset, hyperparameters, and LoRA settings if applicable.
2.  **Initiate fine-tuning**: Use the `/finetune` command with the config path.
3.  **Run the script**: The `finetuner` agent executes `qwen3_asr_sft.py` with config parameters as CLI args.
4.  **Save output**: Checkpoints saved to `/finetuning/runs/`. SFT saves full model weights; LoRA/QLoRA saves adapter weights + `adapter_meta.json`.
5.  **Evaluate**: Use `/evaluate` with `--adapter_path` pointing to a LoRA/QLoRA checkpoint to measure performance.

### Hard Rules

1.  **ALWAYS** use the `/finetune` command to start a fine-tuning run.
2.  **NEVER** modify `finetuning/qwen3_asr_sft.py` without explicit user approval.
3.  **ALWAYS** create a new config file in `/finetuning/configs` for each experiment.
4.  **ALWAYS** install `peft` and `bitsandbytes` before LoRA/QLoRA runs: `pip install -e ".[finetune]"`.
5.  **ALWAYS** verify the `mode` field in the config matches the intended training approach.
