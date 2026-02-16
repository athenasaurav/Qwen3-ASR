# Custom Command: /finetune

**Description**: This command initiates the fine-tuning process for a Qwen3-ASR model, leveraging a specified configuration file.

**Usage**: `/finetune --config <path_to_config_file>`

**Arguments**:

- `--config`: **(Required)** The relative path to the JSON configuration file for the fine-tuning run. This file must be located within the `/finetuning/configs` directory.

**Config File Fields**:

| Field | Required | Description |
|-------|----------|-------------|
| `model_path` | Yes | HF model ID or local path (e.g., `Qwen/Qwen3-ASR-1.7B`) |
| `train_file` | Yes | Path to training JSONL file |
| `output_dir` | Yes | Output directory for checkpoints |
| `mode` | No | `sft` (default), `lora`, or `qlora` |
| `eval_file` | No | Path to evaluation JSONL file |
| `batch_size` | No | Per-device batch size (default: 32) |
| `grad_acc` | No | Gradient accumulation steps (default: 4) |
| `lr` | No | Learning rate (default: 2e-5 for SFT, 1e-4 for LoRA) |
| `epochs` | No | Number of training epochs (default: 1) |
| `lora_r` | No | LoRA rank (default: 16, used with lora/qlora modes) |
| `lora_alpha` | No | LoRA alpha scaling (default: 32) |
| `lora_dropout` | No | LoRA dropout (default: 0.05) |
| `lora_target_modules` | No | Comma-separated target modules (default: `q_proj,v_proj`) |
| `quantization_bits` | No | 4 or 8, for QLoRA mode (default: 4) |

**Workflow**:

1.  Upon invocation, Claude will delegate the fine-tuning task to the `finetuner` sub-agent.
2.  The `finetuner` agent will read and validate the config file. For LoRA/QLoRA modes, it verifies that `peft` and `bitsandbytes` are installed.
3.  It will execute `finetuning/qwen3_asr_sft.py`, passing config parameters as CLI args.
4.  Checkpoints are saved to a timestamped subdirectory within `/finetuning/runs`.
5.  After fine-tuning, use `/evaluate --adapter <checkpoint_path>` to test the model.
