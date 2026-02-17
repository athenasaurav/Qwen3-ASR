# Custom Command: /prepare-data

**Description**: This command prepares mixed training data from multiple HuggingFace datasets using a recipe file. It downloads, processes, and combines data into the JSONL + WAV format required by `qwen3_asr_sft.py`.

**Usage**: `/prepare-data --recipe <path_to_recipe_file>`

**Arguments**:

- `--recipe`: **(Required)** The path to a recipe JSON file, typically located in `/finetuning/recipes/`. The recipe defines which datasets to use, column mappings, language labels, sampling limits, and output directory.

**Recipe File Fields**:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Recipe name (used for logging and manifest) |
| `description` | No | Human-readable description |
| `target_sr` | No | Target audio sampling rate (default: 16000) |
| `output_dir` | Yes | Output directory for combined JSONL and WAV files |
| `datasets` | Yes | Array of dataset configurations (see below) |

**Per-Dataset Fields**:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Short name for this dataset |
| `hf_path` | Yes | HuggingFace dataset path |
| `text_column` | Yes | Column containing ground truth text |
| `audio_column` | No | Column containing audio data (default: `audio`) |
| `splits` | No | Splits to process (default: `["train"]`) |
| `language` | No | Qwen3-ASR language name for all samples |
| `dialect_column` | No | Column with per-sample dialect labels |
| `dialect_map` | No | Object mapping dialect labels to language names |
| `max_samples` | No | Maximum samples to take from this dataset |
| `filter` | No | Object of `{column: value}` filters to apply |
| `configs` | No | For multi-config datasets (e.g., FLEURS), list of configs to load |
| `config_to_language` | No | Object mapping config names to language names |

**Workflow**:

1.  Claude will delegate this task to the `data-mixer` sub-agent.
2.  The agent reads and validates the recipe file.
3.  It loads HF credentials from `.env` automatically.
4.  It executes `finetuning/prepare_mixed_dataset.py --recipe <path>`.
5.  For each dataset in the recipe, it downloads from HuggingFace, extracts WAVs, and writes JSONL entries.
6.  All datasets are combined into a single shuffled `train.jsonl` and `eval.jsonl`.
7.  A `manifest.json` is generated with per-dataset/per-language sample counts.
8.  After preparation, use `/finetune --config <config_file>` pointing to the prepared data.
