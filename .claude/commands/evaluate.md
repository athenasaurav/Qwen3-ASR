# Custom Command: /evaluate

**Description**: This command initiates the evaluation of an ASR model on a specified dataset and language.

**Usage**: `/evaluate --model <model_path> --dataset <dataset_name> --language <language_code> [--adapter <adapter_path>] [--split <split>] [--max-samples <N>] [--device <device>]`

**Arguments**:
- `--model`: **(Required)** The path to the model to be evaluated. This can be a Hugging Face model ID (e.g., `Qwen/Qwen3-ASR-1.7B`) or a local path to a full-finetune checkpoint.
- `--dataset`: **(Required)** The name of the dataset to use for evaluation. Must correspond to a `.md` file in `/evaluation/datasets/` (e.g., `librispeech`, `common_voice`, `fleurs`).
- `--language`: **(Required)** The language code for the evaluation (e.g., `en`, `zh-CN`, `ja`, `en_us`). CER is auto-selected for CJK languages.
- `--adapter`: *(Optional)* Path to a LoRA/QLoRA adapter directory. When provided, the base model is loaded from `--model` and the adapter is applied on top.
- `--split`: *(Optional)* Override the default dataset split from the dataset config (e.g., `test.other` for LibriSpeech).
- `--max-samples`: *(Optional)* Limit evaluation to N samples for quick testing.
- `--device`: *(Optional)* CUDA device (default: `cuda:0`).

**Workflow**:
1.  Claude will delegate this task to the `evaluator` sub-agent.
2.  The `evaluator` agent will parse the `json:config` block from `/evaluation/datasets/<dataset_name>.md` to determine column names, HF path, and gating info.
3.  It will load HF credentials from `.env` automatically.
4.  It will execute `evaluation/scripts/evaluate_model.py` with the specified parameters.
5.  Results will be saved to `evaluation/results/{model_name}/{dataset}_{language}_{timestamp}/`.
6.  The `results-analyzer` agent can then be used to compare results.
