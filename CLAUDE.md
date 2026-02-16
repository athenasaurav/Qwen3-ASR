# CLAUDE.md for the Qwen3-ASR Project

This document serves as the master guide for Claude Code, providing a comprehensive framework for working on the Qwen3-ASR repository. The primary objective of this project is to establish a robust and efficient workflow for evaluating and fine-tuning advanced Automatic Speech Recognition (ASR) models.

## WHAT: The Technical Landscape

This project utilizes a modern Python-based stack for deep learning research and development. The core components are as follows:

| Category | Technology | Version/Details |
| :--- | :--- | :--- |
| **Language** | Python | 3.12 |
| **Core Framework** | PyTorch | Latest stable version |
| **Key Libraries** | `transformers` | 4.57.6 |
| | `datasets` | For data loading and processing |
| | `librosa` | For audio analysis |
| | `soundfile` | For audio file I/O |
| | `accelerate` | 1.12.0, for distributed training |
| | `vllm` | 0.14.0 (optional), for high-throughput inference |
| | `flash-attn` | Recommended for optimized attention mechanism |
| | `jiwer` | >=3.0.0, for WER/CER metric computation |
| | `python-dotenv` | For loading credentials from `.env` |
| | `tqdm` | For progress bars in evaluation |
| **Fine-tuning (optional)** | `peft` | >=0.15.0, for LoRA/QLoRA fine-tuning |
| | `bitsandbytes` | >=0.45.0, for QLoRA quantization |
| **Environment** | `conda` | For environment management |

### Project Structure

The repository is organized into the following key directories:

- `/.claude/`: Houses all Claude Code configurations, including sub-agent definitions (`agents.md`), path-specific rules (`rules.md`), and custom commands.
- `/assets/`: Contains static project assets such as images, logos, and supplementary documents.
- `/docker/`: Includes Dockerfiles for creating containerized and reproducible development environments.
- `/evaluation/`: A dedicated module for model performance assessment. Dataset `.md` files contain a `json:config` code block that the evaluation script parses for column mappings, HF paths, and gating info. Supports WER (non-CJK) and CER (CJK languages) metrics.
- `/examples/`: Provides practical examples and usage scripts for the Qwen3-ASR models.
- `/finetuning/`: Contains the unified fine-tuning script (`qwen3_asr_sft.py`) supporting three modes: SFT (full), LoRA, and QLoRA. Config JSON files in `/finetuning/configs/` drive each experiment.
- `/qwen_asr/`: The core Python package containing the source code for the Qwen3-ASR models, utilities, and inference logic.

## WHY: The Architectural Philosophy

The architecture of this project is founded on the principles of modularity and reproducibility. By leveraging the `transformers` library, we build upon a mature and well-supported foundation for natural language processing and speech tasks. The clear separation of concerns between the core model (`qwen_asr`), fine-tuning (`finetuning`), and evaluation (`evaluation`) modules allows for independent development, testing, and maintenance of each component.

For inference, the project supports both a standard `transformers` backend for flexibility and a `vllm` backend for production-level performance and high throughput. This dual-backend approach enables a seamless transition from research and development to deployment.

## HOW: The Rules of Engagement

### Environment Setup

A consistent development environment is crucial. To create a clean Python 3.12 environment, execute the following commands:

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

Next, install the essential dependencies:

```bash
pip install -U qwen-asr datasets
pip install -U flash-attn --no-build-isolation
```

For development purposes, it is recommended to install the project in editable mode:

```bash
git clone https://github.com/athenasaurav/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .

# For LoRA/QLoRA fine-tuning support:
pip install -e ".[finetune]"
```

### Credential Management

Access to Hugging Face and GitHub is required for downloading models and datasets, and for pushing code. These credentials are to be stored in a `.env` file in the project's root directory. The file should adhere to the following format:

```
GITHUB_USERNAME=athenasaurav
GITHUB_PAT=your_github_personal_access_token
HF_USERNAME=athenasaurav
HF_TOKEN=your_hugging_face_access_token
```

> **Security Note**: The `.env` file is included in the `.gitignore` file and must **NEVER** be committed to the repository. All scripts that require these credentials must be designed to load them from this file at runtime.

### Hard Rules

To ensure the integrity and consistency of the project, the following rules are to be strictly enforced:

1.  **NEVER** commit any secrets, API keys, or personal access tokens to the Git repository.
2.  **ALWAYS** execute the relevant test suite after implementing any significant code modifications.
3.  **ALWAYS** create a new, detailed Markdown configuration file in the `/evaluation/datasets` directory for any new dataset that is introduced.
4.  **NEVER** modify the core model implementation files located in `qwen_asr/core/` without obtaining explicit approval and conducting thorough validation.
5.  **ALWAYS** utilize the custom slash commands, `/evaluate` and `/finetune`, to initiate their respective workflows, and to ensure that all associated workflows and rules are correctly followed.
