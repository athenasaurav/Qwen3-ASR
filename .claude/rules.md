# Path-Specific Rules for the Qwen3-ASR Project

This document defines a set of rules that are enforced based on file paths, using glob patterns for matching. These rules ensure consistency and adherence to the project's architectural principles.

## Rule for Dataset Configurations

**Path**: `/evaluation/datasets/**/*.md`

**Description**: This rule governs the creation and modification of dataset configuration files. All new datasets must be documented in a standardized Markdown format to ensure consistency and ease of use.

**Rule**: **ALWAYS** use the template provided in `/evaluation/datasets/TEMPLATE.md` when creating a new dataset configuration file. This ensures that all necessary information, such as the Hugging Face path, data structure, and preprocessing steps, is included.

## Rule for Evaluation Results

**Path**: `/evaluation/results/**/*`

**Description**: This rule enforces a standardized format for storing evaluation results. To maintain a clean and organized results directory, only specific file types are permitted.

**Rule**: **ONLY** allow the creation of `.md`, `.json`, and `.csv` files within this directory. All other file formats are strictly prohibited. This ensures that all evaluation outputs are in a human-readable or easily parsable format.

## Rule for Fine-tuning Outputs

**Path**: `/finetuning/runs/**/*`

**Description**: The outputs of fine-tuning runs, including model checkpoints and logs, are managed automatically by the `finetuner` agent. This rule prevents manual or accidental modifications to these important artifacts.

**Rule**: **NEVER** permit the direct editing of any files within this directory. All changes must be the result of a properly executed fine-tuning script, ensuring the integrity and reproducibility of the fine-tuning process.

## Rule for Core Model Implementation

**Path**: `/qwen_asr/core/**/*.py`

**Description**: The core model implementation, located in this directory, is the heart of the Qwen3-ASR project. Modifications to these files can have far-reaching and unintended consequences.

**Rule**: **NEVER** modify any file within this directory without obtaining explicit approval from the project owner. Any proposed changes must be accompanied by a thorough justification and a comprehensive testing plan.
