"""
Dataset configuration loader for Qwen3-ASR evaluation.

Parses JSON config blocks from dataset .md files in evaluation/datasets/.
Each .md file must contain a fenced code block tagged as ```json:config
with the dataset's machine-readable configuration.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    """Machine-readable dataset configuration parsed from a .md file."""

    name: str
    hf_path: str
    text_column: str
    audio_column: str
    split: str
    sampling_rate: int
    is_gated: bool
    languages: list = field(default_factory=list)
    language_column: Optional[str] = None
    hf_config: Optional[str] = None  # HF subset/config name; supports {language} placeholder


# Languages where CER is more appropriate than WER
CJK_LANGUAGES = {
    # Codes
    "zh", "zh-cn", "zh-tw", "zh-hk", "ja", "ko", "yue",
    "zh_cn", "zh_tw", "zh_hk", "yue_hant_hk", "ja_jp", "ko_kr",
    # Canonical names (as used by Qwen3-ASR)
    "chinese", "japanese", "korean", "cantonese",
}

# Map dataset language codes to Qwen3-ASR canonical language names
LANG_CODE_TO_NAME = {
    # ISO codes
    "en": "English",
    "zh": "Chinese",
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "zh-HK": "Chinese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "sv-SE": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
    # FLEURS-style codes
    "en_us": "English",
    "zh_cn": "Chinese",
    "de_de": "German",
    "fr_fr": "French",
    "es_419": "Spanish",
    "pt_br": "Portuguese",
    "id_id": "Indonesian",
    "it_it": "Italian",
    "ko_kr": "Korean",
    "ru_ru": "Russian",
    "th_th": "Thai",
    "vi_vn": "Vietnamese",
    "ja_jp": "Japanese",
    "tr_tr": "Turkish",
    "hi_in": "Hindi",
    "ms_my": "Malay",
    "nl_nl": "Dutch",
    "sv_se": "Swedish",
    "da_dk": "Danish",
    "fi_fi": "Finnish",
    "pl_pl": "Polish",
    "cs_cz": "Czech",
    "fil_ph": "Filipino",
    "fa_ir": "Persian",
    "el_gr": "Greek",
    "ro_ro": "Romanian",
    "hu_hu": "Hungarian",
    "mk_mk": "Macedonian",
    "ar_eg": "Arabic",
    "yue_hant_hk": "Cantonese",
}

# Regex to extract ```json:config ... ``` blocks from markdown
_CONFIG_PATTERN = re.compile(
    r"```json:config\s*\n(.*?)\n```",
    re.DOTALL,
)


def _find_datasets_dir() -> str:
    """Locate the evaluation/datasets/ directory relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.normpath(os.path.join(script_dir, "..", "datasets"))
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(
            f"Dataset configs directory not found: {datasets_dir}"
        )
    return datasets_dir


def load_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Load a DatasetConfig by parsing the JSON config block from
    evaluation/datasets/{dataset_name}.md.

    Args:
        dataset_name: Name of the dataset (without .md extension).
                      Must match a file in evaluation/datasets/.

    Returns:
        DatasetConfig with all fields populated from the JSON block.

    Raises:
        FileNotFoundError: If the .md file doesn't exist.
        ValueError: If the .md file has no parseable json:config block.
    """
    datasets_dir = _find_datasets_dir()
    md_path = os.path.join(datasets_dir, f"{dataset_name}.md")

    if not os.path.isfile(md_path):
        available = [
            f.replace(".md", "")
            for f in os.listdir(datasets_dir)
            if f.endswith(".md") and f != "TEMPLATE.md"
        ]
        raise FileNotFoundError(
            f"No dataset config found for '{dataset_name}'. "
            f"Available: {available}. "
            f"Expected file: {md_path}"
        )

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = _CONFIG_PATTERN.search(content)
    if not match:
        raise ValueError(
            f"No ```json:config block found in {md_path}. "
            f"Please add a fenced JSON config block to the .md file."
        )

    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in config block of {md_path}: {e}"
        )

    required_fields = ["hf_path", "text_column", "audio_column"]
    for field_name in required_fields:
        if field_name not in raw:
            raise ValueError(
                f"Missing required field '{field_name}' in config block of {md_path}"
            )

    return DatasetConfig(
        name=dataset_name,
        hf_path=raw["hf_path"],
        text_column=raw["text_column"],
        audio_column=raw["audio_column"],
        split=raw.get("split", "test"),
        sampling_rate=raw.get("sampling_rate", 16000),
        is_gated=raw.get("is_gated", False),
        languages=raw.get("languages", []),
        language_column=raw.get("language_column"),
        hf_config=raw.get("hf_config"),
    )


def resolve_language_name(language_code: str) -> str:
    """
    Convert a dataset language code to the canonical name used by Qwen3-ASR.

    Falls back to the original code if no mapping is found.
    """
    return LANG_CODE_TO_NAME.get(language_code, language_code)


def is_cjk_language(language: str) -> bool:
    """Check if a language code/name should use CER instead of WER."""
    return language.lower() in CJK_LANGUAGES


def list_available_datasets() -> list:
    """Return names of all configured datasets."""
    datasets_dir = _find_datasets_dir()
    return sorted(
        f.replace(".md", "")
        for f in os.listdir(datasets_dir)
        if f.endswith(".md") and f != "TEMPLATE.md"
    )
