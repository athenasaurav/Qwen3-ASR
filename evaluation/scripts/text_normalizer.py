"""
Text normalization for ASR evaluation metrics (WER/CER).

Provides language-aware normalization to ensure fair comparison
between ground truth and model predictions.
"""

import re
import unicodedata


# CJK Unicode ranges for detecting CJK characters
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
]

# Languages where CER is used and spaces should be removed
_CJK_LANGUAGE_CODES = {
    "zh", "zh-cn", "zh-tw", "zh-hk", "ja", "ko", "yue",
    "zh_cn", "zh_tw", "zh_hk", "yue_hant_hk", "ja_jp", "ko_kr",
    "chinese", "japanese", "korean", "cantonese",
}


def _is_cjk_char(char: str) -> bool:
    """Check if a character is in CJK Unicode ranges."""
    cp = ord(char)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _is_punctuation(char: str) -> bool:
    """Check if a character is Unicode punctuation (category P*)."""
    return unicodedata.category(char).startswith("P")


def is_cjk_language(language: str) -> bool:
    """Check if the given language code/name is a CJK language."""
    return language.lower().strip() in _CJK_LANGUAGE_CODES


def normalize_text(text: str, language: str) -> str:
    """
    Normalize text for WER/CER computation.

    Steps:
    1. Strip whitespace
    2. Convert to lowercase
    3. Remove punctuation (Unicode-aware)
    4. For CJK languages: remove all spaces
    5. For non-CJK: collapse multiple spaces to single space

    Args:
        text: Raw text to normalize.
        language: Language code or name (e.g. "en", "zh-CN", "Chinese").

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Lowercase
    text = text.strip().lower()

    # Remove all Unicode punctuation
    text = "".join(c for c in text if not _is_punctuation(c))

    # Remove common symbols that aren't caught by Unicode punctuation category
    text = re.sub(r"[""''「」『』【】《》〈〉（）\(\)\[\]\{\}]", "", text)

    if is_cjk_language(language):
        # For CJK: remove all whitespace (words aren't space-delimited)
        text = re.sub(r"\s+", "", text)
    else:
        # For non-CJK: collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

    return text
