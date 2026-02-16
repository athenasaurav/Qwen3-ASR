# Dataset: SawTarabi (Arabic Multi-Dialect + English)

```json:config
{
  "hf_path": "ArabicSpeech/sawtarabi",
  "text_column": "text_not_diacritized",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 16000,
  "is_gated": false,
  "language_column": "dialect",
  "languages": ["ar"]
}
```

This document describes the configuration and structure of the SawTarabi dataset.

## 1. Dataset Information

- **Hugging Face Path**: `ArabicSpeech/sawtarabi`
- **Description**: SawTarabi is a multi-dialect Arabic speech dataset containing Modern Standard Arabic (MSA), Egyptian Arabic (EGY), code-switched Arabic-English (CS_EGY_ENG), and English speech. It is designed for evaluating ASR systems on dialectal Arabic and mixed-language scenarios.
- **Languages**: Arabic (MSA, Egyptian), English, Code-switched Arabic-English

## 2. Data Structure

### Splits

- `train`: Training data
- `validation`: Validation data
- `test`: Test data (270 samples — 192 Arabic, 78 English as detected by Qwen3-ASR)

### Features

- `audio`: Audio data (array + sampling rate)
- `text_not_diacritized`: Ground truth transcription without diacritical marks
- `dialect`: Dialect/language label per sample (MSA, EGY, CS_EGY_ENG, English)

## 3. Preprocessing

- **Audio Preprocessing**: Audio is provided at 16kHz. No resampling needed.
- **Text Preprocessing**: The `text_not_diacritized` column contains clean text without Arabic diacritics. Standard normalization (lowercase, remove punctuation) applies.

## 4. Dialect-to-Language Mapping

For Qwen3-ASR evaluation and fine-tuning, dialects map to canonical language names:

| Dataset Dialect | Qwen3-ASR Language |
|----------------|-------------------|
| MSA | Arabic |
| EGY | Arabic |
| CS_EGY_ENG | Arabic |
| English | English |

## 5. Example Sample

```json
{
  "audio": {
    "array": [...],
    "sampling_rate": 16000
  },
  "text_not_diacritized": "هذا مثال على النص العربي",
  "dialect": "MSA"
}
```
