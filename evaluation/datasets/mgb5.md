# Dataset: MGB-5 (Moroccan Arabic / Darija)

```json:config
{
  "hf_path": "ArabicSpeech/MGB-5",
  "text_column": "text",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 16000,
  "is_gated": false,
  "languages": ["ar"]
}
```

This document describes the configuration and structure of the MGB-5 dataset.

## 1. Dataset Information

- **Hugging Face Path**: `ArabicSpeech/MGB-5`
- **Description**: MGB-5 is a Moroccan Arabic (Darija) speech recognition dataset. It contains approximately 100+ hours of Moroccan dialectal Arabic speech, providing coverage of a North African Arabic dialect.
- **Languages**: Arabic (Moroccan Darija)

## 2. Data Structure

### Splits

- `train`: 33,300 samples
- `validation`: 6,160 samples
- `test`: 5,750 samples

Total: 45,253 samples. Download size: 6.65 GB.

### Features

- `audio`: Audio data (array + sampling rate)
- `text`: Ground truth transcription in Moroccan Arabic
- `id`: Unique utterance identifier

## 3. Preprocessing

- **Audio Preprocessing**: Audio is provided at 16kHz. No resampling needed.
- **Text Preprocessing**: The `text` column contains Moroccan Arabic transcriptions. Standard normalization applies.

## 4. Language Mapping

All samples map to `Arabic` for Qwen3-ASR language prefix.

## 5. Example Sample

```json
{
  "audio": {
    "array": [...],
    "sampling_rate": 16000
  },
  "text": "هذا مثال ديال الدارجة المغربية",
  "id": "mgb5_001"
}
```
