# Dataset: MGB-3 (Egyptian Arabic)

```json:config
{
  "hf_path": "ArabicSpeech/MGB-3",
  "text_column": "text",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 16000,
  "is_gated": false,
  "languages": ["ar"]
}
```

This document describes the configuration and structure of the MGB-3 dataset.

## 1. Dataset Information

- **Hugging Face Path**: `ArabicSpeech/MGB-3`
- **Description**: MGB-3 is an Egyptian Arabic speech recognition dataset sourced from YouTube channels. It contains ~16 hours of manually transcribed dialectal Egyptian Arabic speech across multiple genres.
- **Languages**: Arabic (Egyptian dialect)

## 2. Data Structure

### Splits

- `train`: 2,150 samples
- `validation`: 1,930 samples
- `test`: 2,470 samples

Total: 6,547 samples (~16 hours). Download size: 1.57 GB.

### Features

- `audio`: Audio data (array + sampling rate)
- `text`: Ground truth transcription in Egyptian Arabic
- `id`: Unique utterance identifier

## 3. Preprocessing

- **Audio Preprocessing**: Audio is provided at 16kHz. No resampling needed.
- **Text Preprocessing**: The `text` column contains Egyptian Arabic transcriptions. Standard normalization applies.

## 4. Language Mapping

All samples map to `Arabic` for Qwen3-ASR language prefix.

## 5. Example Sample

```json
{
  "audio": {
    "array": [...],
    "sampling_rate": 16000
  },
  "text": "ده مثال على كلام مصري",
  "id": "mgb3_001"
}
```
