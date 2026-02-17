# Dataset: MASC (Massive Arabic Speech Corpus)

```json:config
{
  "hf_path": "pain/MASC",
  "text_column": "text",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 16000,
  "is_gated": false,
  "languages": ["ar"]
}
```

This document describes the configuration and structure of the MASC dataset.

## 1. Dataset Information

- **Hugging Face Path**: `pain/MASC`
- **Description**: MASC (Massive Arabic Speech Corpus) is a large-scale multi-dialect Arabic speech dataset containing approximately 1,000 hours of audio sourced from 700+ YouTube channels. It covers multiple Arabic dialects, genres, and recording conditions.
- **Languages**: Arabic (multi-dialect)
- **License**: CC-BY-4.0

## 2. Data Structure

### Splits

- `train`: Training data
- `dev`: Development/validation data
- `test`: Test data

### Features

- `audio`: Audio data (array + sampling rate at 16kHz)
- `text`: Ground truth transcription
- `video_id`: Source YouTube video identifier
- `start`: Segment start time (float)
- `end`: Segment end time (float)
- `duration`: Segment duration in seconds (float)
- `type`: Data quality label — `"c"` for clean, `"n"` for noisy
- `file_path`: Original file path

## 3. Preprocessing

- **Audio Preprocessing**: Audio is provided at 16kHz. No resampling needed.
- **Text Preprocessing**: Standard Arabic text normalization (remove diacritics, normalize characters). The `text` column contains transcriptions ready for use.
- **Filtering**: Use `type == "c"` to select only clean audio samples for higher quality training.

## 4. Language Mapping

All samples map to `Arabic` for Qwen3-ASR language prefix.

## 5. Example Sample

```json
{
  "audio": {
    "array": [...],
    "sampling_rate": 16000
  },
  "text": "مثال على النص العربي",
  "video_id": "abc123",
  "start": 10.5,
  "end": 15.2,
  "duration": 4.7,
  "type": "c",
  "file_path": "path/to/segment.wav"
}
```
