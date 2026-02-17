# Dataset: Common Voice 18 Arabic

```json:config
{
  "hf_path": "MohamedRashad/common-voice-18-arabic",
  "text_column": "sentence",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 48000,
  "is_gated": false,
  "languages": ["ar"]
}
```

This document describes the configuration and structure of the Common Voice 18 Arabic dataset.

## 1. Dataset Information

- **Hugging Face Path**: `MohamedRashad/common-voice-18-arabic`
- **Description**: Arabic-only extraction from Mozilla Common Voice 18.0. Contains community-validated read speech recordings from volunteer contributors. This is the most accessible HuggingFace mirror for Common Voice Arabic since the original mozilla-foundation datasets migrated to Mozilla Data Collective.
- **Languages**: Arabic (mixed accents)
- **License**: CC-0 (Public Domain)

## 2. Data Structure

### Splits

- `train`: 28,410 samples
- `validation`: 10,471 samples
- `test`: 10,471 samples
- `other`: 41,586 samples (unvalidated)
- `invalidated`: 15,120 samples

### Features

- `audio`: Audio data (array + sampling rate at 48kHz)
- `sentence`: Ground truth transcription text
- `client_id`: Anonymous speaker identifier
- `path`: Audio file path
- `up_votes`: Number of validation upvotes
- `down_votes`: Number of validation downvotes
- `age`: Speaker age group
- `gender`: Speaker gender
- `accent`: Speaker accent/dialect
- `locale`: Language locale
- `segment`: Segment identifier
- `variant`: Dataset variant

## 3. Preprocessing

- **Audio Preprocessing**: Audio is at 48kHz. **Resampling to 16kHz is required** for Qwen3-ASR.
- **Text Preprocessing**: The `sentence` column contains clean Arabic text. Standard normalization applies.

## 4. Language Mapping

All samples map to `Arabic` for Qwen3-ASR language prefix.

## 5. Example Sample

```json
{
  "audio": {
    "array": [...],
    "sampling_rate": 48000
  },
  "sentence": "هذا مثال على جملة عربية",
  "client_id": "abc123def456",
  "up_votes": 3,
  "down_votes": 0,
  "age": "thirties",
  "gender": "male",
  "accent": ""
}
```
