# Dataset: Common Voice (Multiple Languages)

```json:config
{
  "hf_path": "mozilla-foundation/common_voice_16_1",
  "text_column": "sentence",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 48000,
  "is_gated": true,
  "languages": ["en", "zh-CN", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "nl", "sv-SE", "da", "fi", "pl", "cs"]
}
```

This document describes the configuration and structure of the Common Voice dataset for various languages.

## 1. Dataset Information

- **Hugging Face Path**: `mozilla-foundation/common_voice_16_1`
- **Description**: Common Voice is a massive, multi-language dataset of transcribed speech. It is one of the largest public domain transcribed speech datasets, containing thousands of hours of audio in many languages.
- **Languages**: The dataset supports a large number of languages. This configuration will focus on the languages supported by Qwen3-ASR, such as English (en), Chinese (zh), German (de), French (fr), etc.

## 2. Data Structure

This section details the structure of the dataset as it is provided on Hugging Face.

### Splits

- `train`: Contains the training data.
- `validation`: Contains the validation data.
- `test`: Contains the test data.

### Features

- `client_id`: A unique identifier for the speaker.
- `path`: The path to the audio file (MP3 format).
- `audio`: The decoded audio array, which needs to be resampled to 16kHz.
- `sentence`: The transcription of the audio.
- `up_votes`: The number of up-votes for the transcription.
- `down_votes`: The number of down-votes for the transcription.
- `age`: The age of the speaker.
- `gender`: The gender of the speaker.
- `accent`: The accent of the speaker.

## 3. Preprocessing

This section outlines the preprocessing steps required to prepare the dataset for evaluation or fine-tuning.

- **Audio Preprocessing**: The audio is provided in MP3 format and needs to be decoded and resampled to 16kHz. The `datasets` library can handle this automatically.
- **Text Preprocessing**: The `sentence` text is generally clean, but may require normalization (e.g., converting to lowercase, removing punctuation) depending on the evaluation requirements.

## 4. Example Sample

This section provides an example of a single sample from the dataset in its raw format.

```json
{
  "client_id": "...",
  "path": "/path/to/common_voice_en_...mp3",
  "audio": {
    "path": "/path/to/common_voice_en_...mp3",
    "array": [...],
    "sampling_rate": 48000
  },
  "sentence": "This is an example transcript.",
  "up_votes": 2,
  "down_votes": 0,
  "age": "20s",
  "gender": "male",
  "accent": "us"
}
```
