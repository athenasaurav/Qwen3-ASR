# Dataset: LibriSpeech ASR

```json:config
{
  "hf_path": "librispeech_asr",
  "text_column": "text",
  "audio_column": "audio",
  "split": "test.clean",
  "sampling_rate": 16000,
  "is_gated": false,
  "languages": ["en"]
}
```

This document describes the configuration and structure of the LibriSpeech ASR dataset.

## 1. Dataset Information

- **Hugging Face Path**: `librispeech_asr`
- **Description**: LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.
- **Languages**: English (en)

## 2. Data Structure

This section details the structure of the dataset as it is provided on Hugging Face.

### Splits

- `train.clean.100`: Training data, 100 hours, "clean" speech.
- `train.clean.360`: Training data, 360 hours, "clean" speech.
- `train.other.500`: Training data, 500 hours, "other" speech (more challenging).
- `validation.clean`: Validation data, "clean" speech.
- `validation.other`: Validation data, "other" speech.
- `test.clean`: Test data, "clean" speech.
- `test.other`: Test data, "other" speech.

### Features

- `file`: The path to the audio file (FLAC format).
- `audio`: The decoded audio array, which needs to be resampled to 16kHz.
- `text`: The transcription of the audio.
- `speaker_id`: A unique identifier for the speaker.
- `chapter_id`: A unique identifier for the chapter.
- `id`: A unique identifier for the utterance.

## 3. Preprocessing

This section outlines the preprocessing steps required to prepare the dataset for evaluation or fine-tuning.

- **Audio Preprocessing**: The audio is provided in FLAC format and needs to be decoded and resampled to 16kHz. The `datasets` library can handle this automatically.
- **Text Preprocessing**: The `text` is already normalized and in uppercase. For some models, it might be necessary to convert it to lowercase.

## 4. Example Sample

This section provides an example of a single sample from the dataset in its raw format.

```json
{
  "file": "/path/to/audio.flac",
  "audio": {
    "path": "/path/to/audio.flac",
    "array": [...],
    "sampling_rate": 16000
  },
  "text": "THIS IS AN EXAMPLE TRANSCRIPT.",
  "speaker_id": 123,
  "chapter_id": 456,
  "id": "123-456-789"
}
```
