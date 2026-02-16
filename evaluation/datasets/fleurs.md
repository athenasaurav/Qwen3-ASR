# Dataset: FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)

```json:config
{
  "hf_path": "google/fleurs",
  "text_column": "transcription",
  "audio_column": "audio",
  "split": "test",
  "sampling_rate": 16000,
  "is_gated": false,
  "hf_config": "{language}",
  "languages": ["en_us", "zh_cn", "de_de", "fr_fr", "es_419", "pt_br", "id_id", "it_it", "ko_kr", "ru_ru", "th_th", "vi_vn", "ja_jp", "tr_tr", "hi_in", "ms_my", "nl_nl", "sv_se", "da_dk", "fi_fi", "pl_pl", "cs_cz", "ar_eg", "yue_hant_hk"]
}
```

This document describes the configuration and structure of the FLEURS dataset.

## 1. Dataset Information

- **Hugging Face Path**: `google/fleurs`
- **Description**: FLEURS is a dataset for evaluating speech recognition systems in a large number of languages. It consists of short audio clips of read speech from Wikipedia articles, and is designed for few-shot learning scenarios.
- **Languages**: The dataset contains over 100 languages. This configuration will focus on the languages supported by Qwen3-ASR.

## 2. Data Structure

This section details the structure of the dataset as it is provided on Hugging Face.

### Splits

- `train`: Training data.
- `validation`: Validation data.
- `test`: Test data.

### Features

- `id`: A unique identifier for the utterance.
- `num_samples`: The number of audio samples.
- `path`: The path to the audio file.
- `audio`: The decoded audio array.
- `transcription`: The transcription of the audio.
- `raw_transcription`: The raw transcription, before normalization.
- `gender`: The gender of the speaker.
- `lang_id`: The language ID.
- `language`: The name of the language.
- `lang_group_id`: The language group ID.

## 3. Preprocessing

- **Audio Preprocessing**: The audio is provided at a 16kHz sampling rate and is ready for use.
- **Text Preprocessing**: The `transcription` field is already normalized and can be used directly for evaluation.

## 4. Example Sample

```json
{
  "id": 1,
  "num_samples": 123456,
  "path": "/path/to/audio.wav",
  "audio": {
    "path": "/path/to/audio.wav",
    "array": [...],
    "sampling_rate": 16000
  },
  "transcription": "This is an example transcript.",
  "raw_transcription": "This is an example transcript.",
  "gender": 1,
  "lang_id": 22,
  "language": "English",
  "lang_group_id": 3
}
```
