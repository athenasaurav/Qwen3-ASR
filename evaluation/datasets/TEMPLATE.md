# Dataset: [Dataset Name]

<!-- REQUIRED: Add this config block so the evaluation script can parse dataset settings. -->
```json:config
{
  "hf_path": "[owner/dataset_name on huggingface]",
  "text_column": "[column containing ground truth text, e.g. sentence, text, transcription]",
  "audio_column": "[column containing audio data, usually audio]",
  "split": "[default evaluation split, e.g. test]",
  "sampling_rate": 16000,
  "is_gated": false,
  "hf_config": "{language}",
  "languages": ["[lang_code_1]", "[lang_code_2]"]
}
```

This document describes the configuration and structure of the [Dataset Name] dataset.

## 1. Dataset Information

- **Hugging Face Path**: `[path/to/dataset/on/huggingface]`
- **Description**: [A brief description of the dataset, including its purpose, size, and any notable characteristics.]
- **Languages**: [List of languages present in the dataset, with their corresponding codes (e.g., English (en), Chinese (zh)).]

## 2. Data Structure

This section details the structure of the dataset as it is provided on Hugging Face. It should include information about the splits (train, validation, test) and the features available in each sample.

### Splits

- `train`: [Number of samples]
- `validation`: [Number of samples]
- `test`: [Number of samples]

### Features

- `audio`: [Description of the audio feature, including its type (e.g., path, array) and sampling rate.]
- `text`: [Description of the text feature, including the format of the transcript.]
- `[other_feature]`: [Description of any other relevant features.]

## 3. Preprocessing

This section outlines any preprocessing steps required to prepare the dataset for evaluation or fine-tuning. This may include resampling audio, cleaning text, or converting the data to a specific format.

- **Audio Preprocessing**: [Steps to preprocess the audio, e.g., resampling to 16kHz.]
- **Text Preprocessing**: [Steps to preprocess the text, e.g., removing punctuation, converting to lowercase.]

## 4. Example Sample

This section provides an example of a single sample from the dataset in its raw format.

```json
{
  "audio": {
    "path": "/path/to/audio.wav",
    "array": [...],
    "sampling_rate": 16000
  },
  "text": "This is an example transcript."
}
```
