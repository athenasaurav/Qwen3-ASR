## Fine-tuning Qwen3-ASR

This script fine-tunes **Qwen3-ASR** using JSONL audio-text pairs. It supports multi-GPU training via `torchrun`.

### 1) Setup

First, please install the two Python packages `qwen-asr` and `datasets` using the command below.

```bash
pip install -U qwen-asr datasets
```

Then, to reduce GPU memory usage and speed up training, it is recommended to install FlashAttention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

### 2) Input JSONL format

Prepare your training file as JSONL (one JSON per line). Each line must contain:

- `audio`: path to a WAV file
- `text`: transcript text (you can include a language prefix)

Example:
```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language English<asr_text>Another example."}
{"audio":"/data/wavs/utt0003.wav","text":"language English<asr_text>Fine-tuning data line."}
```

Language prefix recommendation:

- If you **have** language info, use:
  - `language English<asr_text>...`
  - `language Chinese<asr_text>...`
- If you **do not have** language info, use:
  - `language None<asr_text>...`

Note:
- If you set `language None`, the model will not learn language detection from that prefix.

### 3a) Full Fine-tune / SFT (single GPU)

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --mode sft \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

Checkpoints will be written to:
- `./qwen3-asr-finetuning-out/checkpoint-<global_step>`

### 3b) LoRA Fine-tune (single GPU)

LoRA trains only low-rank adapter weights, keeping the base model frozen. Much faster and uses less VRAM.

First install the required packages:
```bash
pip install -e ".[finetune]"
```

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-lora-out \
  --mode lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,v_proj \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 1e-4 \
  --epochs 3
```

LoRA checkpoints contain only adapter weights (`adapter_model.safetensors`, `adapter_config.json`) plus an `adapter_meta.json` recording the base model path.

### 3c) QLoRA Fine-tune (single GPU, low VRAM)

QLoRA loads the base model in 4-bit quantized form, enabling fine-tuning on consumer GPUs with limited VRAM.

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-qlora-out \
  --mode qlora \
  --quantization_bits 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch_size 8 \
  --grad_acc 16 \
  --lr 2e-4 \
  --epochs 3
```

### 4) Fine-tune (multi GPU with torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200
```

### 5) Resume training

Option A: explicitly set a checkpoint path:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume_from ./qwen3-asr-finetuning-out/checkpoint-200
```

Option B: automatically resume from the latest checkpoint under `output_dir`:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume 1
```

### 6a) Quick inference test (SFT checkpoint)

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
)

print(results[0].language)
print(results[0].text)
```

### 6b) Quick inference test (LoRA/QLoRA adapter)

```python
import torch
from peft import PeftModel
from transformers import AutoModel, AutoProcessor
from qwen_asr import Qwen3ASRModel

# Load base model
base_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qwen3-asr-lora-out/checkpoint-200")
model = model.merge_and_unload()

# Wrap in Qwen3ASRModel
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B", fix_mistral_regex=True)
asr_model = Qwen3ASRModel(
    backend="transformers",
    model=model,
    processor=processor,
    max_inference_batch_size=32,
)

results = asr_model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
)

print(results[0].language)
print(results[0].text)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
TRAIN_FILE="./train.jsonl"
EVAL_FILE="./eval.jsonl"
OUTPUT_DIR="./qwen3-asr-finetuning-out"

torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```