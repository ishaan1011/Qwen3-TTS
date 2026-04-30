## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. Please run `pip install qwen-tts` first, then run the command below:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Then follow the steps below to complete the entire fine-tuning workflow. Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases.

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```

---

## LoRA fine-tuning (validation path for multi-voice / elastic capacity)

The full SFT path above produces a ~3.4 GB checkpoint per voice. To serve N
voices elastically on a single A10G we want LoRA adapters: one shared base
in VRAM + a small adapter per voice (~50–150 MB), routed per request.

Before committing the workflow to LoRA, first validate that LoRA quality
matches the existing full-SFT checkpoint on the same dataset. The scripts
below do exactly that.

### 0) One-time setup

```bash
conda activate qwen3-tts
pip install "peft>=0.18.0"
```

### 1) Train a LoRA on the same dataset

`sft_12hz_lora.py` is a 1:1 mirror of `sft_12hz.py` (same data, same loss,
same optimizer state) with two changes: (a) only LoRA adapters + the small
output heads are trainable, (b) learning rate is bumped 10× since LoRA needs
more signal per step.

**Train more epochs than full SFT.** Full SFT overshoots small datasets
fast (we picked epoch 0 of run 6 as best). LoRA's restricted rank acts as
implicit regularization, so the modal best-epoch shifts later — typically
3–8 on datasets this size. Train 8–10 epochs and audition all of them
rather than guessing.

```bash
python sft_12hz_lora.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path /home/ubuntu/models/ishaan-lora \
  --train_jsonl /home/ubuntu/data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 1e-4 \
  --num_epochs 8 \
  --speaker_name ishaan \
  --lora_rank 32 \
  --lora_alpha 64
```

The optimizer uses two parameter groups: LoRA adapters at `--lr` (default
1e-4) and full-rank `modules_to_save` (codec_head, lm_head, codec_embedding,
text_projection) at `--lr × --modules_to_save_lr_ratio` (default 0.1, so
1e-5). The full-rank LR matches the SFT sweet spot validated in claude.md;
training those modules at the LoRA LR over-rotates the output heads and
produces "gibberish + jelly" output — same failure mode full SFT showed at
lr=2e-5. **Don't pass `--modules_to_save_lr_ratio 1.0`** unless you're
deliberately ablating this.

Per-epoch outputs:

```
ishaan-lora/checkpoint-epoch-N/
├── adapter/                   # PEFT adapter (multi-LoRA inference uses this)
├── speaker_embedding.pt       # row 3000 of codec_embedding (speaker slot)
└── lora_meta.json             # speaker_name, init_model_path, rank/alpha
```

If quality is borderline at rank 32, retrain with `--lora_rank 64
--lora_alpha 128`. The adapter file size scales linearly with rank.

### 2) Audition every epoch against the SFT baseline (one command)

`audition_lora_sweep.py` auditions the SFT baseline once, then for each
LoRA epoch: merges adapter → CustomVoice format, generates audio, frees
the model, and (by default) deletes the merged dir to keep peak disk at
~3.4 GB. Re-runs reuse any merged checkpoint that already exists.

```bash
python audition_lora_sweep.py \
  --baseline_checkpoint /home/ubuntu/models/ishaan-prod/run6-epoch0 \
  --baseline_speaker    ishaan \
  --lora_root           /home/ubuntu/models/ishaan-lora \
  --output_root         /home/ubuntu/audition/lora_sweep
```

Output layout:

```
lora_sweep/
├── sft_baseline_ishaan/sNN.wav
├── lora_epoch_0_ishaan/sNN.wav
├── lora_epoch_1_ishaan/sNN.wav
├── ...
└── AB_manifest.txt              # grouped per-sentence so you can A/B/C/... one at a time
```

Useful flags:
- `--epochs 0,2,4-7` — audition a subset of epochs
- `--keep_merged` — retain merged checkpoints under `<lora_root>/merged/` (~3.4 GB each)
- `--skip_baseline` — re-run only the LoRA epochs without redoing the baseline
- `--sentences_file path.txt` — custom prompt set (defaults to the same 7 held-out sentences as `audition.py`)

Disk note: with default cleanup, peak usage is one merged dir at a time.
With `--keep_merged` and 10 epochs you'll need ~34 GB of free disk — check
with `df -h` first (claude.md flagged peak disk at 84% during SFT).

### 3) Single-pair A/B (lighter alternative)

If you've already decided which LoRA epoch to compare, `ab_audition.py`
takes two specific checkpoints and runs them side-by-side without
auto-merging:

```bash
python ab_audition.py \
  --checkpoint_a /home/ubuntu/models/ishaan-prod/run6-epoch0 \
  --speaker_a    ishaan --label_a sft \
  --checkpoint_b /home/ubuntu/models/ishaan-lora/merged/epoch-5 \
  --speaker_b    ishaan --label_b lora_e5 \
  --output_root  /home/ubuntu/audition/ab_focused
```

For one-off merges (no audition):

```bash
python merge_lora.py \
  --adapter_dir /home/ubuntu/models/ishaan-lora/checkpoint-epoch-5/adapter \
  --output_dir  /home/ubuntu/models/ishaan-lora-merged/epoch-5
```

### Decision rule

- **Indistinguishable on speaker timbre / prosody** → ship LoRA. Refit
  Arjun's voice the same way (`sft_12hz_lora.py` → `merge_lora.py`).
  Adapters are now interchangeable at request time and the same A10G can
  fluidly run 10–12 sessions across both voices.
- **Audibly worse** → retrain at rank 64 (`--lora_rank 64 --lora_alpha
  128`). If still worse, fall back to the multi-process static-split
  deployment (one vLLM instance per voice).

### What's actually trained

Trainable (gradient flows):
- LoRA adapters on `{q,k,v,o}_proj` and `{gate,up,down}_proj` across both
  the talker decoder and the code_predictor (sub-codebook) decoder.
- Full-rank: `codec_embedding`, `codec_head`, `lm_head`, `text_projection`
  (small modules where rank-restriction caps the prosody ceiling).

Frozen:
- `speaker_encoder` (full SFT also bypasses gradients via `.detach()` and
  drops it at save time — LoRA matches that exactly).
- Text input embeddings (phoneme/text representations transfer cleanly from
  the base model).

Speaker slot (row 3000 of codec_embedding) is written at merge time from the
saved `speaker_embedding.pt`, mirroring the full-SFT save logic.