# Project state and learnings

## Git workflow

- After making changes, always ask if you should commit to git. If the user says yes, always run `git add` and `git commit` with a clear, descriptive commit message.
- Commit after each meaningful unit of work (e.g., completing a service, adding a handler, finishing a hook). Do NOT batch all changes into one giant commit.
- Never paste API keys / secrets in chat. The user has had to rotate their OpenAI key twice — paste keys directly on the server terminal, never here.

## LoRA voice-clone training (validated equivalent to full SFT)

Production now ships from a LoRA adapter merged into the base, auditioned as quality-equivalent to the prior full-SFT `run6-epoch0` checkpoint. Default `TTS_CHECKPOINT` is `/home/ubuntu/models/ishaan-lora-prod/epoch-1`; the SFT checkpoint at `~/models/ishaan-prod/run6-epoch0` is preserved as a fallback.

### Working hyperparameters (sft_12hz_lora.py)

```
--lr 1e-5
--modules_to_save_lr_ratio 1.0
--num_epochs 5
--lora_rank 32
--lora_alpha 32       # alpha/rank = 1, NOT the conventional 2
--batch_size 2        # eff. 8 with grad_accum 4
```

Picked epoch 1. Training time ~hours on A10G with 23 min of single-speaker English.

### The non-obvious learning

Conventional LoRA wisdom says use a 10× higher learning rate than full fine-tuning because rank-restricted weights need more signal per step. **For voice clone of this kind, that is wrong.** The first two LoRA runs (lr=1e-4 LoRA / lr=1e-5 modules_to_save) produced gibberish at every epoch even though the LR fix went in the right direction. The diagnostic that pinned it down compared per-layer weight drift between SFT-epoch-0 and LoRA-epoch-7:

```
                    sft%    lora%   ratio(sft/lora)
q_proj L0           0.14    1.55    0.09          # LoRA drifted 11× more
q_proj L_last       0.26    5.23    0.05          # LoRA drifted 20× more
codec_head          0.82    1.99    0.41
lm_head[0]          0.38    1.79    0.21
```

LoRA was over-rotating in its rank-32 subspace. The 10×-higher-LR rule assumes you're nudging an already-capable base model; voice cloning installs a new behavior on a base that has zero concept of "use codec_embedding row 3000 as speaker conditioning," and rank-32 over-rotation pushes the model into useless directions of that subspace. Matching SFT's per-step LR (1e-5 across both LoRA and modules_to_save) and SFT's epoch count (5) lands drift magnitudes comparable to SFT's, in directions the model can use.

Same failure mode is documented for full SFT at lr=2e-5 ("gibberish + jelly sounds") — the LoRA path hit the same wall, just amplified by rank restriction.

### Other things ruled out (so we don't re-investigate)

- **Speaker embedding magnitude:** row 3000 has ||row||≈18 vs ~0.5 mean for other codec_embedding rows. SFT and LoRA both have this and SFT works, so the OOD scale is not the problem.
- **RMSNorm weights:** SFT does not train them either (drift = 0% across all layers). Adding them to `modules_to_save` would not have helped.
- **Linear biases:** the only biases live in `speaker_encoder` (which gets dropped at save), so `bias="all"` in PEFT has nothing to do.
- **modules_to_save not surviving merge:** confirmed via state_dict key inspection — `merge_and_unload` correctly unwraps in PEFT 0.18+.

### Future-debugging quick check

If a LoRA run produces gibberish, run this drift comparison against the working SFT checkpoint before changing anything else:

```python
# delta vs base, for both SFT and LoRA, on the same keys
# ratio = (sft_drift / lora_drift)
#   ratio ≈ 1.0  -> rank-subspace limitation, can't fix with LR
#   ratio >> 1   -> undertrained, raise LR or epochs
#   ratio < 1    -> over-trained, lower LR or fewer epochs
```

Code lives in `finetuning/` — see the diagnostic block we used in the `ishaan-lora` debugging session.

## Server / environment layout

The user runs on a single AWS EC2 box with an A10G GPU (23 GB VRAM), Ubuntu, CUDA 13 driver. Two conda envs:

| Env | Purpose | Python | Key deps |
|---|---|---|---|
| `qwen3-tts` | Original `qwen-tts` package, original streaming server (`serve/main.py`), LoRA fine-tuning (`finetuning/sft_12hz_lora.py`) | 3.12 | torch 2.6.0+cu124, flash-attn 2.7.4.post1, qwen-tts 0.1.1, peft>=0.18 |
| `vllm-omni` | The fast inference path used in production (`serve/vllm_main.py`) | 3.12 | vllm 0.19.0, vllm-omni from source, fastapi, openai |

8 GB swap is enabled (`/swapfile`) — added because flash-attn build OOM'd the 16 GB box during initial setup.

Production checkpoints:
- **Active:** `/home/ubuntu/models/ishaan-lora-prod/epoch-1/` (merged LoRA, default `TTS_CHECKPOINT`)
- **Fallback:** `/home/ubuntu/models/ishaan-prod/run6-epoch0/` (full SFT, prior production)

The LoRA training output (per-epoch adapters + speaker_embedding.pt) lives at `/home/ubuntu/models/ishaan-lora/` and isn't needed at inference time.

## Voice clone fine-tuning data — what worked

- **Dataset**: ~23 minutes total of single-speaker English audio. Mix of long-form monologue (Recordings 1–3 + 13–17, ~16 min) and short utterances (Recordings 4–12, ~5 min).
  - Long-form alone (10.5 min) was data-bound on short utterances. Adding 5 min of explicit short content + 7 min of more long-form essays unlocked another quality tier.
  - Short-utterance recordings need `--min_chunk_s 0.8` in `chunk_and_transcribe.py` to capture sub-2s clips that the default 2.0 threshold drops.
- **VAD/transcription pipeline**: `finetuning/chunk_and_transcribe.py` (Silero VAD → 24 kHz mono → Whisper large-v3). Output is a JSONL ready for `prepare_data.py`.
- **Transcript review tooling**: `finetuning/review_transcripts.py` (interactive macOS afplay-based review). Whisper hits ~90% accuracy out-of-the-box on clean recordings; review is mostly hitting Enter. The `e` command opens a pre-filled prompt for one-word edits.
- **Carry-forward**: `finetuning/carry_forward_fixes.py` (matches by source+start+end timestamps, preserves prior fixes when adding new recordings) and `finetuning/carry_forward_from_jsonl.py` (text-similarity fallback when the manifest is lost but train_raw_fixed.jsonl survives).

## Full SFT hyperparameters that worked (for context / fallback)

The `finetuning/sft_12hz.py` script as-is. After several iterations:
- `lr=1e-5` with `batch_size=2` (effective 8 with grad-accum 4). Higher (`2e-5`) destabilized the model, producing "gibberish + jelly sounds." Lower (`5e-6`) plateaued early without capturing speaker.
- `num_epochs=5` was the right amount on 23 min of data. The user picked **epoch 0** of run 6 as the best-sounding checkpoint — counterintuitive but real. Subsequent epochs sometimes overshoot prosody.
- The model has unstable EOS prediction on out-of-distribution inputs. The qwen-tts inference path needs `max_new_tokens` capping (we used adaptive `8 frames/word + 30 buffer, max 150`). vLLM-omni handles this internally via stage configs.

## Inference: the qwen-tts package was the bottleneck

The `qwen-tts` Python package (the official inference path) runs at **RTF ~2.1× on A10G** for the 1.7B CustomVoice model regardless of optimization. We confirmed this by trying:
- `do_sample=False` on talker (no speedup)
- `subtalker_dosample=False` on the 15-iteration sub-codebook loop (no speedup)
- `torch.compile` on the talker forward (no speedup)
- `torch.compile` on `talker.code_predictor` (the actual hot path — 15 sequential mini-forwards per codec frame) (no speedup)

The bottleneck is compute on this hardware, not Python overhead. Each codec frame requires 1 talker forward + 15 sub-codebook forwards = 16 sequential model passes. At RTF=2.1×, sentence-level streaming has unavoidable gaps because synth produces audio slower than the browser plays it.

## vLLM-omni was the fix

Switching the inference engine from `qwen-tts` to `vllm-omni` (an inference framework purpose-built for speech LMs) dropped RTF from **2.10× → 0.34×** — about a 6× speedup on the same hardware with the same checkpoint. The same SFT'd checkpoint at `/home/ubuntu/models/ishaan-prod/run6-epoch0/` and now the LoRA-merged checkpoint at `/home/ubuntu/models/ishaan-lora-prod/epoch-1/` both load cleanly via vllm-omni's `AsyncOmni`.

The two key implementation details for vllm-omni:
1. **`AsyncOmni.from_cli_args(empty_namespace, model=path)`** is enough — no stage_configs_path needed for the default streaming behavior. Pass requests as `{"prompt_token_ids": [0]*N, "additional_information": {...}}` where N is estimated by `Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(...)`.
2. **vllm-omni emits incremental audio chunks** across multiple `stage_output` events per request. Each event's `mm["audio"]` contains only the NEW chunk, not the cumulative state. We **must** concatenate (or stream) every chunk; just taking the last one yields the last 2-3 words of audio. (Spent half an hour debugging this.)

vllm-omni's chunk pattern per request:
- 12 × 3840-sample warmup chunks (= ~2 s of audio total, ~85 ms TTFA per chunk)
- N × 48000-sample steady-state chunks (= 2 s each)
- 1 variable-size tail chunk

## Streaming TTS chat server (serve/)

The production server is `serve/vllm_main.py`. Architecture:

```
Browser  --WebSocket-->  FastAPI  --HTTP-->  OpenAI Chat Completions (gpt-4o-mini)
                              |
                              v
                       AsyncOmni (vllm-omni) on GPU
```

- `serve/vllm_tts.py`: `VLLMTTSEngine` wrapping `AsyncOmni`. `synthesize(text)` is an async generator that yields PCM bytes per chunk as vllm-omni emits them.
- `serve/llm.py`: `AsyncOpenAI` streaming wrapper.
- `serve/phrases.py`: sentence boundary detection + ElevenLabs-style force-emit schedule.
- `serve/vllm_main.py`: producer/consumer pattern. Producer pulls LLM deltas, drains sentence boundaries, force-emits at clause/word boundaries when the schedule fires. Consumer pops chunks off the queue and synthesizes them.
- `serve/static/`: vanilla JS browser client. Plays PCM gaplessly via Web Audio API scheduling on a single AudioContext at 24 kHz. Defensive `ctx.resume()` on each chunk in case the browser auto-suspends.

### Latency design

- **Sentence-level boundaries** (preferred — best prosody): `.` `!` `?` followed by whitespace, with abbreviation/decimal guards.
- **ElevenLabs `chunk_length_schedule` fallback**: `[150, 200, 260]` chars for the first three chunks, `290` chars steady-state. If a sentence boundary doesn't fire by the threshold, force-emit at the latest clause (`,` `;` `:` `—`) or whitespace within a 60-char window backward.
- **Codec-level streaming**: PCM bytes are forwarded to the WebSocket as soon as each vllm-omni chunk arrives, not after the full sentence completes. This drops per-synth TTFA from ~1.5–2.5 s to **~85 ms**.

End-to-end measured first-audio after user submit: **~600 ms – 1.2 s** (LLM-dominated). RTF=0.34 means audio plays gaplessly across sentence boundaries and within sentences.

### Tuning knobs

- `serve/phrases.py:DEFAULT_FORCE_SCHEDULE` — lower for snappier first chunks, higher for cleaner first-phrase prosody.
- `serve/llm.py:DEFAULT_SYSTEM` — system prompt steers comma vs period frequency, which affects perceived first-audio latency.
- `serve/vllm_main.py:PCM_CHUNK_BYTES` — currently 4096 (~85 ms of audio per WebSocket frame).

### How to run

On the server:
```bash
conda activate vllm-omni
cd ~/Qwen3-TTS && git pull
export OPENAI_API_KEY=sk-...
python -u -m uvicorn serve.vllm_main:app --host 0.0.0.0 --port 8000
```

`TTS_CHECKPOINT` defaults to the LoRA-merged path. Override to roll back to SFT, or to point at a different voice.
