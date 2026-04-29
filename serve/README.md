# Streaming TTS chat (Stage 1)

Browser ↔ WebSocket ↔ FastAPI on EC2 ↔ OpenAI (LLM) + Qwen3-TTS fine-tuned
checkpoint (TTS). Phrase-level streaming: the server synthesizes each
sentence as the LLM completes it, and the browser plays incoming PCM
gaplessly via Web Audio.

## Layout

```
serve/
  main.py                # FastAPI app + WebSocket route
  tts.py                 # singleton TTS engine wrapper
  llm.py                 # OpenAI streaming wrapper
  phrases.py             # sentence boundary detection
  static/
    index.html           # browser test client
    app.js               # WebSocket + Web Audio playback
  requirements.txt
```

## One-time setup on the server

In the `qwen3-tts` conda env (the one with torch + flash-attn already working):

```bash
cd ~/Qwen3-TTS
git pull
conda activate qwen3-tts
pip install -r serve/requirements.txt
```

Make sure the production checkpoint exists at the default location:
```bash
ls -lh ~/models/ishaan-prod/run6-epoch0/model.safetensors
```

## Run

```bash
export OPENAI_API_KEY=sk-...
# optional overrides:
# export OPENAI_MODEL=gpt-4o-mini      # default
# export TTS_CHECKPOINT=/home/ubuntu/models/ishaan-prod/run6-epoch0
# export TTS_SPEAKER=ishaan
# export PORT=8000

cd ~/Qwen3-TTS
python -m uvicorn serve.main:app --host 0.0.0.0 --port 8000
```

Startup logs ~30 s of model load + ~5 s of CUDA warmup, then prints `ready`.

## Open the client

In the EC2 security group, open inbound TCP 8000 to your IP (or 0.0.0.0/0
for testing). Then in a browser:

```
http://<ec2-public-ip>:8000/
```

Type a message, hit Enter. You should see assistant text appear as the LLM
streams, and hear audio start playing when the first sentence completes.

## Notes / known limits (Stage 1)

- **No interrupt**. If you want to start a new turn, wait for the current
  one to finish or refresh the page.
- **No audio queue accounting**. If GPU synthesis is slower than the LLM
  stream, audio may briefly gap between phrases. Stage 2 fixes this.
- **HTTP only**. For production, run behind nginx/Caddy with a real cert
  and switch the WebSocket URL to `wss://`. Web Audio works over `http://`
  on direct-IP connections without issue.
- **Single user**. The TTS engine is a singleton in process memory; running
  multiple browser tabs against the same server will queue serially.
