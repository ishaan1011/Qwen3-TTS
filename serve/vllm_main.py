# coding=utf-8
"""
Streaming TTS chat server (Phase 4) — vLLM-omni inference backend.

Mirrors the protocol of serve/main.py (same WebSocket + same browser
client) but uses serve.vllm_tts.VLLMTTSEngine under the hood. The
producer/consumer queue architecture is unchanged: producer pulls LLM
deltas as fast as OpenAI sends them and queues complete sentences;
consumer pops sentences and synthesizes them. With vllm-omni's RTF~0.3,
synthesis is faster than playback so audio plays gaplessly.

Run on the server in the vllm-omni env:
    conda activate vllm-omni
    pip install fastapi 'uvicorn[standard]' websockets openai
    export OPENAI_API_KEY=sk-...
    python -m uvicorn serve.vllm_main:app --host 0.0.0.0 --port 8000

The browser client (serve/static/index.html, app.js) does NOT change.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .llm import DEFAULT_SYSTEM, make_client, stream_chat
from .phrases import drain_sentences, find_soft_cut, force_emit_threshold
from .vllm_tts import SAMPLE_RATE, make_engine_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("serve_vllm")

PCM_CHUNK_BYTES = 4096

app = FastAPI()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
async def _startup():
    log.info("loading vLLM-omni TTS engine + warming up (~60s on first launch)...")
    app.state.tts = make_engine_from_env()
    await app.state.tts.warmup()
    app.state.openai = make_client()
    log.info("ready on port %s", os.environ.get("PORT", 8000))


async def synthesize_and_send(ws: WebSocket, tts, text: str, voice: str) -> None:
    text = text.strip()
    if not text:
        return
    await ws.send_text(json.dumps({"type": "phrase", "text": text, "voice": voice}))
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    t0 = asyncio.get_event_loop().time()
    bytes_sent = 0
    async for pcm in tts.synthesize(text, voice=voice, request_id=request_id):
        if not pcm:
            continue
        for i in range(0, len(pcm), PCM_CHUNK_BYTES):
            await ws.send_bytes(pcm[i : i + PCM_CHUNK_BYTES])
            bytes_sent += min(PCM_CHUNK_BYTES, len(pcm) - i)
    log.info(
        "sent %d bytes (%.0fms) voice=%s for: %s",
        bytes_sent,
        (asyncio.get_event_loop().time() - t0) * 1000,
        voice,
        text[:80],
    )


async def handle_user_message(
    ws: WebSocket,
    content: str,
    history: list[dict[str, Any]],
    voice: str,
) -> None:
    history.append({"role": "user", "content": content})
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_response_parts: list[str] = []

    await ws.send_text(json.dumps({"type": "audio_start", "sample_rate": SAMPLE_RATE, "voice": voice}))

    async def producer() -> None:
        text_buffer = ""
        chunks_emitted = 0
        try:
            async for delta in stream_chat(app.state.openai, history):
                await ws.send_text(json.dumps({"type": "llm_token", "delta": delta}))
                full_response_parts.append(delta)
                text_buffer += delta

                # 1. Drain any complete sentences (best prosody).
                sentences, text_buffer = drain_sentences(text_buffer)
                for sentence in sentences:
                    await queue.put(sentence)
                    chunks_emitted += 1

                # 2. If we still have a long unfinished buffer, force-emit at
                #    a clause/word boundary once we cross the schedule
                #    threshold (ElevenLabs-style 150 / 200 / 260 / 290).
                while True:
                    threshold = force_emit_threshold(chunks_emitted)
                    if len(text_buffer) < threshold:
                        break
                    cut = find_soft_cut(text_buffer, threshold)
                    if cut is None:
                        break
                    chunk = text_buffer[:cut].strip()
                    text_buffer = text_buffer[cut:]
                    if not chunk:
                        break
                    await queue.put(chunk)
                    chunks_emitted += 1

            tail = text_buffer.strip()
            if tail:
                await queue.put(tail)
        finally:
            await queue.put(None)

    async def consumer() -> None:
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            await synthesize_and_send(ws, app.state.tts, sentence, voice=voice)

    await asyncio.gather(producer(), consumer())

    history.append({"role": "assistant", "content": "".join(full_response_parts)})
    await ws.send_text(json.dumps({"type": "audio_end"}))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    tts = app.state.tts
    voice = tts.default_voice
    # Tell the client which voices it can pick from and what we'll use by
    # default. Existing single-voice clients can ignore this; multi-voice
    # clients show a selector.
    await ws.send_text(json.dumps({
        "type": "voices",
        "available": tts.voices,
        "default": tts.default_voice,
        "current": voice,
    }))
    history: list[dict[str, Any]] = [
        {"role": "system", "content": os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM)},
    ]
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue

            mtype = msg.get("type")
            if mtype == "user_message":
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                # Per-message voice override; falls back to session default.
                # Unknown voice -> friendly error, don't synthesize.
                msg_voice = msg.get("voice")
                if msg_voice and msg_voice != voice:
                    try:
                        voice = tts.resolve_voice(msg_voice)
                    except KeyError as e:
                        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
                        continue
                try:
                    await handle_user_message(ws, content, history, voice)
                except Exception as e:
                    log.exception("error in handle_user_message")
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            elif mtype == "set_voice":
                # Sticky session-level voice selection. Subsequent
                # user_message events without a "voice" field will use this.
                req_voice = msg.get("voice")
                try:
                    voice = tts.resolve_voice(req_voice)
                    await ws.send_text(json.dumps({"type": "voice_set", "voice": voice}))
                except KeyError as e:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            elif mtype == "reset":
                history = [
                    {"role": "system", "content": os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM)},
                ]
                await ws.send_text(json.dumps({"type": "reset_ack"}))
            else:
                await ws.send_text(json.dumps({"type": "error", "message": f"unknown type {mtype!r}"}))
    except WebSocketDisconnect:
        log.info("client disconnected")
