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
from .phrases import drain_sentences
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


async def synthesize_and_send(ws: WebSocket, tts, text: str) -> None:
    text = text.strip()
    if not text:
        return
    await ws.send_text(json.dumps({"type": "phrase", "text": text}))
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    t0 = asyncio.get_event_loop().time()
    bytes_sent = 0
    async for pcm in tts.synthesize(text, request_id=request_id):
        if not pcm:
            continue
        for i in range(0, len(pcm), PCM_CHUNK_BYTES):
            await ws.send_bytes(pcm[i : i + PCM_CHUNK_BYTES])
            bytes_sent += min(PCM_CHUNK_BYTES, len(pcm) - i)
    log.info(
        "sent %d bytes (%.0fms) for: %s",
        bytes_sent,
        (asyncio.get_event_loop().time() - t0) * 1000,
        text[:80],
    )


async def handle_user_message(
    ws: WebSocket,
    content: str,
    history: list[dict[str, Any]],
) -> None:
    history.append({"role": "user", "content": content})
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_response_parts: list[str] = []

    await ws.send_text(json.dumps({"type": "audio_start", "sample_rate": SAMPLE_RATE}))

    async def producer() -> None:
        text_buffer = ""
        try:
            async for delta in stream_chat(app.state.openai, history):
                await ws.send_text(json.dumps({"type": "llm_token", "delta": delta}))
                full_response_parts.append(delta)
                text_buffer += delta
                sentences, text_buffer = drain_sentences(text_buffer)
                for sentence in sentences:
                    await queue.put(sentence)
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
            await synthesize_and_send(ws, app.state.tts, sentence)

    await asyncio.gather(producer(), consumer())

    history.append({"role": "assistant", "content": "".join(full_response_parts)})
    await ws.send_text(json.dumps({"type": "audio_end"}))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
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
                try:
                    await handle_user_message(ws, content, history)
                except Exception as e:
                    log.exception("error in handle_user_message")
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
