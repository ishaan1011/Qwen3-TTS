# coding=utf-8
"""
Streaming TTS chat server.

Architecture:
  Browser  --WebSocket-->  FastAPI on EC2  --HTTP--> OpenAI Chat Completions
                                |
                                v
                          Qwen3-TTS engine (GPU, fine-tuned CustomVoice)

Flow per user message:
  1. Client sends {"type":"user_message","content":"..."} on WS.
  2. Server starts streaming OpenAI; appends text into a buffer.
  3. Each time a complete sentence appears in the buffer, the server
     synthesizes it on the GPU and streams 24 kHz s16le PCM back as
     binary WS frames.
  4. After OpenAI signals done, any leftover text is flushed.
  5. Server sends {"type":"audio_end"}.

Stage 1 is intentionally simple: phrase-level (sentence boundary) emit,
no audio-queue accounting, no interrupt. Add those in Stage 2.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .llm import DEFAULT_SYSTEM, make_client, stream_chat
from .phrases import drain_sentences
from .tts import SAMPLE_RATE, make_engine_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("serve")

PCM_CHUNK_BYTES = 4096  # 4 KB ≈ 85 ms of 24 kHz s16le mono

app = FastAPI()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
async def _startup():
    log.info("loading TTS engine + warming up...")
    app.state.tts = make_engine_from_env()
    app.state.tts.warmup()
    app.state.openai = make_client()
    log.info("ready on port %s", os.environ.get("PORT", 8000))


async def synthesize_and_send(ws: WebSocket, tts, text: str) -> None:
    text = text.strip()
    if not text:
        return
    await ws.send_text(json.dumps({"type": "phrase", "text": text}))
    pcm = await asyncio.to_thread(tts.synthesize, text)
    if not pcm:
        return
    # Stream the PCM out in modest chunks so the browser can start
    # scheduling buffers ASAP rather than waiting for the entire bytes
    # blob to land.
    for i in range(0, len(pcm), PCM_CHUNK_BYTES):
        await ws.send_bytes(pcm[i : i + PCM_CHUNK_BYTES])


async def handle_user_message(
    ws: WebSocket,
    content: str,
    history: list[dict[str, Any]],
) -> None:
    """
    Runs the LLM stream and the TTS pipeline as two concurrent asyncio tasks
    coordinated by a sentence queue. Crucially, synthesis does NOT block
    pulling more tokens from OpenAI, so by the time sentence N's audio is
    playing in the browser, the consumer has already started synthesizing
    sentence N+1.
    """
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
            await queue.put(None)  # sentinel

    async def consumer() -> None:
        """
        Pulls sentences from the queue and synthesizes them. Whenever
        multiple sentences are queued at the time we wake up, they are
        merged into a single generate() call to amortize the fixed
        per-call overhead and give the model more prosodic context.
        """
        while True:
            first = await queue.get()
            if first is None:
                return
            batch = [first]
            # Drain whatever else is queued without blocking. If we hit
            # the sentinel, re-queue it so the next iteration sees it.
            while True:
                try:
                    more = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if more is None:
                    queue.put_nowait(None)
                    break
                batch.append(more)
            text = " ".join(batch)
            t0 = asyncio.get_event_loop().time()
            await synthesize_and_send(ws, app.state.tts, text)
            log.info(
                "synth+send %.0fms (n=%d) for: %s",
                (asyncio.get_event_loop().time() - t0) * 1000,
                len(batch),
                text[:80],
            )

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
