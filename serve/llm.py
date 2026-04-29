# coding=utf-8
"""
Async OpenAI streaming wrapper. Yields raw text deltas as they arrive.
"""
from __future__ import annotations

import os
from typing import AsyncIterator

from openai import AsyncOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM = (
    "You are a friendly conversational assistant. Keep responses natural and "
    "spoken-style — moderate length, varied sentences, and clear punctuation "
    "so the voice synthesis can stream them smoothly."
)


def make_client() -> AsyncOpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var is not set")
    return AsyncOpenAI(api_key=key)


async def stream_chat(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str | None = None,
) -> AsyncIterator[str]:
    """Yield text deltas from the OpenAI chat-completions streaming response."""
    stream = await client.chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        messages=messages,
        stream=True,
    )
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta
