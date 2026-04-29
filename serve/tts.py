# coding=utf-8
"""
Singleton TTS engine wrapper around the fine-tuned Qwen3-TTS CustomVoice
checkpoint. Loads the model + voice clone prompt once at startup and exposes
a blocking synthesize() that returns 24 kHz mono int16 PCM bytes.

Designed to be called from an async server via asyncio.to_thread.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class TTSEngine:
    def __init__(
        self,
        checkpoint_dir: str,
        speaker: str,
        device: str = "cuda:0",
        attn_impl: str = "flash_attention_2",
        max_new_tokens: int = 600,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.speaker = speaker
        self.device = device
        self.max_new_tokens = max_new_tokens

        log.info("loading TTS checkpoint %s", checkpoint_dir)
        t0 = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            checkpoint_dir,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        log.info("loaded in %.1fs", time.time() - t0)
        speakers = self.model.get_supported_speakers() or []
        if speaker not in speakers:
            raise RuntimeError(
                f"speaker '{speaker}' not in checkpoint's supported list: {speakers}"
            )

    def warmup(self) -> None:
        log.info("warming up CUDA kernels")
        t0 = time.time()
        self.synthesize("Warming up.")
        log.info("warmup done in %.1fs", time.time() - t0)

    def synthesize(self, text: str) -> bytes:
        """
        Generate audio for `text` and return s16le 24 kHz mono PCM bytes.
        Blocking; offload via asyncio.to_thread from async code.
        """
        text = text.strip()
        if not text:
            return b""
        with torch.no_grad():
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                max_new_tokens=self.max_new_tokens,
            )
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate {sr}, expected {SAMPLE_RATE}")
        wav = np.asarray(wavs[0], dtype=np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16, copy=False)
        return pcm.tobytes()


def make_engine_from_env() -> TTSEngine:
    checkpoint = os.environ.get(
        "TTS_CHECKPOINT",
        "/home/ubuntu/models/ishaan-prod/run6-epoch0",
    )
    speaker = os.environ.get("TTS_SPEAKER", "ishaan")
    device = os.environ.get("TTS_DEVICE", "cuda:0")
    attn = os.environ.get("TTS_ATTN", "flash_attention_2")
    return TTSEngine(checkpoint, speaker, device=device, attn_impl=attn)
