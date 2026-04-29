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
        max_new_tokens_cap: int = 150,
        frames_per_word: int = 8,
        frames_buffer: int = 30,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.speaker = speaker
        self.device = device
        self.max_new_tokens_cap = max_new_tokens_cap
        self.frames_per_word = frames_per_word
        self.frames_buffer = frames_buffer

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

        # Try to torch.compile the talker LM. This is the largest module
        # in the per-token forward path; reducing its kernel launch
        # overhead is the highest-leverage compile target. Set
        # TTS_COMPILE=0 to skip if it causes problems.
        if os.environ.get("TTS_COMPILE", "1") != "0":
            inner = getattr(self.model, "model", None)
            log.info(
                "compile setup: inner=%s; inner attrs containing 'talk' or 'token' or 'predict': %s",
                type(inner).__name__ if inner is not None else None,
                [a for a in dir(inner) if any(k in a.lower() for k in ("talk", "token", "predict", "decoder"))]
                    if inner is not None else None,
            )
            for attr_name in ("talker", "speech_tokenizer"):
                target = getattr(inner, attr_name, None) if inner is not None else None
                if target is None:
                    log.warning("compile: %s not found, skipping", attr_name)
                    continue
                try:
                    compiled = torch.compile(
                        target,
                        mode="reduce-overhead",
                        dynamic=True,
                        fullgraph=False,
                    )
                    setattr(inner, attr_name, compiled)
                    log.info("compile: %s wrapped (%s)", attr_name, type(target).__name__)
                except Exception as e:
                    import traceback
                    log.warning(
                        "compile: %s FAILED with %s: %r\n%s",
                        attr_name, type(e).__name__, e, traceback.format_exc(),
                    )

    def _estimate_max_new_tokens(self, text: str) -> int:
        """
        Adaptive cap. The fine-tuned checkpoint sometimes fails to emit EOS
        on out-of-distribution inputs, so generation runs to whatever cap we
        set. Sizing the cap to the actual text length keeps wall-clock synth
        time bounded.
        """
        words = max(1, len(text.split()))
        estimated = words * self.frames_per_word + self.frames_buffer
        return min(estimated, self.max_new_tokens_cap)

    def warmup(self) -> None:
        """Warm CUDA kernels and trigger torch.compile graph capture for
        a few representative shapes, so first real request isn't slow."""
        log.info("warming up CUDA kernels (compile + graph capture may take 30s+)")
        t0 = time.time()
        # Two passes at different lengths to capture more graphs under
        # dynamic=True. First pass tends to be slowest due to compile.
        self.synthesize("Warming up.")
        self.synthesize("This is a slightly longer warmup sentence to capture another graph shape.")
        log.info("warmup done in %.1fs", time.time() - t0)

    def synthesize(self, text: str) -> bytes:
        """
        Generate audio for `text` and return s16le 24 kHz mono PCM bytes.
        Blocking; offload via asyncio.to_thread from async code.

        Logs frames-generated, wall-clock, ms/frame, and RTF (real-time
        factor) so we can measure the impact of inference optimizations.
        """
        text = text.strip()
        if not text:
            return b""
        max_new_tokens = self._estimate_max_new_tokens(text)
        t0 = time.time()
        with torch.no_grad():
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                max_new_tokens=max_new_tokens,
                # Greedy decoding: deterministic and ~10-30% faster than
                # default sampling. CustomVoice fine-tunes don't lose
                # meaningfully here -- the speaker bias is in the
                # parameters, not in the sampling stochasticity.
                do_sample=False,
                subtalker_dosample=False,
            )
        wall_ms = (time.time() - t0) * 1000
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate {sr}, expected {SAMPLE_RATE}")
        wav = np.asarray(wavs[0], dtype=np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16, copy=False)
        audio_seconds = len(wav) / SAMPLE_RATE
        # We don't have direct access to codec frame count here; estimate
        # from audio duration at 12 Hz tokenizer rate.
        frames_estimate = int(round(audio_seconds * 12))
        ms_per_frame = wall_ms / max(frames_estimate, 1)
        rtf = wall_ms / (audio_seconds * 1000) if audio_seconds > 0 else float("inf")
        log.info(
            "synth: cap=%d ~frames=%d audio=%.2fs wall=%.0fms ms/frame=%.0f RTF=%.2fx | %s",
            max_new_tokens, frames_estimate, audio_seconds, wall_ms,
            ms_per_frame, rtf, text[:60],
        )
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
