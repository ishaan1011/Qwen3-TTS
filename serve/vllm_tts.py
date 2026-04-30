# coding=utf-8
"""
vLLM-omni-backed TTS engine wrapper.

Replaces serve/tts.py for the vllm-omni env. Same conceptual contract as
TTSEngine but the synthesize entry point is async (because AsyncOmni is
async) and streams PCM bytes via an async generator.

End-to-end speedup vs the qwen-tts package: ~7x (RTF dropped from 2.1x to
~0.3x in our smoke test).
"""
from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator

import numpy as np
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni import AsyncOmni

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class VLLMTTSEngine:
    def __init__(self, model_path: str, speaker: str, language: str = "English"):
        self.model_path = model_path
        self.speaker = speaker
        self.language = language

        # AsyncOmni.from_cli_args accepts a Namespace; an empty parser
        # gives us internal defaults (verified in our smoke test).
        parser = FlexibleArgumentParser()
        args, _ = parser.parse_known_args([])

        log.info("loading vLLM-omni AsyncOmni for %s", model_path)
        t0 = time.time()
        self.omni = AsyncOmni.from_cli_args(args, model=model_path)
        log.info("AsyncOmni init done in %.1fs", time.time() - t0)

        self._prompt_len_cache: dict[str, object] = {}

    def _build_additional_information(self, text: str) -> dict:
        return {
            "task_type": ["CustomVoice"],
            "text": [text],
            "language": [self.language],
            "speaker": [self.speaker],
            "instruct": [""],
            "max_new_tokens": [2048],
        }

    def _estimate_prompt_len(self, text: str) -> int:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        if "tokenizer" not in self._prompt_len_cache:
            from transformers import AutoTokenizer

            self._prompt_len_cache["tokenizer"] = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
            cfg = Qwen3TTSConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self._prompt_len_cache["talker_config"] = getattr(cfg, "talker_config", None)

        tok = self._prompt_len_cache["tokenizer"]
        tcfg = self._prompt_len_cache["talker_config"]

        info = self._build_additional_information(text)
        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=info,
            task_type="CustomVoice",
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(tcfg, "codec_language_id", None),
            spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
            estimate_ref_code_len=lambda *_: None,
        )

    async def warmup(self) -> None:
        log.info("running warmup synthesis (compile + CUDA graphs ~ 60s on first run)")
        t0 = time.time()
        async for _ in self.synthesize("Warming up.", request_id="warmup_1"):
            pass
        async for _ in self.synthesize(
            "This is a slightly longer warmup sentence to capture another graph shape.",
            request_id="warmup_2",
        ):
            pass
        log.info("warmup done in %.1fs", time.time() - t0)

    async def synthesize(self, text: str, request_id: str) -> AsyncIterator[bytes]:
        """Synthesize `text` and yield PCM bytes (s16le 24 kHz mono).

        With default vllm-omni stage config (no async_chunk), the audio is
        emitted as one final chunk after generation completes. We yield
        once. With async_chunk enabled, we'd yield per chunk; that's a
        future enhancement once we wire stage_configs_path.
        """
        text = text.strip()
        if not text:
            return

        request = {
            "prompt_token_ids": [0] * self._estimate_prompt_len(text),
            "additional_information": self._build_additional_information(text),
        }
        t0 = time.time()
        # vllm-omni emits incremental audio chunks across multiple
        # stage_output events. Forward each chunk to the caller as
        # PCM bytes as soon as it arrives — the WebSocket consumer
        # streams them to the browser, where Web Audio schedules
        # them gap-free. This brings TTFA down from "wait for full
        # sentence synth" (~1.5-2.5s) to "first chunk" (~200-500ms).
        chunk_count = 0
        total_samples = 0
        ttfa_ms = -1.0
        async for stage_output in self.omni.generate(request, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            if mm is None:
                continue
            audio = mm.get("audio")
            if audio is None:
                continue

            chunks_this_event: list[torch.Tensor] = []
            if isinstance(audio, list):
                chunks_this_event.extend(t for t in audio if hasattr(t, "shape"))
            elif hasattr(audio, "shape"):
                chunks_this_event.append(audio)

            for chunk in chunks_this_event:
                if chunk_count == 0:
                    ttfa_ms = (time.time() - t0) * 1000
                wav = chunk.float().cpu().numpy().flatten()
                wav = np.clip(wav, -1.0, 1.0)
                pcm = (wav * 32767.0).astype(np.int16, copy=False)
                chunk_count += 1
                total_samples += int(chunk.shape[-1])
                yield pcm.tobytes()

        if chunk_count == 0:
            log.warning("synth produced no audio for: %s", text[:60])
            return

        wall_ms = (time.time() - t0) * 1000
        audio_s = total_samples / SAMPLE_RATE
        rtf = wall_ms / (audio_s * 1000) if audio_s > 0 else float("inf")
        log.info(
            "synth chunks=%d audio=%.2fs ttfa=%.0fms wall=%.0fms RTF=%.2fx | %s",
            chunk_count, audio_s, ttfa_ms, wall_ms, rtf, text[:60],
        )


def make_engine_from_env() -> VLLMTTSEngine:
    # Default points at the merged LoRA epoch-1 checkpoint (validated as
    # equivalent quality to the prior full-SFT run6-epoch0 baseline,
    # auditioned at lora_sweep_v3). The previous SFT checkpoint at
    # /home/ubuntu/models/ishaan-prod/run6-epoch0 is preserved as a
    # fallback — set TTS_CHECKPOINT to switch back if needed.
    model_path = os.environ.get("TTS_CHECKPOINT", "/home/ubuntu/models/ishaan-lora-prod/epoch-1")
    speaker = os.environ.get("TTS_SPEAKER", "ishaan")
    language = os.environ.get("TTS_LANGUAGE", "English")
    return VLLMTTSEngine(model_path, speaker, language=language)
