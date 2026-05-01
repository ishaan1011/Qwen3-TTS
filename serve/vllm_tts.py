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

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni import AsyncOmni

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000


@dataclass
class VoiceSpec:
    """One named voice and the merged CustomVoice checkpoint it loads from.

    `speaker` is the speaker name registered in the checkpoint's spk_id
    map (passed to generate_custom_voice as the speaker arg). For LoRA
    checkpoints produced by sft_12hz_lora.py + merge_lora.py, this matches
    the --speaker_name flag used during training.
    """
    checkpoint_path: str
    speaker: str
    language: str = "English"


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


class MultiVoiceVLLMTTSEngine:
    """Dispatch table of named voices, each backed by its own VLLMTTSEngine.

    This is the static-split implementation: each voice loads its own base
    model into VRAM. On a 23 GB A10G this comfortably fits 2 voices in
    bf16 (~3.4 GB each + activations + KV pool), tightly fits 3.

    The downside is no elasticity — if Ishaan gets 10 concurrent requests
    and Arjun gets 0, you can't borrow capacity. The desired endgame is
    one shared base in VRAM with N LoRA adapters routed per request via
    vLLM's `lora_request` API. That's a single-class swap once we confirm
    vllm-omni's AsyncOmni surfaces lora_request on .generate() — TODO at
    bottom of file.
    """

    def __init__(self, voices: dict[str, VoiceSpec], default_voice: str):
        if not voices:
            raise ValueError("MultiVoiceVLLMTTSEngine needs at least one voice")
        if default_voice not in voices:
            raise ValueError(f"default_voice {default_voice!r} not in voices {list(voices)}")
        self.engines: dict[str, VLLMTTSEngine] = {}
        for name, spec in voices.items():
            log.info("multi-voice: registering %r -> %s (speaker=%s)",
                    name, spec.checkpoint_path, spec.speaker)
            self.engines[name] = VLLMTTSEngine(
                spec.checkpoint_path, spec.speaker, language=spec.language,
            )
        self.default_voice = default_voice

    @property
    def voices(self) -> list[str]:
        return list(self.engines.keys())

    async def warmup(self) -> None:
        for name, eng in self.engines.items():
            log.info("multi-voice: warming up %r", name)
            await eng.warmup()

    def resolve_voice(self, voice: str | None) -> str:
        """Return a valid voice name, falling back to default. Raises if
        the requested voice is unknown (caller should send a friendly
        error to the client rather than 500).
        """
        if voice is None or voice == "":
            return self.default_voice
        if voice not in self.engines:
            raise KeyError(
                f"unknown voice {voice!r}; available: {self.voices}"
            )
        return voice

    async def synthesize(
        self, text: str, voice: str | None, request_id: str,
    ) -> AsyncIterator[bytes]:
        name = self.resolve_voice(voice)
        async for chunk in self.engines[name].synthesize(text, request_id=request_id):
            yield chunk


def make_engine_from_env() -> MultiVoiceVLLMTTSEngine:
    """Build the engine from env config.

    Two modes (in priority order):

    1. TTS_VOICES_CONFIG=/path/to/voices.json
       Multi-voice mode. JSON shape:
         {
           "default": "ishaan",
           "voices": {
             "ishaan": {"checkpoint": "...", "speaker": "ishaan"},
             "arjun":  {"checkpoint": "...", "speaker": "arjun"}
           }
         }

    2. Fallback (back-compat with single-voice deploys):
       TTS_CHECKPOINT, TTS_SPEAKER, TTS_LANGUAGE env vars build a single
       voice named "default". The WebSocket protocol still accepts an
       optional voice field; if omitted, the single voice is used.

    Returns MultiVoiceVLLMTTSEngine in both cases — the single-voice path
    is just a degenerate one-entry dispatch table, so vllm_main.py has
    one code path.
    """
    config_path = os.environ.get("TTS_VOICES_CONFIG")
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        voices = {
            name: VoiceSpec(
                checkpoint_path=v["checkpoint"],
                speaker=v["speaker"],
                language=v.get("language", "English"),
            )
            for name, v in cfg["voices"].items()
        }
        default_voice = cfg.get("default") or next(iter(voices))
        return MultiVoiceVLLMTTSEngine(voices, default_voice=default_voice)

    # Single-voice fallback (back-compat). Default points at the v2 LoRA
    # (ishaan-lora-v2-prod/epoch-1), trained on audio cleaned via
    # finetuning/clean_recordings.py — modestly better stability and noise
    # consistency than the v1 LoRA (ishaan-lora-prod/epoch-1), which is
    # preserved as a fallback. The original full-SFT checkpoint at
    # /home/ubuntu/models/ishaan-prod/run6-epoch0 is the deeper fallback.
    spec = VoiceSpec(
        checkpoint_path=os.environ.get(
            "TTS_CHECKPOINT", "/home/ubuntu/models/ishaan-lora-v2-prod/epoch-1",
        ),
        speaker=os.environ.get("TTS_SPEAKER", "ishaan"),
        language=os.environ.get("TTS_LANGUAGE", "English"),
    )
    return MultiVoiceVLLMTTSEngine({"default": spec}, default_voice="default")


# ---------------------------------------------------------------------------
# TODO: elastic multi-voice via shared base + LoRA adapters.
#
# Current MultiVoiceVLLMTTSEngine partitions VRAM per voice (one base
# model copy per voice). The original motivation for LoRA was elastic
# capacity — one base in VRAM, N LoRA adapters hot-swapped per request,
# so 10–12 concurrent sessions can fluidly mix voices.
#
# The swap-in plan, once Arjun's adapter is trained:
#
#   1. Confirm vllm-omni's AsyncOmni.generate() accepts vLLM's standard
#      `lora_request=LoRARequest(name, id, path)` kwarg. If it does:
#
#   2. New class SharedBaseMultiLoRAEngine:
#        - __init__(base_path, voices: dict[name -> {adapter_dir, speaker}])
#        - One AsyncOmni instance loaded with `enable_lora=True, max_loras=N`.
#        - Build LoRARequest per voice once at init, cache them.
#        - synthesize(text, voice, request_id) passes the cached LoRARequest
#          through to omni.generate.
#        - codec_embedding row 3000 needs to be set per-voice; either
#          (a) keep it in the merged adapter (modules_to_save covers it),
#          or (b) overwrite it at runtime via PEFT's adapter mechanism.
#          Investigate which path AsyncOmni's LoRA loader supports.
#
#   3. Swap make_engine_from_env's two branches:
#        - TTS_BASE_MODEL + TTS_VOICES_CONFIG (with adapter_dir per voice)
#          -> SharedBaseMultiLoRAEngine (elastic).
#        - Existing single-checkpoint mode -> MultiVoiceVLLMTTSEngine
#          (back-compat).
#
# The vllm_main.py side doesn't change — both classes implement the same
# .synthesize(text, voice, request_id) contract.
# ---------------------------------------------------------------------------
