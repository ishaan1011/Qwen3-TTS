# coding=utf-8
"""
Phase 3 smoke test: load the user's fine-tuned CustomVoice checkpoint
through vLLM-omni and synthesize a few English sentences with the
"ishaan" speaker. Verifies that:
  1. vLLM-omni's loader accepts the SFT'd checkpoint (not just stock HF
     CustomVoice variants).
  2. The custom speaker name "ishaan" resolves correctly.
  3. The output audio is real speech, not gibberish.
  4. Steady-state per-request timing is reasonable (after the first
     request's one-time warmup).

Run on the server in the vllm-omni env:
  conda activate vllm-omni
  python serve/vllm_smoke_ishaan.py
"""
import asyncio
import logging
import os
import time

import soundfile as sf
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("vllm_smoke")

MODEL_PATH = "/home/ubuntu/models/ishaan-prod/run6-epoch0"
SPEAKER = "ishaan"
LANGUAGE = "English"
OUTPUT_DIR = "/home/ubuntu/voice_data/vllm_ishaan"

TEST_SENTENCES = [
    "Hi, how are you doing today?",
    "I just wanted to check in and see if you are free this weekend.",
    "It has been one of those weeks where everything goes wrong at the worst possible moment.",
    "Wait, that is actually a great idea. Why did not I think of that?",
]


def _estimate_prompt_len(additional_information: dict, model_name: str, _cache: dict = {}) -> int:
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    if model_name not in _cache:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
        _cache[model_name] = (tok, getattr(cfg, "talker_config", None))

    tok, tcfg = _cache[model_name]
    return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
        additional_information=additional_information,
        task_type=(additional_information.get("task_type") or ["CustomVoice"])[0],
        tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
        codec_language_id=getattr(tcfg, "codec_language_id", None),
        spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
        estimate_ref_code_len=lambda *_: None,
    )


def make_request(text: str, model_name: str) -> dict:
    additional_information = {
        "task_type": ["CustomVoice"],
        "text": [text],
        "language": [LANGUAGE],
        "speaker": [SPEAKER],
        "instruct": [""],
        "max_new_tokens": [2048],
    }
    return {
        "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
        "additional_information": additional_information,
    }


def _save_wav(out_path: str, mm: dict) -> None:
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    sf.write(out_path, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = FlexibleArgumentParser()
    args, _ = parser.parse_known_args([])

    log.info("loading vLLM-omni AsyncOmni for %s", MODEL_PATH)
    t0 = time.time()
    omni = AsyncOmni.from_cli_args(args, model=MODEL_PATH)
    log.info("AsyncOmni init done in %.1fs", time.time() - t0)

    for i, text in enumerate(TEST_SENTENCES, start=1):
        request_id = f"smoke_{i:02d}"
        log.info("[%s] %s", request_id, text)

        request = make_request(text, MODEL_PATH)
        t_start = time.perf_counter()
        chunk_idx = 0
        last_mm = None
        async for stage_output in omni.generate(request, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            if not stage_output.finished:
                t_now = time.perf_counter()
                if chunk_idx == 0:
                    log.info("[%s] TTFA=%.0fms (first audio chunk arrived)",
                             request_id, (t_now - t_start) * 1000)
                chunk_idx += 1
            else:
                last_mm = mm
                t_end = time.perf_counter()
                log.info("[%s] DONE total=%.0fms chunks=%d",
                         request_id, (t_end - t_start) * 1000, chunk_idx)

        if last_mm is not None:
            out_path = os.path.join(OUTPUT_DIR, f"{request_id}.wav")
            _save_wav(out_path, last_mm)
            log.info("[%s] saved -> %s", request_id, out_path)


if __name__ == "__main__":
    asyncio.run(main())
