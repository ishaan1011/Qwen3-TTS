# coding=utf-8
"""
Generate held-out test audio from a fine-tuned Qwen3-TTS CustomVoice checkpoint.

Usage:
  python finetuning/audition.py \
      --checkpoint_dir /home/ubuntu/models/ishaan-ft/checkpoint-epoch-4 \
      --speaker ishaan \
      --output_dir /home/ubuntu/audition/epoch-4

Run separately for each epoch you want to compare; outputs go to distinct
--output_dir directories so you can listen side-by-side.
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

DEFAULT_SENTENCES = [
    "Hey, just wanted to check if you're free this weekend.",
    "Did you see the new restaurant that opened on Third Street?",
    "I have absolutely no idea what I'm doing, but it's working so far.",
    "The meeting got pushed to four thirty, so we should be done by six.",
    "Wait, that's actually a great idea! Why didn't I think of that?",
    "It's been one of those weeks where everything goes wrong at the worst possible moment.",
    "I'll grab coffee on the way over and we can figure out the rest there.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--sentences_file", default=None,
                    help="Optional file with one sentence per line; overrides defaults.")
    ap.add_argument("--max_new_tokens", type=int, default=600,
                    help="Hard cap on generated codec frames per utterance "
                         "(at 12 Hz, 600 ~ 50 s). Prevents runaway generation "
                         "when SFT degraded EOS prediction.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sentences_file:
        with open(args.sentences_file) as f:
            sentences = [s.strip() for s in f if s.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    print(f"[audition] loading {args.checkpoint_dir}")
    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint_dir,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    speakers = model.get_supported_speakers()
    print(f"[audition] supported speakers: {speakers}")
    if args.speaker not in speakers:
        sys.exit(f"[error] speaker '{args.speaker}' not in supported list: {speakers}")

    for i, text in enumerate(sentences, start=1):
        print(f"[s{i:02d}] {text}", flush=True)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=args.language,
            speaker=args.speaker,
            max_new_tokens=args.max_new_tokens,
        )
        out_path = out_dir / f"s{i:02d}.wav"
        sf.write(str(out_path), wavs[0], sr)
        print(f"       -> {out_path}  ({len(wavs[0])/sr:.2f}s)", flush=True)

    print(f"[done] {len(sentences)} files -> {out_dir}")


if __name__ == "__main__":
    main()
