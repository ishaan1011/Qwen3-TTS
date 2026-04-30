# coding=utf-8
"""
Side-by-side A/B audition between two CustomVoice checkpoints.

Loads each checkpoint sequentially (one at a time, to fit on a 23 GB A10G),
generates the SAME prompt set against each, and writes outputs to two
directories with matching filenames so they can be compared by ear:

  <output_root>/A_<speaker>/s01.wav   <- checkpoint A
  <output_root>/B_<speaker>/s01.wav   <- checkpoint B

The default prompt set is the same one as audition.py (held-out sentences
that target everyday phrasing). Override with --sentences_file to use the
same custom set across both runs.

Designed for the LoRA validation A/B:

  python finetuning/ab_audition.py \
      --checkpoint_a /home/ubuntu/models/ishaan-prod/run6-epoch0 \
      --speaker_a    ishaan \
      --label_a      sft \
      --checkpoint_b /home/ubuntu/models/ishaan-lora-merged/checkpoint-epoch-3 \
      --speaker_b    ishaan \
      --label_b      lora_r32 \
      --output_root  /home/ubuntu/audition/ab_lora_vs_sft

If both checkpoints register the same speaker name, --speaker_b can be omitted
and will default to --speaker_a.
"""
import argparse
import gc
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


def synthesize_all(checkpoint_dir, speaker, sentences, out_dir, *,
                   device, attn, language, max_new_tokens):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {checkpoint_dir}")
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    speakers = model.get_supported_speakers()
    if speaker not in speakers:
        sys.exit(f"[error] speaker '{speaker}' not in {speakers} for {checkpoint_dir}")

    for i, text in enumerate(sentences, start=1):
        print(f"  [s{i:02d}] {text}", flush=True)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
        )
        out_path = out_dir / f"s{i:02d}.wav"
        sf.write(str(out_path), wavs[0], sr)
        print(f"        -> {out_path}  ({len(wavs[0]) / sr:.2f}s)", flush=True)

    # Free GPU memory before loading the next checkpoint. Both checkpoints
    # are ~3.4 GB in bf16 plus codec/activation overhead — sequential loading
    # is the safe path on 23 GB.
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_a", required=True)
    ap.add_argument("--speaker_a", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--checkpoint_b", required=True)
    ap.add_argument("--speaker_b", default=None,
                    help="Defaults to --speaker_a if omitted.")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--sentences_file", default=None)
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--max_new_tokens", type=int, default=600)
    args = ap.parse_args()

    speaker_b = args.speaker_b or args.speaker_a

    if args.sentences_file:
        with open(args.sentences_file) as f:
            sentences = [s.strip() for s in f if s.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    out_root = Path(args.output_root).expanduser().resolve()
    dir_a = out_root / f"{args.label_a}_{args.speaker_a}"
    dir_b = out_root / f"{args.label_b}_{speaker_b}"

    print(f"[ab] {len(sentences)} sentences x 2 checkpoints")
    print(f"[ab] A: {args.label_a:>10}  -> {dir_a}")
    print(f"[ab] B: {args.label_b:>10}  -> {dir_b}")

    synthesize_all(
        checkpoint_dir=args.checkpoint_a,
        speaker=args.speaker_a,
        sentences=sentences,
        out_dir=dir_a,
        device=args.device,
        attn=args.attn,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )

    synthesize_all(
        checkpoint_dir=args.checkpoint_b,
        speaker=speaker_b,
        sentences=sentences,
        out_dir=dir_b,
        device=args.device,
        attn=args.attn,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )

    # Manifest for grep-friendly listening order.
    manifest = out_root / "AB_manifest.txt"
    with open(manifest, "w") as f:
        f.write(f"# A = {args.label_a}: {args.checkpoint_a}\n")
        f.write(f"# B = {args.label_b}: {args.checkpoint_b}\n\n")
        for i, text in enumerate(sentences, start=1):
            f.write(f"s{i:02d} | {text}\n")
            f.write(f"     A: {dir_a / f's{i:02d}.wav'}\n")
            f.write(f"     B: {dir_b / f's{i:02d}.wav'}\n")
    print(f"[done] manifest -> {manifest}")


if __name__ == "__main__":
    main()
