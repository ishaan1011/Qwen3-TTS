# coding=utf-8
"""
Audition every epoch of a LoRA training run alongside an SFT baseline.

For LoRA, the "best" epoch tends to be later than for full SFT — full SFT
has high effective rank and overshoots small datasets quickly (you picked
epoch 0 of run 6), while LoRA at rank 32 acts as implicit regularization
and typically peaks at epoch 3–8 on datasets this size.

Workflow per run:
  1. Audition the SFT baseline checkpoint once.
  2. For each epoch in the LoRA training root:
     a. Merge adapter -> CustomVoice checkpoint (--merged_root).
     b. Load merged checkpoint, generate audio for the same prompts.
     c. Free GPU memory.
     d. (Optional) delete the merged checkpoint dir to keep peak disk low.

Output layout:
  <output_root>/
    sft_baseline_<speaker>/sNN.wav
    lora_epoch_0_<speaker>/sNN.wav
    lora_epoch_1_<speaker>/sNN.wav
    ...
    AB_manifest.txt

Usage:
  python finetuning/audition_lora_sweep.py \
      --baseline_checkpoint /home/ubuntu/models/ishaan-prod/run6-epoch0 \
      --baseline_speaker    ishaan \
      --lora_root           /home/ubuntu/models/ishaan-lora \
      --output_root         /home/ubuntu/audition/lora_sweep

By default the merged checkpoints are deleted after each epoch's audition
to keep peak disk usage at one ~3.4 GB merged dir at a time. Pass
--keep_merged to retain them under <merged_root> for re-runs.
"""
import argparse
import gc
import re
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel
from merge_lora import merge_adapter

DEFAULT_SENTENCES = [
    "Hey, just wanted to check if you're free this weekend.",
    "Did you see the new restaurant that opened on Third Street?",
    "I have absolutely no idea what I'm doing, but it's working so far.",
    "The meeting got pushed to four thirty, so we should be done by six.",
    "Wait, that's actually a great idea! Why didn't I think of that?",
    "It's been one of those weeks where everything goes wrong at the worst possible moment.",
    "I'll grab coffee on the way over and we can figure out the rest there.",
]

EPOCH_DIR_RE = re.compile(r"^checkpoint-epoch-(\d+)$")


def discover_epochs(lora_root: Path):
    """Return [(epoch_idx, epoch_dir, adapter_dir), ...] sorted by epoch."""
    found = []
    for child in sorted(lora_root.iterdir()):
        if not child.is_dir():
            continue
        m = EPOCH_DIR_RE.match(child.name)
        if not m:
            continue
        adapter_dir = child / "adapter"
        if not adapter_dir.exists():
            continue
        found.append((int(m.group(1)), child, adapter_dir))
    found.sort(key=lambda t: t[0])
    return found


def parse_epoch_filter(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available
    wanted = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(chunk))
    return [e for e in available if e in wanted]


def synthesize_all(checkpoint_dir, speaker, sentences, out_dir, *,
                   device, attn, language, max_new_tokens):
    """Load checkpoint, generate sentences, write to out_dir, free model."""
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
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
        )
        out_path = out_dir / f"s{i:02d}.wav"
        sf.write(str(out_path), wavs[0], sr)
        print(f"  s{i:02d} -> {out_path}  ({len(wavs[0]) / sr:.2f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_checkpoint", required=True,
                    help="Path to the SFT baseline (e.g. run6-epoch0).")
    ap.add_argument("--baseline_speaker", required=True,
                    help="Speaker name registered in the baseline checkpoint.")
    ap.add_argument("--lora_root", required=True,
                    help="Output root from sft_12hz_lora.py (contains checkpoint-epoch-* subdirs).")
    ap.add_argument("--output_root", required=True,
                    help="Where to write per-checkpoint audio + manifest.")
    ap.add_argument("--merged_root", default=None,
                    help="Where to materialize merged LoRA checkpoints. Defaults to <lora_root>/merged.")
    ap.add_argument("--speaker", default=None,
                    help="Speaker name registered in the LoRA training (defaults to baseline_speaker, "
                         "which matches when both are trained on the same voice).")
    ap.add_argument("--epochs", default="all",
                    help="'all' or a comma/range list, e.g. '0,2,4-7'. Default: all available.")
    ap.add_argument("--keep_merged", action="store_true",
                    help="Keep merged LoRA checkpoints on disk after auditioning (default: delete each "
                         "after use to cap peak disk at ~3.4 GB).")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="Don't re-audition the SFT baseline (e.g. when re-running just the LoRA epochs).")
    ap.add_argument("--sentences_file", default=None)
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--max_new_tokens", type=int, default=600)
    args = ap.parse_args()

    if args.sentences_file:
        with open(args.sentences_file) as f:
            sentences = [s.strip() for s in f if s.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    lora_root = Path(args.lora_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    merged_root = Path(args.merged_root).expanduser().resolve() if args.merged_root \
        else lora_root / "merged"
    output_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    speaker = args.speaker or args.baseline_speaker

    available = discover_epochs(lora_root)
    if not available:
        sys.exit(f"[error] no checkpoint-epoch-*/adapter found under {lora_root}")
    available_idx = [e for e, _, _ in available]
    epochs_to_run = parse_epoch_filter(args.epochs, available_idx)
    if not epochs_to_run:
        sys.exit(f"[error] no epochs match filter '{args.epochs}'. Available: {available_idx}")

    print(f"[sweep] sentences: {len(sentences)}")
    print(f"[sweep] baseline: {args.baseline_checkpoint} (speaker={args.baseline_speaker})")
    print(f"[sweep] lora epochs to run: {epochs_to_run} (available: {available_idx})")
    print(f"[sweep] keep_merged: {args.keep_merged}")
    print()

    runs = []  # (label, source_path, audio_dir)

    if not args.skip_baseline:
        baseline_dir = output_root / f"sft_baseline_{args.baseline_speaker}"
        synthesize_all(
            checkpoint_dir=args.baseline_checkpoint,
            speaker=args.baseline_speaker,
            sentences=sentences,
            out_dir=baseline_dir,
            device=args.device,
            attn=args.attn,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
        )
        runs.append(("sft_baseline", args.baseline_checkpoint, baseline_dir))

    epoch_lookup = {e: (epoch_dir, adapter_dir) for e, epoch_dir, adapter_dir in available}
    for epoch_idx in epochs_to_run:
        epoch_dir, adapter_dir = epoch_lookup[epoch_idx]
        merged_dir = merged_root / f"epoch-{epoch_idx}"

        if not (merged_dir / "model.safetensors").exists():
            print(f"\n[merge] epoch {epoch_idx}: {adapter_dir} -> {merged_dir}")
            merge_adapter(
                adapter_dir=str(adapter_dir),
                output_dir=str(merged_dir),
                device=args.device,
                verbose=True,
            )
        else:
            print(f"\n[merge] epoch {epoch_idx}: reusing existing {merged_dir}")

        # Merge currently leaves the base model + peft model on GPU. Free
        # before loading the merged checkpoint for inference.
        gc.collect()
        torch.cuda.empty_cache()

        audio_dir = output_root / f"lora_epoch_{epoch_idx}_{speaker}"
        synthesize_all(
            checkpoint_dir=str(merged_dir),
            speaker=speaker,
            sentences=sentences,
            out_dir=audio_dir,
            device=args.device,
            attn=args.attn,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
        )
        runs.append((f"lora_epoch_{epoch_idx}", str(merged_dir), audio_dir))

        if not args.keep_merged:
            print(f"[cleanup] removing {merged_dir}")
            shutil.rmtree(merged_dir, ignore_errors=True)

    # Write listening manifest, ordered by sentence so you can A/B/C/...
    # one sentence at a time across all checkpoints.
    manifest_path = output_root / "AB_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"# Sweep: {len(runs)} runs x {len(sentences)} sentences\n")
        for label, src, _ in runs:
            f.write(f"# {label}: {src}\n")
        f.write("\n")
        for i, text in enumerate(sentences, start=1):
            f.write(f"s{i:02d} | {text}\n")
            for label, _, audio_dir in runs:
                f.write(f"  {label:>20} : {audio_dir / f's{i:02d}.wav'}\n")
            f.write("\n")

    print(f"\n[done] {len(runs)} runs x {len(sentences)} sentences")
    print(f"       manifest -> {manifest_path}")
    print(f"       audio    -> {output_root}")


if __name__ == "__main__":
    main()
