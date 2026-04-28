# coding=utf-8
"""
Chunk + transcribe long-form recordings into a Qwen3-TTS SFT JSONL.

Reads from --input_dir, expects:
  - one or more long-form recordings matching --source_glob
  - one reference clip matching --ref_glob

Writes to --output_dir:
  - clips/utt_0001.wav, ...   (24 kHz mono, VAD-segmented, RMS-normalized)
  - ref.wav                    (24 kHz mono, RMS-normalized)
  - train_raw.jsonl            (rows: {audio, text, ref_audio, language})
  - manifest.csv               (for spot-checking before training)

Designed for finetuning/prepare_data.py + sft_12hz.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 24000


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def rms_normalize(audio: np.ndarray, target_dbfs: float, ceiling_dbfs: float = -1.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    target_rms = 10 ** (target_dbfs / 20)
    audio = audio * (target_rms / rms)
    peak = float(np.max(np.abs(audio)))
    ceiling = 10 ** (ceiling_dbfs / 20)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio.astype(np.float32)


def vad_segments(audio: np.ndarray, vad_model, min_s: float, max_s: float,
                 min_silence_s: float = 0.3, pad_ms: int = 80):
    audio_16k = librosa.resample(audio, orig_sr=SR, target_sr=16000)
    ts = get_speech_timestamps(
        torch.from_numpy(audio_16k),
        vad_model,
        sampling_rate=16000,
        min_speech_duration_ms=int(min_s * 1000),
        max_speech_duration_s=max_s,
        min_silence_duration_ms=int(min_silence_s * 1000),
        speech_pad_ms=pad_ms,
    )
    return [(t["start"] / 16000, t["end"] / 16000) for t in ts]


def has_clipping(audio: np.ndarray, thresh: float = 0.998) -> bool:
    return bool(np.max(np.abs(audio)) >= thresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="~/voice_data/raw")
    ap.add_argument("--output_dir", default="~/voice_data/ishaan")
    ap.add_argument("--source_glob", default="Recording*")
    ap.add_argument("--ref_glob", default="ref_raw.*")
    ap.add_argument("--language", default="English",
                    help="Language label written into the JSONL (e.g. 'English').")
    ap.add_argument("--whisper_lang_code", default="en")
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target_rms_db", type=float, default=-20.0)
    ap.add_argument("--min_chunk_s", type=float, default=2.0)
    ap.add_argument("--max_chunk_s", type=float, default=12.0)
    ap.add_argument("--no_speech_threshold", type=float, default=0.5,
                    help="Drop chunks whose worst Whisper segment no_speech_prob exceeds this.")
    ap.add_argument("--dry_run", action="store_true",
                    help="VAD + chunking only; do not write outputs or transcribe.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    clips_dir = out_dir / "clips"
    if not args.dry_run:
        clips_dir.mkdir(parents=True, exist_ok=True)

    ref_files = sorted(in_dir.glob(args.ref_glob))
    if not ref_files:
        sys.exit(f"[error] no reference clip matching {args.ref_glob} in {in_dir}")
    ref_in = ref_files[0]
    print(f"[ref] {ref_in.name}")
    ref_audio = rms_normalize(load_audio(ref_in), args.target_rms_db)
    ref_out = out_dir / "ref.wav"
    if not args.dry_run:
        sf.write(str(ref_out), ref_audio, SR, subtype="PCM_16")
    print(f"[ref] {len(ref_audio) / SR:.2f}s -> {ref_out}")

    print("[vad] loading silero-vad")
    vad_model = load_silero_vad()

    sources = sorted(in_dir.glob(args.source_glob))
    if not sources:
        sys.exit(f"[error] no source files matching {args.source_glob} in {in_dir}")

    segments = []  # (clip_path, src_name, start_s, end_s, dur_s)
    idx = 0
    for src in sources:
        print(f"[src] {src.name}")
        audio = rms_normalize(load_audio(src), args.target_rms_db)
        segs = vad_segments(audio, vad_model, args.min_chunk_s, args.max_chunk_s)
        kept = 0
        for (s, e) in segs:
            seg = audio[int(s * SR):int(e * SR)]
            dur = len(seg) / SR
            if dur < args.min_chunk_s or dur > args.max_chunk_s:
                continue
            if has_clipping(seg):
                print(f"  skip clipped {s:.1f}-{e:.1f}s")
                continue
            idx += 1
            clip_path = clips_dir / f"utt_{idx:04d}.wav"
            if not args.dry_run:
                sf.write(str(clip_path), seg, SR, subtype="PCM_16")
            segments.append((clip_path, src.name, s, e, dur))
            kept += 1
        print(f"  -> {kept}/{len(segs)} segments kept")

    total = sum(s[4] for s in segments)
    print(f"[chunks] {len(segments)} clips, {total / 60:.1f} min total")
    if not segments:
        sys.exit("[error] no usable chunks; aborting before transcription")
    if args.dry_run:
        print("[dry_run] stopping before transcription")
        return

    print(f"[whisper] loading {args.whisper_model} on {args.device}")
    wmodel = whisper.load_model(args.whisper_model, device=args.device)

    rows = []
    manifest = []
    for clip_path, src_name, s, e, dur in segments:
        out = wmodel.transcribe(
            str(clip_path),
            language=args.whisper_lang_code,
            fp16=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        text = (out.get("text") or "").strip()
        no_speech = max((seg.get("no_speech_prob", 0.0) for seg in out.get("segments", [])),
                        default=0.0)
        if not text:
            print(f"  drop empty: {clip_path.name}")
            continue
        if no_speech > args.no_speech_threshold:
            print(f"  drop no_speech={no_speech:.2f}: {clip_path.name} | {text[:60]}")
            continue
        rows.append({
            "audio": str(clip_path.resolve()),
            "text": text,
            "ref_audio": str(ref_out.resolve()),
            "language": args.language,
        })
        manifest.append({
            "clip": clip_path.name,
            "source": src_name,
            "start_s": f"{s:.2f}",
            "end_s": f"{e:.2f}",
            "duration_s": f"{dur:.2f}",
            "no_speech_prob": f"{no_speech:.3f}",
            "text": text,
        })
        print(f"  {clip_path.name} ({dur:.1f}s) {text[:80]}")

    jsonl_path = out_dir / "train_raw.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    print(f"[done] {len(rows)} rows -> {jsonl_path}")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
