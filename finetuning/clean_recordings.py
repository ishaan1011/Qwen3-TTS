# coding=utf-8
"""
Clean source recordings (MP3/WAV/M4A) before VAD/chunking, to make the
training data acoustically more consistent and reduce noise the model
would otherwise learn as part of the voice.

What this does (conservative, audible improvement, no NN artifacts):
  1. Resample to 24 kHz mono (matches dataset.py's required SR).
  2. High-pass filter at 80 Hz (kills mic-stand rumble, AC blower thump).
  3. (Optional) Low-pass filter at 11 kHz (light air-band trim; see --lowpass).
  4. (Optional) Mains hum notches at 50/60 Hz + harmonics (see --hum).
  5. Two-pass EBU R128 loudness normalization to a target LUFS, with
     true-peak limit. Two-pass uses ffmpeg's loudnorm to measure first
     then apply with linear scaling (no transient compression artifacts).
  6. (Optional, off by default) Mild FFT denoise (afftdn). Off by default
     because aggressive denoise creates artifacts the model learns; only
     enable if you have audible broadband noise.

What this does NOT do:
  - De-reverberation (impossible to do cleanly; would add artifacts).
  - NN-based "enhance" (Adobe Enhance / Krisp / RTX Voice). Those baked
    artifacts into the voice in past TTS work.
  - Heavy compression / limiting (kills natural dynamics → robotic output).

Usage:
  python finetuning/clean_recordings.py \
      --input_dir  ~/voice_data/raw \
      --output_dir ~/voice_data/raw_clean \
      --source_glob "Recording*"

Then point chunk_and_transcribe.py at the cleaned dir instead of raw:
  python finetuning/chunk_and_transcribe.py \
      --input_dir  ~/voice_data/raw_clean \
      --output_dir ~/voice_data/ishaan_v2 \
      --source_glob "Recording*.wav"
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# EBU R128 broadcast standard (-23 LUFS) is conservative; -20 LUFS gives a
# slightly hotter, more "podcast-y" target that matches typical dataset
# loudness in voice-clone training. Either works; consistency matters more
# than absolute value.
DEFAULT_TARGET_LUFS = -20.0
DEFAULT_TRUE_PEAK_DB = -1.5
DEFAULT_LRA = 11.0  # loudness range; ffmpeg's default

# Hum notch widths (Hz). Narrow enough to leave voice untouched.
HUM_NOTCH_WIDTH = 2.0


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def build_pre_filter_chain(args) -> str:
    """Filters applied BEFORE the loudnorm pass-1 measurement, so the
    measurement reflects what we'll be normalizing.
    """
    parts = [
        "aformat=channel_layouts=mono",
        f"aresample={args.sample_rate}",
        f"highpass=f={args.hpf}",
    ]
    if args.lowpass:
        parts.append(f"lowpass=f={args.lowpass}")
    if args.hum:
        # Notch at fundamental + 2nd + 3rd harmonics. Narrow Q so voice
        # isn't audibly affected.
        for h in (1, 2, 3):
            f = args.hum * h
            parts.append(f"bandreject=f={f}:width_type=h:w={HUM_NOTCH_WIDTH}")
    if args.denoise:
        # afftdn nr is noise reduction in dB; nf is the noise-floor estimate.
        # Mild settings: nr=10 dB reduction with -40 dBFS noise floor.
        # Defaults of nr=12, nf=-25 are too aggressive for training data.
        parts.append("afftdn=nr=10:nf=-40")
    return ",".join(parts)


_LOUDNORM_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def measure_loudnorm(in_path: Path, pre_filter: str, target_lufs: float,
                     true_peak: float, lra: float) -> dict:
    """Pass 1: run loudnorm in measurement mode, parse the JSON ffmpeg
    prints to stderr.
    """
    chain = (
        f"{pre_filter},"
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(in_path),
        "-af", chain,
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # ffmpeg writes the JSON to stderr after a "[Parsed_loudnorm" preamble.
    m = _LOUDNORM_JSON_RE.search(result.stderr)
    if not m:
        raise RuntimeError(
            f"loudnorm pass-1 measurement failed for {in_path.name}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
    return json.loads(m.group(0))


def apply_loudnorm(in_path: Path, out_path: Path, pre_filter: str,
                   target_lufs: float, true_peak: float, lra: float,
                   measured: dict, sample_rate: int) -> None:
    """Pass 2: apply the measured values with linear=true (no compression)."""
    chain = (
        f"{pre_filter},"
        f"loudnorm="
        f"I={target_lufs}:TP={true_peak}:LRA={lra}:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        f"linear=true:print_format=summary"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-y",
        "-i", str(in_path),
        "-af", chain,
        "-ar", str(sample_rate), "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"loudnorm pass-2 apply failed for {in_path.name}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Source recordings dir (MP3/WAV/M4A).")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write cleaned 24 kHz mono WAVs (mirrors filenames).")
    ap.add_argument("--source_glob", default="*",
                    help="Glob (relative to --input_dir) selecting source files.")
    ap.add_argument("--sample_rate", type=int, default=24000)
    ap.add_argument("--target_lufs", type=float, default=DEFAULT_TARGET_LUFS,
                    help="Integrated loudness target (LUFS). Default -20.")
    ap.add_argument("--true_peak", type=float, default=DEFAULT_TRUE_PEAK_DB,
                    help="True-peak ceiling (dBTP). Default -1.5.")
    ap.add_argument("--lra", type=float, default=DEFAULT_LRA,
                    help="Loudness range target.")
    ap.add_argument("--hpf", type=float, default=80.0,
                    help="High-pass filter cutoff Hz. Default 80.")
    ap.add_argument("--lowpass", type=float, default=None,
                    help="Optional low-pass cutoff Hz (e.g. 11000 to trim mic self-noise above).")
    ap.add_argument("--hum", type=float, default=None,
                    choices=[None, 50.0, 60.0],
                    help="Mains hum fundamental in Hz (50 EU / 60 US). Notches f, 2f, 3f.")
    ap.add_argument("--denoise", action="store_true",
                    help="Apply mild FFT denoise (afftdn nr=10 nf=-40). OFF by default; "
                         "only enable if you can audibly hear broadband noise on the source.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Run the pass-1 measurement only; print stats, write nothing.")
    args = ap.parse_args()

    if not have_ffmpeg():
        sys.exit("[error] ffmpeg not found on PATH. brew install ffmpeg / apt install ffmpeg")

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not in_dir.is_dir():
        sys.exit(f"[error] input_dir not found: {in_dir}")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(in_dir.glob(args.source_glob))
    if not sources:
        sys.exit(f"[error] no files matching {args.source_glob!r} in {in_dir}")

    pre = build_pre_filter_chain(args)
    print(f"[clean] pre-loudnorm chain: {pre}")
    print(f"[clean] target: {args.target_lufs} LUFS, TP <= {args.true_peak} dBTP, LRA {args.lra}")
    print(f"[clean] {len(sources)} files\n")

    summary_rows = []
    for src in sources:
        # Skip files we already wrote (idempotency on re-runs).
        out_path = out_dir / (src.stem + ".wav")
        print(f"[file] {src.name}")
        try:
            measured = measure_loudnorm(
                src, pre, args.target_lufs, args.true_peak, args.lra,
            )
        except RuntimeError as e:
            print(f"  [skip] {e}")
            continue
        print(f"  measured: I={measured['input_i']} LUFS  TP={measured['input_tp']} dBTP  "
              f"LRA={measured['input_lra']}  thresh={measured['input_thresh']}  "
              f"offset={measured['target_offset']}")
        if args.dry_run:
            summary_rows.append((src.name, measured))
            continue
        apply_loudnorm(
            src, out_path, pre,
            args.target_lufs, args.true_peak, args.lra,
            measured, args.sample_rate,
        )
        # Quick post-write sanity: file exists, non-empty.
        if not out_path.exists() or out_path.stat().st_size < 1024:
            print(f"  [error] output suspiciously small: {out_path}")
            continue
        print(f"  -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)\n")
        summary_rows.append((src.name, measured))

    print(f"\n[done] processed {len(summary_rows)}/{len(sources)} files -> {out_dir}")
    if summary_rows:
        # Spread of input loudness across files. If this is very wide
        # (>10 LUFS), original recordings were inconsistent — normalization
        # is helping a lot.
        in_i = [float(m["input_i"]) for _, m in summary_rows
                if m.get("input_i") not in (None, "-inf")]
        if in_i:
            print(f"[stats] source LUFS spread: min={min(in_i):.1f}  "
                  f"max={max(in_i):.1f}  mean={sum(in_i)/len(in_i):.1f}  "
                  f"(after norm: all clamped to {args.target_lufs})")


if __name__ == "__main__":
    main()
