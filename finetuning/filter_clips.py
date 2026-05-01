# coding=utf-8
"""
Quality-filter the per-clip JSONL produced by chunk_and_transcribe.py
before training. Drops chunks that are outliers on simple acoustic
metrics — these tend to be the ones that destabilize the trained voice
(very short fragments, very long ones with multiple breaths, clips that
are mostly silence, clips that are noticeably louder/quieter than the
median).

Reads:  <input_jsonl> with rows {audio, text, ref_audio, language}
Writes: <output_jsonl> with the kept rows
        <output_jsonl>.dropped.csv  (audit trail of dropped clips + reason)

Usage:
  python finetuning/filter_clips.py \
      --input_jsonl  ~/voice_data/ishaan_v2/train_raw.jsonl \
      --output_jsonl ~/voice_data/ishaan_v2/train_filtered.jsonl

Then continue the existing pipeline:
  python finetuning/prepare_data.py \
      --input_jsonl  ~/voice_data/ishaan_v2/train_filtered.jsonl \
      --output_jsonl ~/voice_data/ishaan_v2/train_with_codes.jsonl
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR_REQUIRED = 24000


def measure_clip(path: Path) -> dict:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR_REQUIRED:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_REQUIRED)
        sr = SR_REQUIRED
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    rms_db = 20.0 * float(np.log10(rms))
    peak = float(np.max(np.abs(audio)))
    peak_db = 20.0 * float(np.log10(peak + 1e-12))
    # Silence ratio: fraction of frames below -45 dBFS (a typical
    # "speech vs. silence" threshold for clean recordings).
    frame = 1024
    if len(audio) >= frame:
        rms_per = np.sqrt(
            np.mean(audio[: (len(audio) // frame) * frame].reshape(-1, frame) ** 2, axis=1) + 1e-12
        )
        silence_ratio = float((20.0 * np.log10(rms_per) < -45.0).mean())
    else:
        silence_ratio = 0.0
    return {
        "duration_s": dur,
        "rms_db": rms_db,
        "peak_db": peak_db,
        "silence_ratio": silence_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--min_duration_s", type=float, default=1.2,
                    help="Drop clips shorter than this (sub-1s often partial words).")
    ap.add_argument("--max_duration_s", type=float, default=14.0,
                    help="Drop clips longer than this (multi-breath, hard to model).")
    ap.add_argument("--max_silence_ratio", type=float, default=0.45,
                    help="Drop clips where >45% of frames are below -45 dBFS.")
    ap.add_argument("--rms_zscore_max", type=float, default=2.5,
                    help="Drop clips whose RMS dB deviates by > this many median-absolute-"
                         "deviation-equivalents from the dataset median. Default 2.5 ~ 99th pct.")
    ap.add_argument("--peak_db_min", type=float, default=-30.0,
                    help="Drop clips whose peak is below this (likely too quiet to be useful).")
    args = ap.parse_args()

    in_path = Path(args.input_jsonl).expanduser()
    out_path = Path(args.output_jsonl).expanduser()
    if not in_path.exists():
        sys.exit(f"[error] {in_path} not found")

    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    print(f"[filter] {len(rows)} input clips")

    measurements = []
    for r in rows:
        try:
            m = measure_clip(Path(r["audio"]))
        except Exception as e:
            print(f"  [warn] could not measure {r['audio']}: {e}")
            m = None
        measurements.append(m)

    valid_rms = [m["rms_db"] for m in measurements if m is not None]
    if not valid_rms:
        sys.exit("[error] no measurable clips")
    median_rms = statistics.median(valid_rms)
    # Median Absolute Deviation, scaled to ~stddev for normal distributions.
    mad_rms = statistics.median([abs(x - median_rms) for x in valid_rms]) * 1.4826 + 1e-9

    print(f"[filter] dataset RMS: median={median_rms:.2f} dB  MAD-scaled stddev={mad_rms:.2f} dB")

    kept_rows = []
    kept_meas: list[dict] = []
    dropped = []
    for r, m in zip(rows, measurements):
        reason = None
        if m is None:
            reason = "unmeasurable"
        elif m["duration_s"] < args.min_duration_s:
            reason = f"duration<{args.min_duration_s} ({m['duration_s']:.2f}s)"
        elif m["duration_s"] > args.max_duration_s:
            reason = f"duration>{args.max_duration_s} ({m['duration_s']:.2f}s)"
        elif m["silence_ratio"] > args.max_silence_ratio:
            reason = f"silence_ratio>{args.max_silence_ratio} ({m['silence_ratio']:.2f})"
        elif m["peak_db"] < args.peak_db_min:
            reason = f"peak_db<{args.peak_db_min} ({m['peak_db']:.2f}dB)"
        else:
            z = abs(m["rms_db"] - median_rms) / mad_rms
            if z > args.rms_zscore_max:
                reason = f"rms outlier z={z:.2f} (rms={m['rms_db']:.2f}dB)"

        if reason is None:
            kept_rows.append(r)
            kept_meas.append(m)
        else:
            dropped.append({
                "audio": r["audio"],
                "reason": reason,
                "duration_s": f"{m['duration_s']:.2f}" if m else "",
                "rms_db": f"{m['rms_db']:.2f}" if m else "",
                "peak_db": f"{m['peak_db']:.2f}" if m else "",
                "silence_ratio": f"{m['silence_ratio']:.2f}" if m else "",
                "text": r.get("text", "")[:80],
            })

    print(f"[filter] kept {len(kept_rows)}/{len(rows)} clips  "
          f"({len(dropped)} dropped)")

    # Top reasons summary.
    from collections import Counter
    reason_cat = Counter(d["reason"].split(" ", 1)[0].split("<")[0].split(">")[0]
                         for d in dropped)
    if reason_cat:
        print(f"[filter] drop reason mix: {dict(reason_cat)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if dropped:
        audit_path = Path(str(out_path) + ".dropped.csv")
        with open(audit_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dropped[0].keys()))
            w.writeheader()
            w.writerows(dropped)
        print(f"[filter] dropped audit -> {audit_path}")

    total_dur = sum(m["duration_s"] for m in kept_meas)
    print(f"[done] {len(kept_rows)} clips, ~{total_dur/60:.1f} min total -> {out_path}")


if __name__ == "__main__":
    main()
