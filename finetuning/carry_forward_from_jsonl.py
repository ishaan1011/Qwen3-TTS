# coding=utf-8
"""
Carry forward transcript fixes when only the old train_raw_fixed.jsonl
survives (no manifest_fixed.csv with start/end timestamps).

Matches each old fixed row to a new manifest row by:
  1. Reconstructing the source recording from the old clip index range
     (caller passes --old_index_ranges).
  2. Within the matched source, finding the new-manifest row whose text
     is most similar to the old text (substring containment OR
     SequenceMatcher ratio >= --threshold), greedy by best-similarity-first.

When matched, the old text overwrites the new manifest's text and the row's
status is set to "kept". review_transcripts.py auto-skips such rows.

Usage:
    python finetuning/carry_forward_from_jsonl.py \\
        --old_jsonl ishaan_audio/train_raw_fixed_run1.jsonl \\
        --old_index_ranges "1-65=Recording 1.mp3,66-80=Recording 2.mp3,81-103=Recording 3.mp3" \\
        --new_manifest ishaan_audio/manifest_run3.csv \\
        --out_manifest ishaan_audio/manifest_run3_carried.csv
"""
import argparse
import csv
import difflib
import json
import re
from pathlib import Path


def parse_ranges(spec: str):
    """'1-65=Recording 1.mp3,66-80=...' -> [(1,65,'Recording 1.mp3'), ...]"""
    out = []
    for chunk in spec.split(","):
        rng, src = chunk.strip().split("=", 1)
        lo, hi = rng.split("-")
        out.append((int(lo), int(hi), src.strip()))
    return out


def clip_index(basename: str) -> int | None:
    m = re.match(r"utt_(\d+)\.wav$", basename)
    return int(m.group(1)) if m else None


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return min(len(na), len(nb)) / max(len(na), len(nb))
    return difflib.SequenceMatcher(None, na, nb).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_jsonl", required=True)
    ap.add_argument("--old_index_ranges", required=True,
                    help='Comma-separated, e.g. "1-65=Recording 1.mp3,66-80=Recording 2.mp3"')
    ap.add_argument("--new_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Minimum similarity to accept a match (0-1).")
    args = ap.parse_args()

    ranges = parse_ranges(args.old_index_ranges)

    def index_to_source(n: int) -> str | None:
        for lo, hi, src in ranges:
            if lo <= n <= hi:
                return src
        return None

    with open(args.old_jsonl) as f:
        old_rows = [json.loads(line) for line in f]

    with open(args.new_manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        new_rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "status" not in fields:
        fields.append("status")
    for r in new_rows:
        r.setdefault("status", "")

    new_by_source: dict[str, list[int]] = {}
    for i, r in enumerate(new_rows):
        new_by_source.setdefault(r["source"], []).append(i)

    # Score every (old_row, candidate_new_row) pair, sort desc, greedy assign.
    proposals = []
    for o_idx, old in enumerate(old_rows):
        n = clip_index(Path(old["audio"]).name)
        src = index_to_source(n) if n is not None else None
        if src is None or src not in new_by_source:
            continue
        for n_idx in new_by_source[src]:
            ratio = similarity(old["text"], new_rows[n_idx]["text"])
            if ratio >= args.threshold:
                proposals.append((ratio, o_idx, n_idx))
    proposals.sort(reverse=True)

    claimed_old: set[int] = set()
    claimed_new: set[int] = set()
    matches = 0
    for ratio, o_idx, n_idx in proposals:
        if o_idx in claimed_old or n_idx in claimed_new:
            continue
        claimed_old.add(o_idx)
        claimed_new.add(n_idx)
        new_rows[n_idx]["text"] = old_rows[o_idx]["text"]
        new_rows[n_idx]["status"] = "kept"
        matches += 1

    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(new_rows)

    unmatched_old = [
        Path(old_rows[i]["audio"]).name + ": " + old_rows[i]["text"][:60]
        for i in range(len(old_rows)) if i not in claimed_old
    ]
    todo = sum(1 for r in new_rows if not r["status"])
    print(f"[carry] matched {matches}/{len(old_rows)} old fixed rows above threshold {args.threshold}")
    print(f"[carry] {len(new_rows) - todo}/{len(new_rows)} new rows pre-decided")
    print(f"[carry] {todo} rows still to review")
    if unmatched_old:
        print(f"[carry] {len(unmatched_old)} old rows had no good match (review manually):")
        for s in unmatched_old[:10]:
            print(f"        - {s}")
        if len(unmatched_old) > 10:
            print(f"        ... and {len(unmatched_old) - 10} more")
    print(f"[carry] -> {args.out_manifest}")


if __name__ == "__main__":
    main()
