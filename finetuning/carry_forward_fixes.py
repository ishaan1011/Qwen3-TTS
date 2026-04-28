# coding=utf-8
"""
Carry forward transcript fixes from a prior review into a fresh manifest.

When you add new recordings and re-run chunk_and_transcribe.py, the previously
reviewed clips get re-chunked from the same source files (Silero VAD is
deterministic on the same input). This script matches old and new manifest
rows by (source, start_s, end_s) and copies your prior decisions (kept /
edited text / dropped) into the new manifest, so the next review session only
touches genuinely new clips.

Usage:
    python finetuning/carry_forward_fixes.py \\
        --old_manifest ishaan_audio/manifest_fixed.csv \\
        --new_manifest ishaan_audio/manifest.csv \\
        --out_manifest ishaan_audio/manifest_carried.csv

Then run:
    python finetuning/review_transcripts.py \\
        --clips_dir ishaan_audio/clips \\
        --manifest ishaan_audio/manifest_carried.csv \\
        --jsonl ishaan_audio/train_raw.jsonl

review_transcripts auto-resumes by skipping rows whose `status` is set, so
only the unmatched new clips will need attention.
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_manifest", required=True,
                    help="Previously reviewed manifest with `status` and `text` columns set.")
    ap.add_argument("--new_manifest", required=True,
                    help="Freshly generated manifest from chunk_and_transcribe.py.")
    ap.add_argument("--out_manifest", required=True)
    args = ap.parse_args()

    with open(args.old_manifest, newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    old_by_key = {(r["source"], r["start_s"], r["end_s"]): r for r in old_rows}

    with open(args.new_manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        new_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "status" not in fieldnames:
        fieldnames.append("status")

    carried = {"kept": 0, "edited": 0, "dropped": 0}
    for r in new_rows:
        key = (r["source"], r["start_s"], r["end_s"])
        prior = old_by_key.get(key)
        if prior is None:
            r["status"] = ""
            continue
        status = prior.get("status", "")
        if status == "edited":
            r["text"] = prior["text"]
            r["status"] = "edited"
            carried["edited"] += 1
        elif status == "kept":
            r["status"] = "kept"
            carried["kept"] += 1
        elif status == "dropped":
            r["status"] = "dropped"
            carried["dropped"] += 1
        else:
            r["status"] = ""

    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)

    total_carried = sum(carried.values())
    todo = sum(1 for r in new_rows if not r["status"])
    print(f"[carry] matched {total_carried}/{len(new_rows)} rows  "
          f"(kept={carried['kept']} edited={carried['edited']} dropped={carried['dropped']})")
    print(f"[carry] {todo} rows still to review")
    print(f"[carry] -> {args.out_manifest}")


if __name__ == "__main__":
    main()
