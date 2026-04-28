# coding=utf-8
"""
Interactive transcript review for SFT clips.

Plays each clip via macOS `afplay`, shows the current transcript, and lets
you keep / edit / drop / replay / go back. Auto-resumes from where you
last stopped (rows whose `status` column is already set are skipped
unless --start is passed explicitly).

Inputs:
  --clips_dir   directory containing the utt_*.wav clips
  --manifest    manifest.csv produced by chunk_and_transcribe.py
  --jsonl       train_raw.jsonl produced by chunk_and_transcribe.py

Outputs (alongside the inputs by default):
  manifest_fixed.csv     (manifest with text edits and a `status` column)
  train_raw_fixed.jsonl  (text synced from manifest, dropped rows removed)
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import readline  # noqa: F401  (enables line-edit pre-fill on macOS)
    _HAS_READLINE = True
except ImportError:
    _HAS_READLINE = False

HELP = "Commands: [enter]=keep  e=edit (prefilled)  r=replay  d=drop  b=back  q=save+quit  ?=help"


def edit_with_prefill(prompt: str, text: str) -> str:
    """Open a prompt pre-populated with `text` so the user can arrow-key edit."""
    if not _HAS_READLINE:
        print(f"  current: {text}")
        return input(prompt)

    def hook():
        readline.insert_text(text)
        readline.redisplay()

    readline.set_startup_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook(None)


def play(clip_path: Path):
    return subprocess.Popen(
        ["afplay", str(clip_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_play(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            proc.kill()


def save_outputs(rows, fieldnames, jsonl_by_clip, out_manifest: Path, out_jsonl: Path):
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("status") == "dropped":
                continue
            j = jsonl_by_clip.get(r["clip"])
            if j is None:
                continue
            j = dict(j)
            j["text"] = r["text"]
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_manifest", default=None)
    ap.add_argument("--out_jsonl", default=None)
    ap.add_argument("--start", type=int, default=None,
                    help="1-indexed clip to start from. Default: first row without a status.")
    args = ap.parse_args()

    if not shutil.which("afplay"):
        sys.exit("[error] afplay not found. This tool is macOS-only.")

    clips_dir = Path(args.clips_dir).expanduser()
    manifest_in = Path(args.manifest).expanduser()
    jsonl_in = Path(args.jsonl).expanduser()
    out_manifest = Path(args.out_manifest or str(manifest_in).replace(".csv", "_fixed.csv")).expanduser()
    out_jsonl = Path(args.out_jsonl or str(jsonl_in).replace(".jsonl", "_fixed.jsonl")).expanduser()

    if out_manifest.exists():
        with open(out_manifest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        print(f"[resume] reading prior progress from {out_manifest}")
    else:
        with open(manifest_in, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    if "status" not in fieldnames:
        fieldnames.append("status")
    for r in rows:
        r.setdefault("status", "")

    with open(jsonl_in, "r", encoding="utf-8") as f:
        jsonl_rows = [json.loads(line) for line in f]
    jsonl_by_clip = {Path(r["audio"]).name: r for r in jsonl_rows}

    n = len(rows)
    if args.start is not None:
        i = max(0, args.start - 1)
    else:
        i = next((k for k, r in enumerate(rows) if not r.get("status")), n)

    print(f"[review] {n} clips total | starting at {i+1}")
    print(HELP)
    print()

    while i < n:
        r = rows[i]
        clip_path = clips_dir / r["clip"]
        if not clip_path.exists():
            print(f"[skip] missing file: {clip_path}")
            r["status"] = "missing"
            i += 1
            continue

        prior = f"  ({r['status']})" if r.get("status") else ""
        print(f"[{i+1}/{n}] {r['clip']}  ({r.get('duration_s','?')}s){prior}")
        print(f"      \"{r['text']}\"")
        proc = play(clip_path)
        try:
            ans = input("  > ")
        except (KeyboardInterrupt, EOFError):
            stop_play(proc)
            print()
            ans = "q"
        stop_play(proc)
        ans = ans.strip()

        if ans == "":
            r["status"] = "kept"
            i += 1
        elif ans == "r":
            continue
        elif ans == "d":
            r["status"] = "dropped"
            i += 1
        elif ans == "b":
            if i > 0:
                i -= 1
                rows[i]["status"] = ""
            continue
        elif ans == "q":
            break
        elif ans == "?":
            print(HELP)
            continue
        elif ans == "e":
            new_text = edit_with_prefill("  edit> ", r["text"]).strip()
            if not new_text:
                print("  (empty input -> kept original)")
                r["status"] = "kept"
            elif new_text == r["text"]:
                r["status"] = "kept"
            else:
                r["text"] = new_text
                r["status"] = "edited"
            i += 1
        else:
            # Treat anything else as a direct text replacement (back-compat)
            r["text"] = ans
            r["status"] = "edited"
            i += 1

        if (i % 10) == 0:
            save_outputs(rows, fieldnames, jsonl_by_clip, out_manifest, out_jsonl)

    kept = save_outputs(rows, fieldnames, jsonl_by_clip, out_manifest, out_jsonl)

    n_kept = sum(1 for r in rows if r.get("status") == "kept")
    n_edit = sum(1 for r in rows if r.get("status") == "edited")
    n_drop = sum(1 for r in rows if r.get("status") == "dropped")
    n_todo = sum(1 for r in rows if not r.get("status"))
    print()
    print(f"[done] reviewed: kept={n_kept} edited={n_edit} dropped={n_drop} pending={n_todo}")
    print(f"[done] manifest -> {out_manifest}")
    print(f"[done] jsonl    -> {out_jsonl}  ({kept} rows)")


if __name__ == "__main__":
    main()
