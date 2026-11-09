"""Run the whole pipeline over every video in a folder and summarise what it found.

Usage (from project root):
    python src/batch.py data/raw             # new videos only (skips ones already processed)
    python src/batch.py data/raw --force     # redo everything (saved tracks are re-used)
    python src/batch.py data/raw --retrack   # also run YOLO again

Writes outputs/all_incidents.csv (every incident from every processed video) and prints how
many incidents of each Tier 1 level were mined, which shows how scarce Medium / High are.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retrack", action="store_true")
    args = ap.parse_args()

    vids = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXT)
    if not vids:
        raise SystemExit(f"No videos found in {args.folder}")
    failed = []
    for n, v in enumerate(vids, 1):
        log = ROOT / "outputs" / "phase3" / f"{v.stem}_incident_log.csv"
        if log.exists() and not (args.force or args.retrack):
            print(f"[{n}/{len(vids)}] already done, skipping: {v.name}")
            continue
        print(f"\n[{n}/{len(vids)}] {v.name}")
        cmd = [sys.executable, str(ROOT / "src" / "run_all.py"), str(v), "--no-overlay"]
        if args.retrack:
            cmd.append("--retrack")
        if subprocess.run(cmd).returncode:
            failed.append(v.name)

    logs, per_video = [], {}
    for v in vids:
        p = ROOT / "outputs" / "phase3" / f"{v.stem}_incident_log.csv"
        try:
            d = pd.read_csv(p)
        except FileNotFoundError:
            continue                      # video failed before phase 3
        except pd.errors.EmptyDataError:
            per_video[v.stem] = 0         # processed fine, just no incidents
            continue
        per_video[v.stem] = len(d)
        if len(d):
            logs.append(d)
    allinc = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    allinc.to_csv(ROOT / "outputs" / "all_incidents.csv", index=False)

    print("\n================ summary ================")
    print(f"videos processed: {len(per_video)}   with 0 incidents: {sum(1 for c in per_video.values() if c == 0)}")
    print(f"incidents mined: {len(allinc)}  -> outputs/all_incidents.csv")
    if len(allinc):
        print("by Tier 1 level:", allinc["level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).to_dict())
        print("by kind:", allinc["kinds"].value_counts().to_dict())
    if failed:
        print("FAILED:", ", ".join(failed))


if __name__ == "__main__":
    main()
