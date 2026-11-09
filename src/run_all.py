"""One entry point for the whole project.

  python src/run_all.py data/raw/clip.mp4              one video: track -> signals -> overlay -> phase 3
  python src/run_all.py data/raw                       every video in a folder (batch.py)
  python src/run_all.py label [--sample 40] [--show-pred]   label Tier 1 incident clips
  python src/run_all.py relabel [25]                   blind second pass over N already-labeled clips
  python src/run_all.py agreement                      self-agreement (Cohen's kappa) of the two passes
  python src/run_all.py hazards data/raw/clip.mp4 [--again]   watch a full video, mark real hazards
  python src/run_all.py status                         progress: incidents mined, labels, hazards

Video / folder options:  --retrack (run YOLO again)   --no-overlay (video only)   --force (folder only)
Rubric for labeling: docs/labeling_rubric.md
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
GOLD = ROOT / "data" / "gold"
RUBRIC = ROOT / "docs" / "labeling_rubric.md"
# Config sections that decide WHICH incidents exist. Tier 2 (classifier, clip) may change freely.
TIER1_KEYS = ["model", "imgsz", "conf", "tracker", "roi", "fcw", "pedestrian", "min_track_len",
              "roi_persist_frames", "persist_s", "gap_reset_s", "cooldown_s", "edge_margin_px"]


def run(script, *args):
    args = [str(a) for a in args]
    print(f"\n>>> {script} {' '.join(args)}")
    if subprocess.run([sys.executable, str(SRC / script), *args]).returncode:
        raise SystemExit(f"{script} failed; stopping.")


# ---------- pipeline ----------
def process_video(video, retrack, no_overlay):
    stem = Path(video).stem
    if retrack or not (ROOT / "outputs" / "phase1" / f"{stem}_tracks.csv").exists():
        run("track_video.py", video)
    else:
        print("Tracks already exist, skipping YOLO (use --retrack to redo).")
    run("signals.py", video)
    if not no_overlay:
        run("render_overlay.py", video)
    run("phase3.py", video)


# ---------- labeling support ----------
def _tier1_snapshot():
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    return {k: cfg[k] for k in TIER1_KEYS if k in cfg}


def _save_snapshot(snap):
    GOLD.mkdir(parents=True, exist_ok=True)
    json.dump({"saved_at": datetime.now().isoformat(timespec="seconds"), "config": snap},
              open(GOLD / "tier1_config.json", "w"), indent=1)


def check_freeze():
    """Warn if the Tier 1 settings changed since labeling started (labels are tied to them)."""
    snap, path = _tier1_snapshot(), GOLD / "tier1_config.json"
    if not path.exists():
        _save_snapshot(snap)
        print("Tier 1 settings recorded in data/gold/tier1_config.json. Keep them unchanged while labeling.")
        return
    old = json.load(open(path))["config"]
    changed = sorted({k for k in set(snap) | set(old) if snap.get(k) != old.get(k)})
    if not changed:
        return
    print(f"\nWARNING: Tier 1 settings changed since you started labeling: {', '.join(changed)}")
    print("Your labels belong to incidents the OLD settings produced. Clips made with the new settings")
    print("would make the gold set inconsistent.")
    ans = input("[k] keep labeling anyway   [r] record the new settings as the baseline   [q] quit: ").strip().lower()
    if ans == "r":
        _save_snapshot(snap)
        print("New baseline recorded.")
    elif ans != "k":
        raise SystemExit("Stopped. Restore the old settings, or re-run the pipeline and start a fresh gold set.")


def _logs():
    import pandas as pd
    out = []
    for p in sorted((ROOT / "outputs" / "phase3").glob("*_incident_log.csv")):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if len(d):
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else None


def status():
    import pandas as pd

    def read(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

    n_logs = len(list((ROOT / "outputs" / "phase3").glob("*_incident_log.csv")))
    mined = _logs()
    labels, relabels = read(GOLD / "labels_incidents.csv"), read(GOLD / "labels_incidents_relabel.csv")
    haz, rev = read(GOLD / "hazards.csv"), read(GOLD / "reviewed_videos.csv")

    print(f"videos processed:        {n_logs}")
    print(f"incidents mined:         {0 if mined is None else len(mined)}"
          + ("" if mined is None else "   by Tier 1 level: "
             + str(mined['level'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0).to_dict())))
    lk = {(str(v), round(float(s), 1)) for v, s in zip(labels.get("video", []), labels.get("start_t", []))}
    unl = 0 if mined is None else sum((str(v), round(float(s), 1)) not in lk
                                      for v, s in zip(mined["video"], mined["start_t"]))
    if len(labels):
        print(f"gold labels:             {len(labels)}   "
              + str(labels['risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0).to_dict())
              + f"   false triggers: {int((labels['trigger_valid'] == 0).sum())}")
    else:
        print("gold labels:             0")
    print(f"still unlabeled:         {unl}")
    print(f"second-pass labels:      {len(relabels)}")
    print(f"hazard marks:            {len(haz)} in {len(rev)} fully reviewed video(s)")
    if mined is None:
        print("\nNext: python src/run_all.py data/raw")
    elif unl:
        print("\nNext: python src/run_all.py label")
    elif len(labels) >= 25 and len(relabels) == 0:
        print("\nNext (about a week after the first pass): python src/run_all.py relabel")


# ---------- dispatch ----------
def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]

    if cmd == "label":
        if _logs() is None:
            raise SystemExit("No incidents mined yet. Run first:  python src/run_all.py data/raw")
        if not RUBRIC.exists():
            print(f"Note: rubric not found at {RUBRIC.relative_to(ROOT)}")
        else:
            print(f"Rubric: {RUBRIC.relative_to(ROOT)}   (1 Low, 2 Medium, 3 High, f false trigger)")
        check_freeze()
        run("label_tool.py", "incidents", *rest)
    elif cmd == "relabel":
        n = rest[0] if rest and rest[0].isdigit() else "25"
        run("label_tool.py", "incidents", "--relabel", n)
    elif cmd == "agreement":
        run("label_tool.py", "agreement")
    elif cmd == "hazards":
        if not rest:
            raise SystemExit("Usage: python src/run_all.py hazards data/raw/clip.mp4 [--again]")
        run("label_tool.py", "hazards", *rest)
    elif cmd == "status":
        status()
    else:   # a video file or a folder (also accepts the explicit word "batch" before the folder)
        if cmd == "batch" and rest:
            cmd, rest = rest[0], rest[1:]
        ap = argparse.ArgumentParser()
        ap.add_argument("--retrack", action="store_true")
        ap.add_argument("--no-overlay", action="store_true")
        ap.add_argument("--force", action="store_true")
        opts = ap.parse_args(rest)
        target = Path(cmd)
        if target.is_dir():
            run("batch.py", cmd, *(["--force"] if opts.force else []), *(["--retrack"] if opts.retrack else []))
        elif target.is_file():
            process_video(cmd, opts.retrack, opts.no_overlay)
        else:
            raise SystemExit(f"'{cmd}' is not a video file, a folder, or a command.\n{__doc__}")


if __name__ == "__main__":
    main()