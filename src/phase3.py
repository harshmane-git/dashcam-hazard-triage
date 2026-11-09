"""Phase 3: clip capture, risk classification, actions, logging.

Usage (from project root, after signals.py):
    python src/phase3.py data/raw/clip.mp4              # clips + classify + actions
    python src/phase3.py data/raw/clip.mp4 --no-clips   # dry run: classify and log only (for tuning)

For every incident in outputs/phase2/<name>_incidents.csv:
  1. save a clip from pre_s before to post_s after            -> outputs/archive/<name>/
  2. extract features and classify Low / Medium / High         -> src/classifier.py
  3. cumulative actions:  Low = archive | Medium = + notify | High = + flag for report
Writes outputs/phase3/<name>_incident_log.csv, outputs/notifications.log (JSON lines),
and copies High clips to outputs/flagged/ with a row in outputs/flagged/report_queue.csv.
Re-running a video replaces its previous clips, notifications and report rows.
"""
import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import yaml

from classifier import FEATURES, load_classifier
from signals import ROOT, video_meta

ACTIONS = {"Low": ["archive"], "Medium": ["archive", "notify"],
           "High": ["archive", "notify", "flag"]}


def extract_features(inc, sig, vru_classes):
    w = sig[(sig.frame >= inc.start_frame) & (sig.frame <= inc.end_frame)
            & (sig.track_id == inc.primary_track)]
    growth = w.rel_growth.dropna()
    return {
        "min_ttc": float(inc.min_ttc) if pd.notna(inc.min_ttc) else math.nan,
        "max_h_frac": float(inc.max_h_frac),
        "risk_duration_s": float(inc.risk_duration_s),
        "max_rel_growth": float(growth.max()) if len(growth) else math.nan,
        "min_abs_u": float(w.u.abs().min()) if w.u.notna().any() else math.nan,
        "n_tracks": int(inc.n_tracks),
        # any pedestrian / cyclist hazard in the incident counts, even if a car is the primary object
        "is_vru": int(inc.primary_cls in vru_classes or "proximity" in str(inc.kinds)),
    }


def save_clip(src, f0, f1, out, max_width):
    cap = cv2.VideoCapture(str(src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = min(1.0, max_width / w)
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    n = 0
    for _ in range(f1 - f0 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        wr.write(cv2.resize(frame, (ow, oh)) if scale < 1.0 else frame)
        n += 1
    wr.release(); cap.release()
    return n


QUEUE_COLS = ["video", "incident_id", "level", "start_t", "reason", "clip"]


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _other_lines(path, stem):
    """Existing JSON-lines entries that belong to other videos."""
    if not path.exists():
        return []
    keep = []
    for line in path.read_text().splitlines():
        try:
            if json.loads(line).get("video") != stem:
                keep.append(line)
        except json.JSONDecodeError:
            keep.append(line)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--no-clips", action="store_true", help="dry run: classify and log only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    src = Path(args.video)
    stem = src.stem
    p2 = ROOT / "outputs" / "phase2"
    if not (p2 / f"{stem}_incidents.csv").exists():
        raise SystemExit(f"Run signals.py first, missing {p2 / (stem + '_incidents.csv')}")
    incidents = _read_csv(p2 / f"{stem}_incidents.csv")   # empty file = 0 incidents (older signals.py)
    sig = _read_csv(p2 / f"{stem}_signals.csv")
    w, h, fps = video_meta(src)
    cap = cv2.VideoCapture(str(src)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    clf = load_classifier(cfg)
    vru = set(cfg["pedestrian"]["classes"])
    cc, dry = cfg["clip"], args.no_clips
    arch_dir = ROOT / "outputs" / "archive" / stem
    flag_dir = ROOT / "outputs" / "flagged"
    notif_path = ROOT / "outputs" / "notifications.log"
    queue_path = flag_dir / "report_queue.csv"

    if not dry:
        shutil.rmtree(arch_dir, ignore_errors=True)
        arch_dir.mkdir(parents=True, exist_ok=True)
        flag_dir.mkdir(parents=True, exist_ok=True)
        for old in flag_dir.glob(f"{stem}_incident_*.mp4"):
            old.unlink()

    rows, notes, queue = [], [], []
    for inc in incidents.itertuples(index=False):
        feats = extract_features(inc, sig, vru)
        level, reason = clf.predict(feats)
        actions = ACTIONS[level]
        f0 = max(0, inc.start_frame - round(cc["pre_s"] * fps))
        f1 = min(total - 1, inc.end_frame + round(cc["post_s"] * fps))
        name = f"incident_{int(inc.incident_id):03d}_{level}.mp4"
        clip = ""
        if not dry:
            out = arch_dir / name
            save_clip(src, f0, f1, out, cc["max_width"])
            clip = str(out.relative_to(ROOT))
            if "notify" in actions:
                msg = (f"{level} risk incident #{int(inc.incident_id)} in {src.name} at "
                       f"t={inc.start_t:.1f}s ({inc.primary_cls}): {reason}")
                print(f"[NOTIFY] {msg}")
                notes.append({"time": datetime.now().isoformat(timespec="seconds"), "video": stem,
                              "incident_id": int(inc.incident_id), "level": level, "message": msg,
                              "clip": clip})
            if "flag" in actions:
                shutil.copy2(out, flag_dir / f"{stem}_{name}")
                queue.append({"video": stem, "incident_id": int(inc.incident_id), "level": level,
                              "start_t": inc.start_t, "reason": reason, "clip": clip})
        rows.append({"video": stem, "incident_id": int(inc.incident_id), "start_t": inc.start_t,
                     "end_t": inc.end_t, "kinds": inc.kinds, "primary_cls": inc.primary_cls,
                     "primary_track": int(inc.primary_track), **{k: feats[k] for k in FEATURES},
                     "level": level, "actions": "+".join(actions), "reason": reason,
                     "clip": clip, "clip_start_s": round(f0 / fps, 2), "clip_end_s": round(f1 / fps, 2)})

    out_dir = ROOT / "outputs" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(rows)
    log_path = out_dir / f"{stem}_incident_log{'_dryrun' if dry else ''}.csv"   # never clobber the real log
    log.to_csv(log_path, index=False)

    if not dry:
        notif_path.write_text("\n".join(_other_lines(notif_path, stem) + [json.dumps(n) for n in notes]) + "\n")
        old_q = _read_csv(queue_path) if queue_path.exists() else pd.DataFrame()
        old_q = old_q[old_q.video != stem] if len(old_q) else pd.DataFrame(columns=QUEUE_COLS)
        pd.concat([old_q, pd.DataFrame(queue, columns=QUEUE_COLS)],
                  ignore_index=True).to_csv(queue_path, index=False)

    if log.empty:
        print("No incidents: nothing to archive.")
    else:
        print(f"\n{len(log)} incident(s)" + (" (dry run, no files written)" if dry else ""))
        print(log[["incident_id", "start_t", "primary_cls", "level", "actions", "reason"]]
              .to_string(index=False))
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()