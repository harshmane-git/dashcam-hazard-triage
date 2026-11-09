"""Phase 4 labeling tool: keyboard driven, OpenCV window.

Usage (from project root):
  python src/label_tool.py incidents                  # label Tier 1 incident clips (all unlabeled)
  python src/label_tool.py incidents --sample 40      # ...a sample of 40, spread across Tier 1 levels
  python src/label_tool.py incidents --relabel 25     # blind second pass over 25 already-labeled clips
  python src/label_tool.py agreement                  # Cohen's kappa between first and second pass
  python src/label_tool.py hazards data/raw/clip.mp4  # watch a full video and mark real hazards

Keys, incidents:  1 Low | 2 Medium | 3 High | f false trigger | s skip | u undo
                  space pause | a / d step back / forward (paused) | w slow-mo | r restart | q quit
Keys, hazards:    1 / 2 / 3 mark a hazard of that severity at the current frame | u undo last mark
                  space pause | a / d step | [ ] speed | q quit (video then does NOT count as reviewed)
Rubric: docs/labeling_rubric.md. Everything is saved after each keypress, so quit any time.
Tier 1's prediction is hidden unless you pass --show-pred (labeling blind avoids anchoring).
"""
import argparse
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from signals import ROOT

GOLD = ROOT / "data" / "gold"
LABELS, RELABELS = GOLD / "labels_incidents.csv", GOLD / "labels_incidents_relabel.csv"
HAZARDS, REVIEWED = GOLD / "hazards.csv", GOLD / "reviewed_videos.csv"
WIN, DISPLAY_W, FONT = "label_tool", 1100, cv2.FONT_HERSHEY_SIMPLEX
LABEL_COLS = ["video", "start_t", "end_t", "clip", "clip_start_s", "risk", "trigger_valid",
              "pred_level", "labeled_at"]
HAZ_COLS = ["video", "t_sec", "frame", "severity", "marked_at"]
REV_COLS = ["video", "reviewed_at", "n_marks"]
KEYMAP = {"1": ("Low", 1), "2": ("Medium", 1), "3": ("High", 1), "f": ("Low", 0)}
SEVERITY = {"1": "Low", "2": "Medium", "3": "High"}


# ---------- small helpers ----------
def _read(path, cols):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _key(video, start_t):
    return (str(video), round(float(start_t), 1))


def _now():
    return datetime.now().isoformat(timespec="seconds")


def hud(frame, lines, bar=None, flash=""):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 14 + 24 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, 22 + 24 * i), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if bar:
        pos, a, b = bar
        cv2.rectangle(out, (10, h - 22), (w - 10, h - 10), (90, 90, 90), -1)
        if a is not None:   # red segment = where Tier 1 saw the incident
            cv2.rectangle(out, (int(10 + a * (w - 20)), h - 22), (int(10 + b * (w - 20)), h - 10), (0, 0, 255), -1)
        x = int(10 + pos * (w - 20))
        cv2.line(out, (x, h - 28), (x, h - 4), (255, 255, 255), 2)
    if flash:
        cv2.putText(out, flash, (10, h - 40), FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _resize(f):
    s = DISPLAY_W / f.shape[1]
    return cv2.resize(f, (DISPLAY_W, int(f.shape[0] * s))) if s < 1 else f


def load_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(_resize(f))
    cap.release()
    return frames, fps


def _poll(delay):
    k = cv2.waitKey(max(int(delay), 1))
    return -1 if k == -1 else k & 0xFF


# ---------- incident labeling ----------
def review_clip(frames, fps, lines, seg):
    n, i, paused, slow = len(frames), 0, False, False
    while True:
        cv2.imshow(WIN, hud(frames[i], lines, (i / max(n - 1, 1), *seg)))
        k = _poll(30 if paused else 1000 / fps * (2 if slow else 1))
        if k == -1:
            if not paused:
                i = (i + 1) % n
            continue
        c = chr(k)
        if c in "123fsuq":
            return c
        if c == " ":
            paused = not paused
        elif c == "w":
            slow = not slow
        elif c == "r":
            i = 0
        elif paused and c == "a":
            i = max(0, i - 1)
        elif paused and c == "d":
            i = min(n - 1, i + 1)


def gather_incidents():
    frames = []
    for p in sorted((ROOT / "outputs" / "phase3").glob("*_incident_log.csv")):
        d = _read(p, [])
        if len(d) and "clip" in d.columns:
            frames.append(d)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df[df["clip"].notna() & (df["clip"] != "")]
    df = df[df["clip"].map(lambda c: (ROOT / c).exists())]
    return df.to_dict("records")


def sample_items(items, n, level_key, seed=0):
    rng = random.Random(seed)
    rng.shuffle(items)
    if not n:
        return items
    groups = {}
    for r in items:
        groups.setdefault(r.get(level_key, "?"), []).append(r)
    per = math.ceil(n / max(len(groups), 1))
    picked = [r for g in groups.values() for r in g[:per]]
    rng.shuffle(picked)
    return picked[:n]


def cmd_incidents(args):
    out_path = RELABELS if args.relabel else LABELS
    labels = _read(out_path, LABEL_COLS)
    done = {_key(v, s) for v, s in zip(labels["video"], labels["start_t"])}
    if args.relabel:
        first = _read(LABELS, LABEL_COLS)
        pool = [r for r in first.to_dict("records") if _key(r["video"], r["start_t"]) not in done]
        items, level_key = sample_items(pool, args.relabel, "pred_level", seed=1), "pred_level"
        if len(items) < args.relabel:
            print(f"Only {len(items)} first-pass clips available for a second pass.")
    else:
        pool = [r for r in gather_incidents() if _key(r["video"], r["start_t"]) not in done]
        items, level_key = sample_items(pool, args.sample, "level"), "level"
    if not items:
        print("Nothing to label. Run batch.py / run_all.py first, or everything is already labeled.")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    idx, history = 0, []
    while idx < len(items):
        r = items[idx]
        frames, fps = load_frames(ROOT / r["clip"])
        if not frames:
            print(f"Cannot read {r['clip']}, skipping")
            idx += 1
            continue
        dur = len(frames) / fps
        cs = float(r.get("clip_start_s") or 0.0)
        seg = (max(0.0, (float(r["start_t"]) - cs) / dur), min(1.0, (float(r["end_t"]) - cs) / dur))
        pred = r.get(level_key)
        lines = [f"{'RELABEL ' if args.relabel else ''}{str(r['video'])[:20]}  clip {idx + 1}/{len(items)}"
                 f"   labeled so far: {len(labels)}",
                 "1 Low  2 Medium  3 High  f false-trigger  s skip  u undo | space pause  a/d step  w slow  r restart  q quit"]
        if args.show_pred and pred:
            lines.append(f"Tier 1 says: {pred}")
        key = review_clip(frames, fps, lines, seg)
        if key == "q":
            break
        if key == "s":
            idx += 1
            continue
        if key == "u":
            if history:
                idx = history.pop()
                labels = labels.iloc[:-1]
                _write(labels, out_path)
            continue
        risk, valid = KEYMAP[key]
        clip_rel = r["clip"]
        if not args.relabel:   # freeze a copy: re-running the pipeline deletes outputs/archive
            dst = GOLD / "clips" / f"{r['video']}_{float(r['start_t']):.2f}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / r["clip"], dst)
            clip_rel = str(dst.relative_to(ROOT))
        row = {"video": r["video"], "start_t": r["start_t"], "end_t": r["end_t"], "clip": clip_rel,
               "clip_start_s": cs, "risk": risk, "trigger_valid": valid, "pred_level": pred,
               "labeled_at": _now()}
        labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
        _write(labels, out_path)
        history.append(idx)
        idx += 1
    cv2.destroyAllWindows()
    print(f"\nSaved {len(labels)} labels -> {out_path}")
    if len(labels):
        print("risk:", labels["risk"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).to_dict(),
              "| false triggers:", int((labels["trigger_valid"] == 0).sum()))


def cmd_agreement(_):
    a, b = _read(LABELS, LABEL_COLS), _read(RELABELS, LABEL_COLS)
    if b.empty:
        print("No second-pass labels yet. Run `incidents --relabel 25`, ideally a week after the first pass.")
        return
    a["k"] = [_key(v, s) for v, s in zip(a["video"], a["start_t"])]
    b["k"] = [_key(v, s) for v, s in zip(b["video"], b["start_t"])]
    m = a.merge(b, on="k", suffixes=("_1", "_2"))
    from sklearn.metrics import cohen_kappa_score
    order = ["Low", "Medium", "High"]
    print(f"{len(m)} clips labeled twice")
    print(f"risk:           accuracy {(m.risk_1 == m.risk_2).mean():.0%}   "
          f"kappa {cohen_kappa_score(m.risk_1, m.risk_2, labels=order):.2f}   "
          f"linear-weighted kappa {cohen_kappa_score(m.risk_1, m.risk_2, labels=order, weights='linear'):.2f}")
    if m.trigger_valid_1.nunique() > 1 or m.trigger_valid_2.nunique() > 1:
        print(f"trigger_valid:  agreement {(m.trigger_valid_1 == m.trigger_valid_2).mean():.0%}   "
              f"kappa {cohen_kappa_score(m.trigger_valid_1, m.trigger_valid_2):.2f}")
    print("\nrows = first pass, columns = second pass")
    print(pd.crosstab(m.risk_1, m.risk_2).reindex(index=order, columns=order, fill_value=0))


# ---------- hazard marking (full video) ----------
def cmd_hazards(args):
    video = Path(args.video)
    stem = video.stem
    rev = _read(REVIEWED, REV_COLS)
    haz = _read(HAZARDS, HAZ_COLS)
    if stem in set(rev["video"]) and not args.again:
        print(f"{stem} is already fully reviewed. Use --again to redo it (its old marks are replaced).")
        return
    if args.again:
        haz, rev = haz[haz["video"] != stem], rev[rev["video"] != stem]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speeds, sp = [0.5, 1.0, 2.0], 1
    i, last, paused, frame, finished = 0, -1, False, None, False
    session, flash, flash_until = 0, "", 0.0

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    while True:
        if i >= total:
            finished = True
            break
        if i != last:
            if i != last + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                finished = True
                break
            frame, last = _resize(f), i
        lines = [f"{video.name}   t={i / fps:6.2f}s   frame {i}/{total}   speed {speeds[sp]}x   "
                 f"marks this video: {int((haz['video'] == stem).sum())}",
                 "1/2/3 mark hazard (Low/Med/High) at THIS frame  u undo  space pause  a/d step  [ ] speed  q quit"]
        cv2.imshow(WIN, hud(frame, lines, (i / max(total - 1, 1), None, None),
                            flash if time.time() < flash_until else ""))
        base = 1000 / fps
        k = _poll(30 if paused else base * (2 if speeds[sp] == 0.5 else 1))
        if k == -1:
            if not paused:
                i += 2 if speeds[sp] == 2.0 else 1
            continue
        c = chr(k)
        if c == "q":
            break
        if c in SEVERITY:
            haz = pd.concat([haz, pd.DataFrame([{"video": stem, "t_sec": round(i / fps, 2), "frame": i,
                                                 "severity": SEVERITY[c], "marked_at": _now()}])],
                            ignore_index=True)
            _write(haz, HAZARDS)
            session += 1
            flash, flash_until = f"MARKED {SEVERITY[c]} at {i / fps:.2f}s", time.time() + 1.2
        elif c == "u" and session:
            haz = haz.iloc[:-1]
            _write(haz, HAZARDS)
            session -= 1
            flash, flash_until = "undone", time.time() + 1.0
        elif c == " ":
            paused = not paused
        elif c == "a":
            i, paused = max(0, i - 1), True
        elif c == "d":
            i, paused = min(total - 1, i + 1), True
        elif c == "[":
            sp = max(0, sp - 1)
        elif c == "]":
            sp = min(len(speeds) - 1, sp + 1)
    cap.release()

    n_marks = int((haz["video"] == stem).sum())
    _write(haz, HAZARDS)
    if finished:
        rev = pd.concat([rev, pd.DataFrame([{"video": stem, "reviewed_at": _now(), "n_marks": n_marks}])],
                        ignore_index=True)
        _write(rev, REVIEWED)
        if frame is not None:
            cv2.imshow(WIN, hud(frame, [f"End of video. {n_marks} hazard(s) marked. Press any key."]))
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"{stem}: {n_marks} hazard(s) marked; "
          + ("fully reviewed." if finished else "NOT fully reviewed (quit early), so it will not count for recall."))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("incidents")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--relabel", type=int, default=0)
    p.add_argument("--show-pred", action="store_true")
    p.set_defaults(fn=cmd_incidents)
    p = sub.add_parser("agreement")
    p.set_defaults(fn=cmd_agreement)
    p = sub.add_parser("hazards")
    p.add_argument("video")
    p.add_argument("--again", action="store_true")
    p.set_defaults(fn=cmd_hazards)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
