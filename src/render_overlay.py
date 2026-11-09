"""Phase 2: draw ROI, boxes, TTC and alert banner on the video (no YOLO re-run).

Usage (from project root, after signals.py):
    python src/render_overlay.py data/raw/clip.mp4
Output: outputs/phase2/<name>_overlay.mp4
Large videos are downscaled to 1280 px wide by default; use --scale 1 for full size.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import pandas as pd
import yaml

from signals import ROOT, lane_polygon, roi_polygon, video_meta

RED, YELLOW, GRAY = (0, 0, 255), (0, 255, 255), (170, 170, 170)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    src = Path(args.video)
    sig_path = ROOT / "outputs" / "phase2" / f"{src.stem}_signals.csv"
    if not sig_path.exists():
        raise SystemExit(f"Run signals.py first, missing {sig_path}")

    w, h, fps = video_meta(src)
    scale = args.scale or (1280 / w if w > 1280 else 1.0)
    ow, oh = int(w * scale), int(h * scale)
    sig = pd.read_csv(sig_path)
    by_frame = {f: g for f, g in sig.groupby("frame")}
    calib_path = ROOT / "outputs" / "phase2" / f"{src.stem}_calib.json"
    calib = json.load(open(calib_path)) if calib_path.exists() else {"mode": "static"}
    if calib["mode"] == "auto":
        poly = (lane_polygon(calib, cfg["roi"], w, h) * scale).astype(int)
        horizon_y = int(calib["y_h"] * scale)
    else:
        poly = (roi_polygon(cfg["roi"], w, h) * scale).astype(int)
        horizon_y = None
    fs = max(0.8, ow / 1920 * 0.9)
    th = max(1, round(ow / 1000))
    hold_frames = int(0.5 * fps)

    out_path = ROOT / "outputs" / "phase2" / f"{src.stem}_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    cap = cv2.VideoCapture(str(src))
    banner, banner_left, idx = "", 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (ow, oh))
        layer = frame.copy()
        cv2.fillPoly(layer, [poly], YELLOW)
        frame = cv2.addWeighted(layer, 0.12, frame, 0.88, 0)
        cv2.polylines(frame, [poly], True, YELLOW, th)
        if horizon_y is not None:   # thin cyan line = calibrated horizon, to eyeball the fit
            cv2.line(frame, (0, horizon_y), (ow, horizon_y), (255, 255, 0), 1)

        g = by_frame.get(idx)
        if g is not None:
            for r in g.itertuples(index=False):
                p1, p2 = (int(r.x1 * scale), int(r.y1 * scale)), (int(r.x2 * scale), int(r.y2 * scale))
                if r.fired:
                    color, t = RED, th * 2
                    banner = "COLLISION RISK" if r.kind == "collision" else "PEDESTRIAN / CYCLIST PROXIMITY"
                    banner_left = hold_frames
                elif r.in_roi:
                    color, t = YELLOW, th
                else:
                    color, t = GRAY, 1
                cv2.rectangle(frame, p1, p2, color, t)
                if r.in_roi or r.fired:
                    txt = f"{r.cls}#{r.track_id}"
                    txt += f" TTC {r.ttc:.1f}s" if not math.isnan(r.ttc) else f" {r.h_frac:.0%}"
                    cv2.putText(frame, txt, (p1[0], max(15, p1[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, color, th, cv2.LINE_AA)
        if banner_left > 0:
            bh = int(60 * fs)
            cv2.rectangle(frame, (0, 0), (ow, bh), RED, -1)
            cv2.putText(frame, "ALERT: " + banner, (int(20 * fs), int(bh * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th + 1, cv2.LINE_AA)
            banner_left -= 1
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    print(f"Saved {out_path} ({idx} frames, {ow}x{oh})")


if __name__ == "__main__":
    main()