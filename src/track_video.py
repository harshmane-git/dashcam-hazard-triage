"""Phase 1: YOLO26 detection + ByteTrack on a video.

Usage (from project root):
    python src/track_video.py data/raw/clip.mp4
Outputs (in outputs/phase1/):
    <name>_tracked.mp4   annotated video with track IDs
    <name>_tracks.csv    per-frame tracked boxes (input for Phase 2)
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--show", action="store_true", help="live preview window (q to quit)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    id2name = {v: k for k, v in cfg["classes"].items()}

    src = Path(args.video)
    out_dir = ROOT / "outputs" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {src.name}  {w}x{h} @ {fps_in:.1f} fps, {n_total} frames")

    writer = cv2.VideoWriter(str(out_dir / f"{src.stem}_tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    csv_f = open(out_dir / f"{src.stem}_tracks.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])

    model = YOLO(cfg["model"])
    results = model.track(
        source=str(src), stream=True, persist=True, tracker=cfg["tracker"],
        classes=list(cfg["classes"].values()), conf=cfg["conf"],
        imgsz=cfg["imgsz"], device=cfg["device"], verbose=False,
    )

    t0 = time.perf_counter()
    n = 0
    for n, r in enumerate(results, start=1):
        frame_idx = n - 1
        b = r.boxes
        if b is not None and b.id is not None:
            for tid, c, cf, xyxy in zip(b.id.int().tolist(), b.cls.int().tolist(),
                                        b.conf.tolist(), b.xyxy.tolist()):
                csv_w.writerow([frame_idx, round(frame_idx / fps_in, 3), tid,
                                id2name.get(c, c), round(cf, 3),
                                *[round(v, 1) for v in xyxy]])
        annotated = r.plot()
        writer.write(annotated)
        if args.show:
            cv2.imshow("tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if n % 100 == 0:
            print(f"  {n}/{n_total} frames, {n / (time.perf_counter() - t0):.1f} fps")

    dt = time.perf_counter() - t0
    writer.release()
    csv_f.close()
    cv2.destroyAllWindows()
    print(f"Done: {n} frames in {dt:.1f}s -> {n / dt:.1f} fps (video was {fps_in:.1f} fps)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
