
"""Phase 1: YOLO detection + ByteTrack on a video.

Legacy usage:
    python src/track_video.py data/raw/clip.mp4

Nexar usage:
    python src/track_video.py VIDEO \
        --config config/nexar.yaml \
        --output-dir outputs_nexar/phase1 \
        --no-video

Options:
    --no-video   Disable annotated video generation.
    --show       Display live preview.
    --overwrite  Explicitly allow replacement of existing outputs.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(
        description="YOLO detection and ByteTrack video processing."
    )
    ap.add_argument("video", help="Input video path")
    ap.add_argument(
        "--config",
        default=str(ROOT / "config" / "config.yaml")
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for tracking CSV and annotated video"
    )
    ap.add_argument(
        "--no-video",
        action="store_true",
        help="Disable annotated video generation"
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show live preview (press q to quit)"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files"
    )

    args = ap.parse_args()

    # --------------------------------------------------
    # 1. Validate input and configuration
    # --------------------------------------------------

    src = Path(args.video).expanduser()
    config_path = Path(args.config).expanduser()
    legacy_mode = args.output_dir is None

    out_dir = (
        ROOT / "outputs" / "phase1"
        if legacy_mode
        else Path(args.output_dir).expanduser()
    )

    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {src}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    required_keys = [
        "model", "device", "imgsz", "conf", "tracker", "classes"
    ]

    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")

    if not isinstance(cfg["classes"], dict) or not cfg["classes"]:
        raise ValueError("Configuration classes must be a non-empty mapping")

    id2name = {v: k for k, v in cfg["classes"].items()}

    # --------------------------------------------------
    # 2. Prepare output paths
    # --------------------------------------------------

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{src.stem}_tracks.csv"
    csv_tmp = out_dir / f"{src.stem}_tracks.tmp.csv"

    video_path = out_dir / f"{src.stem}_tracked.mp4"
    video_tmp = out_dir / f"{src.stem}_tracked.tmp.mp4"

    final_paths = [csv_path]
    if not args.no_video:
        final_paths.append(video_path)

    existing = [p for p in final_paths if p.exists()]

    if existing and not (args.overwrite or legacy_mode):
        raise FileExistsError(
            "Output already exists:\n"
            + "\n".join(str(p) for p in existing)
            + "\nUse --overwrite only if replacement is intended."
        )

    # --------------------------------------------------
    # 3. Validate video metadata
    # --------------------------------------------------

    cap = cv2.VideoCapture(str(src))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {src}")

    try:
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps_in <= 0:
        raise ValueError(f"Invalid video FPS: {fps_in}")

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid video dimensions: {w}x{h}")

    print(
        f"Video: {src.name}  {w}x{h} "
        f"@ {fps_in:.1f} fps, {n_total} frames"
    )

    # --------------------------------------------------
    # 4. Initialize YOLO model and output resources
    # --------------------------------------------------

    model = YOLO(cfg["model"])

    writer = None
    csv_f = None
    completed = False
    generator_finished = False
    n = 0
    dt = 0.0

    try:
        if not args.no_video:
            writer = cv2.VideoWriter(
                str(video_tmp),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_in,
                (w, h)
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Cannot initialize video writer: {video_tmp}"
                )

        csv_f = csv_tmp.open("w", newline="")
        csv_w = csv.writer(csv_f)

        csv_w.writerow([
            "frame", "t_sec", "track_id", "cls",
            "conf", "x1", "y1", "x2", "y2"
        ])

        # --------------------------------------------------
        # 5. Run YOLO + ByteTrack
        # --------------------------------------------------

        results = model.track(
            source=str(src),
            stream=True,
            persist=True,
            tracker=cfg["tracker"],
            classes=list(cfg["classes"].values()),
            conf=cfg["conf"],
            imgsz=cfg["imgsz"],
            device=cfg["device"],
            verbose=False,
        )

        t0 = time.perf_counter()

        for n, r in enumerate(results, start=1):
            frame_idx = n - 1
            b = r.boxes

            if b is not None and b.id is not None:
                for tid, c, cf, xyxy in zip(
                    b.id.int().tolist(),
                    b.cls.int().tolist(),
                    b.conf.tolist(),
                    b.xyxy.tolist()
                ):
                    csv_w.writerow([
                        frame_idx,
                        round(frame_idx / fps_in, 3),
                        tid,
                        id2name.get(c, c),
                        round(cf, 3),
                        *[round(v, 1) for v in xyxy]
                    ])

            # Plot only when video output or preview is requested.
            if writer is not None or args.show:
                annotated = r.plot()

                if writer is not None:
                    writer.write(annotated)

                if args.show:
                    cv2.imshow("tracking", annotated)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Preview stopped by user.")
                        break

            if n % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  {n}/{n_total} frames, "
                    f"{n / elapsed:.1f} fps"
                )


        # A video is complete only if its entire result generator finished.
        # Breaking the loop using q must not create a successful final CSV.
        else:
            generator_finished = True

        dt = time.perf_counter() - t0

        if n == 0:
            raise RuntimeError("No frames were processed.")

        if n_total > 0 and n != n_total:
            raise RuntimeError(
                f"Incomplete video processing: "
                f"processed {n}/{n_total} frames."
            )

        if not generator_finished:
            raise RuntimeError(
                "Processing was interrupted. Temporary outputs will not "
                "be promoted to final output files."
            )

        # --------------------------------------------------
        # 6. Finalize outputs
        # --------------------------------------------------

        csv_f.flush()
        csv_f.close()
        csv_f = None

        if writer is not None:
            writer.release()
            writer = None

        # Promote completed temporary files to final filenames.
        if not args.no_video:
            os.replace(video_tmp, video_path)

        os.replace(csv_tmp, csv_path)
        completed = True
    finally:
        # --------------------------------------------------
        # 7. Cleanup resources on success or failure
        # --------------------------------------------------

        if csv_f is not None:
            csv_f.close()

        if writer is not None:
            writer.release()

        if args.show:
            cv2.destroyAllWindows()

        # Remove incomplete temporary outputs.
        if not completed:
            csv_tmp.unlink(missing_ok=True)
            video_tmp.unlink(missing_ok=True)

    print(
        f"Done: {n} frames in {dt:.1f}s "
        f"-> {n / dt:.1f} fps "
        f"(video was {fps_in:.1f} fps)"
    )
    print(f"Tracks CSV: {csv_path}")

    if not args.no_video:
        print(f"Annotated video: {video_path}")


if __name__ == "__main__":
    main()
