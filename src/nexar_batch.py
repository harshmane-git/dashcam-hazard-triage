"""Batch orchestrator for Phase 1: YOLO26-N and ByteTrack video tracking.

Runs object detection and tracking across a directory of dashcam videos (e.g. Nexar dataset)
and exports frame-by-frame tracking data to CSV files.

Usage:
    python src/nexar_batch.py \
        --input_dir data/nexar-collision-prediction/train \
        --output_dir outputs_nexar \
        --config config/nexar.yaml
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import yaml
from ultralytics import YOLO

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexar_batch")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO26-N and ByteTrack batch tracking on dashcam videos."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing input .mp4 videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory (phase1 subdirectory will be created inside).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/nexar.yaml",
        help="Path to YAML configuration file (default: config/nexar.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tracking CSV files.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration from a YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration in {config_path}")

    return cfg


def resolve_model_path(model_name: str, config_dir: Path) -> str:
    """Resolve YOLO model weights path across workspace conventions."""
    p = Path(model_name)
    if p.is_file():
        return str(p)

    # Check relative to config directory
    cand_cfg = config_dir / model_name
    if cand_cfg.is_file():
        return str(cand_cfg)

    # Check relative to project root
    project_root = Path(__file__).resolve().parent.parent
    cand_root = project_root / model_name
    if cand_root.is_file():
        return str(cand_root)

    # Return original string for Ultralytics default model resolution/download
    return model_name


def get_device(configured_device: Any) -> Any:
    """Validate and return appropriate PyTorch/CUDA device."""
    if isinstance(configured_device, int) or (
        isinstance(configured_device, str) and configured_device.isdigit()
    ):
        device_id = int(configured_device)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device_id)
            logger.info("CUDA GPU %d available: %s", device_id, gpu_name)
            return device_id
        else:
            logger.warning(
                "CUDA device %d requested, but CUDA is not available. Falling back to CPU.",
                device_id,
            )
            return "cpu"

    if isinstance(configured_device, str):
        if configured_device.lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested ('%s'), but CUDA is not available. Falling back to CPU.",
                configured_device,
            )
            return "cpu"
        return configured_device

    return 0 if torch.cuda.is_available() else "cpu"


def process_single_video(
    video_path: Path,
    model: YOLO,
    cfg: Dict[str, Any],
    id2name: Dict[int, str],
    device: Any,
    output_csv: Path,
    tmp_csv: Path,
) -> Tuple[int, int, float, float]:
    """Process a single video using YOLO26-N + ByteTrack and save tracking data to CSV.

    Args:
        video_path: Path to input .mp4 video.
        model: Loaded YOLO model instance.
        cfg: Configuration dictionary.
        id2name: Mapping of COCO class IDs to human-readable names.
        device: Target compute device.
        output_csv: Final output CSV path.
        tmp_csv: Temporary output CSV path for atomic writing.

    Returns:
        Tuple of (frame_count, detection_count, elapsed_seconds, video_fps).
    """
    # 1. Read video metadata with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if video_fps <= 0 or math.isnan(video_fps):
        logger.warning(
            "Video %s reported invalid FPS (%s). Defaulting to 30.0 FPS.",
            video_path.name,
            video_fps,
        )
        video_fps = 30.0

    logger.info(
        "Video specs: %s (%dx%d @ %.1f fps, %d estimated frames)",
        video_path.name,
        width,
        height,
        video_fps,
        total_frames,
    )

    # 2. Reset tracker state from any previous video to avoid ID bleed
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            model.predictor.trackers = []

    # 3. Stream tracking results and write frame-by-frame data
    classes_to_track = (
        list(cfg["classes"].values())
        if isinstance(cfg.get("classes"), dict)
        else None
    )

    tracker_config = cfg.get("tracker", "bytetrack.yaml")
    conf_thresh = cfg.get("conf", 0.30)
    img_size = cfg.get("imgsz", 640)

    frame_count = 0
    detection_count = 0
    t0 = time.perf_counter()

    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "t_sec", "track_id", "cls",
            "conf", "x1", "y1", "x2", "y2"
        ])

        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker=tracker_config,
            classes=classes_to_track,
            conf=conf_thresh,
            imgsz=img_size,
            device=device,
            verbose=False,
        )

        for n, r in enumerate(results, start=1):
            frame_idx = n - 1
            frame_count += 1
            t_sec = round(frame_idx / video_fps, 3)

            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().tolist()
                cls_ids = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                coords = boxes.xyxy.tolist()

                for tid, c_id, conf_val, xyxy in zip(track_ids, cls_ids, confs, coords):
                    cls_name = id2name.get(c_id, str(c_id))
                    writer.writerow([
                        frame_idx,
                        t_sec,
                        tid,
                        cls_name,
                        round(conf_val, 3),
                        round(xyxy[0], 1),
                        round(xyxy[1], 1),
                        round(xyxy[2], 1),
                        round(xyxy[3], 1),
                    ])
                    detection_count += 1

    if frame_count == 0:
        raise RuntimeError(f"No frames were processed from {video_path.name}")

    # 4. Atomically promote temporary file to final output CSV
    os.replace(tmp_csv, output_csv)
    elapsed = time.perf_counter() - t0

    return frame_count, detection_count, elapsed, video_fps


def main() -> None:
    """Execute batch video tracking pipeline."""
    args = parse_args()
    batch_start_time = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DASHCAM RISK PREDICTION - PHASE 1 BATCH TRACKING")
    logger.info("=" * 60)
    logger.info("Input directory:  %s", args.input_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Config file:      %s", args.config)

    # 1. Load and validate YAML config
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # Build COCO ID to class name mapping
    raw_classes = cfg.get("classes", {})
    if isinstance(raw_classes, dict):
        id2name = {v: k for k, v in raw_classes.items()}
    else:
        id2name = {}

    # Determine compute device
    device = get_device(cfg.get("device", 0))

    # 2. Initialize YOLO26-N Model
    configured_model = cfg.get("model", "yolo26n.pt")
    resolved_model_path = resolve_model_path(configured_model, config_path.parent)
    logger.info("Initializing YOLO26-N model from: %s", resolved_model_path)
    model = YOLO(resolved_model_path)

    tracker_cfg = cfg.get("tracker", "bytetrack.yaml")
    logger.info("Configured tracker: %s", tracker_cfg)
    logger.info("Confidence threshold: %s", cfg.get("conf", 0.30))
    logger.info("Inference image size: %s", cfg.get("imgsz", 640))
    logger.info("Target classes: %s", list(raw_classes.keys()) if isinstance(raw_classes, dict) else "All")

    # 3. Directory Scanning and Directory Creation
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    base_output_dir = Path(args.output_dir).resolve()
    phase1_dir = base_output_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 1 outputs will be stored in: %s", phase1_dir)

    video_files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp4"
    ])

    total_videos = len(video_files)
    if total_videos == 0:
        logger.warning("No .mp4 video files found in %s.", input_dir)
        logger.info("Batch completed with 0 videos processed.")
        return

    logger.info("Discovered %d video(s) to process.", total_videos)
    logger.info("Starting batch processing loop...")

    # 4. Processing Loop with Fault Tolerance
    successful_videos: List[str] = []
    failed_videos: List[Tuple[str, str]] = []
    skipped_videos: List[str] = []

    for idx, video_path in enumerate(video_files, start=1):
        video_name = video_path.stem
        output_csv = phase1_dir / f"{video_name}_tracking.csv"
        tmp_csv = phase1_dir / f"{video_name}_tracking.tmp.csv"

        logger.info("------------------------------------------------------------")
        logger.info("[%d/%d] Starting processing for: %s", idx, total_videos, video_path.name)

        if output_csv.exists() and not args.overwrite:
            logger.info(
                "[%d/%d] Output '%s' already exists. Skipping (use --overwrite to re-process).",
                idx,
                total_videos,
                output_csv.name,
            )
            skipped_videos.append(video_path.name)
            continue

        try:
            frames, detections, elapsed, fps = process_single_video(
                video_path=video_path,
                model=model,
                cfg=cfg,
                id2name=id2name,
                device=device,
                output_csv=output_csv,
                tmp_csv=tmp_csv,
            )
            proc_fps = frames / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%d/%d] Successfully processed '%s': %d frames, %d tracks in %.2fs (%.1f fps).",
                idx,
                total_videos,
                video_path.name,
                frames,
                detections,
                elapsed,
                proc_fps,
            )
            logger.info("[%d/%d] Saved tracking data to: %s", idx, total_videos, output_csv)
            successful_videos.append(video_path.name)

        except Exception as exc:
            logger.error(
                "[%d/%d] Failed to process '%s': %s",
                idx,
                total_videos,
                video_path.name,
                exc,
                exc_info=True,
            )
            # Ensure partial temporary file is cleaned up
            tmp_csv.unlink(missing_ok=True)
            failed_videos.append((video_path.name, str(exc)))
            logger.info("[%d/%d] Continuing to next video...", idx, total_videos)

    # 5. Final Summary Logging
    total_batch_time = time.perf_counter() - batch_start_time
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total videos discovered: %d", total_videos)
    logger.info("Successfully processed:  %d", len(successful_videos))
    logger.info("Skipped (already done):  %d", len(skipped_videos))
    logger.info("Failed:                  %d", len(failed_videos))
    logger.info("Total elapsed time:      %.2fs", total_batch_time)

    processed_count = len(successful_videos)
    if processed_count > 0:
        logger.info(
            "Average time per processed video: %.2fs",
            total_batch_time / processed_count,
        )

    if failed_videos:
        logger.warning("Summary of failed videos (%d):", len(failed_videos))
        for fname, err in failed_videos:
            logger.warning("  - %s: %s", fname, err)

    logger.info("Phase 1 tracking directory: %s", phase1_dir)
    logger.info("=" * 60)

    if failed_videos and len(successful_videos) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
