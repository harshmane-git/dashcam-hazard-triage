"""Phase 2: Extract collision-risk signals from Phase 1 tracking CSV files.

Calculates bounding box dimensions, rolling-window smoothed heights, frame-to-frame growth rates,
and estimated Time-to-Collision (TTC) for tracked objects across video frames.

Usage:
    python src/signals.py \
        --input_dir outputs_nexar/phase1 \
        --output_dir outputs_nexar \
        --config config/nexar.yaml
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

# Root directory reference for project imports and backward compatibility
ROOT = Path(__file__).resolve().parent.parent

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("signals")
LOG_SEPARATOR = "============================================================"
LOG_SUBSEPARATOR = "------------------------------------------------------------"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments supporting batch and legacy single-video execution."""
    parser = argparse.ArgumentParser(
        description="Extract collision-risk signals (TTC, growth rate, dimensions) from tracking CSVs."
    )
    parser.add_argument(
        "--input_dir",
        "--input-dir",
        type=str,
        dest="input_dir",
        default=None,
        help="Directory containing Phase 1 _tracking.csv files (or single tracking CSV path).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        dest="output_dir",
        default=None,
        help="Base directory where the phase2 subdirectory will be created.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/nexar.yaml",
        help="Path to YAML configuration file (default: config/nexar.yaml).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling average window size in frames for bounding box height smoothing (default: 5).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Default video FPS to assume if not available from data (default: 30.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output signals CSV files.",
    )
    # Optional positional argument for backwards compatibility with legacy pipeline scripts
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Optional single video or CSV file path for legacy invocation.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Backward-compatibility geometry helpers for downstream modules (e.g. render_overlay)
# ----------------------------------------------------------------------

def video_meta(path: Any) -> Tuple[int, int, float]:
    """Retrieve video dimensions and FPS."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return w, h, fps


def roi_polygon(roi: Dict[str, Any], w: int, h: int) -> np.ndarray:
    """Generate ROI polygon array for visualization."""
    cx, ty, by = roi["center_x"], roi["horizon_y"], roi["bottom_y"]
    tw, bw = roi["top_half_width"], roi["bottom_half_width"]
    pts = [(cx - tw, ty), (cx + tw, ty), (cx + bw, by), (cx - bw, by)]
    return np.array([[int(x * w), int(y * h)] for x, y in pts], np.int32)


def lane_polygon(calib: Dict[str, Any], roi: Dict[str, Any], w: int, h: int, widen: float = 1.0) -> np.ndarray:
    """Generate lane perspective polygon array for visualization."""
    y_top = calib.get("y_h", roi.get("horizon_y", 0.5) * h) + 0.06 * h
    y_bot = roi.get("bottom_y", 0.95) * h
    u = calib.get("u_lane", roi.get("u_lane", 1.1)) * widen
    lane_offset = roi.get("lane_offset_u", 0.0)
    x_vp = calib.get("x_vp", roi.get("center_x", 0.5) * w)

    pts = {}
    for y in (y_top, y_bot):
        dy = max(1.0, y - calib.get("y_h", roi.get("horizon_y", 0.5) * h))
        c = x_vp + lane_offset * dy
        pts[y] = (c - u * dy, c + u * dy)
    return np.array([
        [pts[y_top][0], y_top],
        [pts[y_top][1], y_top],
        [pts[y_bot][1], y_bot],
        [pts[y_bot][0], y_bot],
    ], np.int32)


# ----------------------------------------------------------------------
# Core Signal Extraction Logic
# ----------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file if available."""
    if not config_path.is_file():
        # Check relative to project root
        alt_cfg = ROOT / config_path
        if alt_cfg.is_file():
            config_path = alt_cfg
        else:
            logger.warning("Config file not found at %s. Using default parameters.", config_path)
            return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.warning("Failed to parse config at %s (%s). Using defaults.", config_path, e)
        return {}


def extract_video_name(csv_path: Path) -> str:
    """Derive base video name from tracking CSV filename."""
    stem = csv_path.stem
    for suffix in ["_tracking", "_tracks"]:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def discover_tracking_files(input_path: Path) -> List[Path]:
    """Discover all Phase 1 tracking CSV files under input path."""
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path does not exist: {input_path}")

    # Check both input_path and any input_path/phase1 subdirectory
    search_dirs = [input_path]
    phase1_subdir = input_path / "phase1"
    if phase1_subdir.is_dir():
        search_dirs.append(phase1_subdir)

    discovered: List[Path] = []
    for directory in search_dirs:
        # Prioritize *_tracking.csv (produced by Phase 1 batch)
        tracking_csvs = sorted(directory.glob("*_tracking.csv"))
        if tracking_csvs:
            discovered.extend(tracking_csvs)
        else:
            # Fallback to *_tracks.csv (legacy naming)
            tracks_csvs = sorted(directory.glob("*_tracks.csv"))
            if tracks_csvs:
                discovered.extend(tracks_csvs)
            else:
                # General .csv fallback (ignoring signals or temp files)
                gen_csvs = sorted(
                    p for p in directory.glob("*.csv")
                    if not p.name.endswith("_signals.csv")
                    and not p.name.endswith(".tmp.csv")
                )
                discovered.extend(gen_csvs)

    # Deduplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for f in discovered:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(f)

    return sorted(unique_files)


def extract_signals_from_df(
    df: pd.DataFrame,
    window_size: int = 5,
    default_fps: float = 30.0,
) -> pd.DataFrame:
    """Calculate bounding box dimensions, smoothed height, growth rate, and TTC features.

    Args:
        df: DataFrame containing frame-by-frame tracking data.
        window_size: Rolling average window size for height smoothing.
        default_fps: Fallback FPS if not inferrable from timestamps.

    Returns:
        DataFrame enriched with width, height, smoothed_h, growth_rate, and ttc columns.
    """
    signal_cols = [
        "frame", "t_sec", "track_id", "cls", "conf",
        "x1", "y1", "x2", "y2",
        "width", "height", "smoothed_h", "growth_rate", "ttc"
    ]

    if df.empty:
        return pd.DataFrame(columns=signal_cols)

    required_cols = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Tracking data is missing required columns: {sorted(missing)}")

    # Ensure coordinates and frame numbers are numeric
    for col in ["x1", "y1", "x2", "y2", "frame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any unparseable rows
    df = df.dropna(subset=["track_id", "frame", "x1", "y1", "x2", "y2"]).copy()
    if df.empty:
        return pd.DataFrame(columns=signal_cols)

    # 1. Bounding Box Dimensions
    df["width"] = (df["x2"] - df["x1"]).clip(lower=0.0).round(2)
    df["height"] = (df["y2"] - df["y1"]).clip(lower=1e-3).round(2)

    # 2. Determine Video FPS (infer from t_sec if available, otherwise default_fps)
    fps = default_fps
    if "t_sec" in df.columns:
        df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
        # Inquire median time difference between consecutive frames of tracked objects
        dt_series = df.groupby("track_id")["t_sec"].diff()
        positive_dts = dt_series[dt_series > 0]
        if not positive_dts.empty:
            median_dt = positive_dts.median()
            if median_dt > 0:
                inferred_fps = round(1.0 / median_dt)
                if 10 <= inferred_fps <= 120:
                    fps = float(inferred_fps)
    else:
        df["t_sec"] = (df["frame"] / fps).round(3)

    # Ensure strictly sorted by track_id and frame for rolling computations
    df = df.sort_values(by=["track_id", "frame"]).reset_index(drop=True)

    # 3. Smoothing: rolling average over bounding box height per track_id
    # min_periods=1 ensures no NaNs are introduced for initial frames of a track
    df["smoothed_h"] = (
        df.groupby("track_id")["height"]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .round(2)
    )

    # 4. Growth Rate: frame-to-frame percentage change in smoothed height
    frame_diff = df.groupby("track_id")["frame"].diff()
    pct_change = df.groupby("track_id")["smoothed_h"].pct_change()
    # Normalize by frame difference in case the tracker skipped any frames
    growth_rate = np.where(frame_diff > 0, pct_change / frame_diff, pct_change)
    df["growth_rate"] = np.round(growth_rate, 5)

    # 5. Time-to-Collision (TTC): estimate TTC based on growth rate and video FPS
    # Physical relation: TTC = h / (dh/dt) = 1 / (growth_rate * fps)
    # Only approaching objects (growth_rate > 0) have an impending collision; receding = NaN
    df["ttc"] = np.where(
        df["growth_rate"] > 1e-5,
        (1.0 / (df["growth_rate"] * fps)).round(3),
        np.nan,
    )

    # Supplementary columns for downstream compatibility
    df["bbox_w"] = df["width"]
    df["bbox_h"] = df["height"]
    df["rel_growth"] = np.round(df["growth_rate"] * fps, 4)

    # Sort output chronologically by frame, then track_id
    df = df.sort_values(by=["frame", "track_id"]).reset_index(drop=True)
    return df


def process_signals_file(
    csv_path: Path,
    output_csv: Path,
    tmp_csv: Path,
    window_size: int,
    default_fps: float,
) -> Tuple[int, int, float]:
    """Process a single tracking CSV and atomically save signals CSV.

    Returns:
        Tuple of (total_rows, unique_tracks, elapsed_seconds).
    """
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)

    signals_df = extract_signals_from_df(
        df=df,
        window_size=window_size,
        default_fps=default_fps,
    )

    # Write atomically via temporary file
    signals_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, output_csv)

    elapsed = time.perf_counter() - t0
    total_rows = len(signals_df)
    unique_tracks = int(signals_df["track_id"].nunique()) if total_rows > 0 else 0

    return total_rows, unique_tracks, elapsed


def main() -> None:
    """Execute Phase 2 signal extraction pipeline."""
    args = parse_args()
    batch_start_time = time.perf_counter()

    logger.info(LOG_SEPARATOR)
    logger.info("DASHCAM RISK PREDICTION - PHASE 2 SIGNAL EXTRACTION")
    logger.info(LOG_SEPARATOR)

    # Handle argument resolutions (supporting both new CLI and legacy calls)
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # Determine window size (from CLI or config)
    window_size = args.window
    if "fcw" in cfg and "min_samples" in cfg["fcw"]:
        window_size = int(cfg["fcw"].get("min_samples", window_size))

    default_fps = float(args.fps)

    # Resolve input path
    input_str = args.input_dir
    output_str = args.output_dir

    if input_str is None and args.video is not None:
        # Legacy positional invocation support
        candidate_video = Path(args.video)
        legacy_track_cand1 = ROOT / "outputs" / "phase1" / f"{candidate_video.stem}_tracking.csv"
        legacy_track_cand2 = ROOT / "outputs" / "phase1" / f"{candidate_video.stem}_tracks.csv"

        if legacy_track_cand1.is_file():
            input_path = legacy_track_cand1
        elif legacy_track_cand2.is_file():
            input_path = legacy_track_cand2
        else:
            input_path = candidate_video
        output_str = output_str or str(ROOT / "outputs")
    elif input_str is not None:
        input_path = Path(input_str).resolve()
        output_str = output_str or str(input_path.parent)
    else:
        logger.error("Missing required argument: --input_dir (or specify tracking CSV / video path).")
        sys.exit(1)

    base_output_dir = Path(output_str).resolve()
    phase2_dir = base_output_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Input source:      %s", input_path)
    logger.info("Output directory:  %s", phase2_dir)
    logger.info("Smoothing window:  %d frames", window_size)
    logger.info("Default base FPS:  %.1f fps", default_fps)

    # Discover tracking CSV files
    tracking_files = discover_tracking_files(input_path)
    total_files = len(tracking_files)

    if total_files == 0:
        logger.warning("No tracking CSV files found under: %s", input_path)
        logger.info("Phase 2 finished with 0 files processed.")
        return

    logger.info("Discovered %d tracking CSV file(s) to process.", total_files)
    logger.info("Starting signal extraction loop...")

    successful_files: List[str] = []
    skipped_files: List[str] = []
    failed_files: List[Tuple[str, str]] = []

    for idx, csv_path in enumerate(tracking_files, start=1):
        video_name = extract_video_name(csv_path)
        output_csv = phase2_dir / f"{video_name}_signals.csv"
        tmp_csv = phase2_dir / f"{video_name}_signals.tmp.csv"

        logger.info(LOG_SUBSEPARATOR)
        logger.info("[%d/%d] Processing signals for: %s", idx, total_files, csv_path.name)

        if output_csv.exists() and not args.overwrite:
            logger.info(
                "[%d/%d] Output '%s' already exists. Skipping (use --overwrite to re-process).",
                idx,
                total_files,
                output_csv.name,
            )
            skipped_files.append(csv_path.name)
            continue

        try:
            total_rows, unique_tracks, elapsed = process_signals_file(
                csv_path=csv_path,
                output_csv=output_csv,
                tmp_csv=tmp_csv,
                window_size=window_size,
                default_fps=default_fps,
            )
            logger.info(
                "[%d/%d] Successfully generated signals for '%s': %d rows, %d tracks in %.3fs.",
                idx,
                total_files,
                csv_path.name,
                total_rows,
                unique_tracks,
                elapsed,
            )
            logger.info("[%d/%d] Saved signals to: %s", idx, total_files, output_csv)
            successful_files.append(csv_path.name)

        except Exception as exc:
            logger.error(
                "[%d/%d] Failed to process '%s': %s",
                idx,
                total_files,
                csv_path.name,
                exc,
                exc_info=True,
            )
            # Ensure any leftover temporary file is cleaned up
            tmp_csv.unlink(missing_ok=True)
            failed_files.append((csv_path.name, str(exc)))
            logger.info("[%d/%d] Continuing to next file...", idx, total_files)

    # Final Summary Logging
    total_batch_time = time.perf_counter() - batch_start_time
    logger.info(LOG_SEPARATOR)
    logger.info("PHASE 2 SIGNAL EXTRACTION COMPLETE - FINAL SUMMARY")
    logger.info(LOG_SEPARATOR)
    logger.info("Total tracking files discovered: %d", total_files)
    logger.info("Successfully processed:          %d", len(successful_files))
    logger.info("Skipped (already exists):        %d", len(skipped_files))
    logger.info("Failed:                          %d", len(failed_files))
    logger.info("Total elapsed time:              %.2fs", total_batch_time)

    processed_count = len(successful_files)
    if processed_count > 0:
        logger.info(
            "Average time per processed file: %.3fs",
            total_batch_time / processed_count,
        )

    if failed_files:
        logger.warning("Summary of failed files (%d):", len(failed_files))
        for fname, err in failed_files:
            logger.warning("  - %s: %s", fname, err)

    logger.info("Phase 2 signals directory: %s", phase2_dir)
    logger.info(LOG_SEPARATOR)

    if failed_files and len(successful_files) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
