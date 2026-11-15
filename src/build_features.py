"""Phase 3: Consolidated Tabular Feature Dataset Construction.

Converts frame-by-frame object tracking and risk signals (Phase 2 _signals.csv)
into a tabular machine-learning dataset for collision prediction modeling.

Enforces strict anti-leakage temporal truncation prior to the collision point,
extracts both pre-cutoff historical features and focused lookback window dynamics,
and handles missing detections with clean, non-NaN default imputations.

Usage:
    python src/build_features.py \
        --signals_dir outputs_nexar/phase2 \
        --split_csv data/nexar-collision-prediction/splits/dev.csv \
        --output_dir outputs_nexar \
        --cutoff_mode event \
        --lookback_sec 3.0 \
        --ttc_cap 15.0 \
        --config config/nexar.yaml
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Module-level separators resistant to copy-paste wrapping
LOG_SEPARATOR = "============================================================"
LOG_SUBSEPARATOR = "------------------------------------------------------------"

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_features")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature dataset construction."""
    parser = argparse.ArgumentParser(
        description="Construct Phase 3 tabular feature dataset from Phase 2 signal CSVs."
    )
    parser.add_argument(
        "--signals_dir",
        "--signals-dir",
        type=str,
        dest="signals_dir",
        required=True,
        help="Directory containing Phase 2 *_signals.csv files (e.g. outputs_nexar/phase2).",
    )
    parser.add_argument(
        "--split_csv",
        "--split-csv",
        type=str,
        dest="split_csv",
        required=True,
        help="Path to split CSV file (e.g. data/nexar-collision-prediction/splits/dev.csv).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        dest="output_dir",
        required=True,
        help="Base output directory (creates phase3/ subdirectory inside).",
    )
    parser.add_argument(
        "--cutoff_mode",
        "--cutoff-mode",
        type=str,
        dest="cutoff_mode",
        choices=["event", "alert"],
        default="event",
        help="Whether positive videos are truncated strictly before time_of_event or time_of_alert (default: event).",
    )
    parser.add_argument(
        "--lookback_sec",
        "--lookback-sec",
        type=float,
        dest="lookback_sec",
        default=3.0,
        help="Pre-cutoff lookback window duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--ttc_cap",
        "--ttc-cap",
        type=float,
        dest="ttc_cap",
        default=15.0,
        help="Upper cap in seconds for raw TTC values to suppress outliers (default: 15.0).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/nexar.yaml",
        help="Path to YAML configuration file (default: config/nexar.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file if available."""
    if not config_path.is_file():
        logger.warning("Config file not found at %s. Using default parameters.", config_path)
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.warning("Failed to parse config at %s (%s). Using defaults.", config_path, exc)
        return {}


def format_video_id(raw_id: Any) -> str:
    """Safely format video id to 5-digit zero-padded string."""
    try:
        return f"{int(float(raw_id)):05d}"
    except (ValueError, TypeError):
        return str(raw_id).strip().zfill(5)


def extract_features_for_video(
    signals_df: pd.DataFrame,
    target: int,
    cutoff_time: Optional[float],
    lookback_sec: float,
    ttc_cap: float,
) -> Dict[str, Any]:
    """Extract collision-risk features for a single video from pre-cutoff signal data.

    Args:
        signals_df: Raw DataFrame of Phase 2 signals.
        target: Binary label (0 = negative, 1 = collision/near-miss).
        cutoff_time: Collision timestamp in seconds for positive videos, or None for negatives.
        lookback_sec: Duration in seconds of the recent pre-cutoff lookback window.
        ttc_cap: Ceiling value for TTC.

    Returns:
        Dictionary of engineered numerical features.
    """
    # 1. Anti-Leakage Temporal Truncation
    if signals_df.empty or "t_sec" not in signals_df.columns:
        all_df = pd.DataFrame()
        window_df = pd.DataFrame()
    else:
        # Ensure timestamp is numeric and sort chronologically
        signals_df["t_sec"] = pd.to_numeric(signals_df["t_sec"], errors="coerce")
        if "frame" in signals_df.columns:
            signals_df = signals_df.sort_values(by=["frame", "track_id"]).reset_index(drop=True)
        valid_t = signals_df["t_sec"].dropna()

        if target == 1:
            # Strictly filter frames before cutoff_time to eliminate data leakage
            effective_cutoff = cutoff_time if (cutoff_time is not None and not math.isnan(cutoff_time)) else (valid_t.max() if not valid_t.empty else 0.0)
            all_df = signals_df[signals_df["t_sec"] < effective_cutoff].copy()
            window_start = effective_cutoff - lookback_sec
            window_df = all_df[all_df["t_sec"] >= window_start].copy()
        else:
            # Negative clips: use full clip up to max t_sec
            effective_cutoff = float(valid_t.max()) if not valid_t.empty else 0.0
            all_df = signals_df.copy()
            window_start = max(0.0, effective_cutoff - lookback_sec)
            window_df = all_df[all_df["t_sec"] >= window_start].copy()

    # Helper function to extract and cap valid TTC series
    def _clean_ttc_series(df_slice: pd.DataFrame) -> pd.Series:
        if df_slice.empty or "ttc" not in df_slice.columns:
            return pd.Series(dtype=float)
        ttc_num = pd.to_numeric(df_slice["ttc"], errors="coerce").dropna()
        # Only positive finite TTC values represent approaching hazard
        valid = ttc_num[(ttc_num > 0) & np.isfinite(ttc_num)]
        return valid.clip(lower=0.0, upper=ttc_cap)

    # ------------------------------------------------------------------
    # A. TTC Features (capped at ttc_cap)
    # ------------------------------------------------------------------
    ttc_all = _clean_ttc_series(all_df)
    ttc_win = _clean_ttc_series(window_df)

    min_ttc_all = float(ttc_all.min()) if not ttc_all.empty else ttc_cap
    min_ttc_window = float(ttc_win.min()) if not ttc_win.empty else ttc_cap
    p10_ttc_window = float(ttc_win.quantile(0.10)) if not ttc_win.empty else ttc_cap
    mean_ttc_window = float(ttc_win.mean()) if not ttc_win.empty else ttc_cap

    # Linear trend (slope) of frame-level min TTC over lookback window
    # Negative slope = hazard rapidly closing in
    ttc_slope_window = 0.0
    if not window_df.empty and "ttc" in window_df.columns:
        w_ttc_df = window_df.copy()
        w_ttc_df["clean_ttc"] = pd.to_numeric(w_ttc_df["ttc"], errors="coerce")
        w_ttc_df = w_ttc_df[(w_ttc_df["clean_ttc"] > 0) & np.isfinite(w_ttc_df["clean_ttc"])].copy()
        if not w_ttc_df.empty:
            w_ttc_df["clean_ttc"] = w_ttc_df["clean_ttc"].clip(upper=ttc_cap)
            frame_min_s = w_ttc_df.groupby("t_sec")["clean_ttc"].min()
            if len(frame_min_s) >= 2:
                t_arr = frame_min_s.index.to_numpy(dtype=float)
                val_arr = frame_min_s.to_numpy(dtype=float)
                if t_arr[-1] > t_arr[0]:
                    slope, _ = np.polyfit(t_arr, val_arr, 1)
                    ttc_slope_window = float(slope)

    # Frame counts and ratios where min TTC < 1.5s and < 2.5s
    total_frames_win = int(window_df["frame"].nunique()) if (not window_df.empty and "frame" in window_df.columns) else 0
    if total_frames_win > 0 and "ttc" in window_df.columns:
        w_ttc_df = window_df.copy()
        w_ttc_df["clean_ttc"] = pd.to_numeric(w_ttc_df["ttc"], errors="coerce")
        # For each frame with detections, find min TTC across all tracked objects
        frame_min_ttc = w_ttc_df.groupby("frame")["clean_ttc"].min().fillna(np.inf)
        frames_ttc_under_1_5 = int((frame_min_ttc < 1.5).sum())
        ratio_ttc_under_1_5 = float(frames_ttc_under_1_5 / total_frames_win)
        frames_ttc_under_2_5 = int((frame_min_ttc < 2.5).sum())
        ratio_ttc_under_2_5 = float(frames_ttc_under_2_5 / total_frames_win)
    else:
        frames_ttc_under_1_5 = 0
        ratio_ttc_under_1_5 = 0.0
        frames_ttc_under_2_5 = 0
        ratio_ttc_under_2_5 = 0.0

    # ------------------------------------------------------------------
    # B. Bounding-Box Growth & Expansion Features
    # ------------------------------------------------------------------
    gr_col = "growth_rate" if "growth_rate" in window_df.columns else "rel_growth"
    if not window_df.empty and gr_col in window_df.columns:
        gr_series = pd.to_numeric(window_df[gr_col], errors="coerce").dropna()
        max_growth_rate_window = float(gr_series.max()) if not gr_series.empty else 0.0
        p90_growth_rate_window = float(gr_series.quantile(0.90)) if not gr_series.empty else 0.0
    else:
        max_growth_rate_window = 0.0
        p90_growth_rate_window = 0.0

    if not window_df.empty and "rel_growth" in window_df.columns:
        rg_series = pd.to_numeric(window_df["rel_growth"], errors="coerce").dropna()
        max_rel_growth_window = float(rg_series.max()) if not rg_series.empty else 0.0
    else:
        max_rel_growth_window = 0.0

    # Maximum height gain ratio per track in lookback window: max(smoothed_h) / initial(smoothed_h)
    max_height_gain_ratio = 0.0
    h_col = "smoothed_h" if "smoothed_h" in window_df.columns else ("height" if "height" in window_df.columns else "h_px")
    if not window_df.empty and h_col in window_df.columns and "track_id" in window_df.columns:
        gain_ratios: List[float] = []
        for _, t_grp in window_df.groupby("track_id"):
            h_s = pd.to_numeric(t_grp[h_col], errors="coerce").dropna()
            if not h_s.empty:
                h_init = float(h_s.iloc[0])
                h_max = float(h_s.max())
                if h_init > 1.0:
                    gain_ratios.append(h_max / h_init)
        if gain_ratios:
            max_height_gain_ratio = float(max(gain_ratios))

    # ------------------------------------------------------------------
    # C. Proximity & Ego-Lane Geometry Features
    # ------------------------------------------------------------------
    # Bounding box dimensions (width, height, area)
    if not window_df.empty:
        if "width" in window_df.columns:
            w_box = pd.to_numeric(window_df["width"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "x1" in window_df.columns and "x2" in window_df.columns:
            w_box = (pd.to_numeric(window_df["x2"], errors="coerce") - pd.to_numeric(window_df["x1"], errors="coerce")).clip(lower=0.0)
        else:
            w_box = pd.Series([0.0] * len(window_df), index=window_df.index)

        if "height" in window_df.columns:
            h_box = pd.to_numeric(window_df["height"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "h_px" in window_df.columns:
            h_box = pd.to_numeric(window_df["h_px"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "y1" in window_df.columns and "y2" in window_df.columns:
            h_box = (pd.to_numeric(window_df["y2"], errors="coerce") - pd.to_numeric(window_df["y1"], errors="coerce")).clip(lower=0.0)
        else:
            h_box = pd.Series([0.0] * len(window_df), index=window_df.index)

        max_bbox_h_window = float(h_box.max()) if not h_box.dropna().empty else 0.0
        area_box = w_box * h_box
        max_bbox_area_window = float(area_box.max()) if not area_box.dropna().empty else 0.0

        if "y2" in window_df.columns:
            y2_series = pd.to_numeric(window_df["y2"], errors="coerce").dropna()
            max_y2_window = float(y2_series.max()) if not y2_series.empty else 0.0
        else:
            max_y2_window = 0.0

        # Center-weighted min TTC: filtering specifically for objects in middle 50% of frame width
        if "x1" in window_df.columns and "x2" in window_df.columns and "ttc" in window_df.columns:
            x1_num = pd.to_numeric(window_df["x1"], errors="coerce")
            x2_num = pd.to_numeric(window_df["x2"], errors="coerce")
            cx = (x1_num + x2_num) / 2.0

            max_x = float(x2_num.max()) if not x2_num.dropna().empty else 1280.0
            # Standard video width (640 for 360p/downscaled, 1280 for 720p Nexar clips; 1920 if 1080p)
            frame_w = 640.0 if max_x <= 640.0 else (1280.0 if max_x <= 1280.0 else (1920.0 if max_x <= 1920.0 else max_x))

            center_mask = (cx >= 0.25 * frame_w) & (cx <= 0.75 * frame_w)
            center_df = window_df[center_mask]
            center_ttc = _clean_ttc_series(center_df)
            center_weighted_min_ttc = float(center_ttc.min()) if not center_ttc.empty else ttc_cap
        else:
            center_weighted_min_ttc = ttc_cap
    else:
        max_bbox_h_window = 0.0
        max_bbox_area_window = 0.0
        max_y2_window = 0.0
        center_weighted_min_ttc = ttc_cap

    # ------------------------------------------------------------------
    # D. Vulnerable Road Users (VRU: person, bicycle) Features
    # ------------------------------------------------------------------
    if not window_df.empty and "cls" in window_df.columns:
        vru_df = window_df[window_df["cls"].astype(str).str.lower().isin(["person", "bicycle"])]
    else:
        vru_df = pd.DataFrame()

    if not vru_df.empty:
        vru_present_window = 1
        vru_count_window = int(vru_df["track_id"].nunique()) if "track_id" in vru_df.columns else len(vru_df)
        vru_ttc = _clean_ttc_series(vru_df)
        vru_min_ttc_window = float(vru_ttc.min()) if not vru_ttc.empty else ttc_cap
        vru_y2 = pd.to_numeric(vru_df["y2"], errors="coerce").dropna() if "y2" in vru_df.columns else pd.Series(dtype=float)
        vru_max_y2_window = float(vru_y2.max()) if not vru_y2.empty else 0.0
    else:
        vru_present_window = 0
        vru_count_window = 0
        vru_min_ttc_window = ttc_cap
        vru_max_y2_window = 0.0

    # ------------------------------------------------------------------
    # E. Traffic Density & Motion Features
    # ------------------------------------------------------------------
    if not window_df.empty and "track_id" in window_df.columns:
        unique_tracks_window = int(window_df["track_id"].nunique())
    else:
        unique_tracks_window = 0

    if not window_df.empty and "cls" in window_df.columns and "track_id" in window_df.columns:
        veh_df = window_df[window_df["cls"].astype(str).str.lower().isin(["car", "bus", "truck"])]
        vehicle_count_window = int(veh_df["track_id"].nunique())
    else:
        vehicle_count_window = 0

    # Maximum lateral speed (frame-to-frame horizontal center shift in pixels/frame)
    max_lateral_speed_window = 0.0
    if not window_df.empty and "track_id" in window_df.columns and "x1" in window_df.columns and "x2" in window_df.columns:
        w_motion_df = window_df.copy()
        w_motion_df["cx"] = (pd.to_numeric(w_motion_df["x1"], errors="coerce") + pd.to_numeric(w_motion_df["x2"], errors="coerce")) / 2.0
        w_motion_df = w_motion_df.dropna(subset=["cx", "frame"])

        max_lat = 0.0
        for _, t_grp in w_motion_df.groupby("track_id"):
            if len(t_grp) >= 2:
                t_grp = t_grp.sort_values(by="frame")
                f_diff = t_grp["frame"].diff()
                cx_diff = t_grp["cx"].diff().abs()
                shift = cx_diff / np.maximum(f_diff, 1)
                ms = shift.dropna().max()
                if pd.notna(ms) and ms > max_lat:
                    max_lat = float(ms)
        max_lateral_speed_window = max_lat

    # ------------------------------------------------------------------
    # Consolidate and clean feature dict
    # ------------------------------------------------------------------
    features: Dict[str, Any] = {
        # TTC features
        "min_ttc_all": min_ttc_all,
        "min_ttc_window": min_ttc_window,
        "p10_ttc_window": p10_ttc_window,
        "mean_ttc_window": mean_ttc_window,
        "ttc_slope_window": ttc_slope_window,
        "frames_ttc_under_1_5": frames_ttc_under_1_5,
        "ratio_ttc_under_1_5": ratio_ttc_under_1_5,
        "frames_ttc_under_2_5": frames_ttc_under_2_5,
        "ratio_ttc_under_2_5": ratio_ttc_under_2_5,
        # Bounding box growth features
        "max_growth_rate_window": max_growth_rate_window,
        "p90_growth_rate_window": p90_growth_rate_window,
        "max_rel_growth_window": max_rel_growth_window,
        "max_height_gain_ratio": max_height_gain_ratio,
        # Proximity & geometry features
        "max_bbox_h_window": max_bbox_h_window,
        "max_bbox_area_window": max_bbox_area_window,
        "max_y2_window": max_y2_window,
        "center_weighted_min_ttc": center_weighted_min_ttc,
        # Vulnerable road users features
        "vru_present_window": vru_present_window,
        "vru_count_window": vru_count_window,
        "vru_min_ttc_window": vru_min_ttc_window,
        "vru_max_y2_window": vru_max_y2_window,
        # Traffic density & motion features
        "unique_tracks_window": unique_tracks_window,
        "vehicle_count_window": vehicle_count_window,
        "max_lateral_speed_window": max_lateral_speed_window,
    }

    # Sanitize any unexpected NaN or Inf
    for k, v in features.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            features[k] = ttc_cap if "ttc" in k else 0.0

    return features


def main() -> None:
    """Execute Phase 3 feature dataset construction pipeline."""
    args = parse_args()
    start_time = time.perf_counter()

    logger.info(LOG_SEPARATOR)
    logger.info("DASHCAM RISK PREDICTION - PHASE 3 FEATURE DATASET CONSTRUCTION")
    logger.info(LOG_SEPARATOR)
    logger.info("Signals directory: %s", args.signals_dir)
    logger.info("Split metadata:    %s", args.split_csv)
    logger.info("Output directory:  %s", args.output_dir)
    logger.info("Cutoff mode:       %s", args.cutoff_mode)
    logger.info("Lookback window:   %.1f seconds", args.lookback_sec)
    logger.info("TTC outlier cap:   %.1f seconds", args.ttc_cap)
    logger.info("Config file:       %s", args.config)

    # 1. Validate directories and input files
    signals_dir = Path(args.signals_dir).resolve()
    if not signals_dir.is_dir():
        logger.error("Signals directory does not exist: %s", signals_dir)
        sys.exit(1)

    split_csv_path = Path(args.split_csv).resolve()
    if not split_csv_path.is_file():
        logger.error("Split CSV file does not exist: %s", split_csv_path)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    phase3_dir = output_dir / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load split metadata
    split_df = pd.read_csv(split_csv_path, dtype={"id": str})
    if "id" not in split_df.columns or "target" not in split_df.columns:
        logger.error("Split CSV must contain 'id' and 'target' columns: %s", split_csv_path)
        sys.exit(1)

    logger.info("Loaded %d video records from %s", len(split_df), split_csv_path.name)

    # 3. Process video signals and construct features
    dataset_rows: List[Dict[str, Any]] = []
    missing_count = 0
    processed_count = 0

    for idx, row in split_df.iterrows():
        video_id = format_video_id(row["id"])
        target = int(row["target"])

        # Determine cutoff time based on cutoff_mode
        cutoff_time: Optional[float] = None
        if target == 1:
            if args.cutoff_mode == "event":
                raw_time = row.get("time_of_event")
                cutoff_time = float(raw_time) if pd.notna(raw_time) else None
            elif args.cutoff_mode == "alert":
                # Fallback to time_of_event if time_of_alert is missing/NaN
                raw_alert = row.get("time_of_alert")
                raw_event = row.get("time_of_event")
                if pd.notna(raw_alert):
                    cutoff_time = float(raw_alert)
                elif pd.notna(raw_event):
                    cutoff_time = float(raw_event)
                else:
                    cutoff_time = None

        # Locate signals CSV
        signals_file = signals_dir / f"{video_id}_signals.csv"
        if not signals_file.is_file():
            # Check alternative subdirectories or unpadded filenames
            alt_file = signals_dir / f"{int(video_id)}_signals.csv" if video_id.isdigit() else None
            if alt_file and alt_file.is_file():
                signals_file = alt_file
            else:
                logger.warning("[%d/%d] Signals file not found for video %s: %s (skipping)", idx + 1, len(split_df), video_id, signals_file.name)
                missing_count += 1
                continue

        try:
            sig_df = pd.read_csv(signals_file)
            features = extract_features_for_video(
                signals_df=sig_df,
                target=target,
                cutoff_time=cutoff_time,
                lookback_sec=args.lookback_sec,
                ttc_cap=args.ttc_cap,
            )

            # Consolidate row with video metadata first
            record = {
                "video_id": video_id,
                "target": target,
                **features,
            }
            dataset_rows.append(record)
            processed_count += 1

        except Exception as exc:
            logger.error("[%d/%d] Failed processing video %s: %s", idx + 1, len(split_df), video_id, exc, exc_info=True)
            missing_count += 1
            continue

        if (processed_count % 100 == 0) or (idx + 1 == len(split_df)):
            logger.info("Progress: %d / %d videos processed...", processed_count, len(split_df))

    if not dataset_rows:
        logger.error("No features were extracted. Dataset is empty.")
        sys.exit(1)

    # 4. Construct final DataFrame and format outputs
    final_df = pd.DataFrame(dataset_rows)

    # Verify zero NaN or Inf values across final dataset
    null_counts = final_df.isna().sum().sum()
    if null_counts > 0:
        logger.warning("Found %d unexpected NaN values in dataset. Imputing defaults...", null_counts)
        for col in final_df.columns:
            if col not in ("video_id", "target"):
                default_val = args.ttc_cap if "ttc" in col else 0.0
                final_df[col] = final_df[col].fillna(default_val)

    split_stem = split_csv_path.stem
    output_filename = f"{split_stem}_features_{args.cutoff_mode}.csv"
    output_filepath = phase3_dir / output_filename
    final_df.to_csv(output_filepath, index=False)

    total_time = time.perf_counter() - start_time

    # 5. Sanity Check Summary Logging
    feature_cols = [c for c in final_df.columns if c not in ("video_id", "target")]

    logger.info(LOG_SEPARATOR)
    logger.info("PHASE 3 FEATURE DATASET CONSTRUCTION COMPLETE - SUMMARY")
    logger.info(LOG_SEPARATOR)
    logger.info("Total videos in split:       %d", len(split_df))
    logger.info("Videos successfully loaded:  %d", len(final_df))
    logger.info("Videos skipped (missing):    %d", missing_count)
    logger.info("Total feature columns:       %d", len(feature_cols))
    logger.info("Total elapsed time:          %.2fs", total_time)
    logger.info("Class distribution:")
    for target_val, count in final_df["target"].value_counts().sort_index().items():
        logger.info("  - Target %d: %d (%.1f%%)", target_val, count, count / len(final_df) * 100.0)

    # Compute correlations with target for sanity validation
    if len(final_df["target"].unique()) > 1:
        logger.info(LOG_SUBSEPARATOR)
        logger.info("TOP FEATURE CORRELATIONS WITH TARGET (Sanity Check):")
        corrs: Dict[str, float] = {}
        for col in feature_cols:
            if final_df[col].std() > 1e-8:
                r_val = final_df[col].corr(final_df["target"])
                if pd.notna(r_val):
                    corrs[col] = float(r_val)

        sorted_corrs = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 5 Positive Correlations (higher = more danger):")
        for name, val in sorted_corrs[:5]:
            logger.info("  + %-28s : %+.4f", name, val)

        logger.info("Top 5 Negative Correlations (lower TTC / slope = more danger):")
        for name, val in sorted_corrs[-5:][::-1]:
            logger.info("  - %-28s : %+.4f", name, val)

    logger.info(LOG_SUBSEPARATOR)
    logger.info("Saved dataset to: %s", output_filepath)
    logger.info(LOG_SEPARATOR)


if __name__ == "__main__":
    main()
