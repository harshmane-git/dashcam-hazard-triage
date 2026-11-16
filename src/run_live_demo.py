"""End-to-End Single-Video Live Inference & Overlay Demo.

Runs the complete 5-phase Dashcam Risk Prediction pipeline on a single video:
  1. Stages the input video and looks up split metadata (heldout/dev) for
     anti-leakage truncation (falling back to full-clip inference for unseen external videos).
  2. Phase 1: YOLO26-N object detection + ByteTrack multi-object tracking.
  3. Phase 2: 5-frame rolling smoothing + monocular pinhole TTC extraction.
  4. Phase 3: Pre-cutoff 24-feature kinematic extraction.
  5. Phase 4 & 5: Frozen Random Forest inference, 3-tier risk triage, CSV export,
     and dynamic HUD overlay video rendering.

Usage:
    python -u src/run_live_demo.py --video data/nexar-collision-prediction/train/00109.mp4
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# Allow importing helper functions directly from src/postprocess_triage.py
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocess_triage import classify_risk_tier, render_single_overlay_video  # noqa: E402

LOG_SEPARATOR = "=" * 64
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end 5-phase collision prediction & overlay rendering on one video."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input .mp4 dashcam video (e.g., data/nexar-collision-prediction/train/00109.mp4).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_live/out",
        help="Output directory for phase CSVs, prediction CSV, and overlay video (default: demo_live/out).",
    )
    parser.add_argument(
        "--cutoff_mode",
        type=str,
        choices=["alert", "event"],
        default="alert",
        help="Temporal truncation mode for labeled clips: 'alert' or 'event' (default: alert).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/nexar.yaml",
        help="Path to YAML pipeline configuration (default: config/nexar.yaml).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to trained .joblib model (defaults to outputs_nexar/models/dev_features_<mode>_random_forest.joblib).",
    )
    return parser.parse_args()


def normalize_id_str(raw_stem: str) -> str:
    """Format numeric stems as 5-digit zero-padded IDs while preserving custom filenames."""
    try:
        return f"{int(float(raw_stem)):05d}"
    except ValueError:
        return raw_stem.strip()


def stage_split_metadata(video_stem: str, split_out_path: Path) -> Optional[pd.Series]:
    """Look up video metadata in heldout.csv / dev.csv / train.csv or create a fallback row."""
    candidate_splits = [
        PROJECT_ROOT / "data/nexar-collision-prediction/splits/heldout.csv",
        PROJECT_ROOT / "data/nexar-collision-prediction/splits/dev.csv",
        PROJECT_ROOT / "data/nexar-collision-prediction/train.csv",
    ]

    numeric_id: Optional[int] = None
    try:
        numeric_id = int(float(video_stem))
    except ValueError:
        numeric_id = None

    for csv_path in candidate_splits:
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "id" not in df.columns:
            continue

        if numeric_id is not None:
            matched = df[pd.to_numeric(df["id"], errors="coerce") == numeric_id]
        else:
            matched = df[df["id"].astype(str).str.strip() == video_stem]

        if not matched.empty:
            row_df = matched.iloc[[0]].copy()
            row_df.to_csv(split_out_path, index=False)
            logger.info("Matched metadata for video '%s' from %s", video_stem, csv_path.name)
            return row_df.iloc[0]

    # Fallback for external/unlabeled dashcam videos not in Nexar splits
    logger.info(
        "Video '%s' not found in Nexar split CSVs; running full-clip inference (no event truncation).",
        video_stem,
    )
    fallback_df = pd.DataFrame(
        [{"id": video_stem, "target": 0, "time_of_event": None, "time_of_alert": None}]
    )
    fallback_df.to_csv(split_out_path, index=False)
    return None


def run_step(cmd: list[str], step_name: str) -> None:
    """Execute a pipeline phase subprocess and halt immediately on error."""
    logger.info(LOG_SEPARATOR)
    logger.info("RUNNING %s", step_name)
    logger.info(LOG_SEPARATOR)
    subprocess.run(cmd, check=True)


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        logger.error("Input video not found: %s", video_path)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    raw_stage_dir = output_dir.parent / "raw"
    raw_stage_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = normalize_id_str(video_path.stem)
    staged_video_path = raw_stage_dir / f"{video_id}{video_path.suffix.lower()}"

    # Keep demo_live/raw isolated to ONLY the requested video
    for existing in raw_stage_dir.glob("*"):
        if existing.resolve() != staged_video_path.resolve() and existing.is_file():
            existing.unlink()

    if video_path.resolve() != staged_video_path.resolve():
        shutil.copy2(video_path, staged_video_path)

    # Clear stale phase1/phase2 files for this video ID so fresh inference always runs
    for sub, suffix in [("phase1", "_tracking.csv"), ("phase2", "_signals.csv")]:
        stale = output_dir / sub / f"{video_id}{suffix}"
        if stale.is_file():
            stale.unlink()

    split_csv_path = output_dir / f"split_{video_id}.csv"
    meta_row = stage_split_metadata(video_id, split_csv_path)

    # Phase 1: YOLO26-N + ByteTrack
    run_step(
        [
            sys.executable,
            "-u",
            str(SRC_DIR / "nexar_batch.py"),
            "--input_dir",
            str(raw_stage_dir),
            "--output_dir",
            str(output_dir),
            "--config",
            args.config,
        ],
        "PHASE 1: YOLO26-N DETECTION & BYTETRACK TRACKING",
    )

    # Phase 2: Kinematic & Pinhole TTC Signal Extraction
    run_step(
        [
            sys.executable,
            "-u",
            str(SRC_DIR / "signals.py"),
            "--input_dir",
            str(output_dir / "phase1"),
            "--output_dir",
            str(output_dir),
            "--config",
            args.config,
        ],
        "PHASE 2: KINEMATIC & MONOCULAR TTC SIGNAL EXTRACTION",
    )

    # Phase 3: Anti-Leakage Truncation & 24-Feature Engineering
    run_step(
        [
            sys.executable,
            "-u",
            str(SRC_DIR / "build_features.py"),
            "--signals_dir",
            str(output_dir / "phase2"),
            "--split_csv",
            str(split_csv_path),
            "--output_dir",
            str(output_dir),
            "--cutoff_mode",
            args.cutoff_mode,
            "--config",
            args.config,
        ],
        f"PHASE 3: 24-FEATURE EXTRACTION ({args.cutoff_mode.upper()} CUTOFF)",
    )

    # Phase 4 & 5: Frozen Random Forest Inference & Dynamic HUD Overlay
    logger.info(LOG_SEPARATOR)
    logger.info("PHASE 4 & 5: ML RISK INFERENCE & HUD OVERLAY RENDERING")
    logger.info(LOG_SEPARATOR)

    model_path = (
        Path(args.model_path).resolve()
        if args.model_path
        else (PROJECT_ROOT / f"outputs_nexar/models/dev_features_{args.cutoff_mode}_random_forest.joblib").resolve()
    )
    if not model_path.is_file():
        logger.error("Trained model artifact not found: %s", model_path)
        sys.exit(1)

    features_csv = output_dir / "phase3" / f"split_{video_id}_features_{args.cutoff_mode}.csv"
    if not features_csv.is_file():
        logger.error("Phase 3 features CSV was not generated: %s", features_csv)
        sys.exit(1)

    bundle = joblib.load(model_path)
    feats_df = pd.read_csv(features_csv)
    feature_names = bundle["feature_names"]
    opt_threshold = float(bundle["optimal_threshold"])

    X = feats_df[feature_names]
    rf_prob = float(bundle["model"].predict_proba(X)[0, 1])
    cw_ttc = float(feats_df["center_weighted_min_ttc"].iloc[0])
    min_ttc_win = float(feats_df["min_ttc_window"].iloc[0])
    risk_tier = classify_risk_tier(rf_prob, cw_ttc, opt_threshold)

    # Save consolidated prediction & triage CSV in demo_live/out/
    pred_summary = feats_df.copy()
    pred_summary.insert(1, "cutoff_mode", args.cutoff_mode)
    pred_summary.insert(2, "random_forest_prob", round(rf_prob, 4))
    pred_summary.insert(3, "optimal_threshold", round(opt_threshold, 4))
    pred_summary.insert(4, "risk_tier", risk_tier)
    if meta_row is not None:
        pred_summary.insert(5, "time_of_alert", meta_row.get("time_of_alert"))
        pred_summary.insert(6, "time_of_event", meta_row.get("time_of_event"))

    pred_csv_path = output_dir / f"demo_{video_id}_prediction.csv"
    pred_summary.to_csv(pred_csv_path, index=False)

    signals_csv = output_dir / "phase2" / f"{video_id}_signals.csv"
    overlay_mp4_path = output_dir / f"demo_{video_id}_overlay.mp4"

    rendered = render_single_overlay_video(
        video_path=staged_video_path,
        signals_path=signals_csv,
        output_path=overlay_mp4_path,
        video_id=video_id,
        rf_prob=rf_prob,
        opt_threshold=opt_threshold,
        risk_tier=risk_tier,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(LOG_SEPARATOR)
    logger.info("LIVE DEMO COMPLETE (%.2f seconds)", elapsed)
    logger.info("  Video ID            : %s", video_id)
    if meta_row is not None:
        logger.info(
            "  Ground Truth        : target=%s | alert=%ss | event=%ss",
            meta_row.get("target"),
            meta_row.get("time_of_alert"),
            meta_row.get("time_of_event"),
        )
    logger.info("  RF Collision Prob   : %.1f%% (threshold: %.1f%%)", rf_prob * 100, opt_threshold * 100)
    logger.info("  Ego-Lane Min TTC    : %.2fs (overall window min: %.2fs)", cw_ttc, min_ttc_win)
    logger.info("  Final Triage Tier   : %s", risk_tier.upper())
    logger.info("  Tracking CSV        : %s", output_dir / "phase1" / f"{video_id}_tracking.csv")
    logger.info("  Signals CSV         : %s", signals_csv)
    logger.info("  Features CSV        : %s", features_csv)
    logger.info("  Prediction CSV      : %s", pred_csv_path)
    if rendered:
        logger.info("  Annotated Video     : %s", overlay_mp4_path)
    logger.info(LOG_SEPARATOR)


if __name__ == "__main__":
    main()