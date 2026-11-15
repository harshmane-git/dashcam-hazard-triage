"""Phase 5: Post-Processing Triage, Evaluation Plots, and Demo Overlay Videos.

Executes post-processing for Dashcam Hazard Detection and Collision Prediction:
1. Task A - Summary Logging: Displays evaluation comparison metrics for event and alert cutoffs.
2. Task B - Risk Triage Classification: Merges model predictions with Phase 3 collision features
   and categorizes clips into Low, Medium, or High Risk tiers.
3. Task C - Publication-Quality Plots: Generates 200 DPI evaluation figures (ROC curves,
   feature importances, confusion matrix, risk tier distributions).
4. Task D - Demo Overlay Videos: Annotates High-Risk collision clips with bounding boxes,
   TTC alerts, and a real-time HUD dashboard.

Usage:
    python src/postprocess_triage.py \
        --output_dir outputs_nexar \
        --heldout_video_dir data/nexar_heldout_750 \
        --num_demos 3
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure headless matplotlib environment before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Module-level static separators resistant to mangling
LOG_SEPARATOR = "============================================================"
LOG_SUBSEPARATOR = "------------------------------------------------------------"

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("postprocess_triage")


def format_video_id(raw_id: Any) -> str:
    """Safely format video id to 5-digit zero-padded string."""
    try:
        return f"{int(float(raw_id)):05d}"
    except (ValueError, TypeError):
        return str(raw_id).strip().zfill(5)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Post-Processing Triage, Evaluation Plots, and Demo Overlay Videos."
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        dest="output_dir",
        default="outputs_nexar",
        help="Base output directory (default: outputs_nexar).",
    )
    parser.add_argument(
        "--heldout_video_dir",
        "--heldout-video-dir",
        type=str,
        dest="heldout_video_dir",
        default="data/nexar_heldout_750",
        help="Directory of heldout .mp4 videos (default: data/nexar_heldout_750).",
    )
    parser.add_argument(
        "--num_demos",
        "--num-demos",
        type=int,
        dest="num_demos",
        default=3,
        help="Number of High-Risk heldout demo overlay videos to render (default: 3).",
    )
    return parser.parse_args()


# ==============================================================================
# Task A: Summary Logging
# ==============================================================================
def task_a_summary_logging(eval_dir: Path) -> None:
    """Load and log comparison metrics tables for event and alert cutoff modes."""
    logger.info(LOG_SEPARATOR)
    logger.info("TASK A: SUMMARY EVALUATION METRICS LOGGING")
    logger.info(LOG_SEPARATOR)

    for mode in ["event", "alert"]:
        metrics_file = eval_dir / f"dev_features_{mode}_metrics.csv"
        if metrics_file.is_file():
            try:
                df = pd.read_csv(metrics_file)
                logger.info("Evaluation Metrics Summary Table [%s Mode]:", mode.upper())
                summary_cols = [
                    c for c in [
                        "model", "split", "roc_auc", "pr_auc", "optimal_threshold",
                        "f1", "precision", "recall", "tp", "fp", "tn", "fn"
                    ] if c in df.columns
                ]
                logger.info("\n" + df[summary_cols].to_string(index=False))
            except Exception as exc:
                logger.warning("Failed to read metrics file %s: %s", metrics_file, exc)
        else:
            logger.info("Metrics table not found for mode '%s': %s", mode, metrics_file)


# ==============================================================================
# Task B: Low / Medium / High Risk Triage Classification
# ==============================================================================
def classify_risk_tier(rf_prob: float, cw_ttc: float, threshold: float) -> str:
    """Classify a video into Low, Medium, or High Risk tiers.

    Rules:
    - High Risk: random_forest_prob >= 0.65 OR center_weighted_min_ttc < 1.5
    - Low Risk: random_forest_prob < optimal_threshold
    - Medium Risk: All remaining videos (optimal_threshold <= random_forest_prob < 0.65
                   and center_weighted_min_ttc >= 1.5)
    """
    if rf_prob >= 0.65 or cw_ttc < 1.5:
        return "High Risk"
    elif rf_prob < threshold:
        return "Low Risk"
    else:
        return "Medium Risk"


def task_b_risk_triage(output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Execute risk triage classification for both event and alert modes."""
    logger.info(LOG_SEPARATOR)
    logger.info("TASK B: LOW / MEDIUM / HIGH RISK TRIAGE CLASSIFICATION")
    logger.info(LOG_SEPARATOR)

    eval_dir = output_dir / "evaluation"
    phase3_dir = output_dir / "phase3"
    models_dir = output_dir / "models"
    eval_dir.mkdir(parents=True, exist_ok=True)

    triage_results: Dict[str, pd.DataFrame] = {}

    for mode in ["event", "alert"]:
        preds_file = eval_dir / f"dev_features_{mode}_predictions.csv"
        if not preds_file.is_file():
            logger.warning("Predictions file not found: %s. Skipping triage for '%s' mode.", preds_file, mode)
            continue

        logger.info(LOG_SUBSEPARATOR)
        logger.info("Processing Risk Triage for Mode: %s", mode.upper())

        preds_df = pd.read_csv(preds_file, dtype={"video_id": str})
        preds_df["video_id"] = preds_df["video_id"].apply(format_video_id)

        # 1. Load and combine dev and heldout Phase 3 features
        dev_feat_file = phase3_dir / f"dev_features_{mode}.csv"
        heldout_feat_file = phase3_dir / f"heldout_features_{mode}.csv"

        feat_dfs: List[pd.DataFrame] = []
        for feat_path in [dev_feat_file, heldout_feat_file]:
            if feat_path.is_file():
                try:
                    f_df = pd.read_csv(feat_path, dtype={"video_id": str})
                    f_df["video_id"] = f_df["video_id"].apply(format_video_id)
                    feat_dfs.append(f_df)
                except Exception as exc:
                    logger.warning("Could not read feature file %s: %s", feat_path, exc)

        if feat_dfs:
            combined_feat = pd.concat(feat_dfs, ignore_index=True).drop_duplicates(subset=["video_id"])
            cols_to_bring = [
                c for c in ["min_ttc_window", "center_weighted_min_ttc", "max_bbox_h_window"]
                if c in combined_feat.columns
            ]
            merged_df = pd.merge(preds_df, combined_feat[["video_id"] + cols_to_bring], on="video_id", how="left")
        else:
            logger.warning("No Phase 3 feature CSVs found for mode '%s'. Using fallback values.", mode)
            merged_df = preds_df.copy()

        # Fill default values for missing feature columns
        if "center_weighted_min_ttc" not in merged_df.columns:
            merged_df["center_weighted_min_ttc"] = 15.0
        else:
            merged_df["center_weighted_min_ttc"] = pd.to_numeric(
                merged_df["center_weighted_min_ttc"], errors="coerce"
            ).fillna(15.0)

        if "min_ttc_window" not in merged_df.columns:
            merged_df["min_ttc_window"] = 15.0
        else:
            merged_df["min_ttc_window"] = pd.to_numeric(
                merged_df["min_ttc_window"], errors="coerce"
            ).fillna(15.0)

        if "max_bbox_h_window" not in merged_df.columns:
            merged_df["max_bbox_h_window"] = 0.0
        else:
            merged_df["max_bbox_h_window"] = pd.to_numeric(
                merged_df["max_bbox_h_window"], errors="coerce"
            ).fillna(0.0)

        if "random_forest_prob" not in merged_df.columns:
            logger.warning("random_forest_prob not found in %s; defaulting to 0.0", preds_file)
            merged_df["random_forest_prob"] = 0.0
        else:
            merged_df["random_forest_prob"] = pd.to_numeric(
                merged_df["random_forest_prob"], errors="coerce"
            ).fillna(0.0)

        # 2. Load optimal threshold from saved Random Forest model
        opt_threshold = 0.5
        rf_model_path = models_dir / f"dev_features_{mode}_random_forest.joblib"
        if rf_model_path.is_file():
            try:
                rf_payload = joblib.load(rf_model_path)
                opt_threshold = float(rf_payload.get("optimal_threshold", 0.5))
                logger.info("Loaded optimal threshold from %s: %.4f", rf_model_path.name, opt_threshold)
            except Exception as exc:
                logger.warning("Failed to load threshold from %s (%s). Using fallback 0.5.", rf_model_path, exc)
        else:
            # Fallback to metrics file if present
            metrics_path = eval_dir / f"dev_features_{mode}_metrics.csv"
            if metrics_path.is_file():
                try:
                    m_df = pd.read_csv(metrics_path)
                    rf_row = m_df[(m_df["model"] == "random_forest") & (m_df["split"] == "CV_OOF")]
                    if not rf_row.empty and "optimal_threshold" in rf_row.columns:
                        opt_threshold = float(rf_row.iloc[0]["optimal_threshold"])
                        logger.info("Loaded optimal threshold from %s: %.4f", metrics_path.name, opt_threshold)
                except Exception as exc:
                    logger.warning("Failed reading threshold from %s: %s", metrics_path, exc)

        # 3. Apply Triage Rules
        merged_df["risk_tier"] = [
            classify_risk_tier(
                rf_prob=float(row["random_forest_prob"]),
                cw_ttc=float(row["center_weighted_min_ttc"]),
                threshold=opt_threshold,
            )
            for _, row in merged_df.iterrows()
        ]

        # 4. Save triaged table
        triage_path = eval_dir / f"triage_{mode}.csv"
        merged_df.to_csv(triage_path, index=False)
        logger.info("Saved triaged dataset (%d records) to: %s", len(merged_df), triage_path)

        # 5. Print Crosstab Summary
        if "split" in merged_df.columns:
            ct_split = pd.crosstab(merged_df["split"], merged_df["risk_tier"], margins=True)
            logger.info("Split vs. Risk Tier Distribution [%s]:\n%s", mode.upper(), ct_split.to_string())

        if "target" in merged_df.columns:
            ct_target = pd.crosstab(merged_df["target"], merged_df["risk_tier"], margins=True)
            logger.info("Ground Truth Target vs. Risk Tier [%s]:\n%s", mode.upper(), ct_target.to_string())

        triage_results[mode] = merged_df

    return triage_results


# ==============================================================================
# Task C: Publication-Quality Plots (200 DPI)
# ==============================================================================
def task_c_publication_plots(output_dir: Path, triage_dict: Dict[str, pd.DataFrame]) -> None:
    """Generate and save publication-quality evaluation figures at 200 DPI."""
    logger.info(LOG_SEPARATOR)
    logger.info("TASK C: GENERATING PUBLICATION-QUALITY EVALUATION PLOTS")
    logger.info(LOG_SEPARATOR)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "evaluation"
    models_dir = output_dir / "models"

    # Get event mode triage or predictions dataframe
    event_df = triage_dict.get("event")
    if event_df is None:
        preds_file = eval_dir / "dev_features_event_predictions.csv"
        if preds_file.is_file():
            event_df = pd.read_csv(preds_file, dtype={"video_id": str})
            event_df["video_id"] = event_df["video_id"].apply(format_video_id)

    # --------------------------------------------------------------------------
    # Plot 1: Heldout ROC Curves
    # --------------------------------------------------------------------------
    if event_df is not None and "target" in event_df.columns:
        roc_df = event_df[event_df["split"] == "heldout"] if "split" in event_df.columns else event_df
        split_title_suffix = "Heldout Split"

        # Fallback to dev_oof if heldout records are absent or single-class
        if len(roc_df) == 0 or len(roc_df["target"].unique()) < 2:
            roc_df = event_df[event_df["split"] == "dev_oof"] if "split" in event_df.columns else event_df
            split_title_suffix = "Dev (Out-of-Fold) Split"

        if len(roc_df) > 0 and len(roc_df["target"].unique()) >= 2:
            y_true = roc_df["target"].astype(int).values
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

            models_meta = [
                ("random_forest", "Random Forest", "#1f77b4", "-", 2.2),
                ("hist_gradient_boosting", "HistGradientBoosting", "#d62728", "-", 2.0),
                ("logistic_regression", "Logistic Regression", "#2ca02c", "--", 1.8),
                ("heuristic_baseline", "Heuristic Baseline", "#ff7f0e", ":", 1.8),
            ]

            plotted_any = False
            for model_key, display_name, color, style, lw in models_meta:
                prob_col = f"{model_key}_prob"
                if prob_col in roc_df.columns:
                    y_prob = roc_df[prob_col].astype(float).values
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auc_score = roc_auc_score(y_true, y_prob)
                    ax.plot(
                        fpr, tpr,
                        label=f"{display_name} (AUC = {auc_score:.3f})",
                        color=color,
                        linestyle=style,
                        linewidth=lw,
                    )
                    plotted_any = True

            if plotted_any:
                ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.2, alpha=0.75, label="Chance (AUC = 0.500)")
                ax.set_xlim([-0.02, 1.02])
                ax.set_ylim([-0.02, 1.02])
                ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
                ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
                ax.set_title(f"ROC Curves Comparison ({split_title_suffix}, Event Cutoff)", fontsize=13, fontweight="bold", pad=12)
                ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.92, edgecolor="#cccccc")
                ax.grid(True, linestyle="--", alpha=0.35)
                fig.tight_layout()

                roc_plot_path = plots_dir / "heldout_roc_curves.png"
                fig.savefig(roc_plot_path, dpi=200, bbox_inches="tight")
                logger.info("Saved ROC plot: %s", roc_plot_path)
            plt.close(fig)
        else:
            logger.warning("Insufficient target class diversity to generate ROC curves.")
    else:
        logger.warning("Event predictions unavailable; skipping heldout_roc_curves.png.")

    # --------------------------------------------------------------------------
    # Plot 2: Top 10 Random Forest Gini Feature Importances
    # --------------------------------------------------------------------------
    rf_joblib_file = models_dir / "dev_features_event_random_forest.joblib"
    if not rf_joblib_file.is_file():
        rf_joblib_file = models_dir / "dev_features_alert_random_forest.joblib"

    if rf_joblib_file.is_file():
        try:
            payload = joblib.load(rf_joblib_file)
            rf_model = payload.get("model")
            feat_names = payload.get("feature_names", [])

            if hasattr(rf_model, "feature_importances_"):
                importances = rf_model.feature_importances_
                if len(feat_names) != len(importances):
                    feat_names = [f"feat_{i}" for i in range(len(importances))]

                ranked_idx = np.argsort(importances)[::-1][:10]
                top_names = [feat_names[i] for i in ranked_idx]
                top_vals = [importances[i] for i in ranked_idx]

                fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
                y_pos = np.arange(len(top_names))
                ax.barh(y_pos, top_vals, color="#2b5c8f", edgecolor="#1a3654", height=0.65)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_names, fontsize=10)
                ax.invert_yaxis()  # Top feature on top

                max_val = max(top_vals) if top_vals else 1.0
                ax.set_xlim(0, max_val * 1.18)
                for i, v in enumerate(top_vals):
                    ax.text(v + (max_val * 0.015), i, f"{v:.4f}", va="center", ha="left", fontsize=9.5, fontweight="semibold", color="#222222")

                ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="bold")
                ax.set_title("Top 10 Random Forest Feature Importances (Event Cutoff)", fontsize=13, fontweight="bold", pad=12)
                ax.grid(axis="x", linestyle="--", alpha=0.35)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()

                fi_plot_path = plots_dir / "top10_feature_importance.png"
                fig.savefig(fi_plot_path, dpi=200, bbox_inches="tight")
                logger.info("Saved feature importance plot: %s", fi_plot_path)
                plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to generate top10_feature_importance.png: %s", exc)
    else:
        logger.warning("Random Forest model artifact not found; skipping top10_feature_importance.png.")

    # --------------------------------------------------------------------------
    # Plot 3: Heldout Confusion Matrix
    # --------------------------------------------------------------------------
    if event_df is not None and "target" in event_df.columns and "random_forest_pred" in event_df.columns:
        cm_df = event_df[event_df["split"] == "heldout"] if "split" in event_df.columns else event_df
        cm_title_suffix = "Heldout Split"

        if len(cm_df) == 0:
            cm_df = event_df[event_df["split"] == "dev_oof"] if "split" in event_df.columns else event_df
            cm_title_suffix = "Dev (Out-of-Fold) Split"

        if len(cm_df) > 0:
            y_true_cm = cm_df["target"].astype(int).values
            y_pred_cm = cm_df["random_forest_pred"].astype(int).values
            cm = confusion_matrix(y_true_cm, y_pred_cm, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total_samples = len(y_true_cm)

            fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
            cax = ax.matshow(cm, cmap="Blues", alpha=0.88)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

            cell_labels = [
                [f"TN = {tn}\n({tn/total_samples:.1%})", f"FP = {fp}\n({fp/total_samples:.1%})"],
                [f"FN = {fn}\n({fn/total_samples:.1%})", f"TP = {tp}\n({tp/total_samples:.1%})"],
            ]
            for i in range(2):
                for j in range(2):
                    v = cm[i, j]
                    txt_color = "white" if v > (cm.max() / 2) else "black"
                    ax.text(j, i, cell_labels[i][j], ha="center", va="center", color=txt_color, fontsize=12, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Predicted Negative (0)", "Predicted Positive (1)"], fontsize=10, fontweight="bold")
            ax.xaxis.set_ticks_position("bottom")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Actual Negative (0)", "Actual Positive (1)"], fontsize=10, fontweight="bold")
            ax.set_title(f"Random Forest Confusion Matrix ({cm_title_suffix})", fontsize=13, fontweight="bold", pad=12)

            acc = (tp + tn) / max(total_samples, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
            ax.set_xlabel(f"Accuracy: {acc:.1%}  |  Precision: {prec:.1%}  |  Recall: {rec:.1%}  |  F1: {f1:.3f}", fontsize=10, style="italic", labelpad=8)
            fig.tight_layout()

            cm_plot_path = plots_dir / "heldout_confusion_matrix.png"
            fig.savefig(cm_plot_path, dpi=200, bbox_inches="tight")
            logger.info("Saved confusion matrix plot: %s", cm_plot_path)
            plt.close(fig)
    else:
        logger.warning("Event predictions missing required columns; skipping heldout_confusion_matrix.png.")

    # --------------------------------------------------------------------------
    # Plot 4: Risk Tier Distribution by Ground-Truth Target
    # --------------------------------------------------------------------------
    triage_event_file = eval_dir / "triage_event.csv"
    if triage_event_file.is_file():
        try:
            triage_df = pd.read_csv(triage_event_file, dtype={"video_id": str})
            triage_df["video_id"] = triage_df["video_id"].apply(format_video_id)
            if "target" in triage_df.columns and "risk_tier" in triage_df.columns:
                ct = pd.crosstab(triage_df["target"], triage_df["risk_tier"])
                for col in ["Low Risk", "Medium Risk", "High Risk"]:
                    if col not in ct.columns:
                        ct[col] = 0
                for idx in [0, 1]:
                    if idx not in ct.index:
                        ct.loc[idx] = 0
                ct = ct.sort_index()

                fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
                targets = [0, 1]
                x_labels = ["Negative / Safe (Target = 0)", "Collision / Near-Miss (Target = 1)"]
                x_pos = np.arange(len(targets))
                bar_w = 0.52

                low_vals = np.array([ct.loc[t, "Low Risk"] for t in targets])
                med_vals = np.array([ct.loc[t, "Medium Risk"] for t in targets])
                high_vals = np.array([ct.loc[t, "High Risk"] for t in targets])
                totals = low_vals + med_vals + high_vals

                p1 = ax.bar(x_pos, low_vals, bar_w, label="Low Risk", color="#2ecc71", edgecolor="#27ae60")
                p2 = ax.bar(x_pos, med_vals, bar_w, bottom=low_vals, label="Medium Risk", color="#f39c12", edgecolor="#d68910")
                p3 = ax.bar(x_pos, high_vals, bar_w, bottom=low_vals + med_vals, label="High Risk", color="#e74c3c", edgecolor="#c0392b")

                for i in range(len(targets)):
                    t_tot = max(totals[i], 1)
                    if low_vals[i] > 0:
                        y_pos_low = low_vals[i] / 2
                        ax.text(x_pos[i], y_pos_low, f"{low_vals[i]}\n({low_vals[i]/t_tot:.1%})", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                    if med_vals[i] > 0:
                        y_pos_med = low_vals[i] + med_vals[i] / 2
                        ax.text(x_pos[i], y_pos_med, f"{med_vals[i]}\n({med_vals[i]/t_tot:.1%})", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                    if high_vals[i] > 0:
                        y_pos_high = low_vals[i] + med_vals[i] + high_vals[i] / 2
                        ax.text(x_pos[i], y_pos_high, f"{high_vals[i]}\n({high_vals[i]/t_tot:.1%})", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
                ax.set_ylabel("Number of Videos", fontsize=11, fontweight="bold")
                ax.set_title("Risk Tier Distribution by Ground Truth Target (Event Cutoff)", fontsize=13, fontweight="bold", pad=12)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, fontsize=10)
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                max_bar = max(totals) if len(totals) > 0 and max(totals) > 0 else 10
                ax.set_ylim(0, max_bar * 1.15)
                fig.tight_layout()

                tier_plot_path = plots_dir / "risk_tier_distribution.png"
                fig.savefig(tier_plot_path, dpi=200, bbox_inches="tight")
                logger.info("Saved risk tier distribution plot: %s", tier_plot_path)
                plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to generate risk_tier_distribution.png: %s", exc)
    else:
        logger.warning("triage_event.csv not found; skipping risk_tier_distribution.png.")


# ==============================================================================
# Task D: Annotated Demo Overlay Videos
# ==============================================================================
def find_video_file(video_id: str, heldout_video_dir: Path, output_dir: Path) -> Optional[Path]:
    """Find the MP4 video path for a given video ID across candidate directories."""
    padded_id = format_video_id(video_id)
    try:
        raw_id = str(int(float(video_id)))
    except (ValueError, TypeError):
        raw_id = video_id

    candidates = [
        heldout_video_dir / f"{padded_id}.mp4",
        heldout_video_dir / f"{raw_id}.mp4",
        heldout_video_dir / f"{padded_id}.MP4",
        heldout_video_dir / f"{raw_id}.MP4",
        output_dir.parent / "data" / "nexar-collision-prediction" / "videos" / f"{padded_id}.mp4",
        output_dir.parent / "data" / "nexar-collision-prediction" / "videos" / f"{raw_id}.mp4",
        output_dir.parent / "data" / "nexar_heldout_750" / f"{padded_id}.mp4",
        output_dir.parent / "data" / "nexar_heldout_750" / f"{raw_id}.mp4",
        output_dir.parent / "data" / "raw" / f"{padded_id}.mp4",
        output_dir.parent / "data" / "raw" / f"{raw_id}.mp4",
    ]
    for p in candidates:
        if p.is_file():
            return p

    # Fallback glob search in heldout directory
    if heldout_video_dir.is_dir():
        for pat in [f"*{padded_id}*.mp4", f"*{raw_id}*.mp4"]:
            matches = list(heldout_video_dir.glob(pat))
            if matches:
                return matches[0]

    return None


def render_single_overlay_video(
    video_path: Path,
    signals_path: Path,
    output_path: Path,
    video_id: str,
    rf_prob: float,
    opt_threshold: float,
    risk_tier: str,
) -> bool:
    """Render a single demo overlay video with bounding boxes, TTC alerts, and HUD banner."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", video_path)
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    try:
        signals_df = pd.read_csv(signals_path)
    except Exception as exc:
        logger.warning("Failed to read signals CSV %s: %s", signals_path, exc)
        cap.release()
        return False

    frame_col = "frame" if "frame" in signals_df.columns else signals_df.columns[0]
    by_frame: Dict[int, pd.DataFrame] = {
        int(f): group for f, group in signals_df.groupby(frame_col)
    }
    min_frame_in_sig = min(by_frame.keys()) if by_frame else 0
    is_one_indexed = (min_frame_in_sig == 1)

    # Initialize VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.warning("Failed to initialize cv2.VideoWriter for %s", output_path)
        cap.release()
        return False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sig_key = (frame_idx + 1) if is_one_indexed else frame_idx
        frame_sig = by_frame.get(sig_key)

        valid_ttcs = []
        if frame_sig is not None and not frame_sig.empty:
            for _, r in frame_sig.iterrows():
                t = float(r.get("ttc", np.nan))
                if not np.isnan(t) and t > 0.0:
                    valid_ttcs.append(t)

        frame_min_ttc = min(valid_ttcs) if valid_ttcs else np.nan

        # Active Risk Status for HUD banner
        if (not np.isnan(frame_min_ttc) and frame_min_ttc < 1.5) or rf_prob >= 0.65:
            active_status = "HIGH RISK"
            status_color = (0, 0, 255)  # Red BGR
        elif (not np.isnan(frame_min_ttc) and frame_min_ttc < 3.0) or rf_prob >= opt_threshold:
            active_status = "MEDIUM RISK"
            status_color = (0, 255, 255)  # Yellow BGR
        else:
            active_status = "LOW RISK"
            status_color = (0, 255, 0)  # Green BGR

        # 1. Draw Bounding Boxes with TTC Color Coding
        if frame_sig is not None and not frame_sig.empty:
            for _, r in frame_sig.iterrows():
                try:
                    x1 = max(0, min(width - 1, int(round(float(r["x1"])))))
                    y1 = max(0, min(height - 1, int(round(float(r["y1"])))))
                    x2 = max(0, min(width - 1, int(round(float(r["x2"])))))
                    y2 = max(0, min(height - 1, int(round(float(r["y2"])))))
                except (ValueError, TypeError):
                    continue

                ttc_val = float(r.get("ttc", np.nan))

                # Color rule: Green (ttc >= 3.0s or NaN), Yellow (1.5s <= ttc < 3.0s), Red (ttc < 1.5s)
                if np.isnan(ttc_val) or ttc_val >= 3.0 or ttc_val <= 0.0:
                    box_color = (0, 255, 0)    # Green BGR
                elif 1.5 <= ttc_val < 3.0:
                    box_color = (0, 255, 255)  # Yellow BGR
                else:
                    box_color = (0, 0, 255)    # Red BGR

                box_thickness = max(2, int(round(width / 640.0)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                # Label text
                try:
                    t_id = int(float(r.get("track_id", 0)))
                except (ValueError, TypeError):
                    t_id = 0
                cls_name = str(r.get("cls", "car"))
                ttc_str = f"{ttc_val:.1f}s" if (not np.isnan(ttc_val) and ttc_val > 0.0) else "N/A"
                label = f"ID:{t_id} {cls_name} | TTC:{ttc_str}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.42, width / 2400.0)
                font_th = max(1, int(round(font_scale * 2.0)))
                (tw, th_box), _ = cv2.getTextSize(label, font, font_scale, font_th)

                tag_y1 = max(0, y1 - th_box - 6)
                tag_y2 = y1
                cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), box_color, -1)
                text_color = (0, 0, 0) if box_color in [(0, 255, 0), (0, 255, 255)] else (255, 255, 255)
                cv2.putText(frame, label, (x1 + 3, tag_y2 - 3), font, font_scale, text_color, font_th, cv2.LINE_AA)

        # 2. Draw Top-Left HUD Banner
        hud_x = 24
        hud_y = 24
        hud_w = max(340, int(round(width * 0.30)))
        hud_h = max(115, int(round(height * 0.16)))

        # Semi-transparent dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Active status accent border
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), status_color, 2)

        # HUD text lines
        ttc_disp = f"{frame_min_ttc:.1f}s" if (not np.isnan(frame_min_ttc) and frame_min_ttc < 99.0) else "None / >15s"
        hud_scale = max(0.46, width / 2300.0)
        hud_th = max(1, int(round(hud_scale * 1.8)))
        line_step = int(hud_h / 3.8)

        # Line 1: Header & Clip ID
        cv2.putText(
            frame, f"DASHCAM RISK AI  [Clip #{video_id}]",
            (hud_x + 12, hud_y + line_step - 4),
            cv2.FONT_HERSHEY_SIMPLEX, hud_scale * 0.88, (210, 210, 210), hud_th, cv2.LINE_AA
        )
        # Line 2: ML Probability
        cv2.putText(
            frame, f"ML Collision Prob : {rf_prob:.1%}",
            (hud_x + 12, hud_y + 2 * line_step - 2),
            cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (255, 255, 255), hud_th, cv2.LINE_AA
        )
        # Line 3: Current Frame Min TTC
        cv2.putText(
            frame, f"Frame Min TTC     : {ttc_disp}",
            (hud_x + 12, hud_y + 3 * line_step),
            cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (255, 255, 255), hud_th, cv2.LINE_AA
        )
        # Line 4: Active Risk Status Badge
        cv2.putText(
            frame, f"ACTIVE STATUS     : {active_status}",
            (hud_x + 12, hud_y + int(3.8 * line_step) - 2),
            cv2.FONT_HERSHEY_SIMPLEX, hud_scale, status_color, hud_th + 1, cv2.LINE_AA
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Rendered overlay demo video: %s (%d frames, %.1f FPS)", output_path, frame_idx, fps)
    return True


def task_d_demo_overlay_videos(
    output_dir: Path,
    heldout_video_dir: Path,
    num_demos: int,
    triage_dict: Dict[str, pd.DataFrame],
) -> None:
    """Render annotated demo overlay videos for High-Risk collision clips."""
    logger.info(LOG_SEPARATOR)
    logger.info("TASK D: ANNOTATED DEMO OVERLAY VIDEOS")
    logger.info(LOG_SEPARATOR)

    if num_demos <= 0:
        logger.info("num_demos = %d; skipping demo overlay rendering.", num_demos)
        return

    demo_dir = output_dir / "demo_videos"
    phase2_dir = output_dir / "phase2"
    models_dir = output_dir / "models"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Use triage_event.csv as primary source
    triage_df = triage_dict.get("event")
    if triage_df is None:
        triage_path = output_dir / "evaluation" / "triage_event.csv"
        if triage_path.is_file():
            triage_df = pd.read_csv(triage_path, dtype={"video_id": str})
            triage_df["video_id"] = triage_df["video_id"].apply(format_video_id)

    if triage_df is None or len(triage_df) == 0:
        logger.warning("No triage data available for demo selection. Skipping Task D.")
        return

    # Load optimal threshold from model artifact
    opt_threshold = 0.5
    rf_joblib = models_dir / "dev_features_event_random_forest.joblib"
    if rf_joblib.is_file():
        try:
            payload = joblib.load(rf_joblib)
            opt_threshold = float(payload.get("optimal_threshold", 0.5))
        except Exception:
            pass

    # Filter for positive (target == 1) videos classified as High Risk
    candidates = triage_df[
        (triage_df["target"] == 1) & (triage_df["risk_tier"] == "High Risk")
    ].copy()

    if candidates.empty:
        logger.warning("No Target=1 High-Risk videos found in triage table. Checking all High-Risk videos.")
        candidates = triage_df[triage_df["risk_tier"] == "High Risk"].copy()

    if candidates.empty:
        logger.warning("No High-Risk videos found at all. Skipping Task D.")
        return

    # Sort by random_forest_prob descending (or center_weighted_min_ttc ascending)
    if "random_forest_prob" in candidates.columns:
        candidates = candidates.sort_values(by="random_forest_prob", ascending=False)
    elif "center_weighted_min_ttc" in candidates.columns:
        candidates = candidates.sort_values(by="center_weighted_min_ttc", ascending=True)

    # Prioritize heldout split if present
    if "split" in candidates.columns:
        heldout_candidates = candidates[candidates["split"] == "heldout"]
        dev_candidates = candidates[candidates["split"] != "heldout"]
        ordered_candidates = pd.concat([heldout_candidates, dev_candidates], ignore_index=True)
    else:
        ordered_candidates = candidates

    rendered_count = 0
    for _, row in ordered_candidates.iterrows():
        if rendered_count >= num_demos:
            break

        vid_id = format_video_id(row["video_id"])
        video_path = find_video_file(vid_id, heldout_video_dir, output_dir)
        signals_path = phase2_dir / f"{vid_id}_signals.csv"
        if not signals_path.is_file():
            try:
                raw_id = str(int(float(vid_id)))
                alt_sig = phase2_dir / f"{raw_id}_signals.csv"
                if alt_sig.is_file():
                    signals_path = alt_sig
            except (ValueError, TypeError):
                pass

        if video_path is None or not video_path.is_file():
            continue
        if not signals_path.is_file():
            continue

        out_demo_path = demo_dir / f"demo_{vid_id}_overlay.mp4"
        rf_prob = float(row.get("random_forest_prob", 0.0))
        risk_tier = str(row.get("risk_tier", "High Risk"))

        logger.info(
            "Rendering High-Risk demo overlay for Video %s (RF Prob: %.1f%%, Source: %s)...",
            vid_id, rf_prob * 100, video_path.name
        )

        success = render_single_overlay_video(
            video_path=video_path,
            signals_path=signals_path,
            output_path=out_demo_path,
            video_id=vid_id,
            rf_prob=rf_prob,
            opt_threshold=opt_threshold,
            risk_tier=risk_tier,
        )

        if success:
            rendered_count += 1

    if rendered_count == 0:
        logger.warning(
            "No matching High-Risk clips with both .mp4 in '%s' and signals in '%s' could be rendered.",
            heldout_video_dir, phase2_dir
        )
    else:
        logger.info("Successfully rendered %d demo overlay video(s) in: %s", rendered_count, demo_dir)


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def main() -> None:
    """Execute complete Phase 5 post-processing pipeline."""
    start_time = time.perf_counter()
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    heldout_video_dir = Path(args.heldout_video_dir).resolve()
    eval_dir = output_dir / "evaluation"

    logger.info(LOG_SEPARATOR)
    logger.info("DASHCAM RISK PREDICTION - PHASE 5 POST-PROCESSING & TRIAGE")
    logger.info(LOG_SEPARATOR)
    logger.info("Output directory:    %s", output_dir)
    logger.info("Heldout video dir:   %s", heldout_video_dir)
    logger.info("Demo videos to make: %d", args.num_demos)

    # Task A: Summary Logging
    task_a_summary_logging(eval_dir)

    # Task B: Low / Medium / High Risk Triage Classification
    triage_dict = task_b_risk_triage(output_dir)

    # Task C: Publication-Quality Plots
    task_c_publication_plots(output_dir, triage_dict)

    # Task D: Annotated Demo Overlay Videos
    task_d_demo_overlay_videos(
        output_dir=output_dir,
        heldout_video_dir=heldout_video_dir,
        num_demos=args.num_demos,
        triage_dict=triage_dict,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(LOG_SEPARATOR)
    logger.info("PHASE 5 COMPLETE - All triage tables, plots, and demo videos generated successfully.")
    logger.info("Total execution time: %.2f seconds", elapsed)
    logger.info(LOG_SEPARATOR)


if __name__ == "__main__":
    main()
