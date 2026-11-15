"""Phase 4: Model Training, Cross-Validation, and Heldout Evaluation.

Trains and evaluates machine learning models on Phase 3 tabular collision-risk features.
Compares a rule-based Heuristic Baseline against Logistic Regression, Random Forest,
and HistGradientBoosting classifiers.

Performs Stratified K-Fold Cross-Validation, optimizes classification decision thresholds
for F1-score on out-of-fold predictions, extracts top feature importances, saves trained
model artifacts with joblib, and conducts unbiased heldout evaluation.

Usage:
    python src/train_models.py \
        --dev_csv outputs_nexar/phase3/dev_features_event.csv \
        --heldout_csv outputs_nexar/phase3/heldout_features_event.csv \
        --output_dir outputs_nexar \
        --cv_folds 5 \
        --seed 42
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Module-level separators resistant to copy-paste wrapping
LOG_SEPARATOR = "============================================================"
LOG_SUBSEPARATOR = "------------------------------------------------------------"

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_models")


class HeuristicBaseline(BaseEstimator, ClassifierMixin):
    """Rule-based risk score baseline derived from TTC features.

    Serves as an empirical benchmark proving data-driven ML improves over
    heuristic prototype rules.
    """

    def __init__(self, ttc_cap: float = 15.0):
        self.ttc_cap = ttc_cap
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "HeuristicBaseline":
        scores = self._compute_raw_score(X)
        self.min_score_ = float(scores.min()) if len(scores) > 0 else 0.0
        self.max_score_ = float(scores.max()) if len(scores) > 0 else 1.0
        return self

    def _compute_raw_score(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            # Fallback if passed raw numpy array
            c_ttc = X[:, 0]
        else:
            if "center_weighted_min_ttc" in X.columns:
                c_ttc = X["center_weighted_min_ttc"].values
            elif "min_ttc_window" in X.columns:
                c_ttc = X["min_ttc_window"].values
            else:
                c_ttc = X.iloc[:, 0].values

        c_ttc = np.clip(np.nan_to_num(c_ttc, nan=self.ttc_cap), 0.0, self.ttc_cap)
        # Higher score = shorter TTC = impending collision risk
        return 1.0 / (c_ttc + 0.1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._compute_raw_score(X)
        denom = max(self.max_score_ - self.min_score_, 1e-6)
        prob = np.clip((raw - self.min_score_) / denom, 0.0, 1.0)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Model Training, Cross-Validation, and Heldout Evaluation."
    )
    parser.add_argument(
        "--dev_csv",
        "--dev-csv",
        type=str,
        dest="dev_csv",
        required=True,
        help="Path to Phase 3 training features (e.g. outputs_nexar/phase3/dev_features_event.csv).",
    )
    parser.add_argument(
        "--heldout_csv",
        "--heldout-csv",
        type=str,
        dest="heldout_csv",
        default=None,
        help="Optional path to Phase 3 heldout features for unbiased evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        dest="output_dir",
        required=True,
        help="Base output directory (creates models/ and evaluation/ subdirectories).",
    )
    parser.add_argument(
        "--cv_folds",
        "--cv-folds",
        type=int,
        dest="cv_folds",
        default=5,
        help="Number of folds for Stratified K-Fold cross-validation (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Experiment tag prefix for saved files (defaults to dev_csv stem).",
    )
    return parser.parse_args()


def get_models(seed: int) -> Dict[str, Any]:
    """Instantiate candidate models for benchmarking and training."""
    return {
        "heuristic_baseline": HeuristicBaseline(ttc_cap=15.0),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
        ),
    }


def tune_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Find the optimal probability threshold maximizing F1-score on OOF predictions.

    Returns:
        (best_threshold, precision, recall, f1, confusion_matrix)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = -1.0
    best_th = 0.5

    for th in thresholds:
        preds = (probabilities >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # Compute detailed metrics at best threshold
    final_preds = (probabilities >= best_th).astype(int)
    best_p = float(precision_score(y_true, final_preds, zero_division=0))
    best_r = float(recall_score(y_true, final_preds, zero_division=0))
    best_f1 = float(f1_score(y_true, final_preds, zero_division=0))
    best_cm = confusion_matrix(y_true, final_preds, labels=[0, 1])

    return float(best_th), best_p, best_r, best_f1, best_cm


def extract_top_feature_importances(
    model: Any,
    feature_names: List[str],
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Extract top feature importances or standardized coefficients for a fitted model."""
    # 1. Pipeline / Logistic Regression
    if isinstance(model, Pipeline) and hasattr(model.named_steps.get("clf"), "coef_"):
        clf = model.named_steps["clf"]
        importances = np.abs(clf.coef_[0])
    # 2. Random Forest
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # 3. HistGradientBoosting (uses permutation importance)
    elif isinstance(model, HistGradientBoostingClassifier):
        perm_res = permutation_importance(model, X, y, n_repeats=5, random_state=seed)
        importances = perm_res.importances_mean
    # 4. Heuristic Baseline
    elif isinstance(model, HeuristicBaseline):
        return [("center_weighted_min_ttc (rule formula)", 1.0)]
    else:
        return []

    ranked_indices = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[i], float(importances[i])) for i in ranked_indices]


def main() -> None:
    """Execute Phase 4 training, cross-validation, and evaluation pipeline."""
    args = parse_args()
    start_time = time.perf_counter()

    dev_path = Path(args.dev_csv).resolve()
    if not dev_path.is_file():
        logger.error("Dev features CSV not found: %s", dev_path)
        sys.exit(1)

    tag = args.tag or dev_path.stem
    output_dir = Path(args.output_dir).resolve()
    models_dir = output_dir / "models"
    eval_dir = output_dir / "evaluation"
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info(LOG_SEPARATOR)
    logger.info("DASHCAM RISK PREDICTION - PHASE 4 MODEL TRAINING & BENCHMARKING")
    logger.info(LOG_SEPARATOR)
    logger.info("Dev features:       %s", dev_path)
    logger.info("Heldout features:   %s", args.heldout_csv or "None (CV only)")
    logger.info("Output directory:   %s", output_dir)
    logger.info("Experiment tag:     %s", tag)
    logger.info("Stratified CV folds:%d", args.cv_folds)
    logger.info("Random seed:        %d", args.seed)

    # 1. Load Dev Dataset
    dev_df = pd.read_csv(dev_path, dtype={"video_id": str})
    if "target" not in dev_df.columns or "video_id" not in dev_df.columns:
        logger.error("Dev CSV missing 'target' or 'video_id' column: %s", dev_path)
        sys.exit(1)

    X_dev = dev_df.drop(columns=["video_id", "target"])
    y_dev = dev_df["target"].to_numpy(dtype=int)
    feature_names = X_dev.columns.tolist()

    logger.info("Loaded Dev dataset: %d samples, %d features", len(dev_df), len(feature_names))
    logger.info("Dev class balance:  Target 1: %d (%.1f%%), Target 0: %d (%.1f%%)",
                int((y_dev == 1).sum()), float((y_dev == 1).mean() * 100),
                int((y_dev == 0).sum()), float((y_dev == 0).mean() * 100))

    # 2. Cross-Validation & Out-Of-Fold Prediction
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    models_dict = get_models(args.seed)

    oof_predictions: Dict[str, np.ndarray] = {}
    cv_metrics_table: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    optimal_thresholds: Dict[str, float] = {}

    logger.info(LOG_SUBSEPARATOR)
    logger.info("RUNNING %d-FOLD STRATIFIED CROSS-VALIDATION...", args.cv_folds)

    for model_name, model_template in models_dict.items():
        oof_proba = np.zeros(len(y_dev), dtype=float)

        # K-Fold loop
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
            fold_model = clone(model_template)
            X_tr, y_tr = X_dev.iloc[train_idx], y_dev[train_idx]
            X_va = X_dev.iloc[val_idx]

            fold_model.fit(X_tr, y_tr)
            proba_va = fold_model.predict_proba(X_va)[:, 1]
            oof_proba[val_idx] = proba_va

        oof_predictions[model_name] = oof_proba

        # Compute ROC-AUC and PR-AUC on full OOF predictions
        roc_auc = float(roc_auc_score(y_dev, oof_proba))
        pr_auc = float(average_precision_score(y_dev, oof_proba))

        # Tune optimal decision threshold for F1-score
        best_th, prec, rec, f1, cm = tune_optimal_threshold(y_dev, oof_proba)
        optimal_thresholds[model_name] = best_th
        tn, fp, fn, tp = cm.ravel()

        logger.info(
            "[%-23s] ROC-AUC: %.4f | PR-AUC: %.4f | Thresh: %.2f | F1: %.4f | Prec: %.4f | Rec: %.4f",
            model_name, roc_auc, pr_auc, best_th, f1, prec, rec
        )
        logger.info(
            "  -> Confusion Matrix: [TP: %d, FP: %d, TN: %d, FN: %d]",
            tp, fp, tn, fn
        )

        cv_metrics = {
            "model": model_name,
            "split": "CV_OOF",
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "optimal_threshold": round(best_th, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        cv_metrics_table.append(cv_metrics)

        # 3. Fit on Full Dev Dataset & Extract Feature Importances
        full_model = clone(model_template)
        full_model.fit(X_dev, y_dev)
        trained_models[model_name] = full_model

        # Save model artifact with joblib
        model_save_path = models_dir / f"{tag}_{model_name}.joblib"
        artifact_payload = {
            "model": full_model,
            "optimal_threshold": best_th,
            "feature_names": feature_names,
            "cv_metrics": cv_metrics,
            "model_name": model_name,
            "tag": tag,
        }
        joblib.dump(artifact_payload, model_save_path)

        # Extract and log top 10 feature importances
        top_importances = extract_top_feature_importances(
            model=full_model,
            feature_names=feature_names,
            X=X_dev,
            y=y_dev,
            seed=args.seed,
            top_k=10,
        )
        if top_importances:
            logger.info("  -> Top Feature Importances / Coefficients:")
            for feat_name, imp_val in top_importances[:5]:
                logger.info("     * %-28s : %.4f", feat_name, imp_val)

    # 4. Unbiased Heldout Evaluation (if provided)
    heldout_metrics_table: List[Dict[str, Any]] = []
    heldout_predictions: Dict[str, np.ndarray] = {}
    heldout_df = None

    if args.heldout_csv is not None:
        heldout_path = Path(args.heldout_csv).resolve()
        if not heldout_path.is_file():
            logger.warning("Heldout CSV path specified but file not found: %s", heldout_path)
        else:
            heldout_df = pd.read_csv(heldout_path, dtype={"video_id": str})
            logger.info(LOG_SUBSEPARATOR)
            logger.info("RUNNING UNBIASED EVALUATION ON HELDOUT SPLIT (%s)...", heldout_path.name)
            logger.info("Loaded %d heldout records across %d features", len(heldout_df), len(feature_names))

            X_heldout = heldout_df[feature_names]
            y_heldout = heldout_df["target"].to_numpy(dtype=int)

            for model_name, full_model in trained_models.items():
                best_th = optimal_thresholds[model_name]
                p_heldout = full_model.predict_proba(X_heldout)[:, 1]
                heldout_predictions[model_name] = p_heldout
                pred_binary = (p_heldout >= best_th).astype(int)

                h_roc_auc = float(roc_auc_score(y_heldout, p_heldout))
                h_pr_auc = float(average_precision_score(y_heldout, p_heldout))
                h_f1 = float(f1_score(y_heldout, pred_binary, zero_division=0))
                h_prec = float(precision_score(y_heldout, pred_binary, zero_division=0))
                h_rec = float(recall_score(y_heldout, pred_binary, zero_division=0))
                h_cm = confusion_matrix(y_heldout, pred_binary, labels=[0, 1])
                h_tn, h_fp, h_fn, h_tp = h_cm.ravel()

                logger.info(
                    "[Heldout - %-23s] ROC-AUC: %.4f | PR-AUC: %.4f | Thresh: %.2f | F1: %.4f | Prec: %.4f | Rec: %.4f",
                    model_name, h_roc_auc, h_pr_auc, best_th, h_f1, h_prec, h_rec
                )

                heldout_metrics_table.append({
                    "model": model_name,
                    "split": "Heldout",
                    "roc_auc": round(h_roc_auc, 4),
                    "pr_auc": round(h_pr_auc, 4),
                    "optimal_threshold": round(best_th, 4),
                    "f1": round(h_f1, 4),
                    "precision": round(h_prec, 4),
                    "recall": round(h_rec, 4),
                    "tp": int(h_tp),
                    "fp": int(h_fp),
                    "tn": int(h_tn),
                    "fn": int(h_fn),
                })

    # 5. Save Consolidated Metrics Table
    all_metrics = cv_metrics_table + heldout_metrics_table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = eval_dir / f"{tag}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics comparison table to: %s", metrics_path)

    # 6. Save Predictions Table
    pred_records = []
    # Dev out-of-fold records
    for i, row in dev_df.iterrows():
        rec = {
            "video_id": row["video_id"],
            "target": int(row["target"]),
            "split": "dev_oof",
        }
        for model_name in models_dict.keys():
            prob = float(oof_predictions[model_name][i])
            th = optimal_thresholds[model_name]
            rec[f"{model_name}_prob"] = round(prob, 4)
            rec[f"{model_name}_pred"] = int(prob >= th)
        pred_records.append(rec)

    # Heldout records (if present)
    if heldout_df is not None:
        for i, row in heldout_df.iterrows():
            rec = {
                "video_id": row["video_id"],
                "target": int(row["target"]),
                "split": "heldout",
            }
            for model_name in models_dict.keys():
                prob = float(heldout_predictions[model_name][i])
                th = optimal_thresholds[model_name]
                rec[f"{model_name}_prob"] = round(prob, 4)
                rec[f"{model_name}_pred"] = int(prob >= th)
            pred_records.append(rec)

    predictions_df = pd.DataFrame(pred_records)
    predictions_path = eval_dir / f"{tag}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("Saved predictions table to: %s", predictions_path)

    total_time = time.perf_counter() - start_time

    # 7. Final Summary
    logger.info(LOG_SEPARATOR)
    logger.info("PHASE 4 BENCHMARKING COMPLETE - FINAL SUMMARY")
    logger.info(LOG_SEPARATOR)
    logger.info("Comparison Metrics Table:")
    # Print formatted table
    summary_cols = ["model", "split", "roc_auc", "pr_auc", "f1", "precision", "recall", "optimal_threshold"]
    logger.info("\n" + metrics_df[summary_cols].to_string(index=False))
    logger.info(LOG_SUBSEPARATOR)
    logger.info("Model artifacts saved to: %s", models_dir)
    logger.info("Evaluation tables saved to: %s", eval_dir)
    logger.info("Total execution time: %.2f seconds", total_time)
    logger.info(LOG_SEPARATOR)


if __name__ == "__main__":
    main()
