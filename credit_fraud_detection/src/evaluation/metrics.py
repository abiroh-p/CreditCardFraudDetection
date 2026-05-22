# metrics.py
# metrics.py
# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Area Under Precision-Recall Curve.
    Primary metric for imbalanced problems — never use accuracy here.
    """
    return average_precision_score(y_true, scores)


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Secondary metric — useful but can be misleading under severe imbalance."""
    return roc_auc_score(y_true, scores)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, zero_division=0)


def recall_at_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_precision: float = 0.95,
) -> float:
    """
    Business-friendly metric:
    'What % of fraud do we catch while keeping false alarms below X%?'
    target_precision=0.95 means: at the threshold where precision >= 95%,
    what is our recall?
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns arrays in decreasing threshold order
    mask = precision >= target_precision
    if not mask.any():
        logger.warning(f"Precision never reaches {target_precision:.0%} — returning 0.0")
        return 0.0
    return float(recall[mask].max())


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    FPR = FP / (FP + TN)
    Fraction of legitimate transactions incorrectly flagged as fraud.
    This is the customer experience cost metric.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn + 1e-8)


def full_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    config: dict,
) -> dict:
    """Compute and log all metrics in one shot."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    report = {
        "auprc":                  round(compute_auprc(y_true, scores), 4),
        "auroc":                  round(compute_auroc(y_true, scores), 4),
        "f1":                     round(compute_f1(y_true, y_pred), 4),
        "recall_at_95_precision": round(recall_at_precision(y_true, scores, 0.95), 4),
        "false_positive_rate":    round(false_positive_rate(y_true, y_pred), 4),
        "true_positives":         int(tp),
        "false_positives":        int(fp),
        "true_negatives":         int(tn),
        "false_negatives":        int(fn),
        "total_fraud":            int(y_true.sum()),
        "total_normal":           int((y_true == 0).sum()),
    }

    logger.info("── Evaluation Report ──────────────────────────")
    logger.info(f"  AUPRC                  : {report['auprc']:.4f}  ← primary metric")
    logger.info(f"  AUROC                  : {report['auroc']:.4f}")
    logger.info(f"  F1 Score               : {report['f1']:.4f}")
    logger.info(f"  Recall @ 95% Precision : {report['recall_at_95_precision']:.4f}")
    logger.info(f"  False Positive Rate    : {report['false_positive_rate']:.4f}")
    logger.info(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    logger.info("───────────────────────────────────────────────")

    return report


def save_report(report: dict, config: dict):
    out = Path(config["paths"]["reports"]) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved → {out}")