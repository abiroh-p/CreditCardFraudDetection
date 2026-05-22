# threshold.py
# threshold.py
# src/evaluation/threshold.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers and Colab
from sklearn.metrics import precision_recall_curve
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.90,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)

    # compute FPR at each threshold
    normal_total = (y_true == 0).sum()
    fp_at_thresh = np.array([
        ((scores >= t) & (y_true == 0)).sum()
        for t in thresholds
    ])
    fpr_at_thresh = fp_at_thresh / normal_total

    # maximize F1 subject to FPR <= 1%
    fpr_mask = fpr_at_thresh <= 0.01
    if fpr_mask.any():
        masked_f1 = np.where(fpr_mask, f1_scores, 0)
        best_idx  = np.argmax(masked_f1)
    else:
        best_idx  = np.argmax(f1_scores)

    threshold = float(thresholds[best_idx])
    logger.info(
        f"Optimal threshold: {threshold:.4f} — "
        f"recall={recall[best_idx]:.4f}  "
        f"precision={precision[best_idx]:.4f}  "
        f"f1={f1_scores[best_idx]:.4f}  "
        f"fpr={fpr_at_thresh[best_idx]:.4f}"
    )
    return threshold

def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    config: dict,
):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # find the point on the curve closest to our chosen threshold
    idx = np.argmin(np.abs(thresholds - threshold))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(recall, precision, color="#378ADD", linewidth=2, label="PR Curve")
    ax.scatter(
        recall[idx], precision[idx],
        color="#E24B4A", s=100, zorder=5,
        label=f"Chosen threshold={threshold:.3f}\n"
              f"P={precision[idx]:.3f}  R={recall[idx]:.3f}"
    )

    # baseline: random classifier precision = fraud prevalence
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Random classifier (P={baseline:.3f})")

    auprc = np.trapezoid(precision, recall) * -1   # recall decreases → negative area
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve  |  AUPRC = {abs(auprc):.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    out = Path(config["paths"]["plots"]) / "precision_recall_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved → {out}")


def plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    config: dict,
):
    """
    Histogram of anomaly scores split by class.
    The cleaner the separation between fraud and normal, the better.
    This is the most intuitive plot to show in a presentation.
    """
    fraud_scores  = scores[y_true == 1]
    normal_scores = scores[y_true == 0]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(normal_scores, bins=80, alpha=0.6, color="#378ADD",
            label=f"Normal  (n={len(normal_scores):,})", density=True)
    ax.hist(fraud_scores,  bins=80, alpha=0.7, color="#E24B4A",
            label=f"Fraud   (n={len(fraud_scores):,})",  density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Anomaly Score Distribution — Fraud vs Normal", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out = Path(config["paths"]["plots"]) / "score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Score distribution saved → {out}")


def plot_training_loss(history: dict, config: dict):
    """Autoencoder train vs val loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#378ADD", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#E24B4A", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Autoencoder Training Loss", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out = Path(config["paths"]["plots"]) / "training_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training loss plot saved → {out}")