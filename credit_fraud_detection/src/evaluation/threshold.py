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
    """
    Find the lowest threshold that achieves at least target_recall.
    Strategy: we tolerate more false positives to ensure we catch
    most fraud — the cost of missing fraud >> cost of a false alarm.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve has one more entry than thresholds
    # zip stops at the shorter one (thresholds)
    candidates = [
        (thresh, rec, prec)
        for thresh, rec, prec in zip(thresholds, recall[:-1], precision[:-1])
        if rec >= target_recall
    ]

    if not candidates:
        # fallback: just take the threshold with best F1
        logger.warning(
            f"No threshold achieves recall >= {target_recall:.0%}. "
            "Falling back to best F1 threshold."
        )
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx  = np.argmax(f1_scores)
        threshold = float(thresholds[best_idx])
        logger.info(f"Best F1 threshold: {threshold:.4f}")
        return threshold

    # among all thresholds meeting recall target, pick the one with best precision
    best = max(candidates, key=lambda x: x[2])
    threshold, rec, prec = best

    logger.info(
        f"Optimal threshold: {threshold:.4f} — "
        f"recall={rec:.4f}  precision={prec:.4f}"
    )
    return float(threshold)


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

    auprc = np.trapz(precision, recall) * -1   # recall decreases → negative area
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