# src/evaluation/explainability.py

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def explain_autoencoder(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    config: dict,
    n_background: int = 200,
    n_explain: int = 100,
):
    """
    SHAP DeepExplainer for the autoencoder.

    Strategy:
    - background = sample of normal training transactions
    - explain    = top-N highest scoring (most anomalous) test samples
    - SHAP value per feature = how much that feature contributed
      to the reconstruction error for that sample
    """
    import torch
    from src.models.autoencoder import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae     = load_model(X_train.shape[1], config).to(device).eval()

    # background: random sample of normal transactions
    bg_idx = np.random.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)

    # explain the top most anomalous test samples
    explain_data = torch.tensor(X_test[:n_explain], dtype=torch.float32).to(device)

    explainer   = shap.DeepExplainer(ae, background)
    shap_values = explainer.shap_values(explain_data)   # shape: (n_explain, n_features)

    # for autoencoder, shap_values is a list (one per output neuron)
    # mean absolute contribution across all output neurons
    if isinstance(shap_values, list):
        shap_values = np.mean(np.abs(shap_values), axis=0)

    _plot_shap_summary(shap_values, X_test[:n_explain], feature_names, config, tag="autoencoder")
    _plot_shap_bar(shap_values, feature_names, config, tag="autoencoder")

    logger.info("SHAP analysis complete for autoencoder")
    return shap_values


def explain_iforest(
    iforest,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    config: dict,
    n_background: int = 200,
    n_explain: int = 100,
):
    """
    SHAP TreeExplainer for Isolation Forest.
    Tree-based models are natively supported — fast and exact.
    """
    explainer   = shap.TreeExplainer(iforest.model, data=X_train[:n_background])
    shap_values = explainer.shap_values(X_test[:n_explain])

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    _plot_shap_summary(shap_values, X_test[:n_explain], feature_names, config, tag="iforest")
    _plot_shap_bar(shap_values, feature_names, config, tag="iforest")

    logger.info("SHAP analysis complete for isolation forest")
    return shap_values


def explain_single_transaction(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list,
    config: dict,
    idx: int = 0,
    label: str = "fraud",
):
    """
    Waterfall plot for a single transaction.
    Shows exactly which features pushed the anomaly score up or down.
    Great for the 'explainability' section of your report.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    values   = shap_values[idx]
    features = feature_names
    order    = np.argsort(np.abs(values))[::-1][:15]   # top 15 features

    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in values[order]]
    bars   = ax.barh(
        [features[i] for i in order],
        values[order],
        color=colors,
        edgecolor="none",
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on anomaly score)", fontsize=11)
    ax.set_title(f"Feature Contribution — {label} transaction #{idx}", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    out = Path(config["paths"]["plots"]) / f"shap_single_{label}_{idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Single transaction SHAP plot saved → {out}")


def _plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    config: dict,
    tag: str,
):
    """
    Beeswarm plot — each dot is one transaction, colored by feature value.
    Shows both importance and direction of each feature's impact.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
        ax=ax,
    )
    ax.set_title(f"SHAP Summary — {tag}", fontsize=12)
    out = Path(config["paths"]["plots"]) / f"shap_summary_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP summary plot saved → {out}")


def _plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list,
    config: dict,
    tag: str,
    top_n: int = 15,
):
    """
    Mean absolute SHAP values — clean bar chart of global feature importance.
    Most readable for a presentation or report.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in reversed(order)],
        mean_abs[reversed(order)],
        color="#378ADD",
        edgecolor="none",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Global Feature Importance — {tag} (top {top_n})", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    out = Path(config["paths"]["plots"]) / f"shap_importance_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP importance bar chart saved → {out}")# explainability.py
