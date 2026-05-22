
import logging
import numpy as np
from src.utils.config import load_config, setup_logging
from src.data.loader import load_raw_data, split_data, save_processed
from src.data.preprocessor import fit_scalers, apply_scalers, get_X_y, save_scalers
from src.data.feature_engineer import engineer_features
from src.models.autoencoder import build_autoencoder, load_model
from src.models.isolation_forest import IForest
from src.models.ensemble import AnomalyEnsemble
from src.training.train_autoencoder import train_autoencoder, get_reconstruction_scores
from src.training.train_iforest import train_iforest
from src.evaluation.metrics import full_report, save_report
from src.evaluation.threshold import (
    find_optimal_threshold,
    plot_precision_recall_curve,
    plot_score_distribution,
    plot_training_loss,
)
from src.evaluation.explainability import (
    explain_autoencoder,
    explain_iforest,
    explain_single_transaction,
)

logger = logging.getLogger(__name__)


def main():
    # ── 0. Setup ──────────────────────────────────────────────────
    setup_logging()
    config = load_config("config.yaml")
    logger.info("=" * 55)
    logger.info("  Credit Card Fraud Detection — Anomaly Ensemble")
    logger.info("=" * 55)

    # ── 1. Load & split ───────────────────────────────────────────
    logger.info("\n[1/7] Loading data...")
    df           = load_raw_data(config)
    train_df, test_df = split_data(df, config)

    # ── 2. Feature engineering ────────────────────────────────────
    logger.info("\n[2/7] Engineering features...")
    train_df = engineer_features(train_df, config)
    test_df  = engineer_features(test_df,  config)

    # ── 3. Preprocessing ──────────────────────────────────────────
    logger.info("\n[3/7] Preprocessing...")
    scalers  = fit_scalers(train_df)
    train_df = apply_scalers(train_df, scalers)
    test_df  = apply_scalers(test_df,  scalers)
    save_scalers(scalers)
    save_processed(train_df, test_df, config)

    X_train, _      = get_X_y(train_df)
    X_test,  y_test = get_X_y(test_df)
    feature_names   = [c for c in train_df.columns if c != "Class"]

    logger.info(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    logger.info(f"Fraud in test: {y_test.sum()} / {len(y_test)}")

    # ── 4. Train models ───────────────────────────────────────────
    logger.info("\n[4/7] Training Autoencoder...")
    ae      = build_autoencoder(X_train.shape[1], config)
    history = train_autoencoder(ae, X_train, config)
    plot_training_loss(history, config)

    logger.info("\n[4/7] Training Isolation Forest...")
    iforest = train_iforest(X_train, config)

    # ── 5. Score ──────────────────────────────────────────────────
    logger.info("\n[5/7] Scoring test set...")
    ae      = load_model(X_train.shape[1], config)   # reload best checkpoint
    ae_scores = get_reconstruction_scores(ae, X_test, config)
    if_scores = iforest.score(X_test)

    ensemble      = AnomalyEnsemble(config)
    final_scores  = ensemble.combine(ae_scores, if_scores)
    ensemble.score_summary(ae_scores, if_scores, final_scores, y_test)

    # ── 6. Threshold + evaluate ───────────────────────────────────
    logger.info("\n[6/7] Tuning threshold and evaluating...")
    threshold = find_optimal_threshold(
        y_test, final_scores,
        target_recall=config["evaluation"]["target_recall"],
    )
    y_pred = ensemble.predict(final_scores, threshold)

    report = full_report(y_test, final_scores, y_pred, config)
    save_report(report, config)

    plot_precision_recall_curve(y_test, final_scores, threshold, config)
    plot_score_distribution(final_scores, y_test, threshold, config)

    # ── 7. Explainability ─────────────────────────────────────────
    logger.info("\n[7/7] Running SHAP explainability...")

    # explain top anomalous samples — sort test set by score descending
    top_idx      = np.argsort(final_scores)[::-1]
    X_test_sorted = X_test[top_idx]
    y_test_sorted = y_test[top_idx]

    ae_shap = explain_autoencoder(ae, X_train, X_test_sorted, feature_names, config)
    if_shap = explain_iforest(iforest, X_train, X_test_sorted, feature_names, config)

    # single transaction deep-dive: pick first fraud in sorted list
    fraud_idx = int(np.where(y_test_sorted == 1)[0][0])
    explain_single_transaction(ae_shap, X_test_sorted, feature_names, config,
                               idx=fraud_idx, label="fraud")

    # ── Done ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"  AUPRC  : {report['auprc']:.4f}")
    logger.info(f"  F1     : {report['f1']:.4f}")
    logger.info(f"  Recall @ 95% P : {report['recall_at_95_precision']:.4f}")
    logger.info(f"  FPR    : {report['false_positive_rate']:.4f}")
    logger.info("=" * 55)
    logger.info("All outputs saved to outputs/")


if __name__ == "__main__":
    main()