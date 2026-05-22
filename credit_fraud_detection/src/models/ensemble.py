# src/models/ensemble.py

import numpy as np
import logging

logger = logging.getLogger(__name__)


class AnomalyEnsemble:
    def __init__(self, config: dict):
        cfg = config["ensemble"]
        self.ae_weight = cfg["ae_weight"]   # default 0.6
        self.if_weight = cfg["if_weight"]   # default 0.4
        assert abs(self.ae_weight + self.if_weight - 1.0) < 1e-6, \
            "Ensemble weights must sum to 1.0"

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1] so both scores are on the same scale."""
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-8)

    def combine(
        self,
        ae_scores: np.ndarray,
        if_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Weighted combination of normalized anomaly scores.
        Both inputs must already be arrays of shape (N,).
        Higher final score = more likely to be fraud.
        """
        ae_norm = self._normalize(ae_scores)
        if_norm = self._normalize(if_scores)

        final = self.ae_weight * ae_norm + self.if_weight * if_norm

        logger.info(
            f"Ensemble scores — "
            f"mean={final.mean():.4f}  "
            f"max={final.max():.4f}  "
            f"min={final.min():.4f}"
        )
        return final

    def predict(self, final_scores: np.ndarray, threshold: float) -> np.ndarray:
        """Convert continuous scores to binary labels using a threshold."""
        return (final_scores >= threshold).astype(int)

    def score_summary(
        self,
        ae_scores: np.ndarray,
        if_scores: np.ndarray,
        final_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Log mean scores for fraud vs normal — sanity check.
        Good ensemble: fraud should have clearly higher scores than normal.
        """
        fraud_mask  = y_true == 1
        normal_mask = y_true == 0

        summary = {
            "ae_fraud_mean":    ae_scores[fraud_mask].mean(),
            "ae_normal_mean":   ae_scores[normal_mask].mean(),
            "if_fraud_mean":    if_scores[fraud_mask].mean(),
            "if_normal_mean":   if_scores[normal_mask].mean(),
            "final_fraud_mean": final_scores[fraud_mask].mean(),
            "final_normal_mean":final_scores[normal_mask].mean(),
        }

        logger.info("── Score Summary ──────────────────────")
        for k, v in summary.items():
            logger.info(f"  {k:<22}: {v:.4f}")
        logger.info("───────────────────────────────────────")

        return summary# ensemble.py
