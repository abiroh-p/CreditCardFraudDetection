# src/models/ensemble.py

import numpy as np
import logging

logger = logging.getLogger(__name__)


class AnomalyEnsemble:
    def __init__(self, config: dict):
        cfg = config["ensemble"]
        self.ae_weight = cfg["ae_weight"]
        self.if_weight = cfg["if_weight"]
        assert abs(self.ae_weight + self.if_weight - 1.0) < 1e-6, \
            "Ensemble weights must sum to 1.0"

    def _normalize_robust(self, scores: np.ndarray) -> np.ndarray:
        """
        Robust normalization using percentiles instead of min/max or rankdata.
        Clips at p1 and p99 to remove outlier influence, then scales to [0,1].
        This preserves score magnitude relationships unlike rankdata,
        while being robust to extreme outliers unlike min-max.
        """
        p1  = np.percentile(scores, 1)
        p99 = np.percentile(scores, 99)
        clipped = np.clip(scores, p1, p99)
        return (clipped - p1) / (p99 - p1 + 1e-8)

    def combine(
        self,
        ae_scores: np.ndarray,
        if_scores: np.ndarray,
    ) -> np.ndarray:
        ae_norm = self._normalize_robust(ae_scores)
        if_norm = self._normalize_robust(if_scores)
        final   = self.ae_weight * ae_norm + self.if_weight * if_norm

        logger.info(
            f"Ensemble scores — "
            f"mean={final.mean():.4f}  "
            f"max={final.max():.4f}  "
            f"min={final.min():.4f}"
        )
        return final

    def predict(self, final_scores: np.ndarray, threshold: float) -> np.ndarray:
        return (final_scores >= threshold).astype(int)

    def score_summary(
        self,
        ae_scores: np.ndarray,
        if_scores: np.ndarray,
        final_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
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
        return summary