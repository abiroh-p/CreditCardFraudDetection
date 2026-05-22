
# src/models/isolation_forest.py

import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class IForest:
    def __init__(self, config: dict):
        cfg = config["isolation_forest"]
        self.model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            random_state=cfg["random_seed"],
            n_jobs=-1,       # use all CPU cores
            warm_start=False,
        )

    def fit(self, X: np.ndarray) -> None:
        logger.info(f"Fitting Isolation Forest on {len(X):,} samples...")
        self.model.fit(X)
        logger.info("Isolation Forest fitted.")

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1] where higher = more anomalous.
        sklearn's decision_function returns negative scores for anomalies,
        so we negate and normalize to flip the direction intuitively.
        """
        raw = self.model.decision_function(X)   # more negative = more anomalous
        flipped = -raw                           # more positive = more anomalous
        normalized = (flipped - flipped.min()) / (flipped.max() - flipped.min() + 1e-8)
        return normalized

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary labels: 1 = anomaly, 0 = normal."""
        raw = self.model.predict(X)  # sklearn returns -1 = anomaly, 1 = normal
        return (raw == -1).astype(int)

    def save(self, path: str = "saved_models/isolation_forest.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Isolation Forest saved → {path}")

    def load(self, path: str = "saved_models/isolation_forest.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Isolation Forest loaded from {path}")
        return self# isolation_forest.py
