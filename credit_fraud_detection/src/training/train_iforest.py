# src/training/train_iforest.py

import numpy as np
import logging
from src.models.isolation_forest import IForest

logger = logging.getLogger(__name__)


def train_iforest(X_train: np.ndarray, config: dict) -> IForest:
    """
    Isolation Forest is trained on the FULL training set unlike the autoencoder.
    It is unsupervised and handles imbalance natively via the contamination param.
    """
    iforest = IForest(config)
    iforest.fit(X_train)
    iforest.save()
    return iforest# train_iforest.py
