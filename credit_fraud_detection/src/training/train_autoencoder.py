# train_autoencoder.py
# src/training/train_autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from tqdm import tqdm

from src.models.autoencoder import Autoencoder, save_model

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def make_dataloader(X: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(model: Autoencoder, X_train: np.ndarray, config: dict) -> dict:
    cfg    = config["autoencoder"]
    device = get_device()
    model  = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    # Split a small validation chunk from train for early stopping
    split     = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split], X_train[split:]

    train_loader = make_dataloader(X_tr, cfg["batch_size"], shuffle=True)
    val_loader   = make_dataloader(X_val, cfg["batch_size"], shuffle=False)

    best_val_loss   = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ────────────────────────────────────
        model.train()
        train_loss = 0.0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat = model(batch)
            loss  = criterion(x_hat, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
            optimizer.step()
            train_loss += loss.item() * len(batch)

        train_loss /= len(X_tr)

        # ── Validate ─────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_hat = model(batch)
                val_loss += criterion(x_hat, batch).item() * len(batch)
        val_loss /= len(X_val)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ── Early stopping ───────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model)   # save best checkpoint
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch} — best val_loss={best_val_loss:.6f}")
                break

    return history


def get_reconstruction_scores(model: Autoencoder, X: np.ndarray, config: dict) -> np.ndarray:
    """Run trained model over any dataset and return per-sample anomaly scores."""
    device = get_device()
    model  = model.to(device).eval()
    loader = make_dataloader(X, batch_size=config["autoencoder"]["batch_size"], shuffle=False)
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            err   = model.reconstruction_error(batch)
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)