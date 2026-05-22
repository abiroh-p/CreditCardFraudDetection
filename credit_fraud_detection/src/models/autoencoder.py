# src/models/autoencoder.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _build_sequential(dims_in_out: list, dropout: float, final_activation: bool = True) -> nn.Sequential:
    """
    Build a sequential block from a list of (in, out) dimension pairs.
    Applies BN + ReLU + Dropout after each layer except the last.
    """
    layers = []
    for i, (in_d, out_d) in enumerate(dims_in_out):
        layers.append(nn.Linear(in_d, out_d))
        is_last = (i == len(dims_in_out) - 1)
        if not is_last or final_activation:
            layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        """
        hidden_dims: bottleneck path e.g. [64, 32, 16]
        Encoder: input_dim → 64 → 32 → 16
        Decoder: 16 → 32 → 64 → input_dim
        """
        super().__init__()

        # Encoder: input → hidden_dims
        enc_pairs = list(zip([input_dim] + hidden_dims[:-1], hidden_dims))
        self.encoder = _build_sequential(enc_pairs, dropout, final_activation=True)

        # Decoder: hidden_dims reversed → input
        dec_dims    = list(reversed(hidden_dims))
        dec_pairs   = list(zip(dec_dims[:-1], dec_dims[1:]))
        self.decoder = _build_sequential(dec_pairs, dropout, final_activation=True)

        # final reconstruction layer — no BN, no activation
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z     = self.encoder(x)
        z     = self.decoder(z)
        x_hat = self.output_layer(z)
        return x_hat

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE — this is the anomaly score."""
        with torch.no_grad():
            x_hat = self.forward(x)
            error = torch.mean((x - x_hat) ** 2, dim=1)
        return error


def build_autoencoder(input_dim: int, config: dict) -> Autoencoder:
    cfg   = config["autoencoder"]
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Autoencoder built — {total:,} trainable parameters")
    logger.info(f"  Encoder: {input_dim} → {' → '.join(map(str, cfg['hidden_dims']))}")
    logger.info(f"  Decoder: {' → '.join(map(str, reversed(cfg['hidden_dims'])))} → {input_dim}")
    return model


def save_model(model: Autoencoder, path: str = "saved_models/autoencoder.pt"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Autoencoder saved → {path}")


def load_model(input_dim: int, config: dict, path: str = "saved_models/autoencoder.pt") -> Autoencoder:
    model = build_autoencoder(input_dim, config)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    logger.info(f"Autoencoder loaded from {path}")
    return model