# autoencoder.py
# src/models/autoencoder.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, hidden_dims: list, output_dim: int, dropout: float):
        super().__init__()
        layers = []
        dims = list(reversed(hidden_dims))
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        # final layer — no activation, no dropout, raw reconstruction
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        # hidden_dims = [64, 32, 16] — encoder compresses, decoder mirrors
        mid = len(hidden_dims) // 2
        encoder_dims = hidden_dims[:mid + 1]   # [64, 32, 16]
        decoder_dims = hidden_dims[mid:]        # [16, 32, 64]

        self.encoder = Encoder(input_dim, encoder_dims, dropout)
        self.decoder = Decoder(decoder_dims, input_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE — this becomes the anomaly score."""
        with torch.no_grad():
            x_hat = self.forward(x)
            error = torch.mean((x - x_hat) ** 2, dim=1)
        return error


def build_autoencoder(input_dim: int, config: dict) -> Autoencoder:
    cfg = config["autoencoder"]
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Autoencoder built — {total_params:,} trainable parameters")
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