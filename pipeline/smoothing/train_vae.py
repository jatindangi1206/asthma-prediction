"""Stage 3 (training) — train a CNN-VAE on cleaned HRV windows.

Trains on contiguous HRV segments from one or more patients' cleaned CSVs
(``hrv_clean``, the gap-chunked series) and saves ``{model_state, config}`` to
``pipeline/checkpoints/vae_cnn.pt``. Windows never span a gap (NaN), so the VAE
only ever learns within-chunk structure. Values are z-scored using the training
mean/std, which are stored in the checkpoint for the smoother to reuse.

The CNN-VAE architecture is the pasted two-conv design; hyperparameters come
from config.py (tunable defaults).

    python -m pipeline.smoothing.train_vae <patient_id> [<patient_id> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class CNNVAE(nn.Module):
    """Two-conv CNN-VAE (full-resolution latent), as pasted."""

    def __init__(self, window_size, latent_dim, channels_1, channels_2, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels_1, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels_1, channels_2, kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.flatten_dim = channels_2 * window_size
        self.mu = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels_2, channels_1, kernel_size, padding=pad),
            nn.ReLU(),
            nn.ConvTranspose1d(channels_1, 1, kernel_size, padding=pad),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        d = self.decoder_input(z).view(-1, self.flatten_dim // x.size(-1), x.size(-1))
        return self.decoder(d), mu, logvar

    def reconstruct(self, x):
        """Deterministic reconstruction (uses the latent mean, no sampling)."""
        h = self.encoder(x).view(x.size(0), -1)
        d = self.decoder_input(self.mu(h)).view(-1, self.flatten_dim // x.size(-1), x.size(-1))
        return self.decoder(d)


def vae_loss(recon_x, x, mu, logvar, beta):
    recon = nn.functional.mse_loss(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


def finite_segments(values: np.ndarray):
    """Yield (start, end) index ranges of contiguous non-NaN runs."""
    finite = np.isfinite(values)
    i, n = 0, len(values)
    while i < n:
        if finite[i]:
            j = i
            while j < n and finite[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def build_windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Stack length-``window`` windows from each finite segment long enough."""
    out = []
    for a, b in finite_segments(values):
        seg = values[a:b]
        for s in range(0, len(seg) - window + 1, stride):
            out.append(seg[s:s + window])
    if not out:
        return np.empty((0, window), dtype=np.float32)
    return np.stack(out).astype(np.float32)


def _load_clean_hrv(patient_id: str) -> np.ndarray:
    src = config.processed_path(patient_id, "cleaned")
    if not src.exists():
        raise FileNotFoundError(f"Run Stage 2 first; missing {src}")
    return pd.read_csv(src)[config.HRV_CLEAN_COL].to_numpy(dtype=np.float64)


def train(patient_ids, epochs=config.VAE_EPOCHS, verbose=True) -> dict:
    """Train the VAE on the given patients' cleaned HRV; save + return the checkpoint."""
    torch.manual_seed(config.VAE_SEED)
    np.random.seed(config.VAE_SEED)

    raw = np.concatenate([_load_clean_hrv(p) for p in patient_ids])
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError("No finite HRV samples to train on.")
    mean, std = float(finite.mean()), float(finite.std() or 1.0)

    windows = build_windows((raw - mean) / std, config.VAE_WINDOW, config.VAE_STRIDE)
    if len(windows) == 0:
        raise ValueError(
            f"No HRV segment is >= window={config.VAE_WINDOW}; cannot train. "
            f"Tune VAE_WINDOW down in config.py for this data."
        )

    c1, c2 = config.VAE_CHANNELS
    model = CNNVAE(config.VAE_WINDOW, config.VAE_LATENT, c1, c2, config.VAE_KERNEL).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=config.VAE_LR)
    loader = DataLoader(TimeSeriesDataset(windows), batch_size=config.VAE_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, config.VAE_BETA)
            loss.backward()
            opt.step()
            total += loss.item()
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={total / max(len(loader), 1):.5f}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "window_size": config.VAE_WINDOW,
            "stride": config.VAE_STRIDE,
            "latent_dim": config.VAE_LATENT,
            "channels_1": c1,
            "channels_2": c2,
            "kernel_size": config.VAE_KERNEL,
            "mean": mean,
            "std": std,
            "trained_on": list(patient_ids),
        },
    }
    config.ensure_dirs()
    torch.save(ckpt, config.VAE_CHECKPOINT)
    if verbose:
        print(f"  trained on {len(windows)} windows ({len(patient_ids)} patient(s)) "
              f"-> {config.VAE_CHECKPOINT}")
    return ckpt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m pipeline.smoothing.train_vae <patient_id> [<patient_id> ...]")
        raise SystemExit(2)
    print(f"=== Training CNN-VAE on {sys.argv[1:]} ===")
    train(sys.argv[1:])
