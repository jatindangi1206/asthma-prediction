"""Stage 3 (smoothing) — VAE-smooth cleaned HRV; context carried through.

Reads ``<id>_cleaned.csv``, slides the trained CNN-VAE over the cleaned HRV
(``hrv_clean``) within each contiguous segment, averages overlapping window
reconstructions, and appends ``smoothed_value``. Gaps (NaN in ``hrv_clean``)
stay NaN — windows never span them. Every other column rides through unchanged.

If no checkpoint exists yet, the VAE is trained on this patient on the fly.

    python -m pipeline.smoothing.smoothing <patient_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config
from pipeline.smoothing.train_vae import CNNVAE, finite_segments, train, DEVICE


def load_model() -> tuple[CNNVAE, dict]:
    ckpt = torch.load(config.VAE_CHECKPOINT, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = CNNVAE(cfg["window_size"], cfg["latent_dim"],
                   cfg["channels_1"], cfg["channels_2"], cfg["kernel_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def smooth_series(model: CNNVAE, cfg: dict, hrv_clean: np.ndarray) -> tuple[np.ndarray, int]:
    """Reconstruct each finite segment; return (smoothed[N], n_vae_smoothed)."""
    W, S = cfg["window_size"], cfg["stride"]
    mean, std = cfg["mean"], cfg["std"]
    out = np.full(len(hrv_clean), np.nan, dtype=np.float64)
    n_vae = 0

    for a, b in finite_segments(hrv_clean):
        seg = (hrv_clean[a:b] - mean) / std
        L = len(seg)
        if L < W:
            # too short for a full window -> carry cleaned values through as-is
            out[a:b] = hrv_clean[a:b]
            continue
        starts = list(range(0, L - W + 1, S))
        if starts[-1] != L - W:
            starts.append(L - W)
        batch = torch.tensor(np.stack([seg[s:s + W] for s in starts]),
                             dtype=torch.float32, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            rec = model.reconstruct(batch).squeeze(1).cpu().numpy()
        rsum = np.zeros(L)
        rcnt = np.zeros(L)
        for i, s in enumerate(starts):
            rsum[s:s + W] += rec[i]
            rcnt[s:s + W] += 1
        out[a:b] = (rsum / rcnt) * std + mean
        n_vae += L
    return out, n_vae


def smooth_patient(patient_id: str) -> Path:
    config.ensure_dirs()
    src = config.processed_path(patient_id, "cleaned")
    if not src.exists():
        raise FileNotFoundError(f"Run Stage 2 first; missing {src}")
    print(f"=== Stage 3: smoothing {patient_id} ===")

    if not config.VAE_CHECKPOINT.exists():
        print(f"[{patient_id}] no checkpoint -> training VAE on this patient")
        train([patient_id])
    model, cfg = load_model()

    df = pd.read_csv(src)
    hrv_clean = df[config.HRV_CLEAN_COL].to_numpy(dtype=np.float64)
    smoothed, n_vae = smooth_series(model, cfg, hrv_clean)
    df[config.SMOOTHED_COL] = smoothed

    dest = config.processed_path(patient_id, "smoothed")
    df.to_csv(dest, index=False)
    n_finite = int(np.isfinite(hrv_clean).sum())
    print(f"[{patient_id}] VAE-smoothed {n_vae}/{n_finite} finite samples "
          f"(rest in <window segments carried as-is)")
    print(f"[{patient_id}] smoothed -> {dest}")
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.smoothing.smoothing <patient_id>")
        raise SystemExit(2)
    smooth_patient(sys.argv[1])
