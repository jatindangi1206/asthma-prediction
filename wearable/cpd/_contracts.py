"""Shared CPD numerical contracts; no data loading or stage orchestration."""
import numpy as np


def series(time, values, priors=None):
    t, z = np.asarray(time, float), np.asarray(values, float)
    if t.ndim != 1 or z.shape != t.shape or not np.isfinite(t).all():
        raise ValueError("CPD needs aligned 1-D values and finite timestamps")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("CPD timestamps must be unique and strictly increasing")
    pr = priors or {}
    chunk = np.asarray(pr.get("chunk", np.zeros(len(t), int)))
    if chunk.shape != t.shape or not np.isfinite(chunk).all():
        raise ValueError("CPD chunk ids must be finite and aligned")
    return t, z, chunk


def positive(**settings):
    for name, value in settings.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def hit(score, threshold):
    x, th = np.asarray(score, float), np.asarray(threshold, float)
    return np.isfinite(x) & np.isfinite(th) & (x >= th)


def excess(score, threshold):
    """Finite threshold-relative evidence, NOT a probability or cross-method severity."""
    x, th = np.broadcast_arrays(np.asarray(score, float), np.asarray(threshold, float))
    out = np.zeros(x.shape)
    ok = hit(x, th)
    out[ok] = np.clip((x[ok] - th[ok]) / np.maximum(np.abs(th[ok]), 1.0), 0, 1)
    return out


def starts(flag, time, chunk, gap_min):
    f = np.asarray(flag, bool)
    if not len(f):
        return f
    boundary = np.r_[True, (np.diff(time) >= gap_min) | (np.diff(chunk) != 0)]
    return f & (boundary | ~np.r_[False, f[:-1]])
