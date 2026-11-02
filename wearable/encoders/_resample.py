"""Fitting a window onto a model's fixed input length. Shared by the encoders.

Not a method itself — the underscore keeps it out of the registry.

Every encoder here has a pretrained input length it cannot exceed (MOMENT is hard-fixed
at 512; Mantis wants a multiple of 32 and was pretrained at 512), while the window the
`encode` step hands over is `days * profile` slots wide. Something has to give, and
WHICH thing gives is a research decision, so it is a named, keyed choice rather than a
silent reshape:

    pad         keep every value at its own position, pad the rest, and say where the
                real data is via a mask. Nothing is invented. Only works when the
                window is no longer than the ceiling, and only helps a model that
                accepts a mask (MOMENT does; Mantis does not).
    interpolate linear resampling onto the ceiling. Mantis's own recommendation, and
                the only option when the window is longer than the ceiling — but it
                INVENTS values across gaps, and it rescales time differently per row:
                a day with 20 readings and one with 140 both become 512 points, so
                rate-of-change stops being comparable between rows. Read any result
                that depends on it with that in mind.
    decimate    take every k-th slot. Loses detail but never invents a value and keeps
                one slot = one fixed duration across every row.

`wearable/windowing.bins_for` is the other half of this: choose `profile` so the window
fits the ceiling in the first place and none of the above has to do anything lossy.
"""
from __future__ import annotations

import numpy as np

MODES = ("pad", "interpolate", "decimate")


def fit_length(series: np.ndarray, length: int, mode: str = "pad"):
    """(n, T) -> ((n, length), (n, length) observed mask).

    The mask is True where the slot holds a real reading, so a model that takes one is
    never told that padding is data.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    x = np.asarray(series, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"series must be (n, T), got {x.shape}")
    n, T = x.shape
    length = int(length)

    if T == length:
        out = x.copy()
    elif mode == "interpolate":
        src = np.linspace(0.0, 1.0, T)
        dst = np.linspace(0.0, 1.0, length)
        out = np.empty((n, length), dtype=np.float32)
        for i in range(n):
            row = x[i]
            seen = np.isfinite(row)
            if seen.sum() >= 2:
                # interpolate over the OBSERVED points only; np.interp would otherwise
                # propagate NaN across the whole row
                out[i] = np.interp(dst, src[seen], row[seen])
            elif seen.sum() == 1:
                out[i] = row[seen][0]
            else:
                out[i] = np.nan
    elif mode == "decimate":
        idx = np.clip((np.arange(length) * (T / length)).astype(int), 0, T - 1)
        out = x[:, idx]
    else:                                     # pad
        if T > length:
            raise ValueError(
                f"a {T}-slot window does not fit a {length}-slot model under mode="
                f"'pad'. Either lower the window (windowing.bins_for({length}) gives "
                f"the coarsest binning that fits) or choose mode='decimate' / "
                f"'interpolate' and accept what each costs")
        out = np.full((n, length), np.nan, dtype=np.float32)
        out[:, length - T:] = x            # right-align: the classified day sits last

    mask = np.isfinite(out)
    return np.nan_to_num(out, nan=0.0), mask


def standardise(x: np.ndarray, mask: np.ndarray):
    """Per-row zero-mean unit-variance over the OBSERVED slots only.

    Every one of these encoders was pretrained on normalised series, and a per-row
    statistic is the per-patient adaptation H10 asks for — a 40 ms baseline and a 90 ms
    baseline should embed alike if their shapes match. Computed over observed slots so
    padding cannot drag the mean.
    """
    x = np.asarray(x, dtype=np.float32).copy()
    n = mask.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(n > 0, (x * mask).sum(axis=1, keepdims=True) / np.maximum(n, 1), 0.0)
        var = np.where(n > 1, (((x - mu) * mask) ** 2).sum(axis=1, keepdims=True)
                       / np.maximum(n - 1, 1), 1.0)
    sd = np.sqrt(np.maximum(var, 1e-8))
    return np.where(mask, (x - mu) / sd, 0.0).astype(np.float32)


if __name__ == "__main__":
    s = np.array([[1.0, np.nan, 3.0, 4.0]])

    x, m = fit_length(s, 8, "pad")
    # 4 slots right-aligned into 8: padding at 0-3, then the row's own NaN at slot 5
    assert x.shape == (1, 8), x.shape
    assert list(m[0]) == [False, False, False, False, True, False, True, True], m
    assert x[0, 4] == 1.0 and x[0, 7] == 4.0, x          # right-aligned, NaN -> 0 + mask
    assert x[0, 5] == 0.0, x                             # masked, so the 0 is inert
    print("pad: right-aligned, padding AND the row's own hole both masked -> PASS")

    try:
        fit_length(np.zeros((1, 16)), 8, "pad")
    except ValueError as exc:
        assert "bins_for" in str(exc), exc
        print("pad: an over-budget window is refused with the fix named -> PASS")
    else:
        raise AssertionError("pad must refuse an over-budget window")

    x, m = fit_length(s, 8, "interpolate")
    assert m.all() and np.isfinite(x).all(), (x, m)
    assert abs(x[0, 0] - 1.0) < 1e-6 and abs(x[0, -1] - 4.0) < 1e-6, x
    print(f"interpolate: 4 -> 8 slots, endpoints kept, every slot now 'observed' "
          f"(that is the cost) -> PASS")

    x, m = fit_length(np.array([[1.0, 2.0, 3.0, 4.0]]), 2, "decimate")
    assert list(x[0]) == [1.0, 3.0], x
    print("decimate: every k-th slot, no value invented -> PASS")

    z = standardise(*fit_length(np.array([[10.0, 20.0, 30.0]]), 3, "pad"))
    assert abs(z.mean()) < 1e-6 and abs(z.std(ddof=0) - 0.816) < 0.01, (z.mean(), z.std())
    # a row with one observation must not produce NaN
    z1 = standardise(*fit_length(np.array([[np.nan, 5.0, np.nan]]), 3, "pad"))
    assert np.isfinite(z1).all(), z1
    print("standardise: per-row over observed slots, single-reading row stays finite -> PASS")
    print("\nALL RESAMPLE CHECKS PASS")
