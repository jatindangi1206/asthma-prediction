"""Shared machinery for the forecasters: causal block-ahead prediction over a series.

Not a method itself — the underscore keeps it out of the registry. The three adapters
differ only in the vendor call; everything about windowing, causality, gaps and the
predictive distribution lives here, once.

STRICT CAUSALITY (H9) — THE POINT OF THIS FILE
----------------------------------------------
Nine of the ten smoothers fit their per-patient parameters on the whole frame handed in,
so a value at time t moves when later rows arrive (`rspf` deviates 3.16; see CLAUDE.md's
H9 section). These forecasters cannot: the weights are **frozen**, nothing is fitted on
the cohort, and the prediction for grid slot k is computed from `grid[:k]` — readings
strictly before it. Blocks start from the chunk's own first reading, so a prefix run
reproduces a full run exactly where they overlap. Truncation-exact by construction,
which is what makes their surprise a legitimate early-warning signal with no lookahead
to argue about.

THE IRREGULAR SERIES (H7)
-------------------------
These models assume regular spacing; readings do not arrive regularly. Each chunk is
laid on a regular `cadence`-minute grid with each reading in its nearest slot and NaN
everywhere else — which Timestral and Chronos-2 take natively as "missing", so nothing
is interpolated. Predictions are read back at the slots that hold a reading.

Because a block's context is made of OBSERVATIONS and never of earlier predictions,
every block is independent and they all go to the model in ONE batched call: a patient
costs a forward pass or two rather than one per block.

THE OUTPUT IS THREE COLUMNS, NOT ONE
------------------------------------
    predicted_<sig>  what the model expected
    surprise_<sig>   observed - predicted, in the units of the signal
    tail_<sig>       where the observed value fell in the predictive distribution,
                     in [0, 1] — NaN for a method that returns no quantiles

`tail` is the one worth having. "This reading fell at the 99.5th percentile of what the
model expected" is calibrated and comparable across patients and channels; "this reading
was 6 ms off" is neither, because 6 ms means different things at a 30 ms and a 90 ms
baseline. A smoother's contract has nowhere to put it, which is why forecasting is its
own pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wearable.chunking import assign_chunks
from wearable.config import GAP_THRESHOLD_MIN

CADENCE_MIN = 10.0        # the grid step: the cohort's nominal sampling interval
CONTEXT = 512             # grid slots of history a forecast may look back over
STRIDE = 32               # slots predicted per forecast = one patch
MIN_CONTEXT = 64          # observed readings required before the first prediction
MIN_SEGMENT_ROWS = 10     # as every sibling smoother


@dataclass
class ForecastConfig:
    """The knobs every adapter shares; each adds its checkpoint and device on top.

    `stride` is the cost/purity dial: 1 is a true one-step-ahead forecast at a forward
    pass per reading, 32 (one patch) predicts a block at a time from the same strictly
    past context and costs ~32x less. Neither leaks the future; a larger stride only
    means the last slot of a block is predicted from slightly older history.
    """
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    cadence_min: float = CADENCE_MIN
    context: int = CONTEXT
    stride: int = STRIDE
    min_context: int = MIN_CONTEXT


def _grid(minutes: np.ndarray, values: np.ndarray, cadence: float):
    """An irregular series -> (regular grid with NaN holes, each reading's slot).

    The origin is the chunk's own first reading, so the grid a prefix run builds is a
    prefix of the grid a full run builds — which is what makes the result
    truncation-exact.
    """
    slot = np.rint((minutes - minutes[0]) / cadence).astype(np.int64)
    np.maximum.accumulate(slot, out=slot)        # never step backwards on jitter
    grid = np.full(int(slot[-1]) + 1, np.nan, dtype=np.float64)
    grid[slot] = values                          # a collision keeps the later reading
    return grid, slot


def _blocks(grid: np.ndarray, context: int, stride: int, min_context: int):
    """The (start slot, context window) of every block, oldest first.

    A block predicts `grid[start : start+stride]` from observations before `start` only.
    Contexts never contain a prediction, so all blocks are built up front and sent to
    the model together.
    """
    observed = np.cumsum(np.isfinite(grid))
    ready = np.flatnonzero(observed >= min_context)
    if not len(ready):
        return []
    return [(k, grid[max(0, k - context):k])
            for k in range(int(ready[0]) + 1, len(grid), stride)]


def predict_series(minutes: np.ndarray, values: np.ndarray, forecast, levels,
                   cfg: ForecastConfig):
    """Causal block-ahead prediction for one chunk, aligned to the input rows.

    `forecast(contexts, horizon) -> (median, quantiles | None)` is the injected model,
    which is what lets all of this be tested with a two-line stub and no weights.

    Returns (predicted, tail) each the length of `values`, NaN before enough history has
    accumulated.
    """
    n = len(values)
    pred_out = np.full(n, np.nan)
    tail_out = np.full(n, np.nan)
    if n < 2:
        return pred_out, tail_out
    grid, slot = _grid(minutes, values, cfg.cadence_min)
    blocks = _blocks(grid, cfg.context, cfg.stride, cfg.min_context)
    if not blocks:
        return pred_out, tail_out

    # Left-pad every context to one width so they batch; NaN is the models' own
    # "missing", so the padding says nothing and is not a filled value.
    width = max(len(c) for _, c in blocks)
    batch = np.full((len(blocks), width), np.nan)
    for i, (_, c) in enumerate(blocks):
        if len(c):
            batch[i, width - len(c):] = c

    median, quant = forecast(batch, cfg.stride)
    median = np.asarray(median, dtype=float).reshape(len(blocks), cfg.stride)

    pred = np.full(len(grid), np.nan)
    tail = np.full(len(grid), np.nan)
    for i, (start, _) in enumerate(blocks):
        end = min(start + cfg.stride, len(grid))
        take = end - start
        pred[start:end] = median[i, :take]
        if quant is not None:
            q = np.asarray(quant, dtype=float)[i, :take]       # (take, n_levels)
            obs = grid[start:end]
            # Where in the predictive distribution the real value fell: the share of
            # quantile levels at or below it. A calibrated model puts this uniform on
            # [0, 1], so a value near 0 or 1 is a genuine surprise.
            with np.errstate(invalid="ignore"):
                below = (q <= obs[:, None]).sum(axis=1) / max(len(levels), 1)
            tail[start:end] = np.where(np.isfinite(obs), below, np.nan)
    return pred[slot], tail[slot]


def forecast_frame(df, forecast, levels, patient_col: str, timestamp_col: str,
                   value_col: str, cfg: ForecastConfig, out_col: str,
                   tail_col: str, surprise_col: str):
    """The forecast-step contract, driven by an injected `forecast`.

    Per patient: segment at >=180-minute gaps (H8), run each chunk independently so no
    state crosses a gap, and clip the prediction into the patient's own observed range
    (H1 positivity, H10 per-patient).
    """
    missing = {patient_col, timestamp_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan
    work[tail_col] = np.nan

    for _, pdf in work.groupby(patient_col, sort=False):
        t_min = (pdf[timestamp_col] - pdf[timestamp_col].iloc[0]
                 ).dt.total_seconds().to_numpy() / 60.0
        v = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        cid = pdf["chunk_id"].to_numpy()
        observed = np.isfinite(v) & (v > 0)
        if observed.sum() < cfg.min_segment_rows:
            continue
        lo, hi = float(v[observed].min()), float(v[observed].max())

        pred = np.full(v.shape, np.nan)
        tail = np.full(v.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.flatnonzero(sel)
            p, t = predict_series(t_min[idx], v[idx], forecast, levels, cfg)
            pred[idx], tail[idx] = p, t                       # H8: fresh state
        pred = np.where(np.isfinite(pred), np.clip(pred, lo, hi), np.nan)
        work.loc[pdf.index, out_col] = pred
        work.loc[pdf.index, tail_col] = tail

    work[surprise_col] = (pd.to_numeric(work[value_col], errors="coerce")
                          - work[out_col])
    return work


if __name__ == "__main__":
    def persistence(contexts, horizon):
        """Stub: repeat each context's last observed value, with a flat +-5 spread."""
        med = np.empty((len(contexts), horizon))
        for i, c in enumerate(contexts):
            seen = c[np.isfinite(c)]
            med[i] = seen[-1] if len(seen) else 0.0
        levels = np.linspace(-5, 5, 21)
        return med, med[:, :, None] + levels[None, None, :]

    levels = tuple(np.linspace(0.01, 0.99, 21))
    cfg = ForecastConfig(min_context=64, stride=32)

    rng = np.random.default_rng(0)
    n = 400
    minutes = np.cumsum(rng.integers(8, 13, n).astype(float))
    values = 40 + 5 * np.sin(minutes / 300) + rng.normal(0, 1, n)

    p, t = predict_series(minutes, values, persistence, levels, cfg)
    got = np.isfinite(p)
    assert got.sum() > n // 2 and not got[:50].any(), got.sum()
    assert np.nanmin(t) >= 0.0 and np.nanmax(t) <= 1.0, (np.nanmin(t), np.nanmax(t))
    print(f"predicted {got.sum()} of {n} rows, first at {np.argmax(got)}; "
          f"tail in [{np.nanmin(t):.2f}, {np.nanmax(t):.2f}]")

    # H9, the one that matters: a future value must not move an earlier prediction.
    future = values.copy()
    future[300:] = 999.0
    q, _ = predict_series(minutes, future, persistence, levels, cfg)
    same = np.array_equal(np.nan_to_num(p[:280], nan=-1), np.nan_to_num(q[:280], nan=-1))
    print(f"H9 no lookahead: a tail of 999s leaves rows 0-279 untouched -> "
          f"{'PASS' if same else 'FAIL'}")
    assert same

    pre, _ = predict_series(minutes[:250], values[:250], persistence, levels, cfg)
    dev = np.nanmax(np.abs(p[:250] - pre))
    print(f"truncation-exact: max|full-prefix| = {dev:.2e} -> "
          f"{'PASS' if dev == 0 else 'FAIL'}")
    assert dev == 0

    # tail must flag a real outlier: a value far above the predicted spread -> 1.0
    spike = values.copy()
    spike[200] = 500.0
    _, ts = predict_series(minutes, spike, persistence, levels, cfg)
    assert ts[200] == 1.0, ts[200]
    print(f"tail: a 500 against a ~40 baseline reads {ts[200]:.2f} (top of the "
          f"distribution) -> PASS")

    # H8: a >=180-min gap starts a new chunk with its own warm-up
    t0 = pd.Timestamp("2026-01-01 08:00")
    rows = []
    for chunk in range(2):
        tt = t0 + pd.Timedelta(hours=100 * chunk)
        for k in range(150):
            tt += pd.Timedelta(minutes=10)
            rows.append({"patient_id": "A", "timestamp": tt,
                         "hrvValue": 40.0 + (10.0 if chunk else 0.0)})
    demo = pd.DataFrame(rows)
    res = forecast_frame(demo, persistence, levels, "patient_id", "timestamp",
                         "hrvValue", cfg, "predicted_hrv", "tail_hrv", "surprise_hrv")
    assert res["chunk_id"].nunique() == 2, res["chunk_id"].nunique()
    s = res["predicted_hrv"].dropna()
    assert (s > 0).all(), "H1: the prediction must be strictly positive"
    assert res["predicted_hrv"].iloc[150:214].isna().all(), \
        "the second chunk must warm up on its own, not inherit the first"
    assert res["surprise_hrv"].abs().max() < 1e-9, "a flat series should not surprise"
    print(f"H8 chunk reset: {res['chunk_id'].nunique()} chunks, second warms up alone, "
          f"H1 min={s.min():.2f}, flat series surprise=0 -> PASS")
    print("\nALL FORECAST BASE CHECKS PASS")
