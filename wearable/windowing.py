"""The window rule — the ONE implementation, as `chunking.py` is for the H8 gap rule.

`join` and `encode` both need "the classified day plus the N-1 days before it", and a
foundation model needs that window as an ordered series. Duplicating the rule in two
steps is exactly what `chunking.py` exists to prevent, so it lives here and both import
it. Never redefine these; import them.

    day_profile(t, prefix, bins)   one day -> `bins` time-of-day slots of raw signal
    lag_window(feat, days)         per-day features -> `_d<k>` lag columns
    ordered_series(frame)          those columns -> (n, days*bins), oldest slot first

THE CONTEXT BUDGET
------------------
`days * bins` IS the context length a sequence or foundation model receives, which makes
the two knobs one decision rather than two:

    MOMENT      exactly 512 timesteps (padded/truncated; its positional embeddings are
                pretrained at that length)
    Mantis      a multiple of 32, pretrained at 512
    Timestral   up to 32,768 (1,024 patches of 32)

So the window is not a free choice once a foundation model is involved. At HRV's ~55
readings a day, `days=7 bins=55` is 385 points and fits 512 natively. At heart rate's
~744 a day it is 5,208 — ten times over budget, so HR must be binned coarser
(`bins=73` gives 511). Changing `days` against a fixed ceiling trades history length
for time resolution; it does not change how much the model is handed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from wearable.config import CHANNELS

ID = ["p_id", "date"]
# The channel a day profile is built from: whichever known value column the table
# carries, so profiling an `hr` artifact needs no extra argument.
PROFILE_COLS = tuple(CHANNELS.values())
_PROFILE_RE = re.compile(r"_p(\d+)(?:_d(\d+))?$")


def profile_col(t: pd.DataFrame) -> str | None:
    # a smoother's table carries the raw reading beside its estimate; profile the estimate
    smoothed = [c for c in t.columns if str(c).startswith("smoothed_")]
    return smoothed[0] if smoothed else next((c for c in PROFILE_COLS if c in t.columns), None)


def day_profile(t: pd.DataFrame, prefix: str, bins: int,
                col: str | None = None) -> pd.DataFrame | None:
    """Each day's signal binned into `bins` equal time-of-day slots.

    This is what turns a day from four summary statistics into an actual time series.
    A slot with no reading is NaN and stays NaN; nothing is interpolated — the models
    that accept a mask are told where the holes are, and the ones that do not are the
    ones whose results need reading with suspicion.

    Indexed by (p_id, date), so it concatenates with the other day-feature blocks.
    """
    col = col or profile_col(t)
    if col is None or col not in t.columns or bins <= 0:
        return None
    slot = (t["createdTime"].dt.hour * 60 + t["createdTime"].dt.minute) * bins // 1440
    g = (t.assign(_slot=slot, _v=pd.to_numeric(t[col], errors="coerce"))
         .groupby([*ID, "_slot"])["_v"].mean()
         .unstack("_slot").reindex(columns=range(bins)))
    g.columns = [f"{prefix}_p{k:03d}" for k in range(bins)]
    return g


def lag_window(feat: pd.DataFrame, days: int) -> pd.DataFrame:
    """The classified day plus the `days-1` days before it, as `_d<k>` lag columns.

    Day d-k is attached by dating its row forward k days and merging on (p_id, date) —
    so a day missing from the recording contributes NaN rather than silently shifting an
    older day into its place, which a `shift(k)` on row order would do.

    `days=1` is returned untouched, column names and all: the one-day dataset is the
    dataset this repo produced before the window existed.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    if days == 1:
        return feat
    cols = [c for c in feat.columns if c not in ID]
    out = feat.rename(columns={c: f"{c}_d0" for c in cols})
    for k in range(1, days):
        prev = feat.copy()
        prev["date"] = prev["date"] + pd.Timedelta(days=k)
        out = out.merge(prev.rename(columns={c: f"{c}_d{k}" for c in cols}),
                        on=ID, how="left")
    return out


def split_profile_columns(frame: pd.DataFrame):
    """-> (context columns oldest-first, the classified day's columns, everything else).

    The window is recovered from the column NAMES, so no step has to tell another what
    shape it used. `_d0` is the classified day and is the target of a forecast; `_d1`
    and older are the context a forecast may see.
    """
    prof, other = [], []
    for c in frame.columns:
        m = _PROFILE_RE.search(str(c))
        if m:
            prof.append((int(m.group(2) or 0), int(m.group(1)), c))
        else:
            other.append(c)
    prof.sort(key=lambda t: (-t[0], t[1]))      # oldest day first, slot ascending
    return ([c for k, _, c in prof if k >= 1],
            [c for k, _, c in prof if k == 0], other)


def ordered_series(frame: pd.DataFrame) -> np.ndarray:
    """Profile columns -> (n, days*bins) in true time order, NaN where unobserved.

    Every slot of the window, oldest first, including the classified day — this is what
    an ENCODER embeds. A forecaster wants the two halves separately and calls
    `split_profile_columns` instead.
    """
    ctx, tgt, _ = split_profile_columns(frame)
    cols = ctx + tgt
    if not cols:
        raise ValueError(
            "no profile columns in this table. A sequence model reads the day itself, "
            "not summary statistics: build it with `--profile <bins> --days <n>` "
            "(see wearable/windowing.py on the context budget)")
    return frame[cols].to_numpy(dtype=float)


def channel_series(frame: pd.DataFrame):
    """Profile columns -> (names, (n, channels, time)) — one real series per channel.

    `ordered_series` flattens every profile column into ONE series, which is right for a
    single-channel model. A multivariate model needs the channels kept apart, so this
    groups the columns by the prefix before `_p<slot>` and, inside each, orders by
    (oldest day first, slot ascending). The result is the actual recorded series per
    channel, at `days * bins` points.

    Do not confuse this with `models/_seq.sequence`, which reads the SAME columns as
    (days, features) — time is the day there, because `rnn`/`lstm` consume a sequence of
    days. Here time is the reading.
    """
    per: dict[str, dict[tuple[int, int], str]] = {}
    for c in frame.columns:
        m = re.fullmatch(r"(.*)_p(\d+)(?:_d(\d+))?", str(c))
        if m:
            per.setdefault(m.group(1), {})[(int(m.group(3) or 0), int(m.group(2)))] = c
    if not per:
        raise ValueError(
            "no profile columns in this table. A multivariate model reads the recorded "
            "series, not day summaries: build it with `--profile <bins> --days <n>`")
    names = sorted(per)
    width = max(len(v) for v in per.values())
    out = np.full((len(frame), len(names), width), np.nan, dtype=np.float32)
    for j, name in enumerate(names):
        # oldest day first, slots ascending inside each day
        order = sorted(per[name], key=lambda k: (-k[0], k[1]))
        block = frame[[per[name][k] for k in order]].to_numpy(dtype=float)
        out[:, j, :block.shape[1]] = block
    return names, out


def context_length(days: int, bins: int) -> int:
    """What a model will actually receive. Named so the arithmetic is never implicit."""
    return int(days) * int(bins)


def bins_for(days: int, ceiling: int) -> int:
    """The coarsest binning that fits `days` of history into a model's fixed ceiling.

    `bins_for(7, 512) == 73`, so a 7-day window costs 511 of MOMENT's 512 slots.
    """
    if days < 1 or ceiling < 1:
        raise ValueError(f"days and ceiling must be >= 1, got {days}, {ceiling}")
    return max(1, int(ceiling) // int(days))


if __name__ == "__main__":
    # The arithmetic above is the whole reason this module exists; check it.
    assert context_length(7, 144) == 1008
    assert bins_for(7, 512) == 73 and context_length(7, 73) == 511
    assert bins_for(1, 512) == 512
    assert bins_for(512, 512) == 1

    t = pd.DataFrame({
        "p_id": "a", "date": pd.Timestamp("2026-01-01"),
        "createdTime": pd.date_range("2026-01-01 00:00", periods=288, freq="10min"),
        "hrvValue": np.arange(288.0)})
    t["date"] = t["createdTime"].dt.normalize()
    prof = day_profile(t, "raw", 144)
    assert prof is not None and prof.shape == (2, 144), prof.shape
    assert prof.iloc[0]["raw_p000"] == 0.0 and prof.iloc[0]["raw_p143"] == 143.0

    feat = pd.DataFrame({
        "p_id": ["a"] * 3,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "v_p000": [1.0, 2.0, 3.0]})
    w = lag_window(feat, 3).set_index(["p_id", "date"])
    row = w.loc[("a", pd.Timestamp("2026-01-03"))]
    assert (row["v_p000_d0"], row["v_p000_d1"], row["v_p000_d2"]) == (3.0, 2.0, 1.0)

    ctx, tgt, other = split_profile_columns(w.reset_index())
    assert tgt == ["v_p000_d0"] and ctx == ["v_p000_d2", "v_p000_d1"], (ctx, tgt)
    assert set(other) == {"p_id", "date"}
    s = ordered_series(w.reset_index())
    assert s.shape == (3, 3) and list(s[2]) == [1.0, 2.0, 3.0], s
    # two channels, 2 days x 2 slots each: channels stay apart, time is the reading
    mc = pd.DataFrame({"a_p000_d1": [1.0], "a_p001_d1": [2.0],
                       "a_p000_d0": [3.0], "a_p001_d0": [4.0],
                       "b_p000_d1": [10.0], "b_p001_d1": [20.0],
                       "b_p000_d0": [30.0], "b_p001_d0": [40.0]})
    names, cs = channel_series(mc)
    assert names == ["a", "b"] and cs.shape == (1, 2, 4), (names, cs.shape)
    assert list(cs[0, 0]) == [1.0, 2.0, 3.0, 4.0], cs[0, 0]
    assert list(cs[0, 1]) == [10.0, 20.0, 30.0, 40.0], cs[0, 1]
    print("windowing: context budget, profile bins, lag order, series order and "
          "per-channel series all OK")
