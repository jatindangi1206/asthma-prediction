"""Median days from an episode's first flagged day to its onset — the warning time.

The onset date is read from the event id, which is `<pid>_<onset date>` (for example
`a001_20251203`), so this needs the predictions frame and not the two arrays.

Read it against the task. On `onset` it is the warning: 1-3 days before a new episode
begins, so higher is better. On `detection` the flagged day lies inside the episode, so
the value is <= 0 and says how deep into the episode the first flag fell (0 = the onset
day itself). NaN when no episode was caught, and NaN inside a bootstrap draw, whose
integer episode codes carry no date — which is why this metric reports a point estimate
and is not in `config.BOOT_METRICS`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable.metrics._events import positive_events

KIND = "event"
LABEL = "Lead time, days (median)"
HIGHER_IS_BETTER = True


def compute(predictions):
    if "date" not in predictions.columns:
        return float("nan")
    ids, flag = positive_events(predictions)
    if ids is None or not len(ids):
        return float("nan")
    hit = pd.DataFrame({"event": ids[flag],
                        "date": pd.to_datetime(predictions.loc[ids.index, "date"])[flag]})
    if hit.empty:
        return float("nan")
    first = hit.groupby("event")["date"].min()
    onset = pd.to_datetime(pd.Index(first.index).str.rsplit("_", n=1).str[1],
                           format="%Y%m%d", errors="coerce")
    lead = (pd.DatetimeIndex(onset) - pd.DatetimeIndex(first.to_numpy())).days.to_numpy(float)
    lead = lead[np.isfinite(lead)]
    return float(np.median(lead)) if len(lead) else float("nan")
