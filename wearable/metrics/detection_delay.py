"""Days AFTER an episode begins before the model first flags it — the detection delay.

The mirror of `lead_days`. Both read the onset date out of the event id, which is
`<pid>_<onset date>` (for example `a001_20251203`), so both need the predictions frame
rather than the two arrays.

    lead_days        onset date  -  first flagged day      warning, so higher is better
    detection_delay  first flagged day  -  onset date      lateness, so LOWER is better

On the `detection` task the flagged day lies inside the episode, so the delay is >= 0
and says how many days of the episode went by unflagged — 0 means the first day was
caught. On `onset` the flag comes before the episode, so the delay is negative and
`lead_days` is the number to read instead.

NaN when no episode was caught, and NaN inside a bootstrap draw, whose integer episode
codes carry no date — which is why this is a point estimate and not in
`config.BOOT_METRICS`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable.metrics._events import positive_events

KIND = "event"
LABEL = "Detection delay, days (median)"
HIGHER_IS_BETTER = False


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
    delay = (pd.DatetimeIndex(first.to_numpy()) - pd.DatetimeIndex(onset)).days.to_numpy(float)
    delay = delay[np.isfinite(delay)]
    return float(np.median(delay)) if len(delay) else float("nan")
