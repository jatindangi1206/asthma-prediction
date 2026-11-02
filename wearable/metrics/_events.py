"""Episode ids off a predictions frame — shared by the KIND="event" metrics.

A module whose name starts with `_` is skipped by the registry, so this is a helper and
not a metric.

The `event` column arrives in two shapes and both must work:

  strings   a real predictions table: `<pid>_<onset date>` on a day inside an episode,
            blank elsewhere — and a CSV round-trip turns those blanks into NaN
  integers  a bootstrap draw, which relabels episodes as codes so a patient drawn twice
            contributes two distinct episodes; -1 means no episode

Mixing those up is how a mask silently matches nothing, so the conversion lives here
once rather than in each metric.
"""
from __future__ import annotations

import pandas as pd

from wearable.config import THRESHOLD


def episode_ids(predictions) -> pd.Series | None:
    """The episode id per row as a nullable string, <NA> where there is no episode."""
    if "event" not in predictions.columns:
        return None
    ev = predictions["event"]
    if pd.api.types.is_numeric_dtype(ev):
        return ev.where(ev >= 0).astype("Int64").astype("string")
    ev = ev.astype("string")
    return ev.where(ev.notna() & (ev.str.len() > 0) & (ev != "nan"))


def positive_events(predictions):
    """-> (ids on positive days, flagged mask) with non-episode days dropped."""
    ids = episode_ids(predictions)
    if ids is None:
        return None, None
    y = predictions["y"].astype(int)
    flag = predictions["y_prob"].astype(float) >= THRESHOLD
    keep = (y == 1) & ids.notna()
    return ids[keep], flag[keep]
