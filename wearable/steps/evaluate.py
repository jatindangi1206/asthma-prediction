"""Step: evaluate. Scores an artifact with every applicable discovered metric.

    model predictions -> every KIND="model" and KIND="event" metric
    a smooth artifact  -> every KIND="signal" metric   (raw vs smoothed, no model needed)

    output  table `metrics`           metric, label, value, lo, hi, higher_is_better
            table `calibration`       predicted vs observed rate per probability band
            table `metrics_by_group`  the same metrics per cohort group

Depends on no other step: it is handed an artifact and works out which metrics apply.
Adding a metric is one file in `wearable/metrics/` — it then appears here and in every
comparison table with no change to this file.

A prediction is scored three ways, because one number cannot answer the clinical
question on its own:

  per day      every KIND="model" metric, at the fixed `config.THRESHOLD`
  per episode  every KIND="event" metric — an episode is caught if ANY of its days was
               flagged, which is what a clinician means by catching an exacerbation
  per band     the calibration table: does a day given probability 0.3 happen 30% of
               the time? Ranking (AUROC) says nothing about this

`lo`/`hi` are a 95% interval from resampling TEST PATIENTS with replacement
(`config.BOOT` draws), not days: days within a patient are dependent, so resampling
days would report an interval several times too narrow. Only `config.BOOT_METRICS` get
an interval — the rest are point estimates, as in the reference design. A patient drawn
twice contributes two distinct episodes, so an episode cannot be counted as caught
twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable import metrics as metric_pool
from wearable import store
from wearable.config import BOOT, BOOT_METRICS, CALIBRATION_BANDS, THRESHOLD
# The one definition of how to read the `event` column, shared with the event metrics.
from wearable.metrics._events import episode_ids

STEP = "evaluate"
PRED_KINDS = ("model", "event")       # the kinds a predictions table can be scored with
# This step's own logic version, mixed into the key for a predictions artifact. The
# cache key hashes a METHOD's source file, never a step's, so scoring changes that do
# not touch a metric file — a new table, a different interval — would otherwise reuse a
# stale result. Bump it when the scoring output changes. Signal scoring deliberately
# keeps its original key so stored smoothing results stay valid.
PRED_LOGIC = "eval2"


def _rows(kind: str, call) -> list[dict]:
    out = []
    for name, spec in metric_pool.all_metrics(kind).items():
        if spec.label.startswith("<unavailable"):
            continue
        mod = spec.load()
        try:
            value = float(call(mod.compute))
        except Exception as exc:
            value = float("nan")
            print(f"  ! metric {name} failed: {type(exc).__name__}: {exc}")
        out.append({"metric": name, "kind": kind,
                    "label": getattr(mod, "LABEL", name), "value": value,
                    "higher_is_better": bool(getattr(mod, "HIGHER_IS_BETTER", True))})
    return out


def _prediction_rows(table: pd.DataFrame) -> list[dict]:
    """Every day-level and episode-level metric on one predictions frame."""
    y, p = table["y"].to_numpy(), table["y_prob"].to_numpy()
    return (_rows("model", lambda f: f(y, p))
            + _rows("event", lambda f: f(table)))


def _value(name: str, y, p, frame) -> float:
    """One metric by name, given both the arrays and the frame it may need."""
    spec = metric_pool.metric(name)
    try:
        f = spec.load().compute
        return float(f(frame) if spec.kind == "event" else f(y, p))
    except Exception:
        return float("nan")


def _intervals(table: pd.DataFrame, boot: int, seed: int = 0) -> dict:
    """95% interval per metric in BOOT_METRICS, resampling test patients."""
    names = [n for n in BOOT_METRICS if n in metric_pool.names()]
    if boot <= 0 or not names or table.empty:
        return {}
    y = table["y"].to_numpy().astype(int)
    p = table["y_prob"].to_numpy(dtype=float)
    # Episodes as integer codes: distinctness is all an event metric needs, and ints let
    # a draw relabel them without touching strings. -1 on a day with no episode — which
    # is why the ids come from the shared helper and not from a `fillna("")`, whose
    # blanks would factorize into an episode of their own.
    ids = episode_ids(table)
    codes = (pd.factorize(ids.where(y == 1))[0] if ids is not None
             else np.full(len(y), -1))
    n_codes = int(codes.max()) + 1 if len(codes) and codes.max() >= 0 else 1

    rows = [np.flatnonzero(table["p_id"].to_numpy() == q) for q in table["p_id"].unique()]
    sizes = np.array([len(r) for r in rows])
    rng = np.random.default_rng(seed)
    draws: list[dict] = []
    for _ in range(int(boot)):
        pick = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[i] for i in pick])
        # a patient drawn twice: shift its episode codes so they stay separate episodes
        shift = np.repeat(np.arange(len(pick)) * n_codes, sizes[pick])
        c = codes[idx]
        frame = pd.DataFrame({"y": y[idx], "y_prob": p[idx],
                              "event": np.where(c >= 0, c + shift, -1)})
        draws.append({n: _value(n, y[idx], p[idx], frame) for n in names})
    B = pd.DataFrame(draws)
    out = {}
    for n in names:
        col = B[n]
        out[n] = ((float(np.nanpercentile(col, 2.5)), float(np.nanpercentile(col, 97.5)))
                  if col.notna().any() else (float("nan"), float("nan")))
    return out


def _calibration(table: pd.DataFrame, bands: int = CALIBRATION_BANDS) -> pd.DataFrame:
    """Predicted vs observed exacerbation rate per probability band."""
    y = table["y"].to_numpy().astype(int)
    p = table["y_prob"].to_numpy(dtype=float)
    b = np.minimum((p * bands).astype(int), bands - 1)
    rows = []
    for k in range(bands):
        m = b == k
        rows.append({"band": f"{k / bands:.1f}-{(k + 1) / bands:.1f}",
                     "n_days": int(m.sum()), "n_pos_days": int(y[m].sum()),
                     "predicted": float(p[m].mean()) if m.any() else float("nan"),
                     "observed": float(y[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def _by_group(table: pd.DataFrame) -> pd.DataFrame:
    """The same metrics per cohort group — point estimates, no intervals.

    Each group reports its OWN day, positive and episode counts: with 125 exacerbation
    days among the adults and 8 among the children, a group row carrying the cohort's
    totals would be unreadable.
    """
    if "group" not in table.columns:
        return pd.DataFrame()
    rows = []
    for g, part in table.groupby("group", sort=True):
        if not str(g):
            continue
        ids = episode_ids(part)
        n_events = (int(ids[part["y"].astype(int) == 1].nunique())
                    if ids is not None else 0)
        for r in _prediction_rows(part):
            rows.append({"group": str(g), "n_days": len(part),
                         "n_pos_days": int(part["y"].sum()),
                         "n_events": n_events,
                         "n_patients": int(part["p_id"].nunique()), **r})
    return pd.DataFrame(rows)


def run(artifact, force: bool = False, boot: int | None = None):
    """Pick the metric kind from what the artifact actually holds."""
    if artifact.has_table("predictions"):
        kind, table = "model", artifact.read_table("predictions")
    elif artifact.has_table("smoothed"):
        kind, table = "signal", artifact.read_table("smoothed")
    else:
        raise ValueError(f"nothing to evaluate in {artifact.step}/{artifact.method}: "
                         f"tables = {artifact.manifest.get('tables')}")

    name = f"{artifact.step}:{artifact.method}"
    if kind == "model":
        boot = BOOT if boot is None else int(boot)
        applicable = [k for k in PRED_KINDS]
        metric_names = [n for k in applicable for n in metric_pool.names(k)]
        params = {"kind": kind, "metrics": metric_names, "threshold": THRESHOLD,
                  "boot": boot}
        code_hash = PRED_LOGIC + "-" + "-".join(sorted(
            m.source_hash[:8] for k in applicable
            for m in metric_pool.all_metrics(k).values()))
    else:
        # Unchanged from before, so stored signal results stay valid and are reused.
        params = {"kind": kind, "metrics": metric_pool.names(kind)}
        code_hash = "-".join(sorted(
            m.source_hash[:8] for m in metric_pool.all_metrics(kind).values()))

    inputs = [artifact.ref]
    if not force:
        hit = store.find(STEP, name, params, inputs, code_hash)
        if hit is not None:
            return hit

    extra_tables: dict[str, pd.DataFrame] = {}
    if kind == "model":
        rows = _prediction_rows(table)
        ci = _intervals(table, boot)
        for r in rows:
            r["lo"], r["hi"] = ci.get(r["metric"], (float("nan"), float("nan")))
        extra_tables["calibration"] = _calibration(table)
        groups = _by_group(table)
        if len(groups):
            extra_tables["metrics_by_group"] = groups
    else:
        # Judge the method directly: raw against smoothed, per patient, then averaged
        # so one long-recording patient cannot dominate.
        per = []
        for _, g in table.groupby("p_id", sort=True):
            per.append({r["metric"]: r["value"] for r in
                        _rows("signal", lambda f, g=g: f(g["hrvValue"], g["smoothed_hrv"]))})
        mean = pd.DataFrame(per).mean(numeric_only=True)
        rows = _rows("signal", lambda f: np.nan)      # shape + metadata
        for r in rows:
            r["value"] = float(mean.get(r["metric"], np.nan))

    art = store.open_artifact(STEP, name, params, inputs, code_hash,
                              label=f"{kind} metrics for {name}")
    out = pd.DataFrame(rows)
    art.write_table("metrics", out)
    for table_name, frame in extra_tables.items():
        art.write_table(table_name, frame)
    art.complete(kind=kind, n_metrics=len(out),
                 **{r["metric"]: r["value"] for r in rows})
    return art
