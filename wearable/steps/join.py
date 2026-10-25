"""Step: join. Merge any results you already have into one modelling dataset, and split it.

    input   any number of smooth / detect artifacts (or none, for a raw-only baseline)
    output  tables `train` and `test`:
            p_id, date, <features from each input>, y
            plus `split` recording which patient went where

Each input artifact is aggregated to one row per patient-day and merged on (p_id, date).
Day-level ground truth comes from the labels file.

**The split is computed from the patient universe and the seed alone** — never from
feature content or from which artifacts were joined. So every variant in a comparison
sees exactly the same test participants, which is what makes the comparison fair. The
universe is every patient in the labels file, so a method that drops patients yields
fewer rows but cannot shift the split.

Patients are split, never days: no patient appears in both train and test.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from wearable import store
from wearable.config import LABELS_CSV, SEED, TEST_FRACTION

STEP = "join"
TARGETS = {
    # name        -> (label column, validity column)
    "detection": ("label", "sample"),
    "onset": ("onset_label", "onset_valid"),
}
_ID = ["p_id", "date"]
# Never aggregated as features: identifiers, strings, and the labels themselves.
_SKIP = {"p_id", "date", "createdTime", "chunk_id", "label", "sample", "exac_day",
         "episode", "onset_valid", "onset_label", "onset_id", "y"}


def patient_split(patients, seed: int = SEED, test_fraction: float = TEST_FRACTION):
    """Deterministic patient-level split from the id list and seed only.

    Hashing each id with the seed (rather than shuffling) means a patient's side is
    stable even if the universe changes — adding a cohort cannot move existing
    patients between train and test.
    """
    ids = sorted(set(map(str, patients)))
    def _u(pid):                       # stable uniform in [0,1) from (seed, pid)
        h = hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:16]
        return int(h, 16) / 16 ** 16
    return pd.DataFrame({"p_id": ids,
                         "split": ["test" if _u(p) < test_fraction else "train"
                                   for p in ids]})


def _day_features(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """One row per patient-day. Numeric columns -> mean/std/min/max; `*_anomaly` flags
    -> rows flagged plus distinct events (contiguous runs), which is the honest way to
    count a sustained-regime detector."""
    t = table.copy()
    t["createdTime"] = pd.to_datetime(t["createdTime"])
    t["date"] = t["createdTime"].dt.normalize()

    num = [c for c in t.columns
           if c not in _SKIP and pd.api.types.is_numeric_dtype(t[c])]
    flags = [c for c in num if c.endswith("_anomaly") or c == "anomaly_present"]
    plain = [c for c in num if c not in flags]

    out = []
    if plain:
        agg = t.groupby(_ID)[plain].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{prefix}_{c}_{s}" for c, s in agg.columns]
        out.append(agg)
    if flags:
        g = t.sort_values(["p_id", "createdTime"]).groupby(_ID)
        fl = g[flags].sum()
        fl.columns = [f"{prefix}_{c}_rows" for c in fl.columns]
        ev = g[flags].agg(lambda s: int(((s.to_numpy() == 1) &
                                        (np.r_[0, s.to_numpy()[:-1]] == 0)).sum()))
        ev.columns = [f"{prefix}_{c}_events" for c in flags]
        out.extend([fl, ev])
    n = t.groupby(_ID).size().rename(f"{prefix}_n_rows").to_frame()
    out.append(n)
    return pd.concat(out, axis=1).reset_index()


def _labels(target: str) -> pd.DataFrame:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"{LABELS_CSV} not found — run ingest with a labelled source first")
    y_col, valid_col = TARGETS[target]
    lab = pd.read_csv(LABELS_CSV, parse_dates=["date"],
                      dtype={"p_id": str, "episode": str, "onset_id": str})
    lab = lab[lab[valid_col] == 1].copy()          # only days this target can score
    lab["y"] = lab[y_col].astype(int)
    return lab[["p_id", "date", "y"]]


def run(artifacts=(), target: str = "detection", seed: int = SEED,
        test_fraction: float = TEST_FRACTION, force: bool = False,
        name: str | None = None):
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}, got {target!r}")
    artifacts = list(artifacts)

    # The artifact's name describes the combination, so variants sit side by side.
    name = name or ("+".join(f"{a.step}:{a.method}" for a in artifacts) or "raw_only")
    params = {"target": target, "seed": seed, "test_fraction": test_fraction,
              "n_inputs": len(artifacts)}
    inputs = [a.ref for a in artifacts]
    code_hash = "join1"            # this step's own logic version

    if not force:
        hit = store.find(STEP, name, params, inputs, code_hash)
        if hit is not None:
            print(f"[join/{name}] reusing {hit.key[:12]}")
            return hit

    y = _labels(target)
    # Universe = every patient the labels know about, independent of the inputs.
    split = patient_split(y["p_id"].unique(), seed, test_fraction)

    frame = y
    for a in artifacts:
        tbl = a.read_table(a.manifest["tables"][0])
        tbl["p_id"] = tbl["p_id"].astype(str)
        feats = _day_features(tbl, prefix=f"{a.step[:2]}_{a.method}".replace("+", "_"))
        frame = frame.merge(feats, on=_ID, how="left")

    if len(artifacts) == 0:
        # Raw-only baseline: features straight off the ingested signal.
        from wearable.steps.detect import _load_input
        raw = _load_input(None, "raw")
        raw["p_id"] = raw["p_id"].astype(str)
        frame = frame.merge(_day_features(raw, "raw"), on=_ID, how="left")

    frame = frame.merge(split, on="p_id", how="left")
    frame = frame[frame["split"].notna()]

    art = store.open_artifact(STEP, name, params, inputs, code_hash, label=name)
    feature_cols = [c for c in frame.columns if c not in {"p_id", "date", "y", "split"}]
    for part in ("train", "test"):
        art.write_table(part, frame[frame["split"] == part]
                        .drop(columns="split").reset_index(drop=True))
    art.write_table("split", split)

    tr = frame[frame["split"] == "train"]
    te = frame[frame["split"] == "test"]
    art.manifest["feature_cols"] = feature_cols
    art.complete(target=target, n_features=len(feature_cols),
                 n_train=len(tr), n_test=len(te),
                 n_train_patients=tr["p_id"].nunique(),
                 n_test_patients=te["p_id"].nunique(),
                 train_positives=int(tr["y"].sum()), test_positives=int(te["y"].sum()))
    print(f"\n[join/{name}] target={target}")
    print(f"  train {len(tr):,} days / {tr['p_id'].nunique()} patients "
          f"({int(tr['y'].sum())} positive)")
    print(f"  test  {len(te):,} days / {te['p_id'].nunique()} patients "
          f"({int(te['y'].sum())} positive)")
    print(f"  {len(feature_cols)} features, key {art.key[:12]}")
    return art
