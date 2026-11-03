"""Compare the two CPD threshold calibrations, "shuffle" (whole days of the patient's
own past, resampled) and "sts" (a structural time-series model fitted on that past,
simulated on its own observation mask) — see wearable/cpd/_sts.py.

For each channel, takes the newest complete detect artifact of each calibration (same
ensemble, same budget) and reports:

  fitted STS parameters       median [IQR] over patients (each patient's median over
                              its recalibrations), and how often each model was used
  thresholds                  per detector statistic, median [IQR] over recalibrations
  null alarm rate             achieved on the no-change data, against its target
  real alarm rate             events per armed worn day on the patients' own data,
                              per detector and for the ensemble, and time flagged

Writes data/reports/cpd_calibration/<signal>_{params,thresholds,rates}.csv.

    python scripts/cpd_calibration_compare.py [--budget 0.3333] [--signal hrv hr]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wearable import config, cpd                                   # noqa: E402
from wearable.cpd import _whiten as W                              # noqa: E402

RUNS = config.RUNS_DIR / "detect"
OUT = config.REPORTS_DIR / "cpd_calibration"


def latest(signal, calibration, budget):
    name = "+".join(cpd.DEFAULT_ENSEMBLE) + ("" if signal == config.DEFAULT_CHANNEL else f"@{signal}")
    best = None
    for m in (RUNS / name).glob("*/manifest.json"):
        man = json.loads(m.read_text())
        p = man.get("params", {})
        if man.get("status") != "complete" or abs(p.get("budget", -1) - budget) > 1e-6:
            continue
        if p.get("calibration", "shuffle") != calibration or p.get("limit_patients"):
            continue
        if best is None or man["created"] > best[1]["created"]:
            best = (m.parent, man)
    return best


def q(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.4g} [{s.quantile(.25):.4g}–{s.quantile(.75):.4g}]" if len(s) else "—"


def real_rates(path, step_min):
    cols = ["p_id", "armed", "anomaly_present"] + [f"{m}_anomaly" for m in cpd.DEFAULT_ENSEMBLE]
    t = pd.read_csv(path / "detected.csv", usecols=cols)
    a = t[t["armed"] == 1]
    rows = []
    for pid, g in a.groupby("p_id"):
        worn = len(g) * step_min / 1440
        r = {"p_id": pid, "worn_days": worn, "flagged": g["anomaly_present"].mean()}
        for c in ["anomaly_present"] + [f"{m}_anomaly" for m in cpd.DEFAULT_ENSEMBLE]:
            r[c] = W.events(g[c].to_numpy() == 1) / worn
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=config.CPD_BUDGET)
    ap.add_argument("--signal", nargs="+", default=["hrv", "hr"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    share = a.budget / len(cpd.DEFAULT_ENSEMBLE)
    for sig in a.signal:
        step_min = config.CPD_SETTINGS[sig]["detect"]["step_min"]
        found = {c: latest(sig, c, a.budget) for c in config.CPD_CALIBRATIONS}
        print(f"\n{'=' * 78}\n{sig.upper()}  budget {a.budget:.3g}/worn day "
              f"(per detector {share:.3g})\n{'=' * 78}")
        for c, f in found.items():
            print(f"  {c:8s} " + (f"{f[0].name[:12]}  {f[1]['stats'].get('calibration_used')}"
                                  if f else "no complete artifact"))
        cal = {}
        for c, f in found.items():
            if f and (f[0] / "calibration.csv").exists():
                cal[c] = pd.read_csv(f[0] / "calibration.csv")
            elif f:
                print(f"  ! {c}: artifact has no calibration table (built before it existed; "
                      f"rerun detect with force=True)")
        # fitted STS parameters
        if "sts" in cal:
            s = cal["sts"]
            pcols = [c for c in s if c.startswith("sts_") and c not in ("sts_model", "sts_exog")]
            per_patient = s[s["calibration"].str.startswith("sts")].groupby("p_id")[pcols].median()
            params = pd.DataFrame({"median [IQR] over patients": per_patient.apply(q)})
            params.to_csv(OUT / f"{sig}_params.csv")
            print("\n  STS model used per recalibration:",
                  s["calibration"].value_counts().to_dict())
            print("  STS exogenous terms:", s["sts_exog"].value_counts().to_dict())
            print("  fitted parameters (log scale; beta.x1/x2 = the listed exog in order):")
            print("   " + params.to_string().replace("\n", "\n   "))
        # thresholds and null rates, both methods
        th_rows = []
        for c, s in cal.items():
            for col in [k for k in s if k.startswith("thr_")]:
                key = col[4:]
                det = key.split("_")[0]
                n_streams = sum(1 for k in s if k.startswith(f"thr_{det}_"))
                nr = s.get(f"null_rate_{key}")
                th_rows.append({"calibration": c, "statistic": key, "threshold": q(s[col]),
                                "target": round(share / n_streams, 4),
                                "null_rate": q(nr) if nr is not None else "analytic"})
        if th_rows:
            th = pd.DataFrame(th_rows).pivot(index="statistic", columns="calibration")
            th.to_csv(OUT / f"{sig}_thresholds.csv")
            print("\n  thresholds and achieved no-change rate (median [IQR] over recalibrations):")
            print("   " + th.to_string().replace("\n", "\n   "))
        # real rates
        rr = []
        for c, f in found.items():
            if not f:
                continue
            r = real_rates(f[0], step_min)
            w = r["worn_days"]
            row = {"calibration": c, "patients": len(r), "worn_days": round(w.sum()),
                   "time flagged": round(float((r["flagged"] * w).sum() / w.sum()), 4)}
            for col, lab in [("anomaly_present", "ensemble")] + \
                    [(f"{m}_anomaly", m) for m in cpd.DEFAULT_ENSEMBLE]:
                row[f"{lab} pooled"] = round(float((r[col] * w).sum() / w.sum()), 3)
            row["ensemble per-patient median"] = round(float(r["anomaly_present"].median()), 3)
            rr.append(row)
        if rr:
            rates = pd.DataFrame(rr).set_index("calibration").T
            rates.to_csv(OUT / f"{sig}_rates.csv")
            print(f"\n  real alarm rates, events per armed worn day (target: ensemble "
                  f"{a.budget:.3g}, each detector {share:.3g} — on no-change data):")
            print("   " + rates.to_string().replace("\n", "\n   "))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
