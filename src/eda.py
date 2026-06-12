
"""src/eda.py — First-principles EDA of RAW observed HRV."""
import sys
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis, gaussian_kde

INPUT_DIR = Path("./data/processed")
OUT_DIR = Path("./data/results/eda")
GRID_MIN = 10
STEPS_DAY = 144
ROLL_WIN = 24

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["createdTime"] = pd.to_datetime(df["createdTime"])
    return df.sort_values("createdTime").reset_index(drop=True)

def analyze(df: pd.DataFrame, stem: str) -> dict:
    y = df["hrvValue"].dropna().to_numpy()
    if len(y) < ROLL_WIN:
        return {"patient": stem, "valid": False}

    # H2: KDE Bimodal check
    kde = gaussian_kde(y)
    x_grid = np.linspace(y.min(), y.max(), 100)
    peaks = len(find_peaks(kde(x_grid))[0])

    # H4: Non-stationary rolling variance
    roll_mu = pd.Series(y).rolling(ROLL_WIN, min_periods=2).mean().dropna()
    var_ratio = roll_mu.var() / y.var() if y.var() > 0 else 0

    # H5: Circadian Modulo
    hr = df["createdTime"].dt.hour
    prof = df.groupby(hr)["hrvValue"].median()
    amp = float(prof.max() - prof.min())

    # H7/H8: Sampling & Missingness
    dt = df["createdTime"].diff().dt.total_seconds() / 60.0
    pct_on_grid = float(((dt - GRID_MIN).abs() < 0.5).mean() * 100)
    nan_pct = float(df["hrvValue"].isna().mean() * 100)

    # H3: Welch on longest segment (approximating 1/f)
    obs = df["hrvValue"].notna()
    segs = [g for _, g in df[obs].groupby((obs != obs.shift()).cumsum()[obs])]
    long_y = max(segs, key=len)["hrvValue"].to_numpy() if segs else y
    if len(long_y) > 50:
        f, psd = welch(long_y, fs=1.0/GRID_MIN, nperseg=min(len(long_y)//4, 256))
        slope = float(np.polyfit(np.log10(f[f>0]), np.log10(psd[f>0]), 1)[0])
    else:
        slope = np.nan

    return {
        "patient": stem, "valid": True, "n": len(y),
        "min": float(y.min()), "max": float(y.max()),
        "median": float(np.median(y)), "std": float(y.std()),
        "kde_peaks": peaks, "var_ratio": var_ratio,
        "amp": amp, "kurt": float(kurtosis(y)),
        "pct_on_grid": pct_on_grid, "nan_pct": nan_pct,
        "slope": slope
    }

def analyze_file(path: Path) -> dict:
    return analyze(load(path), path.stem)

def plot_patient(df: pd.DataFrame, m: dict) -> None:
    if not m.get("valid"): return
    t, y = df["createdTime"], df["hrvValue"]
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"HRV First Principles — {m['patient']} | n={m['n']}", weight="bold")

    ax[0,0].scatter(t, y, s=2, c="gray", alpha=0.5)
    ax[0,0].plot(t, y.rolling(ROLL_WIN).mean(), c="red", lw=2)
    ax[0,0].set(title=f"H1/H4: Drift Ratio = {m['var_ratio']:.2f}")

    kde_x = np.linspace(m['min'], m['max'], 100)
    ax[0,1].plot(kde_x, gaussian_kde(y.dropna())(kde_x), c="blue")
    ax[0,1].set(title=f"H2: Probability Density (Peaks: {m['kde_peaks']})")

    prof = df.groupby(t.dt.hour)["hrvValue"].median()
    ax[0,2].bar(prof.index, prof.values, color="green", alpha=0.6)
    ax[0,2].set(title=f"H5: Modulo 24h (Amp: {m['amp']:.1f}ms)")

    ax[1,0].boxplot(y.dropna(), vert=False)
    ax[1,0].set(title=f"H6: Outlier Spread (Kurtosis: {m['kurt']:.1f})")

    if not np.isnan(m["slope"]):
        ax[1,1].text(0.5, 0.5, f"1/f Slope: {m['slope']:.2f}", fontsize=12)
    ax[1,1].set(title="H3: Spectral Decay")

    dt = t.diff().dt.total_seconds().dropna() / 60.0
    ax[1,2].hist(dt[dt < 120], bins=30, color="purple")
    ax[1,2].set(title=f"H7/H8: Sampling ($\Delta t$) | Grid: {m['pct_on_grid']:.0f}%")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{m['patient']}_eda.png", dpi=100)
    plt.close(fig)

def verdicts(d: pd.DataFrame) -> pd.DataFrame:
    # H10: Strict ANOVA Variance Decomposition
    v_total = d["std"].pow(2).mean() # Within-patient variance
    v_betwn = d["median"].var()      # Between-patient variance
    h10_res = "HIGH (Must Normalize)" if v_betwn > v_total else "LOW"

    m = d.median(numeric_only=True)
    table = [
        ("H1", "Bounded Positive", f"Min {d['min'].min():.0f}", 
         "PASS" if d["min"].min() > 0 else "FAIL"),
        ("H2", "Bimodal", f"Median Peaks: {m['kde_peaks']:.0f}", 
         "PASS" if m["kde_peaks"] >= 2 else "FAIL"),
        ("H3", "Long Memory", f"Median Slope: {m['slope']:.2f}", 
         "PASS" if m["slope"] < -0.5 else "FAIL"),
        ("H4", "Non-Stationary", f"Var Ratio: {m['var_ratio']:.2f}", 
         "PASS" if m["var_ratio"] > 0.1 else "FAIL"),
        ("H5", "Circadian", f"Median Amp: {m['amp']:.1f}ms", 
         "PASS" if m["amp"] > 10.0 else "FAIL"),
        ("H6", "Heavy Tails", f"Median Kurtosis: {m['kurt']:.1f}", 
         "PASS" if m["kurt"] > 1.0 else "FAIL"),
        ("H7", "Irregular Grid", f"On-Grid: {m['pct_on_grid']:.0f}%", 
         "PASS" if m["pct_on_grid"] < 95 else "FAIL"),
        ("H8", "Missingness", f"Median NaN: {m['nan_pct']:.1f}%", 
         "PASS" if m["nan_pct"] > 0 else "FAIL"),
        ("H10", "Between-Patient Variance", 
         f"Betwn: {v_betwn:.0f} | Within: {v_total:.0f}", h10_res)
    ]
    return pd.DataFrame(table, columns=["ID", "Trait", "Evidence", "Verdict"])

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*_processed.csv"))
    if not files: sys.exit("No CSVs found.")

    with cf.ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(analyze_file, files) if r["valid"]]

    if not rows:
        sys.exit("No valid patient files found.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_patient.csv", index=False)
    
    metrics = {row["patient"]: row for row in rows}
    for f in files:
        if f.stem in metrics:
            plot_patient(load(f), metrics[f.stem])

    v = verdicts(df)
    v.to_csv(OUT_DIR / "h1_h10_verdicts.csv", index=False)
    print("\n" + v.to_string(index=False) + "\n\nOutputs → " + str(OUT_DIR))

if __name__ == "__main__":
    main()
