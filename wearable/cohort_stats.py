"""Data characteristics that smoothing settings are derived from.

`patient_stats(df)` measures one patient; `summarise(stats)` takes the median across
patients, so every patient counts once however long they recorded. The same per-patient
dict can later drive per-patient settings.

The model every derivation shares: a reading is a slowly wandering level plus white
noise. The level is what a smoother should recover, the noise what it should remove.
Its parameters come from a robust variogram, fitted within H8 chunks up to a 24 h lag:

    gamma(h) = R + S (1 - exp(-h / tau))      h = time lag (min)
    R    white noise variance (measurement noise + sub-10-min physiology)
    S    variance of the wandering level
    tau  how long the level stays correlated (min)

fitted in three spaces, because the smoothers work in three: raw HRV, log HRV, and the
logit of the patient's own [min, max] range (rspf).

HRV readings are strongly autocorrelated, so every significance test and estimation-
noise correction uses an effective sample size n (1 - rho) / (1 + rho), rho being the
lag-1 autocorrelation of the residuals. Without it, thousands of correlated readings
look like thousands of independent ones and everything is "significant".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import f as fdist, wilcoxon

from wearable.chunking import assign_chunks

DEVICE_CAP = 129.0
DAY_MIN = 1440.0
STATES = ("sleep", "rest", "active")


def _mad_noise_var(d: np.ndarray) -> float:
    """The noise variance the smoothers compute themselves: (1.4826 MAD(diff) / sqrt 2)^2."""
    return float((1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) ** 2)


def _ch_noise_var(d: np.ndarray) -> float:
    """Cressie-Hawkins robust half-variance of differences. Unlike the MAD it is not
    collapsed by the many exact-zero differences of integer, ceiling-capped HRV."""
    n = len(d)
    return float(np.mean(np.sqrt(np.abs(d))) ** 4 / (2 * (0.457 + 0.494 / n))) if n else np.nan


def _logit(y, lo, hi, eps=1e-4):
    u = np.clip((y - lo) / (hi - lo), eps, 1 - eps)
    return np.log(u / (1 - u))


def _n_eff(resid: np.ndarray) -> float:
    r = resid - resid.mean()
    rho = float(np.dot(r[1:], r[:-1]) / max(np.dot(r, r), 1e-12)) if len(r) > 2 else 0.0
    rho = min(max(rho, 0.0), 0.99)
    return len(r) * (1 - rho) / (1 + rho)


def _variogram(t, z, chunk, dt_bin, max_lag_steps=200, max_h=DAY_MIN):
    """Cressie-Hawkins robust variogram binned by time lag, within chunks only."""
    sums: dict[int, tuple[int, float]] = {}
    for k in range(1, max_lag_steps + 1):
        same = chunk[k:] == chunk[:-k]
        if not same.any():
            break
        h = (t[k:] - t[:-k])[same]
        d = np.sqrt(np.abs((z[k:] - z[:-k])[same]))
        keep = h <= max_h
        b = np.round(h[keep] / dt_bin).astype(int)
        cnt, tot = np.bincount(b), np.bincount(b, weights=d[keep])
        for bi in np.nonzero(cnt)[0]:
            c, s = sums.get(bi, (0, 0.0))
            sums[bi] = (c + int(cnt[bi]), s + float(tot[bi]))
    hs, vals = [], []
    for bi in sorted(sums):
        n, s = sums[bi]
        if bi == 0 or n < 30:
            continue
        hs.append(bi * dt_bin)
        vals.append((s / n) ** 4 / (2 * (0.457 + 0.494 / n)))
    return np.array(hs), np.array(vals)


def _fit_variogram(h, g, dt):
    """gamma(h) = R + S (1 - exp(-h/tau)) -> (R, S, tau). None if too little data."""
    if len(h) < 8:
        return None
    model = lambda x, R, S, tau: R + S * (1 - np.exp(-x / tau))
    g0 = float(np.median(g[:3]))
    p0 = [max(g0 / 2, 1e-6), max(float(g.max()) - g0 / 2, 1e-6), 120.0]
    try:
        (R, S, tau), _ = curve_fit(model, h, g, p0=p0,
                                   bounds=([0, 0, dt / 2], [np.inf, np.inf, 10 * DAY_MIN]),
                                   maxfev=20000)
    except Exception:
        return None
    return float(R), float(S), float(tau)


def _design(tod_min, H):
    w = 2 * np.pi * tod_min / DAY_MIN
    cols = [np.ones_like(w)]
    for h in range(1, H + 1):
        cols += [np.cos(h * w), np.sin(h * w)]
    return np.column_stack(cols)


def _harmonics(tod_min, z, H=3):
    """Amplitude per harmonic + partial-F p-value with an autocorrelation-corrected n."""
    X = _design(tod_min, H)
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ beta
    rss_full = float(resid @ resid)
    df2 = max(_n_eff(resid) - X.shape[1], 1.0)
    amps, pvals = [], []
    for h in range(1, H + 1):
        amps.append(float(np.hypot(beta[2 * h - 1], beta[2 * h])))
        Xr = np.delete(X, [2 * h - 1, 2 * h], axis=1)
        br, *_ = np.linalg.lstsq(Xr, z, rcond=None)
        rss_r = float(np.sum((z - Xr @ br) ** 2))
        F = ((rss_r - rss_full) / 2) / (rss_full / df2)
        pvals.append(float(fdist.sf(F, 2, df2)))
    return amps, pvals


def _daily_harmonic_change(day, tod_min, z, dt, min_cover=0.5):
    """Variance per day of the change in the first circadian harmonic's coefficients,
    net of estimation noise (autocorrelation-corrected). Consecutive covered days only."""
    coefs = {}
    for d in np.unique(day):
        s = day == d
        if s.sum() < min_cover * DAY_MIN / dt:
            continue
        X = _design(tod_min[s], 1)
        b, *_ = np.linalg.lstsq(X, z[s], rcond=None)
        resid = z[s] - X @ b
        var = float(resid @ resid) / max(_n_eff(resid) - 3, 1.0)
        cov = np.linalg.pinv(X.T @ X) * var * (len(resid) / max(_n_eff(resid), 1.0))
        coefs[d] = (b[1:], float(np.mean(np.diag(cov)[1:])))
    days = sorted(coefs)
    diffs, noise = [], []
    for a, b in zip(days[:-1], days[1:]):
        if (b - a) == np.timedelta64(1, "D"):
            diffs.append(coefs[b][0] - coefs[a][0])
            noise.append(coefs[a][1] + coefs[b][1])
    if len(diffs) < 3:
        return None
    return max(float(np.mean(np.concatenate(diffs) ** 2)) - float(np.mean(noise)), 0.0)


def _hourly_vol(t, z, chunk):
    """Robust noise level per clock hour within a chunk -> (hour index, log noise var)."""
    same = chunk[1:] == chunk[:-1]
    hour = np.floor(t[1:] / 60.0)[same].astype(int)
    dz = np.diff(z)[same]
    out = {}
    for hh in np.unique(hour):
        s = dz[hour == hh]
        if len(s) >= 4:
            v = _ch_noise_var(s)
            if v > 0:
                out[int(hh)] = np.log(v)
    return out


def _persistence_min(series: dict[int, float], unit_min=60.0, max_lag=48):
    """Minutes until the autocorrelation of an hourly series first drops below 1/e."""
    if len(series) < 24:
        return None
    keys = np.array(sorted(series))
    vals = np.array([series[k] for k in keys])
    vals = vals - vals.mean()
    var = float(np.mean(vals ** 2))
    if var <= 0:
        return None
    lookup = dict(zip(keys, vals))
    prev = 1.0
    for k in range(1, max_lag + 1):
        pairs = [(lookup[a], lookup[a + k]) for a in keys if a + k in lookup]
        if len(pairs) < 10:
            return None
        r = float(np.mean([a * b for a, b in pairs]) / var)
        if r < 1 / np.e:
            return unit_min * (k - 1 + (prev - 1 / np.e) / max(prev - r, 1e-9))
        prev = r
    return unit_min * max_lag


def _gd_rank(zc, sigma, window, l_frac=0.5, n_windows=40, seed=0):
    """Median Gavish-Donoho rank (known noise sigma) of Hankel windows of length `window`."""
    from wearable.smoothing.ossa import _hankel
    starts = np.arange(0, len(zc) - window + 1)
    if len(starts) == 0 or window < 8:
        return None
    rng = np.random.default_rng(seed)
    ranks = []
    for s in rng.choice(starts, min(n_windows, len(starts)), replace=False):
        y = zc[s:s + window]
        L = int(max(2, min(round(l_frac * window), window - 1)))
        X = _hankel(y - y.mean(), L)
        m, n = sorted(X.shape)
        beta = m / n
        lam = np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta ** 2 + 14 * beta + 1)))
        S = np.linalg.svd(X, compute_uv=False)
        ranks.append(int((S > lam * np.sqrt(n) * sigma).sum()))
    return float(np.median(ranks))


def state_medians(raw: pd.DataFrame) -> dict:
    """Median log HRV per activity state, from the raw pass1 minute file
    (columns hrv, sleep_active, steps_active). sleep wins over steps."""
    h = raw[raw["hrv"].notna() & (raw["hrv"] > 0)]
    state = np.where(h["sleep_active"] == 1, "sleep",
                     np.where(h["steps_active"] == 1, "active", "rest"))
    z = np.log(h["hrv"].to_numpy(float))
    out = {}
    for s in STATES:
        v = z[state == s]
        out[f"state_med_{s}"] = float(np.median(v)) if len(v) >= 30 else np.nan
        out[f"state_n_{s}"] = int(len(v))
    return out


def patient_stats(df: pd.DataFrame, pid: str) -> dict | None:
    """df: createdTime, hrvValue (observed rows only). Returns the per-patient measures."""
    from wearable.smoothing.logmhw import LogMHWConfig, _optimize_params, _seasonal_profile, _tod_bucket
    d = df[np.isfinite(df["hrvValue"]) & (df["hrvValue"] > 0)].copy()
    if len(d) < 30:
        return None
    d["patient_id"] = pid
    d = assign_chunks(d, timestamp_col="createdTime")
    t = (d["createdTime"] - d["createdTime"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
    y = d["hrvValue"].to_numpy(float)
    chunk = d["chunk_id"].to_numpy()
    dts = d["time_diff_min"].to_numpy()
    inside = np.isfinite(dts) & ~d["gap_boundary"].to_numpy()
    dt = float(np.median(dts[inside])) if inside.any() else 10.0
    tod = (d["createdTime"].dt.hour * 60 + d["createdTime"].dt.minute).to_numpy(float)
    day = d["createdTime"].dt.normalize().to_numpy()
    same = chunk[1:] == chunk[:-1]
    sizes = pd.Series(chunk).value_counts()
    longest = chunk == sizes.index[0]

    lo, hi = float(y.min()), float(y.max())
    z = np.log(y)
    spaces = {"raw": y, "log": z, "logit": _logit(y, lo, hi) if hi > lo else None}
    out = {"p_id": pid, "n_obs": int(len(y)), "n_chunks": int(chunk.max()), "dt_min": dt,
           "frac_at_cap": float((y >= DEVICE_CAP).mean()), "median_hrv": float(np.median(y)),
           "hrv_min": lo, "hrv_max": hi}
    for name, v in spaces.items():
        if v is None:
            continue
        dz = np.diff(v)[same]
        out[f"mad_noise_{name}"] = _mad_noise_var(dz) if len(dz) > 5 else np.nan
        out[f"spread_{name}"] = float(1.4826 * np.median(np.abs(v - np.median(v))))
        fit = _fit_variogram(*_variogram(t, v, chunk, dt), dt)
        if fit:
            out[f"R_{name}"], out[f"S_{name}"], out[f"tau_{name}"] = fit
    # What the smoothers calibrate internally, exactly as they do it (differences across
    # ALL observed readings, gaps included), so a derived scale factor corrects THAT.
    out["module_noise_log"] = max(_mad_noise_var(np.diff(z)), 1e-4)
    out["module_noise_raw"] = max(_mad_noise_var(np.diff(y)), 1e-6)
    out["gamma_shape_module"] = float(np.clip(np.median(y) ** 2 / out["module_noise_raw"], 1, 1e4))
    from wearable.smoothing.gpssm import GPSSMConfig, _fit_patient as _gp_fit
    zc = [z[chunk == c] - z.mean() for c in np.unique(chunk) if (chunk == c).sum() >= 10]
    if zc:
        F, Q, R = _gp_fit(zc, GPSSMConfig(q_scale=1.0, r_scale=1.0))
        out.update(gpssm_F=F, gpssm_Q_module=Q, gpssm_R_module=R)

    # circadian rhythm and its day-to-day change
    if len(y) >= 144:
        amps, pv = _harmonics(tod, z)
        out.update({f"amp{h + 1}_log": a for h, a in enumerate(amps)})
        out.update({f"p_harm{h + 1}": p for h, p in enumerate(pv)})
        dc = _daily_harmonic_change(day, tod, z, dt)
        if dc is not None:
            out["dcoef_var_log"] = dc
        if spaces["logit"] is not None:
            out["amp1_logit"] = _harmonics(tod, spaces["logit"], H=1)[0][0]
            dc = _daily_harmonic_change(day, tod, spaces["logit"], dt)
            if dc is not None:
                out["dcoef_var_logit"] = dc

    # volatility: hourly noise level, how it varies and how long a spell lasts
    hv = _hourly_vol(t, z, chunk)
    out["tau_vol"] = _persistence_min(hv)
    out["hour_vol"] = [float(np.exp(v)) for v in hv.values()]

    # noise when HRV is high vs low (Kim's regime-1 vs regime-0 noise)
    mid = ((z[1:] + z[:-1]) / 2)[same]
    dzs = np.diff(z)[same]
    high = mid > np.median(mid)
    if high.sum() > 20 and (~high).sum() > 20:
        out["noise_ratio_high_low"] = _ch_noise_var(dzs[high]) / _ch_noise_var(dzs[~high])

    # day-to-day memory of the daily median: consecutive-day pairs, patient-centred
    dm = pd.Series(z).groupby(day).median()
    dm = dm.reindex(pd.date_range(dm.index.min(), dm.index.max(), freq="D")) - dm.mean()
    a, b = dm.to_numpy()[:-1], dm.to_numpy()[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    out["day_pairs"] = np.column_stack([a[ok], b[ok]]).tolist()

    # SSA rank at a one-day window on the longest chunk, noise = variogram white noise
    if "R_log" in out:
        zc = z[longest]
        out["ssa_rank_1day"] = _gd_rank(zc, np.sqrt(out["R_log"]),
                                        min(int(round(DAY_MIN / dt)), len(zc)))

    # Holt (damped) MLE on the deseasonalised longest chunk, as logmhw does internally
    if longest.sum() >= 30:
        m = int(round(DAY_MIN / dt))
        bk = _tod_bucket(d["createdTime"], m)
        prof = _seasonal_profile(z, bk, m)
        a_, b_, phi_ = _optimize_params(z[longest] - prof[bk[longest]], LogMHWConfig())
        out.update(holt_alpha=a_, holt_beta=b_, holt_phi=phi_)

    out["inside_dt"] = dts[inside].tolist()
    out["all_dt"] = dts[np.isfinite(dts)].tolist()
    out["chunk_len"] = sizes.to_numpy().tolist()
    return out


LIST_KEYS = ("hour_vol", "day_pairs", "inside_dt", "all_dt", "chunk_len")


def summarise(stats: list[dict]) -> dict:
    """Median across patients for scalars; pooled quantities for the per-reading lists."""
    df = pd.DataFrame([{k: v for k, v in s.items() if k not in LIST_KEYS} for s in stats])
    out = {"n_patients": int(len(df))}
    for c in df.columns:
        if c == "p_id":
            continue
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(col):
            out[c] = float(col.median())
            out[f"{c}__n"] = int(len(col))
            out[f"{c}__iqr"] = [float(col.quantile(.25)), float(col.quantile(.75))]

    out["frac_harm_sig"] = {h: float((df[f"p_harm{h}"].dropna() < 0.05).mean())
                            for h in (1, 2, 3) if f"p_harm{h}" in df}

    vol = np.concatenate([np.asarray(s["hour_vol"]) for s in stats if s["hour_vol"]])
    out["hour_vol_p10_p50_p90"] = [float(x) for x in np.quantile(vol, [0.1, 0.5, 0.9])]
    out["n_patient_hours_vol"] = int(len(vol))

    pairs = np.concatenate([np.asarray(s["day_pairs"]).reshape(-1, 2) for s in stats])
    r1 = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) > 10 else np.nan
    out["daymedian_lag1_corr"] = r1
    out["n_day_pairs"] = int(len(pairs))
    out["tau_day_min"] = float(-DAY_MIN / np.log(r1)) if 0 < r1 < 1 else np.nan

    # activity states: which pairs differ across patients (paired Wilcoxon, Bonferroni 3)
    if "state_med_sleep" in df:
        tests = {}
        for a, b in (("sleep", "rest"), ("rest", "active"), ("sleep", "active")):
            x = df[[f"state_med_{a}", f"state_med_{b}"]].dropna()
            if len(x) >= 10:
                p = float(wilcoxon(x.iloc[:, 0], x.iloc[:, 1]).pvalue)
                diff = float((x.iloc[:, 0] - x.iloc[:, 1]).median())
                tests[f"{a}_vs_{b}"] = {"n": int(len(x)), "median_diff_log": diff,
                                        "p": p, "distinct": bool(p * 3 < 0.05)}
        out["state_tests"] = tests
        # K = number of groups left after merging every pair that is NOT distinct
        group = {s: s for s in STATES}
        for key, t in tests.items():
            if not t["distinct"]:
                a, b = key.split("_vs_")
                ga, gb = group[a], group[b]
                group = {s: (ga if g == gb else g) for s, g in group.items()}
        out["n_distinct_states"] = len(set(group.values()))

    ins = np.concatenate([np.asarray(s["inside_dt"]) for s in stats])
    out["inside_dt_quantiles"] = {str(q): float(np.quantile(ins, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    alld = np.concatenate([np.asarray(s["all_dt"]) for s in stats])
    edges = [0, 5, 15, 30, 60, 120, 180, 360, 720, 1440, np.inf]
    out["gap_hist_min"] = {f"{a:g}-{b:g}": int(((alld >= a) & (alld < b)).sum())
                           for a, b in zip(edges[:-1], edges[1:])}
    cl = np.concatenate([np.asarray(s["chunk_len"]) for s in stats])
    out["chunk_len_quantiles"] = {str(q): float(np.quantile(cl, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out["frac_chunks_ge_10"] = float((cl >= 10).mean())
    out["frac_readings_in_chunks_ge_10"] = float(cl[cl >= 10].sum() / cl.sum())
    return out


if __name__ == "__main__":
    # The variogram fit must recover a known level + noise: AR(1) level with variance S
    # and correlation time tau, plus white noise R, sampled every 10 min with gaps.
    rng = np.random.default_rng(0)
    R, S, tau, dt, n = 0.05, 0.15, 150.0, 10.0, 20000
    phi = np.exp(-dt / tau)
    x = np.empty(n); x[0] = rng.normal(0, np.sqrt(S))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, np.sqrt(S * (1 - phi ** 2)))
    t = np.arange(n) * dt
    chunk = np.cumsum(rng.random(n) < 0.01)          # ~ every 100 readings a new segment
    fit = _fit_variogram(*_variogram(t, x + rng.normal(0, np.sqrt(R), n), chunk, dt), dt)
    print("true R, S, tau =", (R, S, tau), " fitted =", tuple(round(v, 3) for v in fit))
    assert abs(fit[0] - R) / R < 0.2 and abs(fit[1] - S) / S < 0.3 and abs(fit[2] - tau) / tau < 0.3
    print("OK")
