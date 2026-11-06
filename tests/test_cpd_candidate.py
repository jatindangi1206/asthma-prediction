"""Candidate CPD regression specifications. Written, NOT executed during the fix.

Use unittest on the data machine. No raw cohort or foundation/GPU model is required.
"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from wearable.cpd import _whiten as W, bocpd, drift, hmm, kalman, kcpd
from wearable.cpd._contracts import hit, excess
from wearable.steps import detect as D, join as J


def flags():
    return [pd.Series(dtype=float, index=pd.DatetimeIndex([])) for _ in range(2)]


class CandidateCPDTests(unittest.TestCase):
    def test_fillers_do_not_hide_observation_gap(self):
        df = pd.DataFrame({"createdTime": pd.date_range("2026-01-01", periods=25, freq="10min"),
                           "hrvValue": [60.0] + [np.nan] * 23 + [60.0]})
        got = D._ensure_chunk_id(df, "hrvValue")
        self.assertNotEqual(got.chunk_id.iloc[0], got.chunk_id.iloc[-1])
        self.assertTrue(got.chunk_id.iloc[1:-1].isna().all())

    def test_shuffle_cutoff_excludes_later_same_day(self):
        stamp = pd.date_range("2026-01-01 08:00", periods=12, freq="1h").to_numpy()
        t = np.arange(12.0) * 60
        values = np.r_[np.arange(4.0), np.full(8, 999.0)]
        def noise(stamp, t, r, raw, chunk, cfg, state=None, exposure=None):
            return {"z": r, "armed": np.ones(len(r), bool)}
        cfg = {**W.DEFAULTS, "step_min": 60.0, "null_days": 1 / 24, "burn_in_min": 60.0}
        with patch.object(W, "noise_model", side_effect=noise):
            twin = W.twin(stamp, values, values, np.ones(12),
                          [stamp[0].astype("datetime64[D]")], cfg, cutoff=stamp[4])
        self.assertLess(twin[1].max(), 999.0)

    def test_shuffle_preserves_internal_resets(self):
        stamp = pd.date_range("2026-01-01", periods=6, freq="1h").to_numpy()
        cfg = {**W.DEFAULTS, "step_min": 60.0, "null_days": 0.1, "burn_in_min": 60.0}
        def noise(stamp, t, r, raw, chunk, cfg, state=None, exposure=None):
            return {"z": r, "armed": np.ones(len(r), bool)}
        with patch.object(W, "noise_model", side_effect=noise):
            twin = W.twin(stamp, np.ones(6), np.ones(6), np.array([1, 1, 1, 2, 2, 2]),
                          [stamp[0].astype("datetime64[D]")], cfg)
        self.assertNotEqual(twin[2][2], twin[2][3])

    def test_closed_bins_do_not_publish_terminal_partial(self):
        def bins(n):
            t = np.arange(float(n))
            stamp = (np.datetime64("2026-01-01", "ns") +
                     (t * 60e9).astype("timedelta64[ns]"))
            return W.bins(stamp, t, t + 60, t + 60, np.ones(n), 5.0)
        prefix, full = bins(7), bins(11)
        np.testing.assert_array_equal(prefix["t"], [5.0])
        np.testing.assert_array_equal(prefix["r"], full["r"][:1])
        self.assertEqual(prefix["holder"][4], -1)
        self.assertEqual(prefix["last"][0], 5)

    def test_mapped_alarms_do_not_cross_chunk_settling(self):
        t = np.r_[np.arange(61.0), np.arange(300.0, 311.0)]
        stamp = pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="m")
        df = pd.DataFrame({"p_id": "fixture", "createdTime": stamp, "hr": 80.0})
        cfg = {**W.DEFAULTS, "step_min": 1.0, "bin_min": 5.0, "burn_in_min": 1.0}
        def noise(stamp, t, r, raw, chunk, cfg, state=None, exposure=None):
            n = len(t)
            return {"arm": 0, "arms": [0], "z": np.ones(n), "scale": np.ones(n),
                    "tail_nu": np.full(n, 200.0), "tail_scale": np.ones(n), "fits": [],
                    "armed": np.ones(n, bool), "reentry_jump_z": np.full(n, np.nan),
                    "reentry_gap_min": np.full(n, np.nan)}
        def detect(t, z, priors, **kw):
            return {"anomaly": np.ones(len(t), int), "phenotype": np.full(len(t), "fixture", object),
                    "magnitude": np.ones(len(t)), "severity": np.ones(len(t)),
                    "available": np.ones(len(t), bool)}
        module = SimpleNamespace(detect=detect)
        spec = SimpleNamespace(load=lambda: module)
        with patch.object(D.registry, "get", return_value=spec), patch.object(D, "_flags", return_value=flags()), \
                patch.object(W, "noise_model", side_effect=noise), \
                patch.object(D, "_thresholds", return_value=({"bocpd": {}}, [])):
            got = D._detect_patient(df, "hr", "hr", ["bocpd"], {"bocpd": {}}, cfg, 1 / 3)
        self.assertEqual(got.loc[t >= 300, "bocpd_anomaly"].sum(), 0)
        self.assertEqual(got.loc[t >= 300, "armed"].sum(), 0)

    def test_missing_scores_never_hit_nonpositive_threshold(self):
        self.assertFalse(hit([np.nan, np.inf, -np.inf], 0).any())
        self.assertTrue(np.isfinite(excess([0.0, 1.0, np.nan], 0.0)).all())

    def test_constant_null_does_not_pass_occupancy_guard(self):
        threshold = W.pick_threshold(np.zeros(1000), np.ones(1000, bool), 60.0, 1 / 3)
        self.assertGreater(threshold, 0.0)

    def test_kcpd_invalid_windows_do_not_alarm(self):
        t = np.arange(10.0) * 10
        got = kcpd.detect(t, np.ones(10), {"thresholds": {"shift": -1.0}})
        self.assertEqual(got["anomaly"].sum(), 0)

    def test_kcpd_jittered_prefix_is_immutable(self):
        t = np.arange(90.0) * 11
        z = np.sin(np.arange(90.0))
        full = kcpd.score(t, z, {"step_min": 11.0}, n_perm=4)["shift"]
        pre = kcpd.score(t[:59], z[:59], {"step_min": 11.0}, n_perm=4)["shift"]
        np.testing.assert_array_equal(full[:59], pre)

    def test_future_first_sleep_does_not_redefine_past_nights(self):
        stamp = pd.date_range("2026-01-01", periods=7 * 24, freq="1h").to_numpy()
        raw = np.full(len(stamp), 60.0)
        state = np.zeros(len(stamp), int)
        state[-12:-6] = W.SLEEP
        p = dict(drift.DEFAULTS)
        full = drift._nights(stamp, raw, 60.0, p, state=state)
        pre = drift._nights(stamp[:4 * 24], raw[:4 * 24], 60.0, p, state=state[:4 * 24])
        for day, (level, index) in pre.items():
            self.assertEqual((level, index), full[day])

    def test_consecutive_nightly_verdicts_remain_distinct(self):
        t = np.arange(7.0) * 1440
        stamp = np.datetime64("2026-01-01") + np.arange(7).astype("timedelta64[D]")
        nights = {d: (60.0 if i < 3 else 100.0, i) for i, d in enumerate(stamp)}
        pri = {"stamp": stamp, "raw": np.ones(7), "thresholds": {"level": 0.1}}
        with patch.object(drift, "_nights", return_value=nights):
            got = drift.detect(t, np.ones(7), pri, direction=1)
        self.assertEqual(got["anomaly"].sum(), 4)

    def test_cusum_resets_at_h8(self):
        t = np.r_[np.arange(5.0) * 10, 400.0]
        z = np.full(6, 2.0)
        got = kalman.score(t, z, {"chunk": np.r_[np.ones(5), 2], "step_min": 10.0})
        self.assertAlmostEqual(got["drift"][-1], 1.5)

    def test_long_absence_restarts_burn_in(self):
        t = np.r_[np.arange(160.0) * 10, 30 * 1440 + np.arange(5.0) * 10]
        stamp = np.datetime64("2026-01-01", "ns") + (t * 60e9).astype("timedelta64[ns]")
        chunk = np.r_[np.ones(160), np.full(5, 2)]
        r = 80 + np.sin(np.arange(len(t)))
        got = W.noise_model(stamp, t, r, r, chunk, W.DEFAULTS)
        self.assertFalse(got["armed"][-5:].any())
        self.assertTrue(np.isnan(got["z"][-5:]).all())

    def test_hmm_scale_changes_emission_likelihood(self):
        t, z = np.arange(30.0) * 10, np.full(30, 4.0)
        unit = hmm.score(t, z, {"tail_scale": 1.0})["volatile"]
        scaled = hmm.score(t, z, {"tail_scale": 4.0})["volatile"]
        self.assertGreater(unit[-1], scaled[-1])

    def test_cpd_daily_counts_do_not_restart_at_midnight(self):
        frame = pd.DataFrame({"p_id": ["x", "x"],
                              "createdTime": pd.to_datetime(["2026-01-01 23:59", "2026-01-02 00:01"]),
                              "anomaly_present": [1, 1], "anomaly_present_event_start": [1, 0]})
        got = J._day_features(frame, "cpd", cpd_schema=True)
        self.assertEqual(got["cpd_anomaly_present_bouts"].tolist(), [1, 0])
        self.assertIn("cpd_anomaly_present_count", got)

    def test_patient_failure_preserves_raw_support(self):
        g = pd.DataFrame({"p_id": ["x", "x"], "createdTime": ["bad", "bad"], "hr": [80.0, 81.0]})
        result, _, _, status = D._patient_job(g, col="hr", raw_col="hr", method_names=["bocpd"],
                                             settings={}, cfg=W.DEFAULTS, budget=1 / 3)
        self.assertEqual(len(result), len(g))
        self.assertTrue(result.bocpd_anomaly.isna().all())
        self.assertEqual(status["status"], "failed")

    def test_invalid_series_is_rejected(self):
        for module in (bocpd, hmm, kalman, kcpd):
            with self.assertRaises(ValueError):
                module.score([0.0, 0.0], [1.0, 1.0])

    def test_dependency_key_tracks_state_content_and_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "x_processed.csv"
            sleep = Path(directory) / "x__sleep_active_processed.csv"
            raw.write_text("createdTime,hrvValue\n2026-01-01,60\n")
            def paths(signal):
                return [raw] if signal == "hrv" else [sleep] if signal == "sleep_active" and sleep.exists() else []
            with patch.object(D, "processed_files", side_effect=paths):
                absent = D._dependencies("hrv")["key"]
                sleep.write_text("createdTime,sleep_active\n2026-01-01,0\n")
                present = D._dependencies("hrv")["key"]
                sleep.write_text("createdTime,sleep_active\n2026-01-01,1\n")
                changed = D._dependencies("hrv")["key"]
            self.assertEqual(len({absent, present, changed}), 3)

    def test_sleep_file_survives_missing_activity_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sleep = root / "sleep_active.csv"
            sleep.write_text("createdTime,sleep_active\n2026-01-01,1\n")
            with patch.object(D, "processed_path", side_effect=lambda pid, ch: root / f"{ch}.csv"):
                sl, st = D._flags("fixture")
            self.assertEqual(sl.iloc[0], 1)
            self.assertTrue(st.empty)

    def test_transform_mask_does_not_remove_raw_day_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture_processed.csv"
            raw = pd.DataFrame({"createdTime": pd.date_range("2026-01-01", periods=3, freq="10min"),
                                "hrvValue": [60.0, 60.0, 60.0]})
            raw.to_csv(path, index=False)
            transformed = raw.iloc[[0, 2]].copy()
            transformed["p_id"] = "fixture"
            transformed["residual_hrv"] = [1.0, 2.0]
            artifact = SimpleNamespace(manifest={"tables": ["smoothed"]}, read_table=lambda name: transformed)
            with patch.object(D, "processed_files", return_value=[path]), \
                    patch.object(D, "processed_pid", return_value="fixture"):
                got = D._load_input(artifact, "residual", "hrv")
            self.assertEqual(len(got), 3)
            self.assertEqual(got.hrvValue.notna().sum(), 3)
            self.assertEqual(got.residual_hrv.notna().sum(), 2)


if __name__ == "__main__":
    unittest.main()
