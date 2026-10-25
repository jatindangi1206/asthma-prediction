# legacy/

Code kept for reference, not part of the step pipeline. Nothing in `wearable/` imports
from here.

| file | why it is here |
|---|---|
| `run_pf_filters.py` | The SMC + FFBS particle filter. Its backward smoothing pass makes its output **non-causal**, so it cannot satisfy the smoothing contract. Useful only as an offline reference. |
| `pf_hyper_tuning.py` | Particle-filter hyperparameter search (N*, M*) for the above. |
| `calibrate_adaptive_baseline.py` | The stage-3a causal adaptive noise floor: grows while the circadian-detrended residual is stationary, freezes when the estimate stops moving, caps at the ~21-day drift horizon, with gap-aware carry/verify/restart. **Verified working**, but it reads the old per-patient `data/smoothed_<method>/` layout that the `smooth` step no longer writes, so it is unwired. Port it to an artifact-store step when you want adaptive arming back — the detect step currently uses the inline 144-sample burn-in only. Self-test: `python legacy/calibrate_adaptive_baseline.py selftest`. |
| `build_modeling_dataset.py` | The old single-method stage 4. Superseded by the `join` + `model` steps, which take any combination of artifacts. |
| `patch_add_trend_level.py` | A one-off migration script. |
