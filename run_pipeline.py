#!/usr/bin/env python3
"""Full pipeline runner — every stage, every method, one command.

Order per method (dependency-correct):

    00 preprocess ─┐ (shared, runs once)
                   └─> 02x smooth ─> 03a calibrate ─> 03 annotate ─> 04 model

NOTE: calibration (03a) runs BEFORE annotation (03) because 03 consumes 03a's
adaptive baseline. (Listed the other way round in the request; the dependency
is calibrate → annotate.)

Output management — one self-contained folder per method:

    data/runs/<method>/
        smoothed/     -> symlink to data/smoothed_<method>/     (Stage 02x)
        calibrated/   -> symlink to data/calibrated_<method>/   (Stage 03a)
        annotated/    -> symlink to data/annotated_<method>/    (Stage 03)
        modeling/     real dir, one <pid>__sm-<method>__model.csv per patient (Stage 04)
        logs/         stage stdout for the cohort prerequisites (00, 02x)
        manifest.json what ran, when, git rev, per-patient row counts, config

The flat data/*_<method>/ dirs stay exactly where the existing stages write them
(nothing in src/ changes); this runner only orchestrates and organizes. Symlinks,
not copies, so there is no multi-GB duplication.

Usage (from asthma-prediction/):
    python run_pipeline.py --preflight              # readiness report, runs nothing
    python run_pipeline.py --methods all            # FULL COHORT, every method
    python run_pipeline.py --patient 0010           # one patient, all methods
    python run_pipeline.py --methods gammadglm,rspf --patient 0010
    python run_pipeline.py --dry-run                # print the plan, do nothing
    python run_pipeline.py --stages calibrate,annotate,model
    python run_pipeline.py --skip-existing          # idempotent resume after an interruption
    python run_pipeline.py --jobs 8                 # parallel per-patient stages
    python run_pipeline.py --selftest               # assert wiring (no data needed)

Cohort-safety properties (all learned the hard way on patient 0010):
  * Stage 00 runs ONCE, not once per method.
  * --skip-existing is SCHEMA- and COVERAGE-aware for smoothing: a method is only skipped
    when it has >= the expected patient count AND every file carries `residual_hrv`.
    Presence of a populated dir is not proof it is usable.
  * One patient can never abort the run: every per-patient failure is captured as a record
    field (with its reason) instead of raising.
  * --jobs parallelism is result-safe: outputs are byte-identical to a serial run (verified).
    This depends on detectors seeding their RNG per call, not at module scope.
  * Re-running is idempotent; --skip-existing resumes calibrate/annotate/model per patient.

Single-patient mode reuses existing data/processed + data/smoothed_<method> and
skips the cohort-wide 00/02x stages; if a method's smoothed data is missing it
tells you which cohort command to run first rather than silently running it.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# method -> (flat smoothed dir, 02x orchestrator, FILE TOKEN). Mirrors the CLAUDE.md table.
# The file token is what actually appears in filenames: <pid>_<token>.csv. It equals the
# method key for every method EXCEPT pf, whose files are <pid>_smoothed.csv.
METHOD_SPECS = {
    "pf":        ("data/smoothed",           "02_run_filters.py",   "smoothed"),  # non-causal (FFBS) — offline only
    "rspf":      ("data/smoothed_rspf",      "02b_run_rspf.py",     "rspf"),
    "krlst":     ("data/smoothed_krlst",     "02c_run_krlst.py",    "krlst"),
    "gpssm":     ("data/smoothed_gpssm",     "02d_run_gpssm.py",    "gpssm"),
    "ossa":      ("data/smoothed_ossa",      "02e_run_ossa.py",     "ossa"),
    "kim":       ("data/smoothed_kim",       "02f_run_kim.py",      "kim"),
    "gammadglm": ("data/smoothed_gammadglm", "02g_run_gammadglm.py", "gammadglm"),
    "logmhw":    ("data/smoothed_logmhw",    "02h_run_logmhw.py",   "logmhw"),
    "hrkf":      ("data/smoothed_hrkf",      "02i_run_hrkf.py",     "hrkf"),
    "anrewma":   ("data/smoothed_anrewma",   "02j_run_anrewma.py",  "anrewma"),
    "ctimm":     ("data/smoothed_ctimm",     "02k_run_ctimm.py",    "ctimm"),
}


def token_of(method):
    return METHOD_SPECS[method][2]
# Default set excludes pf (its residual is non-causal — offline comparison only).
DEFAULT_METHODS = [m for m in METHOD_SPECS if m != "pf"]
ALL_STAGES = ["preprocess", "smooth", "calibrate", "annotate", "model"]

PROCESSED_DIR = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "data" / "runs"


# --------------------------------------------------------------------------- #
# module loading (stage files start with digits -> import by path, not name)
# --------------------------------------------------------------------------- #
_MOD_CACHE = {}


def load_stage(fname, alias):
    if alias not in _MOD_CACHE:
        spec = importlib.util.spec_from_file_location(alias, SRC / fname)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MOD_CACHE[alias] = m
    return _MOD_CACHE[alias]


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_cohort_stage(script, arg, log_path, dry):
    """Run a whole-cohort stage (00 / 02x) via its existing CLI, teeing to a log."""
    cmd = [sys.executable, str(SRC / script)] + ([arg] if arg else [])
    printable = " ".join(["python", f"src/{script}"] + ([arg] if arg else []))
    if dry:
        print(f"    [dry-run] {printable}")
        return True
    print(f"    $ {printable}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
    ok = proc.returncode == 0
    print(f"      {'✓' if ok else '✗ FAILED'} ({time.time()-t0:.0f}s)  log: {log_path.relative_to(ROOT)}")
    return ok


def relink(link_path: Path, target_dir: Path):
    """Point link_path at target_dir with a RELATIVE symlink (idempotent)."""
    if link_path.is_symlink() or link_path.exists():
        try:
            link_path.unlink()
        except IsADirectoryError:
            return  # a real dir already sits here (e.g. modeling/) — leave it
    import os
    rel = Path(*[".."] * len(link_path.relative_to(ROOT).parent.parts)) / target_dir.relative_to(ROOT)
    os.symlink(rel, link_path, target_is_directory=True)


def organize(method, manifest):
    """Assemble data/runs/<method>/ from the flat stage outputs."""
    run_dir = RUNS_DIR / method
    (run_dir / "modeling").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    relink(run_dir / "smoothed",   ROOT / METHOD_SPECS[method][0])
    relink(run_dir / "calibrated", ROOT / "data" / f"calibrated_{method}")
    relink(run_dir / "annotated",  ROOT / "data" / f"annotated_{method}")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir


# --------------------------------------------------------------------------- #
# per-patient stages (03a, 03, 04) — one code path for cohort & single-patient
# --------------------------------------------------------------------------- #
def patient_files(method, patient):
    """Smoothed CSVs to process: one file if --patient, else all in the method dir."""
    sdir = ROOT / METHOD_SPECS[method][0]
    if not sdir.exists():
        return None
    tok = token_of(method)
    if patient:
        f = sdir / f"{patient}_{tok}.csv"
        return [f] if f.exists() else []
    return sorted(sdir.glob(f"*_{tok}.csv"))


def pid_of(path, method):
    return path.stem.rsplit(f"_{token_of(method)}", 1)[0]


# Stage 03a/03 need `residual_hrv`, which older smoother versions did not emit. Presence of a
# smoothed dir is therefore NOT proof it is usable — checking only for files would silently skip
# re-smoothing stale outputs and fail the whole cohort at the calibrate stage.
REQUIRED_SMOOTHED_COL = "residual_hrv"


def smoothed_status(method):
    """(n_files, n_with_required_col, first_missing_name) for a method's flat smoothed dir.

    Scans EVERY file, not a sample: files sort alphabetically, so sampling files[0] tests
    whichever patient happens to sort first and can report a whole method healthy while later
    patients are stale. Reading one header per file is cheap next to a multi-hour cohort run.
    """
    files = patient_files(method, None)
    if not files:
        return 0, 0, None
    import csv
    n_ok, first_missing = 0, None
    for f in files:
        try:
            with open(f, newline="", encoding="utf-8-sig") as fh:
                header = [h.strip() for h in next(csv.reader(fh))]
            if REQUIRED_SMOOTHED_COL in header:
                n_ok += 1
            elif first_missing is None:
                first_missing = f.name
        except Exception:
            if first_missing is None:
                first_missing = f.name
    return len(files), n_ok, first_missing


def expected_cohort_size():
    """Patient count the cohort should have = number of Stage-00 processed files."""
    return len(list(PROCESSED_DIR.glob("*_processed.csv"))) if PROCESSED_DIR.exists() else 0


def _patient_worker(args):
    """One patient through calibrate -> annotate -> model. Module-level and picklable so it
    can run in a ProcessPoolExecutor. NEVER raises: a single bad patient must not abort a
    multi-hour cohort run, so every failure is captured as a record field instead."""
    (method, f, stages, cpd_mode, model_dir, calib_dir, annot_dir,
     pop_c, pop_a, skip_existing) = args
    pid = pid_of(f, method)
    rec = {"patient": pid, "smoothed_file": f.name}
    try:
        m03a = load_stage("03a_calibrate.py", "stage03a")
        m03 = load_stage("03_annotate.py", "stage03")
        m04 = load_stage("04_build_modeling_dataset.py", "stage04")

        if "calibrate" in stages:
            out = calib_dir / f"{f.stem}_calib.csv"
            if skip_existing and out.exists():
                rec["calibrate"] = "skipped (exists)"
            else:
                r = m03a.calibrate_patient(f, calib_dir, pop_c)
                rec["calibrate"] = r.get("status")
                rec["n_segments"] = r.get("n_segments")
                if r.get("status") == "failed":
                    rec["calibrate_reason"] = r.get("reason")

        if "annotate" in stages:
            out = annot_dir / f"{f.stem}_annotated.csv"
            if skip_existing and out.exists():
                rec["annotate"] = "skipped (exists)"
            else:
                cd = calib_dir if (calib_dir / f"{f.stem}_calib.csv").exists() else None
                r = m03.annotate_patient(f, annot_dir, pop_a, calib_dir=cd)
                rec["annotate"] = r.get("status")
                rec["used_calibration"] = cd is not None
                rec["ensemble_events"] = r.get("ensemble_events")
                if r.get("status") == "failed":
                    rec["annotate_reason"] = r.get("reason")

        if "model" in stages:
            out_csv = model_dir / f"{pid}__sm-{method}__model.csv"
            if skip_existing and out_csv.exists():
                rec["model"] = "skipped (exists)"
            else:
                m04.ANNOT_DIR = annot_dir        # override the module's import-time constant
                try:
                    dfm = m04.build_patient_dataset(pid, token_of(method), token_of(method), cpd_mode)
                    dfm.to_csv(out_csv, index=False)
                    rec["model"] = "success"
                    rec["model_rows"] = len(dfm)
                except Exception as e:            # missing raw channels, unreadable spine, ...
                    rec["model"] = "failed"
                    rec["model_reason"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def run_per_patient(method, files, stages, cpd_mode, run_dir, skip_existing, dry, jobs=1):
    """Calibrate -> annotate -> model across patients (serial or ProcessPool)."""
    calib_dir = ROOT / "data" / f"calibrated_{method}"
    annot_dir = ROOT / "data" / f"annotated_{method}"
    model_dir = run_dir / "modeling"
    for d in (calib_dir, annot_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)     # must exist BEFORE any write; organize() runs after

    if dry:
        todo = [s for s in ("calibrate", "annotate", "model") if s in stages]
        for f in files:
            print(f"    [dry-run] {pid_of(f, method)}: {' -> '.join(todo)}")
        return [{"patient": pid_of(f, method), "planned": todo} for f in files]

    # Population noise priors: computed ONCE and passed to every worker so they don't each
    # rescan the cohort. Deliberately derived from the method's FULL cohort (every file in
    # the smoothed dir), NOT from `files` — otherwise a --patient run would build a
    # "population" prior out of that one patient and produce different detections than the
    # same patient inside a cohort run, defeating single-patient verification.
    m03a = load_stage("03a_calibrate.py", "stage03a")
    m03 = load_stage("03_annotate.py", "stage03")
    cohort = patient_files(method, None) or files
    pop_c = m03a._population_sigma0(cohort) if "calibrate" in stages else None
    pop_a = m03._population_sigma0(cohort) if "annotate" in stages else None

    argv = [(method, f, stages, cpd_mode, model_dir, calib_dir, annot_dir,
             pop_c, pop_a, skip_existing) for f in files]

    def show(rec):
        keys = ("calibrate", "annotate", "ensemble_events", "model", "model_rows",
                "calibrate_reason", "annotate_reason", "model_reason", "error")
        print(f"    {rec['patient']}: " + "  ".join(f"{k}={rec[k]}" for k in keys if k in rec))

    records = []
    if jobs > 1 and len(files) > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            for rec in ex.map(_patient_worker, argv):
                show(rec); records.append(rec)
    else:
        for a in argv:
            rec = _patient_worker(a); show(rec); records.append(rec)
    return records


# --------------------------------------------------------------------------- #
def preflight(methods, stages):
    """Readiness report for a full-cohort run — what is present, what is stale, what will run."""
    print(f"\n{'='*74}\nPREFLIGHT — full-cohort readiness\n{'='*74}")
    n_raw = len([p for p in (ROOT / "raw_data").glob("*") if p.is_dir()]) if (ROOT / "raw_data").exists() else 0
    n_proc = len(list(PROCESSED_DIR.glob("*_processed.csv"))) if PROCESSED_DIR.exists() else 0
    plist = ROOT / "raw_data" / "processed_users.txt"
    print(f"  raw_data patient dirs : {n_raw}")
    print(f"  processed_users.txt   : {'present' if plist.exists() else 'MISSING (Stage 00 needs it)'}")
    print(f"  data/processed        : {n_proc} files"
          f"{'' if n_proc else '  → Stage 00 must run'}")
    exp = expected_cohort_size()
    print(f"  expected cohort size  : {exp} patients (= data/processed files)\n")
    print(f"  {'method':<11}{'smoothed':>9}{'/exp':>5} {'w/resid':>8} {'calib':>7} {'annot':>7} {'model':>7}   action")
    print(f"  {'-'*11}{'-'*9}{'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*7}   {'-'*30}")
    todo, ready = [], []
    for m in methods:
        n, n_ok, missing = smoothed_status(m)
        nc = len(list((ROOT / "data" / f"calibrated_{m}").glob("*_calib.csv")))
        na = len(list((ROOT / "data" / f"annotated_{m}").glob("*_annotated.csv")))
        nm = len(list((RUNS_DIR / m / "modeling").glob("*__model.csv")))
        complete = exp > 0 and n >= exp and n_ok == n
        if complete:
            act = "reuse smoothed"; ready.append(m)
        elif n == 0:
            act = "SMOOTH (no output)"; todo.append(m)
        elif n_ok < n:
            act = f"RE-SMOOTH ({n-n_ok} stale, e.g. {missing})"; todo.append(m)
        else:
            act = f"RE-SMOOTH (only {n}/{exp} patients)"; todo.append(m)
        print(f"  {m:<11}{n:>9}{'/'+str(exp):>5} {n_ok:>8} {nc:>7} {na:>7} {nm:>7}   {act}")
    print(f"\n  cohort-complete : {len(ready)}/{len(methods)}  {ready if ready else '(none)'}")
    if todo:
        print(f"  needs smoothing : {len(todo)}  {todo}")
        print(f"  → run with 'smooth' in --stages; incompleteness/staleness overrides --skip-existing")
    stale = todo
    print(f"{'='*74}\n")
    return {"ready": ready, "stale": stale, "n_processed": n_proc, "n_raw": n_raw}


def run_method(method, patient, stages, cpd_mode, skip_existing, dry, jobs=1):
    print(f"\n{'='*70}\nMETHOD: {method}\n{'='*70}")
    run_dir = RUNS_DIR / method
    manifest = {"method": method, "git_rev": git_rev(), "cpd_mode": cpd_mode, "jobs": jobs,
                "started": datetime.now(timezone.utc).isoformat(),
                "patient": patient or "cohort", "stages": stages}

    # --- cohort prerequisites (00 preprocess, 02x smooth) --------------------
    if patient:
        # single-patient mode: reuse existing data, don't run cohort stages
        if patient_files(method, patient) == []:
            sm_cmd = f"python src/{METHOD_SPECS[method][1]}"
            print(f"  ! No smoothed file for patient {patient} in {METHOD_SPECS[method][0]}/.")
            print(f"    Run the cohort smoother first:  {sm_cmd}")
            manifest["status"] = "missing_smoothed_input"
            organize(method, manifest)
            return manifest
        print(f"  (single-patient mode: reusing existing processed + smoothed data)")
    else:
        # NOTE: Stage 00 is method-independent and is run ONCE by main() before the method
        # loop — running it inside this per-method function would re-run it 11 times.
        if "smooth" in stages:
            print(f"  [02x] smooth via {METHOD_SPECS[method][1]}")
            n, n_ok, missing = smoothed_status(method)
            exp = expected_cohort_size()
            complete = exp > 0 and n >= exp and n_ok == n
            if skip_existing and complete:
                print(f"    ✓ skipped ({n}/{exp} files, all carry {REQUIRED_SMOOTHED_COL})")
            else:
                if n and not complete:
                    why = (f"{n-n_ok} lack {REQUIRED_SMOOTHED_COL} (e.g. {missing})"
                           if n_ok < n else f"only {n}/{exp} patients")
                    print(f"    ! incomplete: {why} — re-smoothing (skip-existing overridden)")
                run_cohort_stage(METHOD_SPECS[method][1], None, run_dir / "logs" / "02_smooth.log", dry)

    # --- per-patient stages (03a calibrate, 03 annotate, 04 model) -----------
    files = patient_files(method, patient)
    if files is None:
        print(f"  ! Smoothed dir {METHOD_SPECS[method][0]} does not exist — run smooth stage first.")
        manifest["status"] = "no_smoothed_dir"
        organize(method, manifest)
        return manifest
    print(f"  [03a->03->04] {len(files)} patient(s)")
    manifest["patients"] = run_per_patient(method, files, stages, cpd_mode, run_dir, skip_existing, dry, jobs)
    manifest["status"] = "dry-run" if dry else "complete"
    manifest["finished"] = datetime.now(timezone.utc).isoformat()

    run_dir = organize(method, manifest)
    if not dry:
        print(f"  → organized: {run_dir.relative_to(ROOT)}/  "
              f"(smoothed/ calibrated/ annotated/ modeling/ + manifest.json)")
    return manifest


def _selftest():
    """Assert wiring without touching data: script mapping, pid round-trip, symlink math."""
    for m, (sdir, script, tok) in METHOD_SPECS.items():
        assert (SRC / script).exists(), f"missing orchestrator for {m}: {script}"
    # pid extraction round-trips the smoothed filename convention
    p = Path(f"data/smoothed_gammadglm/0010_gammadglm.csv")
    assert pid_of(p, "gammadglm") == "0010", "pid extraction broken"
    p2 = Path("data/smoothed_rspf/A12_rspf.csv")
    assert pid_of(p2, "rspf") == "A12", "pid extraction broken (alnum id)"
    # pf diverges: its files are <pid>_smoothed.csv, not <pid>_pf.csv
    assert pid_of(Path("data/smoothed/0010_smoothed.csv"), "pf") == "0010", "pf token handling broken"
    assert token_of("pf") == "smoothed" and token_of("ctimm") == "ctimm"
    # relative symlink target: data/runs/<m>/smoothed -> ../../smoothed_<m>
    link = ROOT / "data" / "runs" / "gammadglm" / "smoothed"
    depth = len(link.relative_to(ROOT).parent.parts)          # data/runs/gammadglm -> 3
    assert depth == 3 and (Path(*[".."]*depth)) == Path("../../.."), "symlink depth math off"
    assert set(ALL_STAGES) == {"preprocess", "smooth", "calibrate", "annotate", "model"}
    print("run_pipeline self-test OK — orchestrators resolve, pid round-trips, symlink math holds.")


def main():
    ap = argparse.ArgumentParser(description="Full HRV pipeline runner (all stages, all methods).")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help=f"comma list or 'all' (default: {','.join(DEFAULT_METHODS)}; 'all' adds non-causal pf)")
    ap.add_argument("--patient", default=None, help="single patient id (skips cohort 00/02x, reuses existing data)")
    ap.add_argument("--stages", default=",".join(ALL_STAGES), help=f"subset of {ALL_STAGES}")
    ap.add_argument("--cpd-mode", default="all", help="Stage-04 CPD columns: ensemble | all | comma list")
    ap.add_argument("--skip-existing", action="store_true", help="idempotent: skip stages whose output exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, execute nothing")
    ap.add_argument("--selftest", action="store_true", help="assert wiring and exit")
    ap.add_argument("--preflight", action="store_true", help="print cohort-readiness report and exit")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel workers for the per-patient stages (default: cpu_count-1)")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    methods = list(METHOD_SPECS) if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    bad = [m for m in methods if m not in METHOD_SPECS]
    if bad:
        print(f"Unknown method(s): {bad}. Choose from: {list(METHOD_SPECS)}"); sys.exit(2)
    stages = [s.strip() for s in args.stages.split(",")]
    cpd_mode = args.cpd_mode if args.cpd_mode in ("ensemble", "all") else [s.strip() for s in args.cpd_mode.split(",")]

    if args.preflight:
        preflight(methods, stages); return

    jobs = 1 if args.patient else max(1, args.jobs)
    print(f"Pipeline run — methods={methods}  patient={args.patient or 'cohort'}  "
          f"stages={stages}  jobs={jobs}  {'[DRY-RUN]' if args.dry_run else ''}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.patient:
        preflight(methods, stages)
        # Stage 00 is method-independent: run it ONCE here, not once per method.
        if "preprocess" in stages:
            print("[00] preprocess (cohort, shared, runs once)")
            have = PROCESSED_DIR.exists() and any(PROCESSED_DIR.glob("*_processed.csv"))
            if args.skip_existing and have:
                print("    skipped (data/processed already populated)")
            else:
                run_cohort_stage("00_preprocess_raw.py", None,
                                 RUNS_DIR / "_shared" / "logs" / "00_preprocess.log", args.dry_run)

    summary = [run_method(m, args.patient, stages, cpd_mode, args.skip_existing, args.dry_run, jobs)
               for m in methods]

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for man in summary:
        n = len(man.get("patients", []))
        print(f"  {man['method']:<10} {man.get('status','?'):<24} patients={n}  → data/runs/{man['method']}/")


if __name__ == "__main__":
    main()
