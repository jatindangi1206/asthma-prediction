"""Artifact store: every result records how it was made, and nothing is computed twice.

An artifact is `data/runs/<step>/<method>/<key>/` holding its tables and a
`manifest.json`, where

    key = sha256(step, method, params, input artifact keys, method source hash)

so the same computation is found and reused, while a changed parameter, input or method
source recomputes. The manifest is the provenance record; following `inputs` back from
any artifact reaches the raw source. Table format is a setting (`WEARABLE_FORMAT`).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from wearable.config import RUNS_DIR, STORE_FORMAT

_WRITERS = {
    "csv": lambda df, p: df.to_csv(p, index=False),
    "parquet": lambda df, p: df.to_parquet(p, index=False),
}
_READERS = {
    "csv": lambda p: pd.read_csv(p),
    "parquet": lambda p: pd.read_parquet(p),
}


def store_format() -> str:
    fmt = os.environ.get("WEARABLE_FORMAT", STORE_FORMAT).lower()
    if fmt not in _WRITERS:
        raise ValueError(f"unsupported WEARABLE_FORMAT {fmt!r}; "
                         f"available: {sorted(_WRITERS)}")
    return fmt


_git: dict[str, Any] | None = None


def code_version() -> dict[str, Any]:
    """Repo revision + dirty flag, for tracing a result back to the code that made it."""
    global _git
    if _git is None:
        def _run(*a):
            try:
                return subprocess.run(a, capture_output=True, text=True,
                                      timeout=5).stdout.strip()
            except Exception:
                return ""
        rev = _run("git", "rev-parse", "HEAD")
        _git = {"git_rev": rev or "unknown",
                "git_dirty": bool(_run("git", "status", "--porcelain"))}
    return dict(_git)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class Artifact:
    step: str
    method: str
    key: str
    dir: Path
    manifest: dict = field(default_factory=dict)

    @property
    def ref(self) -> dict:
        """The compact reference recorded in a downstream artifact's `inputs`."""
        return {"step": self.step, "method": self.method, "key": self.key}

    def table_path(self, name: str) -> Path:
        return self.dir / f"{name}.{self.manifest.get('format', store_format())}"

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        p = self.table_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        _WRITERS[self.manifest.get("format", store_format())](df, p)
        self.manifest.setdefault("tables", [])
        if name not in self.manifest["tables"]:
            self.manifest["tables"].append(name)
        return p

    def read_table(self, name: str) -> pd.DataFrame:
        p = self.table_path(name)
        if not p.exists():
            raise FileNotFoundError(f"{self.step}/{self.method}/{self.key[:12]} "
                                    f"has no table {name!r} ({p})")
        return _READERS[self.manifest.get("format", store_format())](p)

    def has_table(self, name: str) -> bool:
        return self.table_path(name).exists()

    def complete(self, **stats) -> "Artifact":
        self.manifest["status"] = "complete"
        self.manifest["finished"] = datetime.now(timezone.utc).isoformat()
        self.manifest.setdefault("stats", {}).update(stats)
        self._save()
        return self

    def fail(self, reason: str) -> "Artifact":
        self.manifest["status"] = "failed"
        self.manifest["reason"] = reason
        self._save()
        return self

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2,
                                                           default=str))

    def lineage(self) -> list[dict]:
        """This artifact's manifest, then its inputs', recursively back to the source."""
        out = [self.manifest]
        for ref in self.manifest.get("inputs", []):
            if {"step", "method", "key"} <= set(ref):
                parent = load(ref["step"], ref["method"], ref["key"])
                if parent is not None:
                    out.extend(parent.lineage())
        return out


def make_key(step: str, method: str, params: dict, inputs: list[dict],
             code_hash: str) -> str:
    payload = _canonical({
        "step": step, "method": method, "params": params,
        # only the input KEYS matter: they already encode their own provenance
        "inputs": [i.get("key", _canonical(i)) for i in inputs],
        "code": code_hash,
    })
    return hashlib.sha256(payload.encode()).hexdigest()


def artifact_dir(step: str, method: str, key: str) -> Path:
    return RUNS_DIR / step / method / key[:12]


def load(step: str, method: str, key: str) -> Artifact | None:
    d = artifact_dir(step, method, key)
    mf = d / "manifest.json"
    if not mf.exists():
        return None
    return Artifact(step, method, key, d, json.loads(mf.read_text()))


def find(step: str, method: str, params: dict, inputs: list[dict],
         code_hash: str) -> Artifact | None:
    """An existing COMPLETE artifact for exactly this computation, or None."""
    a = load(step, method, make_key(step, method, params, inputs, code_hash))
    return a if a is not None and a.manifest.get("status") == "complete" else None


def open_artifact(step: str, method: str, params: dict, inputs: list[dict],
                  code_hash: str, **extra) -> Artifact:
    """Start a new artifact (overwriting any incomplete one at the same key)."""
    key = make_key(step, method, params, inputs, code_hash)
    a = Artifact(step, method, key, artifact_dir(step, method, key), {
        "step": step, "method": method, "key": key,
        "params": params, "inputs": inputs,
        "code": {"method_source_hash": code_hash, **code_version()},
        "format": store_format(),
        "created": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "tables": [],
        **extra,
    })
    a._save()
    return a


def ls(step: str | None = None, method: str | None = None) -> pd.DataFrame:
    """Every artifact on disk, newest first — the run log."""
    root = RUNS_DIR / step if step else RUNS_DIR
    if step and method:
        root = RUNS_DIR / step / method
    rows = []
    if root.exists():
        for mf in root.rglob("manifest.json"):
            try:
                m = json.loads(mf.read_text())
            except Exception:
                continue
            rows.append({
                "step": m.get("step"), "method": m.get("method"),
                "key": (m.get("key") or "")[:12], "status": m.get("status"),
                "created": m.get("created"),
                "params": _canonical(m.get("params", {}))[:60],
                "n_inputs": len(m.get("inputs", [])),
                "dirty": m.get("code", {}).get("git_dirty"),
            })
    df = pd.DataFrame(rows)
    return df.sort_values("created", ascending=False).reset_index(drop=True) if len(df) else df
