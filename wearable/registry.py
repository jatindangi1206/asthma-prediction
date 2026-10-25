"""Method discovery: drop a file in a pool directory and it is registered.

There is no list to maintain. Each pool is a package directory, and every module in it
that exposes the pool's entry point becomes an available method:

    pool          directory             entry point          extra module attributes
    ----------    -------------------   ------------------   ------------------------
    smoothing     wearable/smoothing/   smooth_dataframe     CONFIG, DEFAULTS, LABEL
    cpd           wearable/cpd/         detect               DEFAULTS, LABEL
    models        wearable/models/      fit + predict_proba  DEFAULTS, LABEL
    metrics       wearable/metrics/     compute              KIND, LABEL

The extras are optional. A module starting with `_`, or not exposing the entry point,
is skipped — so helpers can live in a pool directory. `source_hash` (of the module's
own file) goes into the artifact key, so editing a method invalidates only its results.
"""
from __future__ import annotations

import hashlib
import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# pool -> the callable(s) a module must expose to count as a method
POOLS: dict[str, tuple[str, ...]] = {
    "smoothing": ("smooth_dataframe",),
    "cpd": ("detect",),
    "models": ("fit", "predict_proba"),
    "metrics": ("compute",),
}


@dataclass(frozen=True)
class Method:
    pool: str
    name: str
    module: str
    label: str
    defaults: dict[str, Any] = field(default_factory=dict)
    kind: str = ""          # metrics only: "model" (needs y_true/y_prob) or "signal"
    source_hash: str = ""

    def load(self):
        return importlib.import_module(self.module)

    def entry(self, which: int = 0):
        return getattr(self.load(), POOLS[self.pool][which])

    def config(self, **overrides):
        """The method's Config dataclass filled from defaults + overrides, or the merged
        plain dict when the method declares no CONFIG class."""
        mod = self.load()
        params = {**self.defaults, **overrides}
        cls = getattr(mod, "CONFIG", None)
        return cls(**params) if cls is not None else params

    def params(self, **overrides) -> dict:
        """The resolved parameters, as plain data for the provenance record."""
        return {**self.defaults, **overrides}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def discover(pool: str) -> dict[str, Method]:
    """Import every module in the pool directory and keep the ones that qualify."""
    if pool not in POOLS:
        raise KeyError(f"unknown pool {pool!r}; available: {sorted(POOLS)}")
    pkg = importlib.import_module(f"wearable.{pool}")
    required = POOLS[pool]
    found: dict[str, Method] = {}

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        dotted = f"wearable.{pool}.{info.name}"
        try:
            mod = importlib.import_module(dotted)
        except Exception as exc:                 # a broken/optional method must not
            found[info.name] = Method(           # take the whole registry down
                pool, info.name, dotted,
                label=f"<unavailable: {type(exc).__name__}: {exc}>")
            continue
        if not all(hasattr(mod, fn) for fn in required):
            continue
        src = Path(mod.__file__)
        found[info.name] = Method(
            pool=pool, name=info.name, module=dotted,
            label=getattr(mod, "LABEL", info.name),
            defaults=dict(getattr(mod, "DEFAULTS", {})),
            kind=getattr(mod, "KIND", ""),
            source_hash=_hash_file(src),
        )
    return dict(sorted(found.items()))


_cache: dict[str, dict[str, Method]] = {}


def registry(pool: str) -> dict[str, Method]:
    """Discovery is cached per process — the filesystem is only scanned once."""
    if pool not in _cache:
        _cache[pool] = discover(pool)
    return _cache[pool]


def get(pool: str, name: str) -> Method:
    reg = registry(pool)
    if name not in reg:
        raise KeyError(f"unknown {pool} method {name!r}; available: {sorted(reg)}")
    m = reg[name]
    if m.label.startswith("<unavailable"):
        raise ImportError(f"{pool}/{name} failed to import: {m.label}")
    return m


def available(pool: str) -> list[str]:
    return [n for n, m in registry(pool).items() if not m.label.startswith("<unavailable")]
