"""Bounded omitted-default versus explicit-canonical-default runtime probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


ROWS, ASSETS = 48, 4
t = np.arange(ROWS, dtype=float)[:, None]
a = np.arange(ASSETS, dtype=float)[None, :]
X = pd.DataFrame(10 + 0.08 * t + np.sin(t / 3 + a), columns=list("ABCD"))
COND = pd.DataFrame(((t.astype(int) + a.astype(int)) % 5 == 0), columns=X.columns)

CASES = {
    "MACD_line": ((X,), {"signal": 9}),
    "power": ((X,), {"y": 2.0}),
    "scale": ((X,), {"to": 1.0}),
    "signed_power": ((X - 12,), {"c": 2.0}),
    "trade_when": ((COND, X), {"fallback": 0.0}),
    "ts_bottomk_mean": ((X,), {"window": 20}),
    "ts_bottomk_std": ((X,), {"window": 20}),
    "ts_bottomk_sum": ((X,), {"window": 20}),
    "ts_tail_mean": ((X,), {"min_periods": 2}),
    "ts_topk_mean": ((X,), {"window": 20}),
    "ts_topk_std": ((X,), {"window": 20}),
    "ts_topk_sum": ((X,), {"d": 20}),
}


def backend_arg(value, backend):
    if backend == "pandas_numpy":
        return value
    return pl.DataFrame({"date": np.arange(len(value)), **{c: value[c] for c in value.columns}})


def array(value):
    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=float)
    cols = [c for c in value.columns if c not in {"date", "timestamp", "trade_date", "datetime", "__fe_time__"}]
    return value.select(cols).to_numpy().astype(float)


def main():
    ensure_cleaned_loaded()
    rows = []
    for canonical, (args, explicit) in CASES.items():
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(canonical, backend, mode="any")
            if op is None:
                rows.append({"canonical": canonical, "backend": backend, "status": "NO_BACKEND"})
                continue
            converted = tuple(backend_arg(x, backend) for x in args)
            try:
                omitted = array(op.calculate(*converted))
                spelled = array(op.calculate(*converted, **explicit))
                equal = omitted.shape == spelled.shape and np.allclose(
                    omitted, spelled, equal_nan=True, rtol=0, atol=0
                )
                rows.append({
                    "canonical": canonical,
                    "backend": backend,
                    "explicit": explicit,
                    "equal": bool(equal),
                    "shape": list(omitted.shape),
                    "finite_omitted": int(np.isfinite(omitted).sum()),
                    "finite_explicit": int(np.isfinite(spelled).sum()),
                    "max_abs_diff": 0.0 if equal else float(np.nanmax(np.abs(omitted - spelled))),
                    "status": "EQUIVALENT" if equal else "DIFFERENT",
                })
            except Exception as exc:
                rows.append({
                    "canonical": canonical, "backend": backend,
                    "explicit": explicit, "status": "ERROR",
                    "error_type": type(exc).__name__, "error": str(exc),
                })
    sources = {}
    for path in (
        "factor_engine/cleaned_operators/overhaul/technical.py",
        "factor_engine/cleaned_operators/common/elementwise.py",
        "factor_engine/cleaned_operators/common/cross_sectional.py",
        "factor_engine/cleaned_operators/technical/signal.py",
        "factor_engine/cleaned_operators/layer_topk_compat.py",
        "factor_engine/cleaned_operators/overhaul/daily.py",
    ):
        p = Path(path)
        sources[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    print(json.dumps({"rows": rows, "source_sha256": sources}, sort_keys=True))


if __name__ == "__main__":
    main()
