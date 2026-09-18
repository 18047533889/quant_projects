"""Fresh paired evidence for Poly2 metadata isolation after robust-stats repair."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).with_name("ledger_attempt_4.jsonl")
SOURCES = {
    "numpy_kernels": ROOT / "factor_engine/cleaned_operators/_numpy_kernels.py",
    "polars_robust_stats": ROOT / "factor_engine/cleaned_operators/common/polars_robust_stats.py",
}
PROTOCOL = "r37:timestamp-irregular:rows=72:cut=48:d=16:seed=3702:x/y-independent-missing:v4"


def emit(row):
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def numeric(value):
    if isinstance(value, pl.DataFrame):
        return value.select([c for c in value.columns if c not in {"date", "timestamp"}]).to_numpy()
    return value.to_numpy(dtype=float)


def main():
    load_all()
    rng = np.random.default_rng(3702)
    n, cut = 72, 48
    bar = np.arange(n, dtype=float)
    x = np.sin(bar / 4.3) + 0.03 * bar + 0.08 * rng.normal(size=n)
    y = 0.8 - 0.5 * x + 0.12 * x ** 2 + 0.09 * rng.normal(size=n)
    x[[7, 23, 41]] = np.nan
    y[[11, 23, 55]] = np.nan
    timestamp = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        np.cumsum(np.resize([1, 4, 2, 1, 6], n)), unit="D"
    )
    pandas_y = pd.DataFrame({"A": y}, index=pd.DatetimeIndex(timestamp, name="timestamp"))
    pandas_x = pd.DataFrame({"A": x}, index=pandas_y.index)
    polars_y = pl.DataFrame({"timestamp": timestamp, "A": y})
    polars_x = pl.DataFrame({"timestamp": timestamp, "A": x})
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in SOURCES.items()}
    hashes["campaign"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256(
        pd.util.hash_pandas_object(pandas_y, index=True).values.tobytes()
        + pd.util.hash_pandas_object(pandas_x, index=True).values.tobytes()
    ).hexdigest()
    recipes = {
        "ts_poly2_coeff": ((pandas_y,), (polars_y,)),
        "ts_poly2_resid": ((pandas_y, pandas_x), (polars_y, polars_x)),
    }
    for name, (pandas_inputs, polars_inputs) in recipes.items():
        outputs = {}
        for backend, inputs in (("pandas_numpy", pandas_inputs), ("polars", polars_inputs)):
            op = OperatorRegistry.get(name, backend, mode="any")
            actual = numeric(op.calculate(*inputs, d=16))
            changed = []
            for frame in inputs:
                if isinstance(frame, pl.DataFrame):
                    changed.append(frame.with_columns(
                        pl.when(pl.int_range(pl.len()) >= cut)
                        .then(pl.col("A") * -9 + 13).otherwise(pl.col("A")).alias("A")
                    ))
                else:
                    altered = frame.copy()
                    altered.iloc[cut:, :] = altered.iloc[cut:, :] * -9 + 13
                    changed.append(altered)
            future = numeric(op.calculate(*changed, d=16))
            finite = int(np.isfinite(actual).sum())
            prefix = bool(np.allclose(actual[:cut], future[:cut], equal_nan=True, rtol=1e-10, atol=1e-12))
            outputs[backend] = actual
            emit({"canonical": name, "backend": backend, "status": "EXECUTED_FINITE" if finite else "FAILED",
                  "finite": finite, "future_prefix_invariant_all_inputs": prefix,
                  "shape": list(actual.shape), "sha256": hashes,
                  "fixture_sha256": fixture_hash, "protocol": PROTOCOL, "params": {"d": 16}})
        parity = bool(np.allclose(outputs["pandas_numpy"], outputs["polars"], equal_nan=True, rtol=1e-10, atol=1e-12))
        emit({"canonical": name, "status": "CANONICAL_PARITY" if parity else "CANONICAL_PARITY_FAILED",
              "pandas_polars_equal": parity, "sha256": hashes,
              "fixture_sha256": fixture_hash, "protocol": PROTOCOL, "params": {"d": 16}})


if __name__ == "__main__":
    main()
