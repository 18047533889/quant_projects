"""Independent bounded NumPy oracles for a representative R24 subset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


ROWS = 96
WINDOW = 20


def fixture() -> pd.DataFrame:
    rng = np.random.default_rng(2309)
    t = np.arange(ROWS, dtype=float)
    values = 0.01 * t + np.sin(t / 3.0) + 0.3 * np.sin(t / 11.0) + 0.08 * rng.normal(size=ROWS)
    return pd.DataFrame({"A": values, "B": values[::-1] + 0.04 * rng.normal(size=ROWS)})


def rolling(frame: pd.DataFrame, window: int, fn) -> np.ndarray:
    values = frame.to_numpy(float)
    out = np.full_like(values, np.nan)
    for row in range(window - 1, len(values)):
        for col in range(values.shape[1]):
            out[row, col] = fn(values[row - window + 1:row + 1, col])
    return out


def backend_array(op, frame: pd.DataFrame, backend: str, params: dict) -> np.ndarray:
    value = pl.from_pandas(frame) if backend == "polars" else frame
    result = op.calculate(value, **params)
    return result.to_numpy() if isinstance(result, pl.DataFrame) else result.to_numpy(dtype=float)


def main() -> None:
    load_all()
    frame = fixture()
    checks = {
        "ts_mean_abs_deviation": (
            {"window": WINDOW},
            rolling(frame, WINDOW, lambda x: np.mean(np.abs(x - np.mean(x)))),
        ),
        "ts_median_abs_deviation": (
            {"window": WINDOW},
            rolling(frame, WINDOW, lambda x: np.median(np.abs(x - np.median(x)))),
        ),
        "ts_product": (
            {"window": 12, "min_periods": 3, "skipna": True},
            rolling(frame, 12, np.prod),
        ),
        "ts_valid_count": (
            {"window": WINDOW, "min_periods": 1},
            rolling(frame, WINDOW, lambda x: float(np.isfinite(x).sum())),
        ),
        "ts_coverage_ratio": (
            {"window": WINDOW, "min_periods": 1},
            rolling(frame, WINDOW, lambda x: float(np.isfinite(x).mean())),
        ),
        "ts_positive_ratio": (
            {"window": WINDOW, "threshold": 0.0, "min_periods": 3},
            rolling(frame, WINDOW, lambda x: float(np.mean(x > 0.0))),
        ),
        "ts_negative_ratio": (
            {"window": WINDOW, "threshold": 0.0, "min_periods": 3},
            rolling(frame, WINDOW, lambda x: float(np.mean(x < 0.0))),
        ),
    }
    results = []
    for canonical, (params, expected) in checks.items():
        row = {"canonical": canonical, "params": params, "backends": {}}
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(canonical, backend, mode="any")
            if op is None:
                raise AssertionError(f"missing {canonical}/{backend}")
            actual = backend_array(op, frame, backend, params)
            mask = np.isfinite(expected)
            equal = actual.shape == expected.shape and np.allclose(
                actual[mask], expected[mask], rtol=1e-10, atol=1e-12, equal_nan=True
            )
            if not equal:
                delta = float(np.nanmax(np.abs(actual[mask] - expected[mask])))
                raise AssertionError(f"{canonical}/{backend} oracle mismatch max_delta={delta}")
            row["backends"][backend] = {"oracle_equal": True, "checked_cells": int(mask.sum())}
        results.append(row)
    output = Path(__file__).with_name("oracle_results.json")
    output.write_text(json.dumps({"rows": ROWS, "checks": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"oracle_checks": len(results), "all_equal": True, "output": str(output)}))


if __name__ == "__main__":
    main()
