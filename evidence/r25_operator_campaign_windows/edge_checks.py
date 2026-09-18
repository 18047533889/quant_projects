"""Bounded minimum-window, constant, NaN-gap, parity and prefix probes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


MIN_PARAMS = {
    "ts_chord_excursion_area": {"window": 2, "min_periods": 2},
    "ts_cumulative_deviation_score": {"window": 3, "min_periods": 1},
    "ts_current_drawdown_area": {"window": 2},
    "ts_current_drawdown_duration": {"window": 2},
    "ts_cusum_vol_break_score": {"window": 4, "min_periods": 4},
    "ts_effective_turning_rate": {"window": 3, "epsilon": 0.0, "min_periods": 3},
    "ts_endpoint_deviation": {"window": 3, "min_periods": 3},
    "ts_max_chord_excursion": {"window": 2, "min_periods": 2},
    "ts_max_drawdown": {"window": 2, "min_periods": 2},
    "ts_recovery_fraction": {"window": 2},
    "ts_time_under_water": {"window": 2},
    "ts_lo_mackinlay_vr": {"window": 6, "q": 2, "min_periods": 1},
    "ts_lo_mackinlay_z": {"window": 6, "q": 2, "min_periods": 1},
    "ts_variance_ratio_proxy": {"window": 7, "q": 2, "min_periods": 1},
    "ts_level_shift_score": {"window": 20, "min_periods": 1},
    "ts_vol_shift_score": {"window": 4, "min_periods": 1},
    "ts_rolling_median_causal": {"window": 2, "min_periods": 1},
    "ts_causal_local_linear_smoother": {"window": 2, "min_periods": 2},
    "ts_tail_imbalance": {"window": 8, "k": 1.0, "min_periods": 8},
    "ts_time_slope": {"window": 2, "min_periods": 1},
}


def as_backend(frame: pd.DataFrame, backend: str):
    return pl.from_pandas(frame) if backend == "polars" else frame


def array(value) -> np.ndarray:
    return value.to_numpy() if isinstance(value, pl.DataFrame) else value.to_numpy(dtype=float)


def main() -> None:
    load_all()
    rows = 32
    t = np.arange(rows, dtype=float)
    varying = pd.DataFrame({"A": np.sin(t / 3) + t / 50, "B": np.cos(t / 5) - t / 70})
    constant = pd.DataFrame({"A": np.full(rows, 1.25), "B": np.full(rows, -0.75)})
    nan_gap = varying.copy()
    nan_gap.iloc[[3, 4, 11, 19], 0] = np.nan
    nan_gap.iloc[[2, 9, 10, 25], 1] = np.nan
    fixtures = {"minimum_window": varying, "constant": constant, "nan_gap": nan_gap}
    cut = rows - 7
    results = []
    for canonical, params in MIN_PARAMS.items():
        row = {"canonical": canonical, "params": params, "fixtures": {}}
        for fixture_name, frame in fixtures.items():
            outputs = {}
            prefixes = {}
            changed = frame.copy()
            changed.iloc[cut:] = changed.iloc[cut:] * -13.0 + 17.0
            for backend in ("pandas_numpy", "polars"):
                op = OperatorRegistry.get(canonical, backend, mode="any")
                if op is None:
                    raise AssertionError(f"missing {canonical}/{backend}")
                actual = array(op.calculate(as_backend(frame, backend), **params))
                future = array(op.calculate(as_backend(changed, backend), **params))
                outputs[backend] = actual
                prefixes[backend] = bool(np.allclose(
                    actual[:cut], future[:cut], equal_nan=True, rtol=1e-10, atol=1e-12
                ))
            parity = bool(np.allclose(
                outputs["pandas_numpy"], outputs["polars"],
                equal_nan=True, rtol=1e-10, atol=1e-12,
            ))
            if not parity or not all(prefixes.values()):
                raise AssertionError(
                    f"{canonical}/{fixture_name}: parity={parity}, prefixes={prefixes}"
                )
            row["fixtures"][fixture_name] = {
                "pandas_polars_equal": True,
                "future_prefix_invariant": True,
                "finite_pandas": int(np.isfinite(outputs["pandas_numpy"]).sum()),
                "finite_polars": int(np.isfinite(outputs["polars"]).sum()),
            }
        results.append(row)
    output = Path(__file__).with_name("edge_results.json")
    output.write_text(json.dumps({"rows": rows, "cut": cut, "checks": results}, indent=2) + "\n")
    print(json.dumps({"canonicals": len(results), "fixtures_each": 3, "all_equal_and_prefix_stable": True}))


if __name__ == "__main__":
    main()
