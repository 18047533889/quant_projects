#!/usr/bin/env python3
"""Audit every retained Factor DSL canonical against production invariants.

For every Daily or Extended factor operator this executes the Pandas/Numpy
semantic-reference implementation on a deterministic synthetic panel and checks
production admission, same-shape output, determinism and prefix invariance.
ResearchToolRegistry utilities are intentionally outside this audit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


PANEL_PARAM_NAMES = frozenset({
    "x", "y", "z", "a", "b", "w", "g", "left", "right",
    "numerator", "denominator", "ret", "returns", "benchmark_ret",
    "market_ret", "benchmark", "market", "open", "high", "low", "close",
    "price", "volume", "amount", "vwap", "weight", "weights", "signal",
    "fallback", "condition", "group", "industry", "sector", "fiscal_quarter",
    "period_id", "quarter", "revision_id", "decision_time", "available_time",
    "available_at", "exposure", "exposures", "control", "controls", "factor",
    "target", "mask", "event", "value", "values", "sort_col",
})

SCALAR_VALUES: dict[str, Any] = {
    "window": 20, "d": 20, "n": 20, "m": 2, "span": 20, "period": 20,
    "periods": 4, "lag": 1, "lags": 1, "k": 3, "q": 0.2,
    "quantile": 0.2, "threshold": 0.0, "run": 2, "hump": 0.02,
    "min_periods": 5, "min_obs": 3, "ddof": 1, "ann_factor": 252,
    "decimals": 2, "to": 1.0, "lower": -2.0, "upper": 2.0, "eps": 1e-8,
    "epsilon": 1e-8, "alpha": 0.2, "fast": 12, "slow": 26,
    "fast_period": 12, "slow_period": 26, "signal_span": 9,
    "signal_window": 9, "signal_period": 9, "side": "lower",
    "order": "largest", "add_intercept": True, "clip": 3.0, "limit": 3,
    "max_gap": 3, "max_periods": 3, "max_lookback": 60, "power": 2.0,
    "exponent": 2.0, "p": 0.5, "c": 1.0, "fraction": 0.5,
    "buckets": 5, "top": 3, "asc": True, "annualization": 252,
    "annualization_factor": 252, "periods_per_year": 4, "method": "average",
    "interpolation": "linear", "center": True, "ascending": True,
    "inclusive": True, "offset": 0, "require_consecutive": True,
    "trim_pct": 0.1, "sign_policy": "strict", "denominator": "signed",
    "aggr_func": "sum",
}

# Canonical-specific legal parameter points. A generic name such as ``mode`` or
# ``method`` has different semantic domains across operators and must never be
# guessed globally.
SPECIAL_SCALARS: dict[tuple[str, str], Any] = {
    ("cs_rank_gaussian", "method"): "blom",
    ("cs_regression", "mode"): 0,
    ("cs_quantile", "p"): 0.5,
    ("group_percentile", "p"): 0.5,
    ("revision_delta", "mode"): "absolute",
    ("period_change", "mode"): "absolute",
    ("ts_nth_value", "order"): "largest",
}

SPECIAL_POSITIONAL: dict[str, tuple[str, ...]] = {
    # Canonical metadata contains an ellipsis marker because this operator has a
    # variable number of exposures. The smoke case supplies two explicit ones.
    "cs_multi_resid": ("target", "exposure", "control", "add_intercept", "min_obs"),
}


def _panels(rows: int = 96, cols: int = 6) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    assets = [f"A{i}" for i in range(cols)]
    t = np.arange(rows, dtype=float)[:, None]
    j = np.arange(cols, dtype=float)[None, :]
    base = 50.0 + 0.15 * t + 0.7 * j + np.sin(t / 5.0 + j / 3.0)
    close = pd.DataFrame(base, index=dates, columns=assets)
    open_ = close * (1.0 + 0.002 * np.cos(t / 4.0 + j))
    high = pd.DataFrame(np.maximum(open_, close) * 1.01, index=dates, columns=assets)
    low = pd.DataFrame(np.minimum(open_, close) * 0.99, index=dates, columns=assets)
    volume = pd.DataFrame(1_000_000.0 + 5000.0 * t + 10000.0 * j, index=dates, columns=assets)
    amount = volume * close
    ret = close.pct_change().fillna(0.0)
    market = pd.DataFrame(
        np.repeat(ret.mean(axis=1).to_numpy()[:, None], cols, axis=1),
        index=dates,
        columns=assets,
    )
    group_values = np.tile(
        np.array(["G0", "G1", "G2", "G0", "G1", "G2"], dtype=object)[:cols],
        (rows, 1),
    )
    group = pd.DataFrame(group_values, index=dates, columns=assets)
    condition = volume.gt(volume.rolling(5, min_periods=1).mean())
    quarters = pd.DataFrame(
        np.repeat((((np.arange(rows) // 20) % 4) + 1)[:, None], cols, axis=1),
        index=dates,
        columns=assets,
    )
    revision = pd.DataFrame(
        np.repeat((np.arange(rows) // 5)[:, None], cols, axis=1),
        index=dates,
        columns=assets,
    )
    decision_time = pd.DataFrame(
        np.repeat(dates.to_numpy()[:, None], cols, axis=1),
        index=dates,
        columns=assets,
    )
    available_time = decision_time - pd.Timedelta(days=3)
    weights = volume.div(volume.sum(axis=1), axis=0)
    control = volume.pct_change().fillna(0.0)
    zero = pd.DataFrame(0.0, index=dates, columns=assets)
    return {
        "x": close, "y": open_, "z": control, "a": close, "b": open_,
        "w": weights, "g": group, "left": close, "right": open_,
        "numerator": close, "denominator": open_.abs() + 1.0,
        "ret": ret, "returns": ret, "benchmark_ret": market,
        "market_ret": market, "benchmark": market, "market": market,
        "open": open_, "high": high, "low": low, "close": close,
        "price": (high + low + close) / 3.0, "volume": volume, "amount": amount,
        "vwap": amount / volume, "weight": weights, "weights": weights,
        "signal": ret, "fallback": zero, "condition": condition, "mask": condition,
        "event": condition, "group": group, "industry": group, "sector": group,
        "fiscal_quarter": quarters, "period_id": quarters, "quarter": quarters,
        "revision_id": revision, "decision_time": decision_time,
        "available_time": available_time, "available_at": available_time,
        "exposure": market, "exposures": market, "control": control,
        "controls": control, "factor": market, "target": ret, "sort_col": volume,
        "value": close, "values": close,
    }


def _value_for_parameter(
    canonical: str,
    name: str,
    panels: dict[str, pd.DataFrame],
) -> Any:
    key = str(name)
    special = SPECIAL_SCALARS.get((canonical, key))
    if (canonical, key) in SPECIAL_SCALARS:
        return special
    if key in panels:
        return panels[key]
    if key in SCALAR_VALUES:
        return SCALAR_VALUES[key]
    if key.startswith("window") or key.endswith("_window"):
        return 20
    if key.startswith("min_period"):
        return 5
    if key.endswith("_period"):
        return 20
    if key.endswith("_id"):
        return panels["period_id"]
    if key in PANEL_PARAM_NAMES:
        return panels["x"]
    raise KeyError(key)


def _build_call(
    canonical: str,
    op: Any,
    panels: dict[str, pd.DataFrame],
) -> tuple[list[Any], dict[str, Any]]:
    """Build calls from the canonical Registry contract, never wrapper introspection."""
    from cleaned_operators.registry import OperatorRegistry

    catalog = OperatorRegistry._catalog.get(canonical, {})
    names = tuple(str(x) for x in (catalog.get("param_names") or ()))
    if not names:
        names = ("x",)

    if canonical in SPECIAL_POSITIONAL:
        return [
            _value_for_parameter(canonical, name, panels)
            for name in SPECIAL_POSITIONAL[canonical]
        ], {}

    positional: list[Any] = []
    for name in names:
        if name == "...":
            continue
        positional.append(_value_for_parameter(canonical, name, panels))
    return positional, {}


def _slice_value(value: Any, rows: int) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.iloc[:rows]
    return value


def _to_frame(value: Any, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        if value.index.equals(template.index):
            return pd.DataFrame(
                np.repeat(value.to_numpy()[:, None], template.shape[1], axis=1),
                index=template.index,
                columns=template.columns,
            )
        raise TypeError("Series result does not share the time index")
    arr = np.asarray(value)
    if arr.shape == template.shape:
        return pd.DataFrame(arr, index=template.index, columns=template.columns)
    if np.isscalar(value):
        return pd.DataFrame(value, index=template.index, columns=template.columns)
    raise TypeError(f"unsupported result shape/type: {type(value).__name__} {arr.shape}")


def _numeric_delta(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    try:
        av = a.to_numpy(dtype=float)
        bv = b.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    finite = np.isfinite(av) & np.isfinite(bv)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(av[finite] - bv[finite])))


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.shape != b.shape or not a.index.equals(b.index) or not a.columns.equals(b.columns):
        return False
    av, bv = a.to_numpy(), b.to_numpy()
    if av.dtype.kind in "biufc" and bv.dtype.kind in "biufc":
        return bool(np.allclose(av, bv, equal_nan=True, rtol=1e-8, atol=1e-10))
    return a.astype(object).where(pd.notna(a), None).equals(
        b.astype(object).where(pd.notna(b), None)
    )


def audit() -> list[str]:
    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import (
        check_factor_production_hardening,
        factor_production_targets,
    )
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.registry import OperatorRegistry
    from backend.operator_capability import production_eligible_backends

    load_all()
    errors = list(check_factor_production_hardening())
    full_panels = _panels()
    template = full_panels["x"]
    prefix_rows = 72

    for canonical in sorted(factor_production_targets()):
        spec = build_operator_spec(canonical)
        if spec is None:
            errors.append(f"{canonical}: no OperatorSpec")
            continue
        if not spec.allow_in_production:
            errors.append(f"{canonical}: OperatorSpec.allow_in_production=False")
            continue
        if not production_eligible_backends(canonical):
            errors.append(f"{canonical}: no production eligible backend")
            continue
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        if op is None:
            errors.append(f"{canonical}: no pandas semantic reference")
            continue
        try:
            args, kwargs = _build_call(canonical, op, full_panels)
            result1 = _to_frame(op.calculate(*args, **kwargs), template)
            result2 = _to_frame(op.calculate(*args, **kwargs), template)
            if not result1.index.equals(template.index) or not result1.columns.equals(template.columns):
                errors.append(f"{canonical}: output does not preserve panel axes")
                continue
            if not _equal(result1, result2):
                errors.append(f"{canonical}: non-deterministic repeated evaluation")
                continue

            p_args = [_slice_value(x, prefix_rows) for x in args]
            p_kwargs = {k: _slice_value(v, prefix_rows) for k, v in kwargs.items()}
            p_template = template.iloc[:prefix_rows]
            prefix = _to_frame(op.calculate(*p_args, **p_kwargs), p_template)
            historical = result1.iloc[:prefix_rows]
            if not _equal(historical, prefix):
                delta = _numeric_delta(historical, prefix)
                errors.append(
                    f"{canonical}: prefix invariance / causality violation"
                    + (f" (max_abs_delta={delta:.6g})" if delta is not None else "")
                )
        except Exception as exc:
            errors.append(f"{canonical}: {type(exc).__name__}: {exc}")

    return errors


def main() -> int:
    errors = audit()
    if errors:
        print(f"factor production audit FAILED ({len(errors)} issues)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    from cleaned_operators.production_hardening import factor_production_targets
    print(f"factor production audit passed ({len(factor_production_targets())} canonicals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
