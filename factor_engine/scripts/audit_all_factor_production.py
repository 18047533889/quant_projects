#!/usr/bin/env python3
"""Audit every retained Factor DSL canonical against production invariants.

Runtime-only mode bootstraps semantic evidence by executing the Pandas reference,
checking axes, determinism and prefix causality. Strict mode additionally
requires OperatorSpec admission and at least one evidence-backed physical
backend. Public parameters are generated from the final Registry contract;
unknown parameters fail closed instead of receiving an arbitrary panel.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(FE_ROOT.parent), str(FE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_PANEL_PARAMETERS = frozenset({
    "x", "y", "z", "a", "b", "w", "g", "left", "right",
    "numerator", "denominator", "ret", "returns", "benchmark_ret",
    "market_ret", "benchmark", "market", "open", "high", "low", "close",
    "price", "volume", "amount", "vwap", "turnover", "weight", "weights",
    "signal", "fallback", "condition", "group", "industry", "sector",
    "fiscal_quarter", "period_id", "target_period_id", "quarter", "revision_id",
    "decision_time", "available_time", "available_at", "exposure", "exposures",
    "control", "controls", "factor", "target", "mask", "event", "value",
    "values", "v1", "v2", "sort_col", "float_shares", "flow", "balance",
    "earnings", "cashflow", "assets", "working_capital", "base",
    "dollar_volume", "scale_base", "fundamental_x", "fundamental_y",
    "fundamental_scale", "actual", "expected", "expected_std", "expected_mean",
    "current_assets", "current_liabilities", "inventory", "total_debt",
    "total_equity", "short_debt", "long_debt", "cash", "total_assets",
    "operating_income", "revenue", "gross_profit", "net_income",
    "operating_cash_flow", "research_development", "capex", "invested_capital",
    "nopat", "receivables", "cost_of_goods_sold", "interest_expense",
    "market_cap",
})

_SCALAR_VALUES: dict[str, Any] = {
    "window": 20,
    "d": 20,
    "n": 3,
    "m": 2,
    "span": 20,
    "period": 20,
    "periods": 4,
    "lag": 1,
    "lags": 1,
    "k": 3,
    "q": 0.2,
    "quantile": 0.2,
    "threshold": 0.01,
    "run": 2,
    "hump": 0.02,
    "min_periods": 5,
    "min_obs": 8,
    "ddof": 1,
    "ann_factor": 252.0,
    "decimals": 2,
    "to": 1.0,
    "lower": -2.0,
    "upper": 2.0,
    "eps": 1e-8,
    "epsilon": 1e-8,
    "alpha": 0.2,
    "fast": 12,
    "slow": 26,
    "fast_period": 12,
    "slow_period": 26,
    "fast_window": 12,
    "slow_window": 26,
    "signal_span": 9,
    "signal_window": 9,
    "signal_period": 9,
    "side": "lower",
    "order": "largest",
    "add_intercept": True,
    "clip": 3.0,
    "limit": 3,
    "max_gap": 3,
    "max_periods": 8,
    "max_lookback": 60,
    "power": 2.0,
    "exponent": 2.0,
    "p": 0.5,
    "c": 1.0,
    "fraction": 0.5,
    "buckets": 5,
    "top": 3,
    "asc": True,
    "annualization": 252.0,
    "annualization_factor": 252.0,
    "periods_per_year": 4,
    "method": "std",
    "mode": "absolute",
    "interpolation": "linear",
    "center": True,
    "ascending": True,
    "inclusive": True,
    "offset": 0,
    "require_consecutive": True,
    "trim_pct": 0.1,
    "sign_policy": "strict",
    "denominator_policy": "signed",
    "aggr_func": "sum",
    "lo": -2.0,
    "hi": 2.0,
    "left_window": 3,
    "right_window": 3,
    "history_window": 60,
    "points": 3,
    "std_dev": 2.0,
    "skipna": True,
    "revision_policy": "latest_available",
    "missing_group_policy": "raise",
    "short_window": 7,
    "medium_window": 14,
    "long_window": 28,
    "ema_window": 20,
    "atr_window": 14,
    "tenkan_window": 9,
    "kijun_window": 26,
    "senkou_b_window": 52,
    "er_window": 10,
    "vol_window": 20,
    "baseline_window": 40,
    "price_window": 20,
    "volume_window": 20,
    "turnover_window": 20,
    "adl_window": 60,
    "impulse_window": 20,
    "flag_window": 10,
    "pennant_window": 12,
    "cup_window": 80,
    "handle_window": 15,
    "growth_periods": 4,
    "compare_periods": 1,
    "window_periods": 8,
    "average_periods": 2,
    "short_periods": 1,
    "long_periods": 4,
    "window_days": 60,
    "max_days": 252,
    "max_wait": 10,
    "body_window": 10,
    "shadow_window": 10,
    "tolerance": 0.03,
    "min_depth": 0.05,
    "min_spacing": 5,
    "max_spacing": 40,
    "shoulder_tolerance": 0.05,
    "head_min_prominence": 0.05,
    "max_neckline_slope": 0.02,
    "slope_threshold": 0.001,
    "parallel_tolerance": 0.001,
    "min_impulse": 0.08,
    "min_swing": 0.05,
    "min_fit": 0.01,
    "max_retracement": 0.5,
    "max_handle_retracement": 0.5,
    "max_width": 0.12,
    "max_edge_diff": 0.10,
    "volume_decay_threshold": 0.0,
    "multiplier": 2.0,
    "volume_scale": 1_000_000.0,
    "acceleration": 0.02,
    "maximum": 0.2,
    "penetration": 0.3,
    "short_weight": 4.0,
    "medium_weight": 2.0,
    "long_weight": 1.0,
    "pattern": "high_wave",
    "body_factor": 1.0,
    "shadow_factor": 1.0,
}

_SPECIAL_SCALARS: dict[tuple[str, str], Any] = {
    ("cs_rank_gaussian", "method"): "blom",
    ("cs_regression", "mode"): 0,
    ("cs_quantile", "p"): 0.5,
    ("group_percentile", "p"): 0.5,
    ("group_percentile", "side"): "top",
    ("group_percentile", "missing_group_policy"): "raise",
    ("revision_delta", "mode"): "absolute",
    ("period_change", "mode"): "absolute",
    ("period_lag", "revision_policy"): "latest_available",
    ("period_stability", "method"): "std",
    ("ts_nth_value", "order"): "largest",
    ("ts_mad", "scale"): 1.0,
    ("ts_product", "skipna"): True,
    ("MACD_line", "signal"): 9,
    ("MACD_signal", "signal"): 9,
    ("MACD_hist", "signal"): 9,
    ("fillna_const", "value"): 0.0,
    ("group_winsorize", "a"): 0.05,
    ("winsorize", "lower"): 0.05,
    ("winsorize", "upper"): 0.95,
}
_SPECIAL_POSITIONAL = {
    "cs_multi_resid": ("target", "exposure", "control"),
    "cs_neutralize": ("target", "exposure", "control"),
}
_SPECIAL_KWARGS = {
    "cs_multi_resid": {"add_intercept": True, "min_obs": 8},
    "cs_neutralize": {"add_intercept": True, "min_obs": 8},
}


def _panels(rows: int = 220, columns: int = 6) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    assets = [f"A{index}" for index in range(columns)]
    time = np.arange(rows, dtype=float)[:, None]
    asset = np.arange(columns, dtype=float)[None, :]
    close = pd.DataFrame(
        50.0 + 0.15 * time + 0.7 * asset + np.sin(time / 5.0 + asset / 3.0),
        index=dates,
        columns=assets,
    )
    open_ = close * (1.0 + 0.002 * np.cos(time / 4.0 + asset))
    high = pd.DataFrame(
        np.maximum(open_, close) * 1.01, index=dates, columns=assets
    )
    low = pd.DataFrame(
        np.minimum(open_, close) * 0.99, index=dates, columns=assets
    )
    volume = pd.DataFrame(
        1_000_000.0 + 5_000.0 * time + 10_000.0 * asset,
        index=dates,
        columns=assets,
    )
    amount = volume * close
    returns = close.pct_change().fillna(0.0)
    market = pd.DataFrame(
        np.repeat(returns.mean(axis=1).to_numpy()[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    groups = np.array(["G0", "G1", "G2", "G0", "G1", "G2"], dtype=object)
    group = pd.DataFrame(
        np.tile(groups[:columns], (rows, 1)), index=dates, columns=assets
    )
    condition = volume.gt(volume.rolling(5, min_periods=1).mean())

    report_number = np.arange(rows) // 10
    period = pd.DataFrame(
        np.repeat(report_number[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    target_period = pd.DataFrame(
        np.repeat((report_number + 1)[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    fiscal_quarter = pd.DataFrame(
        np.repeat((report_number % 4 + 1)[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    revision = pd.DataFrame(
        np.repeat((np.arange(rows) // 5)[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    decision = pd.DataFrame(
        np.repeat(dates.to_numpy()[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    available = decision - pd.Timedelta(days=3)
    weights = volume.div(volume.sum(axis=1), axis=0)
    control = volume.pct_change().fillna(0.0)
    zero = pd.DataFrame(0.0, index=dates, columns=assets)
    float_shares = pd.DataFrame(
        np.repeat(
            (100_000_000.0 + np.arange(rows, dtype=float)[:, None] * 10_000.0),
            columns,
            axis=1,
        ),
        index=dates,
        columns=assets,
    )
    turnover = volume / float_shares

    panels: dict[str, pd.DataFrame] = {
        "x": close,
        "y": open_,
        "z": control,
        "a": close,
        "b": open_,
        "w": weights,
        "g": group,
        "left": close,
        "right": open_,
        "numerator": close,
        "denominator": open_.abs() + 1.0,
        "ret": returns,
        "returns": returns,
        "benchmark_ret": market,
        "market_ret": market,
        "benchmark": market,
        "market": market,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price": (high + low + close) / 3.0,
        "volume": volume,
        "amount": amount,
        "vwap": amount / volume,
        "turnover": turnover,
        "weight": weights,
        "weights": weights,
        "signal": returns,
        "fallback": zero,
        "condition": condition,
        "mask": condition,
        "event": condition,
        "group": group,
        "industry": group,
        "sector": group,
        "fiscal_quarter": fiscal_quarter,
        "period_id": period,
        "target_period_id": target_period,
        "quarter": fiscal_quarter,
        "revision_id": revision,
        "decision_time": decision,
        "available_time": available,
        "available_at": available,
        "exposure": market,
        "exposures": market,
        "control": control,
        "controls": control,
        "factor": market,
        "target": returns,
        "sort_col": volume,
        "value": close,
        "values": close,
        "v1": close,
        "v2": open_,
        "float_shares": float_shares,
        "flow": close,
        "balance": open_.abs() + 10.0,
        "earnings": close,
        "cashflow": open_,
        "assets": close.abs() + 100.0,
        "working_capital": close,
        "base": close.abs() + 100.0,
        "dollar_volume": amount,
        "scale_base": close.abs() + 100.0,
        "actual": close * 1.02,
        "expected": close,
        "expected_std": close.abs() * 0.1 + 1.0,
        "expected_mean": close,
    }
    for name in sorted(_PANEL_PARAMETERS):
        if name in panels:
            continue
        panels[name] = close * (1.0 + 0.001 * (len(panels) % 17)) + 10.0
    return panels


def _value(canonical: str, name: str, panels: dict[str, pd.DataFrame]) -> Any:
    key = str(name)
    special = _SPECIAL_SCALARS.get((canonical, key))
    if special is not None:
        return special
    if key in panels:
        return panels[key]
    if key in _SCALAR_VALUES:
        return _SCALAR_VALUES[key]
    if key.endswith("_window") or key.startswith("window"):
        return 20
    if key.endswith("_periods"):
        return 4
    if key.endswith("_days"):
        return 60
    if key.startswith("min_"):
        return 0.01
    if key.startswith("max_"):
        return 0.5
    if key.endswith("_id"):
        return panels["period_id"]
    if key in _PANEL_PARAMETERS:
        return panels["x"]
    raise KeyError(f"unclassified public parameter {canonical}.{key}")


def _build_call(canonical: str, operator: Any, panels: dict[str, pd.DataFrame]):
    from cleaned_operators.registry import OperatorRegistry

    names = tuple(
        str(value)
        for value in (
            OperatorRegistry._catalog.get(canonical, {}).get("param_names") or ()
        )
    ) or ("x",)
    if canonical in _SPECIAL_POSITIONAL:
        arguments = [
            _value(canonical, name, panels)
            for name in _SPECIAL_POSITIONAL[canonical]
        ]
        return arguments, dict(_SPECIAL_KWARGS.get(canonical, {}))
    return [
        _value(canonical, name, panels) for name in names if name != "..."
    ], dict(_SPECIAL_KWARGS.get(canonical, {}))


def _slice(value: Any, rows: int) -> Any:
    return value.iloc[:rows] if isinstance(value, (pd.DataFrame, pd.Series)) else value


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
    array = np.asarray(value)
    if array.shape == template.shape:
        return pd.DataFrame(array, index=template.index, columns=template.columns)
    if np.isscalar(value):
        return pd.DataFrame(value, index=template.index, columns=template.columns)
    raise TypeError(
        f"unsupported result shape/type: {type(value).__name__} {array.shape}"
    )


def _equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if (
        left.shape != right.shape
        or not left.index.equals(right.index)
        or not left.columns.equals(right.columns)
    ):
        return False
    try:
        return bool(
            np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            )
        )
    except (TypeError, ValueError):
        return left.astype(object).where(pd.notna(left), None).equals(
            right.astype(object).where(pd.notna(right), None)
        )


def _numeric_delta(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
    try:
        left_values = left.to_numpy(dtype=float)
        right_values = right.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left_values[finite] - right_values[finite])))


def _implementation_label(operator: Any) -> str:
    cls = type(operator)
    source = inspect.getsourcefile(cls)
    try:
        source = str(Path(source).resolve().relative_to(FE_ROOT)) if source else "?"
    except Exception:
        source = str(source or "?")
    return f"{cls.__module__}.{cls.__name__}@{source}"


def audit(*, require_admission: bool = True) -> list[str]:
    from backend.operator_capability import production_eligible_backends
    from cleaned_operators import load_all
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    errors: list[str] = []
    panels = _panels()
    template = panels["x"]
    prefix_rows = 160

    for canonical in sorted(factor_production_targets()):
        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        if operator is None:
            errors.append(f"{canonical}: no pandas semantic reference")
            continue
        catalog = OperatorRegistry._catalog.get(canonical, {})
        policy = infer_operator_policy(operator, canonical=canonical)
        if str(catalog.get("status")) != "production":
            errors.append(f"{canonical}: lifecycle status is not production")
        if not policy.pit_safe or policy.lag < 0:
            errors.append(f"{canonical}: PIT policy is not causal")
        if not policy.shape_preserving:
            errors.append(f"{canonical}: shape_preserving=False")
        if require_admission:
            specification = build_operator_spec(canonical)
            if specification is None or not specification.allow_in_production:
                errors.append(f"{canonical}: OperatorSpec.allow_in_production=False")
                continue
            if not production_eligible_backends(canonical):
                errors.append(f"{canonical}: no production eligible backend")
                continue
        try:
            arguments, keyword_arguments = _build_call(canonical, operator, panels)
            first = _to_frame(
                operator.calculate(*arguments, **keyword_arguments), template
            )
            second = _to_frame(
                operator.calculate(*arguments, **keyword_arguments), template
            )
            if not first.index.equals(template.index) or not first.columns.equals(
                template.columns
            ):
                errors.append(
                    f"{canonical}: output axes changed [{_implementation_label(operator)}]"
                )
                continue
            if not _equal(first, second):
                errors.append(
                    f"{canonical}: non-deterministic repeated evaluation "
                    f"[{_implementation_label(operator)}]"
                )
                continue
            prefix_arguments = [_slice(value, prefix_rows) for value in arguments]
            prefix_kwargs = {
                key: _slice(value, prefix_rows)
                for key, value in keyword_arguments.items()
            }
            prefix_template = template.iloc[:prefix_rows]
            prefix = _to_frame(
                operator.calculate(*prefix_arguments, **prefix_kwargs),
                prefix_template,
            )
            historical = first.iloc[:prefix_rows]
            if not _equal(historical, prefix):
                delta = _numeric_delta(historical, prefix)
                details = (
                    f" (max_abs_delta={delta:.6g})" if delta is not None else ""
                )
                errors.append(
                    f"{canonical}: prefix invariance / causality violation{details} "
                    f"[{_implementation_label(operator)}]"
                )
        except Exception as error:
            errors.append(
                f"{canonical}: {type(error).__name__}: {error} "
                f"[{_implementation_label(operator)}]"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="skip admission evidence and audit semantic runtime only",
    )
    arguments = parser.parse_args()
    errors = audit(require_admission=not arguments.runtime_only)
    if errors:
        print(
            f"factor production audit FAILED ({len(errors)} issues)",
            file=sys.stderr,
        )
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    from cleaned_operators.production_hardening import factor_production_targets

    phase = "runtime" if arguments.runtime_only else "admission"
    print(
        f"factor production {phase} audit passed "
        f"({len(factor_production_targets())} canonicals)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
