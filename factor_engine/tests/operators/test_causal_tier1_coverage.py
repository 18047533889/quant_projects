# -*- coding: utf-8 -*-
"""Tier-1 算子因果 / prefix-invariant 自动扫测。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_policy import TIER1_CANONICALS, infer_operator_policy
from cleaned_operators.registry import OperatorRegistry

pd = pytest.importorskip("pandas")

# 可用简单单变量 panel 做 prefix-invariant 测试的 Tier-1 算子
_TIER1_UNIVARIATE = frozenset(
    {
        "ts_delay", "ts_delta", "ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_min", "ts_max",
        "ts_ema", "ewm_mean", "SMA", "EMA", "WMA", "decay_linear", "ts_decay_linear",
        "hump_decay", "rank", "zscore", "winsorize", "scale", "normalize", "standardize",
        "quantile", "fillna_const", "ffill", "cum_prod", "cum_delta", "cum_first",
        "expanding_rank", "abs", "log", "clip",
    }
)

_SKIP = frozenset({"col", "corr_test", "bfill", "fillna_interpolate"})


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _panel(n: int = 8, cols: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    rng = np.arange(1, n + 1, dtype=float)
    data = {c: rng * (i + 1) for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=_dates(n))


def _op(name: str):
    op = OperatorRegistry.get(name)
    assert op is not None, name
    return op


def _assert_prefix_invariant(calc, x: pd.DataFrame) -> None:
    full = calc(x)
    col = x.columns[0]
    for t in range(len(x)):
        part = calc(x.iloc[: t + 1])
        got = full.iloc[t][col]
        exp = part.iloc[t][col]
        if pd.isna(got) and pd.isna(exp):
            continue
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-9), f"t={t}"


@pytest.mark.parametrize("canonical", sorted(TIER1_CANONICALS - _SKIP))
def test_tier1_prefix_invariant_or_pit_marked(canonical: str):
    if canonical not in _TIER1_UNIVARIATE:
        pytest.skip(f"{canonical} 需双变量/特殊 fixture")
    if not OperatorRegistry.backends_for(canonical):
        pytest.skip(f"{canonical} 未实现")
    op = OperatorRegistry.get(canonical)
    policy = infer_operator_policy(op, canonical=canonical)
    if not policy.pit_safe:
        pytest.skip(f"{canonical} pit_safe=False")

    x = _panel(8)

    def calc(df):
        kwargs = {}
        if canonical.startswith(("ts_", "SMA", "EMA", "WMA", "ewm_", "decay", "hump")):
            kwargs["window"] = 3
        if canonical == "quantile":
            kwargs["bins"] = 2
        if canonical == "ts_delay":
            kwargs["window"] = 1
        if canonical == "ts_delta":
            kwargs["window"] = 1
        return op.calculate(df, **kwargs)

    _assert_prefix_invariant(calc, x)


_TIER1_BIVARIATE = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "ts_corr",
        "ewm_corr",
    }
)


def _assert_prefix_invariant_on_result(calc, x: pd.DataFrame) -> None:
    full = calc(x)
    out_col = full.columns[0]
    for t in range(len(x)):
        part = calc(x.iloc[: t + 1])
        got = full.iloc[t][out_col]
        exp = part.iloc[t][out_col]
        if pd.isna(got) and pd.isna(exp):
            continue
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-9), f"t={t}"


@pytest.mark.parametrize("canonical", sorted(_TIER1_BIVARIATE & TIER1_CANONICALS))
def test_tier1_bivariate_prefix_invariant(canonical: str):
    if not OperatorRegistry.backends_for(canonical):
        pytest.skip(f"{canonical} 未实现")
    op = OperatorRegistry.get(canonical)
    policy = infer_operator_policy(op, canonical=canonical)
    if not policy.pit_safe:
        pytest.skip(f"{canonical} pit_safe=False")

    x = _panel(8, cols=("A", "B"))

    def calc(df):
        kwargs = {"window": 3}
        return op.calculate(df[["A"]], df[["B"]], **kwargs)

    _assert_prefix_invariant_on_result(calc, x)
