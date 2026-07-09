# -*- coding: utf-8
"""OperatorPolicy PIT 安全：intentional pandas-only / 前视算子标记。"""
from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_policy import (
    INTENTIONALLY_PANDAS_ONLY,
    PANDAS_ONLY_PIT_SAFE,
    PIT_UNSAFE_CANONICALS,
    infer_operator_policy,
)
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


@pytest.mark.parametrize(
    "canonical",
    sorted(INTENTIONALLY_PANDAS_ONLY - PANDAS_ONLY_PIT_SAFE),
)
def test_intentionally_pandas_only_marked_pit_unsafe(canonical: str):
    if canonical not in OperatorRegistry._operators:
        pytest.skip(f"{canonical} 未注册")
    op = OperatorRegistry.get(canonical, backend="pandas_numpy")
    policy = infer_operator_policy(op, canonical=canonical)
    assert policy.pit_safe is False, canonical


def test_ts_mean_pit_safe():
    op = OperatorRegistry.get("ts_mean", backend="pandas_numpy")
    assert infer_operator_policy(op, canonical="ts_mean").pit_safe is True


def test_shuffle_in_pit_unsafe_set():
    assert "shuffle" in PIT_UNSAFE_CANONICALS
    assert "fft" in PIT_UNSAFE_CANONICALS
