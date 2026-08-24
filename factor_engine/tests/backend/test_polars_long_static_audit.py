# -*- coding: utf-8
"""PolarsLong NATIVE tier 静态审计 CI。"""
from __future__ import annotations

from factor_engine.backend.polars_long_policy import (
    POLARS_LONG_NATIVE,
    POLARS_LONG_NONSTANDARD_ALG,
    POLARS_LONG_STATEFUL,
)
from factor_engine.backend.polars_long_static_audit import (
    assert_polars_long_native_clean,
    audit_polars_long_native,
)


def test_native_tier_disjoint_from_python_rolling():
    assert_polars_long_native_clean()


def test_nonstandard_alg_not_in_native():
    assert not (POLARS_LONG_NONSTANDARD_ALG & POLARS_LONG_NATIVE)


def test_stateful_wilder_not_in_native():
    assert "RSI_WILDER" in POLARS_LONG_STATEFUL
    assert "ATR_WILDER" in POLARS_LONG_STATEFUL
    assert not (POLARS_LONG_STATEFUL & POLARS_LONG_NATIVE)


def test_audit_returns_no_violations_for_current_native():
    assert audit_polars_long_native() == []
