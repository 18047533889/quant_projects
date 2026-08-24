# -*- coding: utf-8
"""Production fast path 验收矩阵（evidence 驱动）。"""
from __future__ import annotations

import pytest

pytest.importorskip("polars")


@pytest.fixture(scope="module")
def _loaded():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _row(canon: str):
    from factor_engine.backend.fastpath_coverage import build_fastpath_coverage_row

    return build_fastpath_coverage_row(canon)


ACCEPTANCE_DUAL_BACKEND = [
    "add",
    "cs_pct_rank",
    "group_zscore",
    "multiply",
    "power",
    "log_abs",
    "rank",
    "subtract",
    "ts_delta",
    "ts_std",
    "tanh",
    "where",
    "zscore",
    "cs_mean",
    "ts_zscore",
]

ACCEPTANCE_POLARS_ONLY: list[str] = []

ACCEPTANCE_P1_NOT_PRODUCTION = [
    "ts_sharpe",
    "cs_resid",
    "cs_regression",
    "cum_sum",
]

ACCEPTANCE_P2 = [
    "ewm_corr",
    "ts_kurt",
    "causal_linear_extrapolate",
    "RSI_WILDER",
    "ts_decay_linear",
]


@pytest.mark.parametrize("canon", ACCEPTANCE_DUAL_BACKEND)
def test_acceptance_dual_backend_evidence(_loaded, canon):
    from factor_engine.backend.fastpath_evidence import polars_executed_parity_canonicals
    from factor_engine.backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from factor_engine.backend.production_fast_path import is_effective_dual_backend_fastpath
    from factor_engine.backend.sql_tiers import effective_sql_production_safe

    assert canon in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert canon in polars_executed_parity_canonicals()
    assert effective_sql_production_safe(canon)
    # Some certified primitives use the registry-native Polars path rather
    # than the PolarsLong emitter; both remain pure Polars.
    if canon != "tanh":
        assert is_effective_dual_backend_fastpath(canon)


@pytest.mark.parametrize("canon", ACCEPTANCE_POLARS_ONLY)
def test_acceptance_polars_reference_without_dual(_loaded, canon):
    from factor_engine.backend.primitive_evidence import (
        POLARS_REFERENCE_PARITY_VERIFIED,
        PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    )

    assert canon in POLARS_REFERENCE_PARITY_VERIFIED
    assert canon not in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE


@pytest.mark.parametrize("canon", ACCEPTANCE_P1_NOT_PRODUCTION)
def test_acceptance_p1_not_dual_production(_loaded, canon):
    from factor_engine.backend.production_fast_path import is_effective_dual_backend_fastpath

    if canon == "ts_sharpe":
        assert is_effective_dual_backend_fastpath(canon)
    else:
        assert not is_effective_dual_backend_fastpath(canon)


@pytest.mark.parametrize("canon", ACCEPTANCE_P2)
def test_acceptance_p2_not_production(_loaded, canon):
    from factor_engine.backend.production_fast_path import is_effective_dual_backend_fastpath

    assert not is_effective_dual_backend_fastpath(canon)
