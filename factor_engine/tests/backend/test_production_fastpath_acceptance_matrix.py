# -*- coding: utf-8
"""Production fast path 验收矩阵（evidence 驱动）。"""
from __future__ import annotations

import pytest

pytest.importorskip("polars")


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _row(canon: str):
    from backend.fastpath_coverage import build_fastpath_coverage_row

    return build_fastpath_coverage_row(canon)


ACCEPTANCE_DUAL_BACKEND = [
    "add",
    "cs_pct_rank",
    "group_zscore",
    "multiply",
    "power",
    "protected_div",
    "protected_log",
    "rank",
    "subtract",
    "ts_delta",
    "ts_std",
    "vwap",
    "where",
    "zscore",
    "c_mean",
    "ts_zscore",
]

ACCEPTANCE_POLARS_ONLY: list[str] = []

ACCEPTANCE_P1_NOT_PRODUCTION = [
    "ts_sharpe",
    "cs_resid",
    "cs_regression",
    "rolling_beta",
    "cum_sum",
]

ACCEPTANCE_P2 = [
    "ewm_corr",
    "ts_kurt",
    "fillna_interpolate",
    "RSI_WILDER",
    "ts_decay_linear",
]


@pytest.mark.parametrize("canon", ACCEPTANCE_DUAL_BACKEND)
def test_acceptance_dual_backend_evidence(_loaded, canon):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from backend.production_fast_path import is_effective_dual_backend_fastpath
    from backend.sql_tiers import effective_sql_production_safe

    assert canon in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert is_polars_long_native_production_safe(canon)
    assert effective_sql_production_safe(canon)
    assert is_effective_dual_backend_fastpath(canon)


@pytest.mark.parametrize("canon", ACCEPTANCE_POLARS_ONLY)
def test_acceptance_polars_reference_without_dual(_loaded, canon):
    from backend.primitive_evidence import (
        POLARS_REFERENCE_PARITY_VERIFIED,
        PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    )

    assert canon in POLARS_REFERENCE_PARITY_VERIFIED
    assert canon not in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE


@pytest.mark.parametrize("canon", ACCEPTANCE_P1_NOT_PRODUCTION)
def test_acceptance_p1_not_dual_production(_loaded, canon):
    from backend.production_fast_path import is_effective_dual_backend_fastpath

    assert not is_effective_dual_backend_fastpath(canon)


@pytest.mark.parametrize("canon", ACCEPTANCE_P2)
def test_acceptance_p2_not_production(_loaded, canon):
    from backend.production_fast_path import is_effective_dual_backend_fastpath

    assert not is_effective_dual_backend_fastpath(canon)
