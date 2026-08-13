# -*- coding: utf-8
"""Backend capability 注册表与 cost routing 测试。"""

from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE
from cleaned_operators.registry import OperatorRegistry
from backend.operator_capability import (
    ExecutionKind,
    build_capability_matrix,
    get_best_backend,
    resolve_canonical,
    summarize_operator,
)


def test_execution_kind_has_single_authority():
    from backend.polars_backend_kind import (
        BackendKind as PolarsBackendKind,
        ExecutionKind as PolarsExecutionKind,
        PhysicalImplementationSpec,
        polars_backend_kind,
    )

    assert PolarsExecutionKind is ExecutionKind
    assert ExecutionKind.POLARS_NATIVE_KERNEL is ExecutionKind.POLARS_NUMPY_KERNEL
    assert ExecutionKind.SQL_NATIVE is ExecutionKind.DUCKDB_NATIVE_SQL

    class DeclaredPolarsKernel:
        canonical = "declared_polars_kernel"
        _physical_spec = PhysicalImplementationSpec(
            canonical=canonical,
            backend="polars",
            execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
        )

    assert polars_backend_kind(DeclaredPolarsKernel()) is PolarsBackendKind.POLARS_NATIVE


@pytest.fixture(scope="module")
def loaded():
    load_all()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()
    yield


def test_ts_mean_production_capabilities(loaded):
    s = summarize_operator("ts_mean")
    assert s.pandas_numpy in {"parity_verified", "production_safe", "implemented"}
    assert s.polars in {"parity_verified", "production_safe"}
    assert s.duckdb_sql in {"parity_verified", "production_safe"}
    assert s.allow_in_production


def test_bfill_is_removed_from_runtime(loaded):
    assert OperatorRegistry.backends_for("bfill") == []
    assert OperatorRegistry.get("bfill") is None


def test_if_else_polars_long_tier(loaded):
    s = summarize_operator("if_else")
    assert s.polars_long_tier == "native"
    assert summarize_operator("where").polars in {"parity_verified", "production_safe"}


def test_get_best_backend_respects_production_safe(loaded):
    _, backend = get_best_backend("ts_mean", mode="production", prefer="auto")
    assert backend == "polars"
    canon = resolve_canonical("rank")
    _, rank_backend = get_best_backend(canon, mode="production", prefer="auto")
    assert rank_backend == ("polars" if canon in POLARS_PRODUCTION_SAFE else "pandas_numpy")


def test_capability_matrix_covers_implemented(loaded):
    matrix = build_capability_matrix()
    assert len(matrix) >= 200
    ts = next(r for r in matrix if r.canonical == "ts_std")
    assert ts.polars in {"parity_verified", "production_safe"}
    ts_mean = next(r for r in matrix if r.canonical == "ts_mean")
    assert ts_mean.polars in {"parity_verified", "production_safe", "implemented"}


def test_production_fast_path_whitelist(loaded):
    from backend.production_fast_path import (
        is_production_fast_path,
        summarize_production_fast_path,
    )

    assert is_production_fast_path("rank")
    assert is_production_fast_path("zscore")
    assert is_production_fast_path("ts_mean")
    assert not is_production_fast_path("RSI_WILDER")
    assert not is_production_fast_path("cs_resid")
    assert is_production_fast_path("ts_sharpe")
    summary = summarize_production_fast_path()
    assert summary["primitive_dual_backend_evidence_count"] >= 5
    assert "rank" in summary["production_fast_path"]


def test_get_best_backend_matches_registry_preferred(loaded):
    for name in ("ts_mean", "rank", "MACD"):
        op_a, b_a = OperatorRegistry.get_preferred(name, prefer="auto")
        op_b, b_b = get_best_backend(name, mode="production", prefer="auto")
        assert b_a == b_b
        assert (op_a is None) == (op_b is None)
