# -*- coding: utf-8
"""统一 OPERATOR_CASES 注册表：Batch A 认证与 evidence 同步单一来源。"""
from __future__ import annotations

from typing import Callable

from factor_engine.backend.operator_upgrade_matrix import BATCH_A_CERTIFICATION_ONLY

CaseList = list[tuple[str, Callable]]


def batch_a_pending() -> list[str]:
    """Batch A 中尚未六证齐全的算子。"""
    from factor_engine.backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    return sorted(c for c in BATCH_A_CERTIFICATION_ONLY if c not in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE)


def collect_all_evidence_cases() -> dict[str, CaseList]:
    """合并各 parity 套件 case 列表（按 canonical 去重）。"""
    from tests.backend_parity.evidence_case_registry import merge_case_lists
    from tests.backend_parity.test_p0_edge_cases_triple_parity import (
        DUCKDB_EDGE_CASES,
        EDGE_CASES,
    )
    from tests.backend_parity.test_polars_long_no_pandas_path import NO_PANDAS_CASES
    from tests.backend_parity.test_production_core_triple_parity import DUCKDB_CASES, MEMORY_CASES
    from tests.backend_parity.test_production_safe_bulk_parity import (
        DUCKDB_BULK_CASES,
        POLARS_BULK_CASES,
    )

    return {
        "polars_reference": merge_case_lists(MEMORY_CASES, POLARS_BULK_CASES),
        "polars_edge": merge_case_lists(EDGE_CASES),
        "duckdb_reference": merge_case_lists(DUCKDB_CASES, DUCKDB_BULK_CASES),
        "duckdb_edge": merge_case_lists(DUCKDB_EDGE_CASES),
        "no_fallback": merge_case_lists(NO_PANDAS_CASES),
    }
