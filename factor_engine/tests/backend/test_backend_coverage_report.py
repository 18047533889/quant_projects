# -*- coding: utf-8
"""Backend 三层覆盖与默认配置门禁。"""

from __future__ import annotations

import pytest

from factor_engine.runtime.config import BackendConfig, FactorEngineConfig, FactorDefinitionConfig, DataSourceConfig
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_policy import (
    POLARS_PARITY_VERIFIED,
    POLARS_PRODUCTION_SAFE,
    POLARS_PRODUCTION_SAFE_CORE,
    check_polars_production_gate,
    polars_implemented_canonicals,
)
from factor_engine.cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS


def test_backend_config_default_is_auto():
    assert BackendConfig().type == "auto"


def test_factor_engine_config_default_backend_is_auto():
    cfg = FactorEngineConfig(
        factor=FactorDefinitionConfig(name="t", expr="close"),
        data_source=DataSourceConfig(type="data_access", options={"dataset": "x"}),
    )
    assert cfg.backend.type == "auto"


def test_production_core_is_not_self_certifying(loaded):
    assert POLARS_PRODUCTION_SAFE <= POLARS_PARITY_VERIFIED


def test_polars_production_gate_contract(loaded):
    assert check_polars_production_gate() == []


def test_polars_safe_requires_independent_edge_and_no_fallback_evidence(loaded):
    from factor_engine.backend.primitive_evidence import POLARS_EDGE_VERIFIED, POLARS_NO_FALLBACK_VERIFIED

    assert POLARS_PRODUCTION_SAFE <= POLARS_EDGE_VERIFIED
    assert POLARS_PRODUCTION_SAFE <= POLARS_NO_FALLBACK_VERIFIED


def test_polars_implemented_superset_production_safe(loaded):
    assert POLARS_PRODUCTION_SAFE <= polars_implemented_canonicals()


def test_production_safe_capabilities_have_explicit_contract_metadata(loaded):
    from factor_engine.backend.operator_capability import capability_for
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS

    required = {
        "execution_kind", "supports_lazy", "supports_streaming",
        "materializes_full_panel", "supports_nulls", "supports_nan",
        "supports_inf", "supports_scalar_broadcast", "supports_group",
        "supports_window", "supports_min_periods",
    }
    for canonical in DAILY_CANONICALS:
        for backend in ("polars", "duckdb_sql"):
            capability = capability_for(canonical, backend)
            assert capability.status == "production_safe", (canonical, backend)
            row = capability.to_csv_row()
            assert required <= set(row)
            assert capability.execution_kind != "unsupported"


def test_policy_table_contains_only_active_canonicals_and_covers_daily(loaded):
    from factor_engine.cleaned_operators.operator_policy import _EXPLICIT_POLICIES
    from factor_engine.cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        INTERNAL_ONLY_CANONICALS,
        LEGACY_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
    )

    active = (
        DAILY_CANONICALS | EXTENDED_ONLY_CANONICALS | INTERNAL_ONLY_CANONICALS
        | LEGACY_ONLY_CANONICALS | RESEARCH_ONLY_CANONICALS
    )
    assert set(_EXPLICIT_POLICIES) <= active
    assert DAILY_CANONICALS <= set(_EXPLICIT_POLICIES)
    assert all(_EXPLICIT_POLICIES[name].get("scope") != "unknown" for name in active)


@pytest.fixture(scope="module")
def loaded():
    load_all()
    yield
