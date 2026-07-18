# -*- coding: utf-8
"""Backend 三层覆盖与默认配置门禁。"""

from __future__ import annotations

import pytest

from runtime.config import BackendConfig, FactorEngineConfig, FactorDefinitionConfig, DataSourceConfig
from cleaned_operators import load_all
from cleaned_operators.operator_policy import (
    POLARS_PARITY_VERIFIED,
    POLARS_PRODUCTION_SAFE,
    POLARS_PRODUCTION_SAFE_CORE,
    check_polars_production_gate,
    polars_implemented_canonicals,
)
from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS


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
    from backend.primitive_evidence import POLARS_EDGE_VERIFIED, POLARS_NO_FALLBACK_VERIFIED

    assert POLARS_PRODUCTION_SAFE <= POLARS_EDGE_VERIFIED
    assert POLARS_PRODUCTION_SAFE <= POLARS_NO_FALLBACK_VERIFIED


def test_polars_implemented_superset_production_safe(loaded):
    assert POLARS_PRODUCTION_SAFE <= polars_implemented_canonicals()


@pytest.fixture(scope="module")
def loaded():
    load_all()
    yield
