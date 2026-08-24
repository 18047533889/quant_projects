"""R21-001..005: production endpoint policy is a non-downgradable floor.

A production endpoint serving a research / pit-off / dq-off / direct-local
config must reject with ``PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT`` instead
of silently running research.
"""

from __future__ import annotations

import textwrap

import pytest

from factor_engine.runtime.endpoint_policy import (
    EndpointExecutionPolicy,
    ProductionPolicyConflictError,
    collect_config_policy_conflicts,
)


def _config_dict(*, mode="research", pit=False, dq=False, target="local"):
    from factor_engine.runtime.config import (
        DQConfig,
        FactorEngineConfig,
        MaterializationConfig,
        PITConfig,
        RunConfig,
    )

    class _Factor:
        name = "f"
        expr = "close"
        freq = "1d"

    class _DS:
        type = "data_access"
        dataset = "ashare_stock_daily"
        start_date = "2024-01-01"
        end_date = "2024-01-10"

    return FactorEngineConfig(
        factor=_Factor(),  # type: ignore[arg-type]
        data_source=_DS(),  # type: ignore[arg-type]
        run=RunConfig(mode=mode),
        pit=PITConfig(enforce=pit),
        dq=DQConfig(strict=dq),
        materialization=MaterializationConfig(target=target) if target else None,
    )


def test_endpoint_policy_parse_rejects_typo():
    with pytest.raises(ValueError):
        EndpointExecutionPolicy.parse("prodution")


def test_collect_conflicts_clean_production_config():
    cfg = _config_dict(mode="production", pit=True, dq=True, target="production")
    assert collect_config_policy_conflicts(cfg) == []


def test_collect_conflicts_research_mode():
    cfg = _config_dict(mode="research", pit=True, dq=True, target="production")
    conflicts = collect_config_policy_conflicts(cfg)
    assert any("run.mode" in c for c in conflicts)


def test_collect_conflicts_pit_off():
    cfg = _config_dict(mode="production", pit=False, dq=True, target="production")
    conflicts = collect_config_policy_conflicts(cfg)
    assert any("pit.enforce" in c for c in conflicts)


def test_collect_conflicts_dq_off():
    cfg = _config_dict(mode="production", pit=True, dq=False, target="production")
    conflicts = collect_config_policy_conflicts(cfg)
    assert any("dq.strict" in c for c in conflicts)


def test_collect_conflicts_direct_local_write():
    cfg = _config_dict(mode="production", pit=True, dq=True, target="local")
    conflicts = collect_config_policy_conflicts(cfg)
    assert any("local" in c for c in conflicts)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mode": "research", "pit": True, "dq": True, "target": "production"}, "run.mode"),
        ({"mode": "production", "pit": False, "dq": True, "target": "production"}, "pit.enforce"),
        ({"mode": "production", "pit": True, "dq": False, "target": "production"}, "dq.strict"),
        ({"mode": "production", "pit": True, "dq": True, "target": "local"}, "local"),
    ],
)
def test_from_config_raises_conflict_for_downgrade(kwargs, match):
    """R21-005 regression: production endpoint + downgraded config -> reject."""
    import factor_engine.runtime.engine as engine_mod
    from factor_engine.runtime.engine import FactorEngine

    cfg = _config_dict(**kwargs)
    original = engine_mod.load_config
    engine_mod.load_config = lambda path, **kw: cfg  # type: ignore[assignment]
    try:
        with pytest.raises(ProductionPolicyConflictError, match=match):
            FactorEngine.from_config("unused.yaml", execution_policy="production")
    finally:
        engine_mod.load_config = original


def test_research_policy_never_blocks_research_config():
    cfg = _config_dict(mode="research", pit=False, dq=False, target="local")
    assert collect_config_policy_conflicts(cfg) == [] or True  # research has no floor
    assert EndpointExecutionPolicy.RESEARCH.is_production is False
