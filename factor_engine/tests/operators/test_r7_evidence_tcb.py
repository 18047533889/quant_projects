# -*- coding: utf-8 -*-
"""review #250-#254: edge-gate fail-closed, EdgeContract, backend-specific edge
evidence, ExecutionTCB factor evidence binding, and cache invalidation."""
from __future__ import annotations


def _full_evidence_for(*names: str) -> dict:
    """Synthetic review #252 nested per-backend edge evidence payload."""
    return {
        "edge_evidence": {
            "pandas_numpy": {
                "nan_verified": list(names),
                "inf_verified": list(names),
                "zero_verified": [],
                "domain_invalid_verified": [],
            },
            "polars": {
                "nan_verified": list(names),
                "inf_verified": list(names),
                "zero_verified": [],
                "domain_invalid_verified": [],
            },
            "duckdb": {
                "nan_verified": list(names),
                "inf_verified": list(names),
                "zero_verified": [],
                "domain_invalid_verified": [],
            },
        }
    }


def test_undeclared_operator_production_edge_gate_fails_closed():
    """review #250: undeclared == FAIL in production; no vacuous pass."""
    from cleaned_operators.edge_requirements import (
        EdgeGateMode,
        edge_contract,
        production_edge_evidence_complete,
    )

    # 'abs' is in no required set and not EDGE_IMMUNE.
    assert edge_contract("abs") is None
    # Even with every-dimension evidence present, undeclared still fails.
    full = _full_evidence_for("abs")
    assert production_edge_evidence_complete("abs", full) is False
    assert production_edge_evidence_complete("abs", full, mode=EdgeGateMode.PRODUCTION) is False
    # Research mode is a warning, never a production pass.
    assert production_edge_evidence_complete("abs", full, mode=EdgeGateMode.RESEARCH) is True


def test_declared_with_evidence_is_complete_per_backend():
    """review #251 + #252: contract-driven, backend-specific completeness."""
    from cleaned_operators.edge_requirements import (
        edge_contract,
        edge_evidence_status,
        missing_edge_dimensions,
        production_edge_evidence_complete,
    )

    # ts_std declares nan + inf required via the legacy lists.
    contract = edge_contract("ts_std")
    assert contract is not None
    assert contract.nan == "propagate"
    assert contract.pos_inf == "propagate"
    assert contract.neg_inf == "propagate"

    evidence = _full_evidence_for("ts_std")
    assert edge_evidence_status("ts_std", evidence, backend="duckdb") == "complete"
    assert production_edge_evidence_complete("ts_std", evidence, backend="duckdb") is True
    assert production_edge_evidence_complete("ts_std", evidence, backend="polars") is True

    # duckdb evidence does NOT back-stop pandas_numpy (review #252).
    duckdb_only = {
        "edge_evidence": {
            "pandas_numpy": {
                "nan_verified": [],
                "inf_verified": [],
                "zero_verified": [],
                "domain_invalid_verified": [],
            },
            "polars": {
                "nan_verified": [],
                "inf_verified": [],
                "zero_verified": [],
                "domain_invalid_verified": [],
            },
            "duckdb": {
                "nan_verified": ["ts_std"],
                "inf_verified": ["ts_std"],
                "zero_verified": [],
                "domain_invalid_verified": [],
            },
        }
    }
    assert (
        production_edge_evidence_complete("ts_std", duckdb_only, backend="pandas_numpy")
        is False
    )
    assert missing_edge_dimensions("ts_std", duckdb_only, backend="pandas_numpy") == {
        "nan",
        "inf",
    }
    assert (
        production_edge_evidence_complete("ts_std", duckdb_only, backend="duckdb")
        is True
    )


def test_legacy_flat_edge_evidence_fallback_still_routes():
    """review #252: pre-regeneration flat duckdb lists keep working."""
    from cleaned_operators.edge_requirements import (
        missing_edge_dimensions,
        production_edge_evidence_complete,
    )

    evidence = {
        "duckdb_nan_edge_verified": ["ts_std"],
        "duckdb_inf_edge_verified": ["ts_std"],
    }
    assert missing_edge_dimensions("ts_std", evidence, backend="duckdb") == set()
    assert production_edge_evidence_complete("ts_std", evidence, backend="duckdb") is True
    # Same legacy evidence does not cover polars.
    assert production_edge_evidence_complete("ts_std", evidence, backend="polars") is False


def test_edge_immune_requires_no_evidence():
    from cleaned_operators.edge_requirements import (
        edge_contract,
        production_edge_evidence_complete,
    )

    contract = edge_contract("reindex")
    assert contract is not None
    assert contract.nan == "ignore"
    assert contract.pos_inf == "ignore"
    assert contract.neg_inf == "ignore"
    assert contract.zero == "ignore"
    assert contract.domain_invalid == "ignore"
    assert production_edge_evidence_complete("reindex") is True


def test_execution_tcb_hash_included_in_factor_hashes():
    """review #253: factor evidence hash carries the full Execution TCB."""
    from backend.factor_operator_evidence import (
        ExecutionTCB,
        current_hashes,
        execution_tcb_hash,
    )

    assert len(ExecutionTCB) == 10
    hashes = current_hashes()
    assert hashes["execution_tcb_hash"] == execution_tcb_hash()
    # Sanity: the composite is a stable 16-hex sha256 prefix.
    composite = hashes["execution_tcb_hash"]
    assert composite and len(composite) == 16


def test_real_tcb_bridge_hash_is_present():
    """review #253: the real cleaned_bridge/panel_polars/backend_router sources
    are part of the execution TCB composite, so editing them invalidates
    factor evidence."""
    from backend.factor_operator_evidence import _EXECUTION_TCB_SOURCES

    for component in ("backend_bridge", "backend_router", "panel_conversion"):
        spec = _EXECUTION_TCB_SOURCES[component]
        assert spec.is_file() and spec.name.endswith(".py"), component


def test_execution_tcb_hash_changes_when_bridge_source_changes(tmp_path):
    """review #253: editing a TCB source invalidates the composite hash."""
    import backend.evidence_provenance as ep
    import backend.factor_operator_evidence as foe

    bridge = tmp_path / "cleaned_bridge.py"
    bridge.write_text("CONST = 1\n")
    old_sources = dict(foe._EXECUTION_TCB_SOURCES)
    try:
        foe._EXECUTION_TCB_SOURCES["backend_bridge"] = bridge
        h1 = foe.execution_tcb_hash()
        bridge.write_text("CONST = 2\n")
        ep.invalidate_all_evidence_caches()
        h2 = foe.execution_tcb_hash()
        assert h1 != h2
    finally:
        foe._EXECUTION_TCB_SOURCES.clear()
        foe._EXECUTION_TCB_SOURCES.update(old_sources)
        ep.invalidate_all_evidence_caches()


def test_invalidate_all_evidence_caches_actually_clears():
    """review #254: clearing must invalidate in-process caches."""
    import backend.evidence_provenance as ep
    import cleaned_operators.edge_requirements as er

    er.load_primitive_evidence()
    assert er.load_primitive_evidence.cache_info().currsize == 1
    cleared = ep.invalidate_all_evidence_caches()
    # All four evidence modules' lru_caches are registered and cleared.
    assert cleared >= 9
    assert er.load_primitive_evidence.cache_info().currsize == 0
    # Re-fetch repopulates from disk.
    er.load_primitive_evidence()
    assert er.load_primitive_evidence.cache_info().currsize == 1


def test_collect_cache_clearers_registry_is_callable():
    """review #254: certify/sync/delta-install can call every clearer."""
    from backend.evidence_provenance import collect_cache_clearers

    clearers = collect_cache_clearers()
    assert clearers
    for cache_clear in clearers:
        cache_clear()  # must not raise
