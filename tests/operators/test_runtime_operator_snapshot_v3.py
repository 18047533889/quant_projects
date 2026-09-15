from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.operator_snapshot import (
    _evidence_for, _native_accelerated, _parameter_contract, _research_callable,
    build_runtime_operator_snapshot, load_evidence_records,
    query_runtime_operator_snapshot,
)
import factor_engine.runtime.operator_snapshot as snapshot_module
from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec


class _Op:
    metadata = OperatorMetadata(
        name="demo", category="test", param_names=["x", "window", "mystery"],
        panel_params=("x",),
        param_specs={"window": ParamSpec(dtype=int, min=1, default=5, param_role=ParamRole.HORIZON)},
    )


class _FunctionBackedOp:
    metadata = OperatorMetadata(
        name="function_backed", category="test",
        param_names=["high", "low", "window", "enabled"],
    )

    @staticmethod
    def _kernel(high: "pd.DataFrame", low: "pd.DataFrame", window: int = 14,
                enabled: bool = True):
        return high

    _fn = _kernel

    def calculate(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _identity():
    return {"canonical": "demo", "backend": "pandas_numpy", "semantic_version": 1,
            "implementation_digest": "impl", "parameter_schema_digest": "schema"}


def test_parameter_schema_is_typed_and_unknown_is_fail_closed():
    schema, digest, verified = _parameter_contract(_Op(), ("x",))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["window"]["type"] == "integer"
    assert schema["properties"]["window"]["minimum"] == 1
    assert schema["properties"]["mystery"]["x-factor-engine-verification"] == "unknown"
    assert schema["x-factor-engine-contract-status"] == "unknown"
    assert len(digest) == 64
    assert verified is False


def test_function_backed_signature_is_authoritative_without_name_guesses():
    from factor_engine.cleaned_operators.operator_spec import _infer_panel_params

    op = _FunctionBackedOp()
    panels = _infer_panel_params(op, op.metadata, {})
    assert panels == ("high", "low")
    schema, _, verified = _parameter_contract(
        op, panels
    )
    assert schema["properties"]["high"] == {
        "x-factor-engine-role": "panel", "type": "array"
    }
    assert schema["properties"]["low"] == {
        "x-factor-engine-role": "panel", "type": "array"
    }
    assert schema["properties"]["window"] == {
        "x-factor-engine-verification": "unknown"
    }
    assert schema["properties"]["enabled"] == {
        "x-factor-engine-verification": "unknown"
    }
    assert verified is False


def test_pass_requires_actual_linked_test_artifact():
    rec = {**_identity(), "status": "PASS"}
    assert _evidence_for(_identity(), {("demo", "pandas_numpy"): rec})["status"] == "NOT_RUN"


def test_strings_that_look_linked_cannot_forge_pass():
    rec = {**_identity(), "status": "PASS", "test_node_id": "tests/x.py::test_x",
           "artifact_reference": "fake.json", "artifact_digest": "a" * 64}
    assert _evidence_for(_identity(), {("demo", "pandas_numpy"): rec})["status"] == "NOT_RUN"


def test_pandas_and_delegate_slots_never_count_as_native():
    rows = [
        {"backend": "pandas", "physical_execution_kind": "pandas_reference", "evidence": {}},
        {"backend": "polars", "physical_execution_kind": "polars_pandas_delegate", "evidence": {}},
    ]
    assert _native_accelerated(rows, {"pandas", "polars"}) is False


def test_registered_noop_without_typed_contract_is_not_research_callable():
    assert _research_callable([{"research_supported": True, "contract_status": "unknown"}]) is False


def test_snapshot_uses_authoritative_inferred_panel_topology():
    snapshot = build_runtime_operator_snapshot(profile="runtime")
    rows = {row["canonical"]: row for row in snapshot["operators"]}

    for canonical, expected_panels in {
        "ts_mean": {"x"},
        "ts_corr": {"x", "y"},
    }.items():
        row = rows[canonical]
        assert row["research_callable"] is True
        assert row["execution_state"] == "EXECUTABLE_UNVERIFIED"
        assert row["production_callable"] is False
        binding = next(b for b in row["backends"] if b["backend"] == "pandas_numpy")
        properties = binding["parameter_schema"]["properties"]
        assert {
            name for name, contract in properties.items()
            if contract.get("x-factor-engine-role") == "panel"
        } == expected_panels
        assert binding["evidence"]["status"] == "NOT_RUN"


def test_snapshot_exposes_declared_bool_scalar_contract_and_runtime_gate():
    snapshot = build_runtime_operator_snapshot(profile="runtime")
    row = next(r for r in snapshot["operators"] if r["canonical"] == "cs_huber_resid")
    binding = next(b for b in row["backends"] if b["backend"] == "pandas_numpy")
    assert binding["parameter_schema"]["properties"]["add_intercept"] == {
        "x-factor-engine-role": "policy",
        "x-searchable": False,
        "type": "boolean",
        "default": True,
    }
    assert binding["contract_status"] == "declared"
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    op = OperatorRegistry.get("cs_huber_resid", "pandas_numpy", mode="any")
    x = pd.DataFrame(np.arange(12.0).reshape(3, 4))
    with pytest.raises((TypeError, ValueError)):
        op.calculate(x, x, add_intercept=1)


def test_binding_digest_change_invalidates_old_pass():
    rec = {**_identity(), "implementation_digest": "old", "status": "PASS",
           "test_node_id": "tests/test_demo.py::test_demo", "artifact_reference": "evidence/run.json"}
    assert _evidence_for(_identity(), {("demo", "pandas_numpy"): rec})["status"] == "STALE"


def test_absent_evidence_is_explicit_not_run():
    assert _evidence_for(_identity(), {})["status"] == "NOT_RUN"


def test_legacy_stale_block_expands_without_recertifying(tmp_path):
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"affected": [{"canonical": "demo"}]}))
    rows = load_evidence_records(path)
    supplied = {(r["canonical"], r["backend"]): r for r in rows}
    assert _evidence_for(_identity(), supplied)["status"] == "STALE"


def test_query_is_bounded_and_unknown_context_fails_closed():
    row = {"canonical": "demo", "frequency": "daily", "execution_kind": "primitive",
           "markets": [], "data_capabilities": [], "budget_class": None,
           "backends": [{"backend": "pandas_numpy", "research_supported": True,
                         "contract_status": "declared"}],
           "capabilities": {"discoverable": True, "research_registered": True,
                            "research_callable": True,
                            "production_callable": False, "native_accelerated": False}}
    snapshot = {"schema_version": "v3", "catalog_digest": "d", "operators": [row]}
    assert query_runtime_operator_snapshot(snapshot, actor="research_callable")["total_matches"] == 1
    assert query_runtime_operator_snapshot(snapshot, market="ashare")["total_matches"] == 0
    assert query_runtime_operator_snapshot(snapshot, data_capability="minute")["total_matches"] == 0


def test_backend_filter_applies_capability_to_that_binding():
    row = {"canonical": "demo", "frequency": "daily", "execution_kind": "primitive",
           "markets": [], "data_capabilities": [], "budget_class": None,
           "backends": [
               {"backend": "pandas_numpy", "research_supported": True, "contract_status": "declared"},
               {"backend": "polars", "research_supported": False, "contract_status": "declared"},
           ], "capabilities": {"discoverable": True, "research_registered": True,
                               "research_callable": True, "production_callable": False,
                               "native_accelerated": False}}
    snapshot = {"schema_version": "v3", "catalog_digest": "d", "operators": [row]}
    assert query_runtime_operator_snapshot(snapshot, actor="research_callable", backend="pandas_numpy")["total_matches"] == 1
    assert query_runtime_operator_snapshot(snapshot, actor="research_callable", backend="polars")["total_matches"] == 0


def test_native_requires_explicit_empty_fallback_record():
    row = {"backend": "polars", "physical_execution_kind": "polars_native_expr",
           "evidence": {}}
    assert _native_accelerated([row], {"polars"}) is False


def test_cached_snapshot_singleflights_concurrent_builds(monkeypatch):
    snapshot_module._SNAPSHOT_CACHE.clear()
    calls = []
    monkeypatch.setattr(snapshot_module, "_live_runtime_digest", lambda: "runtime")
    def build(**kwargs):
        calls.append(kwargs)
        return {"operators": [], "catalog_digest": "d"}
    monkeypatch.setattr(snapshot_module, "build_runtime_operator_snapshot", build)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: snapshot_module.get_cached_runtime_operator_snapshot(), range(8)))
    assert len(calls) == 1
    assert all(result is results[0] for result in results)
