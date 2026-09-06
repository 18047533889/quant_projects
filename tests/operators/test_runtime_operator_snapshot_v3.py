from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from factor_engine.runtime.operator_snapshot import (
    _evidence_for, _native_accelerated, _parameter_contract, _research_callable,
    load_evidence_records,
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
