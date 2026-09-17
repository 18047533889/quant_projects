"""Source dependency collection must scale with unique nodes and bind v2 identity."""
import json
from dataclasses import replace

import pytest

from factor_engine.api.source_ref import SourceRefSpec, encode_source_ref
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner import source_dependencies as deps


def column(spec=None):
    return PlanNode("column", attrs={"name": encode_source_ref(
        spec or SourceRefSpec(table="StockDailyBarAdj", field="Close")
    )})


def test_deep_plan_without_recursion():
    leaf = column()
    plan = leaf
    for _ in range(5000):
        plan = PlanNode("neg", (plan,))
    assert deps.build_source_dependency_manifest(plan) == deps.build_source_dependency_manifest(leaf)


def test_shared_diamond_decoded_once(monkeypatch):
    plan = column()
    original = deps.decode_source_ref
    calls = []
    def counted(name):
        calls.append(name)
        return original(name)
    monkeypatch.setattr(deps, "decode_source_ref", counted)
    for _ in range(60):
        plan = PlanNode("add", (plan, plan))
    assert len(deps.build_source_dependency_manifest(plan)) == 1
    assert len(calls) == 1


def test_malformed_cycle_rejected():
    plan = PlanNode("neg")
    object.__setattr__(plan, "inputs", (plan,))
    with pytest.raises(ValueError, match="cycle"):
        deps.build_source_dependency_manifest(plan)


def test_corrupt_encoded_reference_still_rejected():
    with pytest.raises(ValueError):
        deps.build_source_dependency_manifest(
            PlanNode("column", attrs={"name": "__fe_source_ref_v1__broken"})
        )


@pytest.mark.parametrize("key,value", [
    ("market", "US"), ("concept_id", "price.close"), ("field_id", "close-v2"),
    ("provider_id", "provider-b"), ("dataset", "other-daily"),
    ("timeframe", "1m"), ("temporal_policy_digest", "pit-v2"),
    ("catalog_hash", "catalog-v2"), ("source_version", "revision-2"),
])
def test_v2_identity_dimensions_do_not_collide(key, value):
    base = SourceRefSpec(table="StockDailyBarAdj", field="Close", market="ASHARE")
    assert deps.source_dependency_hash(column(base)) != deps.source_dependency_hash(
        column(replace(base, **{key: value}))
    )


def test_legacy_manifest_shape_preserved_and_deduplicated():
    spec = SourceRefSpec(table="StockDailyBarAdj", field="Close")
    plan = PlanNode("add", (column(spec), column(spec)))
    manifest = deps.build_source_dependency_manifest(plan)
    assert len(manifest) == 1
    assert json.loads(manifest[0]) == {
        "table": spec.table, "field": spec.field, "params": {},
        "transform": None, "transform_params": {},
        "dialect": spec.dialect, "dialect_version": spec.dialect_version,
    }
