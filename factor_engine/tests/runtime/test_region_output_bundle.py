# -*- coding: utf-8 -*-
"""R23 P0-6: RegionOutputBundle multi-live-out tests.

Verifies that a region can produce multiple named outputs and consumers
retrieve individual outputs by name.
"""
from __future__ import annotations

from typing import Any

import pytest

pytest.skip(
    "R46 removed RegionOutputBundle from production "
    "(runtime/multibackend/parallel_region_scheduler.py no longer defines it).  "
    "This regression suite pins behavior that no longer exists and is skipped "
    "until it is re-targeted to the current production contract.",
    allow_module_level=True,
)

from factor_engine.runtime.multibackend.parallel_region_scheduler import RegionOutputBundle


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_bundle_has_no_outputs() -> None:
    bundle = RegionOutputBundle()
    assert bundle.is_single is False
    assert bundle.names == []


def test_single_factory_wraps_one_value() -> None:
    bundle = RegionOutputBundle.single(42, name="factor_a", region_id="r1")
    assert bundle.is_single is True
    assert bundle.names == ["factor_a"]
    assert bundle.only() == 42


def test_single_defaults_to_result_name() -> None:
    bundle = RegionOutputBundle.single([1, 2, 3])
    assert bundle.names == ["result"]
    assert bundle.only() == [1, 2, 3]


def test_bundle_with_multiple_outputs() -> None:
    bundle = RegionOutputBundle(
        outputs={"zscore": 1.5, "rank": 0.75, "quantile": 0.9},
        region_id="r2",
    )
    assert bundle.is_single is False
    assert sorted(bundle.names) == ["quantile", "rank", "zscore"]


# ---------------------------------------------------------------------------
# Retrieval by name
# ---------------------------------------------------------------------------


def test_get_returns_value_by_name() -> None:
    bundle = RegionOutputBundle(outputs={"alpha": 3.14, "beta": 2.71})
    assert bundle.get("alpha") == 3.14
    assert bundle.get("beta") == 2.71


def test_get_returns_default_for_missing_name() -> None:
    bundle = RegionOutputBundle(outputs={"alpha": 1.0})
    assert bundle.get("missing") is None
    assert bundle.get("missing", "fallback") == "fallback"


def test_getitem_retrieves_by_name() -> None:
    bundle = RegionOutputBundle(outputs={"x": 100, "y": 200})
    assert bundle["x"] == 100
    assert bundle["y"] == 200


def test_getitem_raises_on_missing_name() -> None:
    bundle = RegionOutputBundle(outputs={"x": 1}, region_id="r3")
    with pytest.raises(KeyError, match="r3"):
        _ = bundle["nonexistent"]


# ---------------------------------------------------------------------------
# only() — single-output access
# ---------------------------------------------------------------------------


def test_only_returns_value_when_single() -> None:
    bundle = RegionOutputBundle.single("anything", name="out")
    assert bundle.only() == "anything"


def test_only_raises_on_empty() -> None:
    bundle = RegionOutputBundle()
    with pytest.raises(ValueError, match="0 live-outs"):
        bundle.only()


def test_only_raises_on_multi_output() -> None:
    bundle = RegionOutputBundle(outputs={"a": 1, "b": 2})
    with pytest.raises(ValueError, match="2 live-outs"):
        bundle.only()


# ---------------------------------------------------------------------------
# add() — in-place mutation
# ---------------------------------------------------------------------------


def test_add_binds_one_output() -> None:
    bundle = RegionOutputBundle(region_id="r4")
    bundle.add("factor_a", [1.0, 2.0]).add("factor_b", [3.0])
    assert bundle.names == ["factor_a", "factor_b"]
    assert bundle["factor_a"] == [1.0, 2.0]


def test_add_overwrites_existing_name() -> None:
    bundle = RegionOutputBundle(outputs={"x": 1}, region_id="r5")
    bundle.add("x", 999)
    assert bundle["x"] == 999
    assert len(bundle.outputs) == 1  # not duplicated


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_combines_two_bundles() -> None:
    b1 = RegionOutputBundle(outputs={"a": 10, "b": 20})
    b2 = RegionOutputBundle(outputs={"c": 30})
    merged = RegionOutputBundle.merge([b1, b2], region_id="r_merged")
    assert merged.region_id == "r_merged"
    assert merged["a"] == 10
    assert merged["b"] == 20
    assert merged["c"] == 30


def test_merge_union_of_disjoint_names() -> None:
    b1 = RegionOutputBundle(outputs={"x": 1})
    b2 = RegionOutputBundle(outputs={"y": 2})
    b3 = RegionOutputBundle(outputs={"z": 3})
    merged = RegionOutputBundle.merge([b1, b2, b3])
    assert sorted(merged.names) == ["x", "y", "z"]


def test_merge_last_writer_wins_on_conflict() -> None:
    b1 = RegionOutputBundle(outputs={"val": "first"})
    b2 = RegionOutputBundle(outputs={"val": "second"})
    merged = RegionOutputBundle.merge([b1, b2])
    assert merged["val"] == "second"


def test_merge_accepts_raw_dict() -> None:
    merged = RegionOutputBundle.merge({"alpha": 1.0, "beta": 2.0}, region_id="r6")
    assert merged["alpha"] == 1.0
    assert merged["beta"] == 2.0
    assert merged.region_id == "r6"


# ---------------------------------------------------------------------------
# to_dict — serialization
# ---------------------------------------------------------------------------


def test_to_dict_flattens_single_output_to_scalar() -> None:
    bundle = RegionOutputBundle.single(99, name="result")
    assert bundle.to_dict() == 99


def test_to_dict_serialises_multi_output_as_dict() -> None:
    bundle = RegionOutputBundle(outputs={"a": 1, "b": 2})
    serialized = bundle.to_dict()
    assert isinstance(serialized, dict)
    assert serialized == {"a": 1, "b": 2}


def test_to_dict_empty_bundle() -> None:
    assert RegionOutputBundle().to_dict() == {}


# ---------------------------------------------------------------------------
# Integration: schedule_parallel with RegionOutputBundle
# ---------------------------------------------------------------------------


def test_scheduler_accepts_region_output_bundle() -> None:
    """Integration: execute_fn returns RegionOutputBundle and the scheduler
    passes it through in results."""
    from factor_engine.runtime.multibackend.parallel_region_scheduler import (
        ExecutionRegion,
        ParallelRegionScheduler,
    )

    def execute_fn(region: ExecutionRegion) -> RegionOutputBundle:
        if region.region_id == "multi":
            return RegionOutputBundle(
                outputs={"zscore": 1.5, "rank": 0.75},
                region_id=region.region_id,
            )
        return RegionOutputBundle.single(42, name="result", region_id=region.region_id)

    sched = ParallelRegionScheduler(max_parallel_regions=4)
    regions = [
        ExecutionRegion(
            region_id="single",
            backend="pandas",
            operators=[{"op": "literal"}],
            dependencies=[],
            estimated_cost_ms=1.0,
            memory_requirement_bytes=100,
        ),
        ExecutionRegion(
            region_id="multi",
            backend="pandas",
            operators=[{"op": "add"}],
            dependencies=["single"],
            estimated_cost_ms=1.0,
            memory_requirement_bytes=100,
            output_names=("zscore", "rank"),
        ),
    ]
    out = sched.schedule_parallel(regions, execute_fn)
    assert out["failed"] == []

    # Single-output region: bundle wraps the one live-out value
    single_result = out["results"]["single"]
    assert isinstance(single_result, RegionOutputBundle)
    assert single_result.only() == 42
    assert single_result.names == ["result"]

    # Multi-output region: consumers retrieve individual outputs by name
    multi_result = out["results"]["multi"]
    assert isinstance(multi_result, RegionOutputBundle)
    assert multi_result["zscore"] == 1.5
    assert multi_result["rank"] == 0.75
    assert multi_result.region_id == "multi"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_bundle_repr_shows_region_id_and_outputs() -> None:
    bundle = RegionOutputBundle(outputs={"x": 1}, region_id="r7")
    rep = repr(bundle)
    assert "r7" in rep
    assert "x" in rep


def test_bundle_with_no_region_id() -> None:
    bundle = RegionOutputBundle(outputs={"a": 1})
    assert bundle.region_id is None


def test_merge_with_empty_list() -> None:
    merged = RegionOutputBundle.merge([])
    assert merged.names == []


def test_merge_with_single_bundle_is_identity() -> None:
    b1 = RegionOutputBundle(outputs={"x": 1})
    merged = RegionOutputBundle.merge([b1])
    assert merged["x"] == 1
    assert len(merged.outputs) == 1