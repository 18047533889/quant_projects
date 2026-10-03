from __future__ import annotations

import pandas as pd
import pytest

from factor_optimizer.adapters.repair_execution import compile_value_repair
from factor_optimizer.adapters.repair_execution import ValueRepairPlan
from factor_optimizer.adapters.repair_execution_identity import (
    build_direct_fp_execution_identity,
)
from factor_preprocess.transforms import repair_shapes
from factor_optimizer.search.execution_dedup import _execution_signature


def _plan():
    return compile_value_repair(
        "REPRESENTATION_ZSCORE",
        {"zscore_axis": "cross_sectional", "cap": 3.0},
        natural_time_scale=10.0,
        training_context_ref="controlled-identity-test",
    )


def _changed_capped_zscore(frame, *, cap):
    return pd.Series(17.0, index=frame.index, name="value")


def test_direct_fp_identity_tracks_the_callable_selected_by_execute(monkeypatch):
    plan = _plan()
    original_identity = build_direct_fp_execution_identity(plan)
    calls = []

    def selected(frame, *, cap):
        calls.append(cap)
        return _changed_capped_zscore(frame, cap=cap)

    monkeypatch.setattr(repair_shapes, "capped_zscore", selected)
    changed_identity = build_direct_fp_execution_identity(plan)

    assert original_identity["function_source_sha256"] != changed_identity["function_source_sha256"]
    assert changed_identity["qualname"].endswith("selected")
    assert changed_identity["scope"] == "selected_callable_and_containing_module_source"
    assert "outside the containing module" in changed_identity["coverage_note"]
    assert set(changed_identity["selector_source_sha256"]) == {
        "value_repair_execute", "identity_route_selector",
        "identity_callable_resolver",
    }

    frame = pd.DataFrame({"asset_id": ["a"], "date": ["d"], "value": [1.0]})
    actual = plan.execute(frame, allow_research=True)
    assert calls == [3.0]
    assert actual.tolist() == [17.0]


def test_direct_fp_identity_rejects_routes_outside_its_explicit_surface():
    class UnknownPlan:
        transform = "unknown"
        parameters = ()

    with pytest.raises(ValueError, match="no direct FP identity route"):
        build_direct_fp_execution_identity(UnknownPlan())


def test_average_cs_rank_is_not_misidentified_as_a_direct_fp_route():
    class AverageRankPlan:
        transform = "cs_rank"
        parameters = (("method", "average"),)

    with pytest.raises(ValueError, match="FE route"):
        build_direct_fp_execution_identity(AverageRankPlan())


@pytest.mark.parametrize(
    "transform,parameters,module_name,attribute",
    [
        ("rank_shape", {"center": 0.5, "power": 2.0,
                         "inverted": False, "asymmetric": False},
         "factor_preprocess.transforms.repair_shapes", "rank_shape"),
        ("fp_cs_rank_min", {},
         "factor_preprocess.transforms.repair_shapes", "cross_sectional_rank"),
        ("cs_rank", {"method": "min"},
         "factor_preprocess.transforms.repair_shapes", "cross_sectional_rank"),
        ("ts_rank_history", {"method": "average", "window": 3},
         "factor_preprocess.transforms.temporal_representation", "time_series_rank"),
        ("ts_zscore_history", {"cap": 3.0, "window": 3},
         "factor_preprocess.transforms.temporal_representation", "capped_time_series_zscore"),
        ("capped_zscore", {"cap": 3.0},
         "factor_preprocess.transforms.repair_shapes", "capped_zscore"),
        ("tail_hinge", {"hinge": "top", "hinge_value": 1.0},
         "factor_preprocess.transforms.repair_shapes", "tail_hinge"),
        ("robust_scale", {"scale": "mad", "center": "median"},
         "factor_preprocess.transforms.repair_shapes", "robust_scale"),
    ],
)
def test_execution_signature_changes_when_selected_direct_fp_callable_changes(
    monkeypatch, transform, parameters, module_name, attribute,
):
    import importlib

    plan = ValueRepairPlan(
        "controlled-test", transform, tuple(sorted(parameters.items())),
        "controlled-execution-context", 10.0,
    )
    baseline = _execution_signature(plan, 1, {"train": "same"})
    module = importlib.import_module(module_name)

    def replacement(*args, **kwargs):
        return None

    monkeypatch.setattr(module, attribute, replacement)
    changed = _execution_signature(plan, 1, {"train": "same"})

    assert changed != baseline


def test_selector_source_unavailable_fails_closed(monkeypatch):
    import factor_optimizer.adapters.repair_execution_identity as identity

    plan = _plan()
    original_getsource = identity.inspect.getsource

    def unavailable(value):
        from factor_optimizer.adapters.repair_execution import ValueRepairPlan
        if value is ValueRepairPlan.execute:
            raise OSError("controlled unavailable source")
        return original_getsource(value)

    monkeypatch.setattr(identity.inspect, "getsource", unavailable)
    with pytest.raises(ValueError, match="cannot certify value_repair_execute source"):
        build_direct_fp_execution_identity(plan)
