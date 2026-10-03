from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from factor_optimizer.candidate_catalog import optimizer_candidate_specs
from factor_optimizer.research_batch import (
    BatchOptimizationConfig, _specs, optimize_factor_batch,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def load_method_audit():
    path = Path(__file__).parents[1] / "examples/method_audit.py"
    spec = importlib.util.spec_from_file_location("method_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_catalog_exact_order_and_parameter_values():
    config = BatchOptimizationConfig(families=(
        "U_SHAPE_REPAIR", "INVERTED_U_REPAIR", "DECAY_REFINEMENT",
        "MISSINGNESS_FRESHNESS", "REPRESENTATION_RANK", "REPRESENTATION_ZSCORE"))
    expected = [
        ("DECAY_REFINEMENT", {"decay": .5, "half_life_relative": False}),
        ("DECAY_REFINEMENT", {"decay": .5, "half_life_relative": True}),
        ("DECAY_REFINEMENT", {"decay": .8, "half_life_relative": False}),
        ("DECAY_REFINEMENT", {"decay": .8, "half_life_relative": True}),
        ("INVERTED_U_REPAIR", {"asymmetry": False, "center": .35, "power": 1.}),
        ("INVERTED_U_REPAIR", {"asymmetry": False, "center": .35, "power": 2.}),
        ("INVERTED_U_REPAIR", {"asymmetry": False, "center": .5, "power": 1.}),
        ("INVERTED_U_REPAIR", {"asymmetry": False, "center": .5, "power": 2.}),
        ("INVERTED_U_REPAIR", {"asymmetry": False, "center": .65, "power": 1.}),
        ("INVERTED_U_REPAIR", {"asymmetry": False, "center": .65, "power": 2.}),
        ("INVERTED_U_REPAIR", {"asymmetry": True, "center": .35, "power": 1.}),
        ("INVERTED_U_REPAIR", {"asymmetry": True, "center": .35, "power": 2.}),
        ("INVERTED_U_REPAIR", {"asymmetry": True, "center": .5, "power": 1.}),
        ("INVERTED_U_REPAIR", {"asymmetry": True, "center": .5, "power": 2.}),
        ("INVERTED_U_REPAIR", {"asymmetry": True, "center": .65, "power": 1.}),
        ("INVERTED_U_REPAIR", {"asymmetry": True, "center": .65, "power": 2.}),
        ("MISSINGNESS_FRESHNESS", {"freshness_window": 5, "mode": "flag"}),
        ("MISSINGNESS_FRESHNESS", {"freshness_window": 1, "mode": "fill"}),
        ("MISSINGNESS_FRESHNESS", {"freshness_window": 3, "mode": "fill"}),
        ("MISSINGNESS_FRESHNESS", {"freshness_window": 5, "mode": "fill"}),
        ("REPRESENTATION_RANK", {"rank_axis": "cross_sectional", "tie_method": "average", "window": 20}),
        ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "average", "window": 5}),
        ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "average", "window": 10}),
        ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "average", "window": 20}),
        ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "min", "window": 5}),
        ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "min", "window": 10}),
        ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "min", "window": 20}),
        ("REPRESENTATION_RANK", {"rank_axis": "cross_sectional", "tie_method": "min", "window": 20}),
        ("REPRESENTATION_ZSCORE", {"cap": 3.0, "window": 20, "zscore_axis": "cross_sectional"}),
        ("REPRESENTATION_ZSCORE", {"cap": 3.0, "window": 5, "zscore_axis": "ts"}),
        ("REPRESENTATION_ZSCORE", {"cap": 3.0, "window": 10, "zscore_axis": "ts"}),
        ("REPRESENTATION_ZSCORE", {"cap": 3.0, "window": 20, "zscore_axis": "ts"}),
        ("U_SHAPE_REPAIR", {"asymmetry": False, "center": .35, "power": 1.}),
        ("U_SHAPE_REPAIR", {"asymmetry": False, "center": .35, "power": 2.}),
        ("U_SHAPE_REPAIR", {"asymmetry": False, "center": .5, "power": 1.}),
        ("U_SHAPE_REPAIR", {"asymmetry": False, "center": .5, "power": 2.}),
        ("U_SHAPE_REPAIR", {"asymmetry": False, "center": .65, "power": 1.}),
        ("U_SHAPE_REPAIR", {"asymmetry": False, "center": .65, "power": 2.}),
        ("U_SHAPE_REPAIR", {"asymmetry": True, "center": .35, "power": 1.}),
        ("U_SHAPE_REPAIR", {"asymmetry": True, "center": .35, "power": 2.}),
        ("U_SHAPE_REPAIR", {"asymmetry": True, "center": .5, "power": 1.}),
        ("U_SHAPE_REPAIR", {"asymmetry": True, "center": .5, "power": 2.}),
        ("U_SHAPE_REPAIR", {"asymmetry": True, "center": .65, "power": 1.}),
        ("U_SHAPE_REPAIR", {"asymmetry": True, "center": .65, "power": 2.}),
    ]
    assert optimizer_candidate_specs(config) == expected
    assert _specs(config) == expected


def test_static_budget_rejection_does_not_mutate_config():
    config = BatchOptimizationConfig(families=("DECAY_REFINEMENT",), maximum_candidates=1)
    original = replace(config)
    with pytest.raises(ValueError, match="candidate budget"):
        optimizer_candidate_specs(config)
    assert config == original


def test_audit_executes_optimizer_static_grid_once():
    audit = load_method_audit()
    config = BatchOptimizationConfig()
    cases = [(family, parameters) for family, parameters, _ in audit.method_cases(config)]
    assert len(cases) == len(set((family, tuple(sorted(parameters.items())))
                                 for family, parameters in cases))
    static = optimizer_candidate_specs(config)
    assert all(cases.count((family, parameters)) == 1 for family, parameters in static)


def test_inventory_uses_actual_adaptive_candidates_and_preserves_inputs():
    rng = np.random.default_rng(317)
    times, assets = 240, 24
    values = rng.normal(size=(times, assets, 1))
    returns = .002 * values[:, :, 0] + rng.normal(0, .01, size=(times, assets))
    ta = AxisRef("time", "int", times, np.arange(times))
    aa = AxisRef("asset", "str", assets, np.array([f"a{i}" for i in range(assets)]))
    batch = FactorBatch(("probe",), ta, aa, values)
    labels = LabelBundle("returns", returns, 1, decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=aa)
    values_before, returns_before = values.copy(), returns.copy()
    config = BatchOptimizationConfig(families=("TAIL_HINGE", "CAUSAL_SMOOTHING"),
        selection_objective="rank_ic", bootstrap_draws=99)
    result = optimize_factor_batch(batch, labels, config=config, allow_research=True)

    inventory = load_method_audit().candidate_universe(result)
    static = inventory["optimizer_static"]
    actual = result.factors["probe"].candidates
    adaptive = inventory["optimizer_adaptive_actual_by_factor"]["probe"]
    assert {(item["family"], tuple(sorted(item["parameters"].items())))
            for item in static} == {(family, tuple(sorted(parameters.items())))
                                    for family, parameters in optimizer_candidate_specs(config)}
    assert adaptive
    assert all(any(record == candidate for candidate in actual) for record in adaptive)
    assert any(record["family"] == "CAUSAL_SMOOTHING" for record in adaptive)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(returns, returns_before)


def test_audit_config_controls_split_and_static_catalog():
    audit = load_method_audit()
    config = BatchOptimizationConfig(
        train_fraction=.50, validation_fraction=.25, warmup_bars=10,
        families=("REPRESENTATION_RANK",))
    split = audit.automatic_time_split(
        type("Labels", (), {"decision_time": tuple(range(240)),
                             "label_end_time": tuple(range(2, 242))})(), config)
    assert (split.validation_start, split.test_start) == (120, 180)
    cases = list(audit.method_cases(config))
    static = optimizer_candidate_specs(config)
    assert {family for family, _ in static} == {"REPRESENTATION_RANK"}
    assert {params["window"] for family, params in static if family == "REPRESENTATION_RANK"} == {5, 10, 20}
    assert any(family != "REPRESENTATION_RANK" for family, _, _ in cases)
