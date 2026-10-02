from itertools import permutations

import pytest

from quant_evaluator.runtime.source_auto_evidence import select_source_auto_route

F48_SHAPE = (2586, 5461, 48)
F48_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")


def _activate_f48_candidate(monkeypatch, status="measured_source_ab"):
    from dataclasses import replace

    import quant_evaluator.runtime.source_auto_evidence as registry

    entry = next(item for item in registry.SOURCE_AUTO_EVIDENCE
                 if item.evidence_id == "real_cos_f48_mixed_three_tile2")
    activated = replace(entry, evidence_status=status)
    monkeypatch.setattr(
        registry, "SOURCE_AUTO_EVIDENCE",
        tuple(activated if item.evidence_id == entry.evidence_id else item
              for item in registry.SOURCE_AUTO_EVIDENCE),
    )


def _select(**overrides):
    args = dict(shape=F48_SHAPE, metrics=F48_METRICS, source_dtype="float64",
                label_dtype="float64", requested_tile_width=2)
    args.update(overrides)
    return select_source_auto_route(**args)


@pytest.mark.parametrize("metrics", list(permutations(F48_METRICS)))
def test_f48_source_ab_pending_auto_does_not_select_route(monkeypatch, metrics):
    _activate_f48_candidate(monkeypatch, "measured_source_ab_pending_auto")
    assert _select(metrics=metrics) is None


@pytest.mark.parametrize("metrics", list(permutations(F48_METRICS)))
def test_f48_registered_auto_routes_without_candidate_injection(metrics):
    route = _select(metrics=metrics)
    assert route.evidence_id == "real_cos_f48_mixed_three_tile2"
    assert route.evidence_status == "measured_source_ab"
    assert route.effective_tile_width == 2


def test_f48_pending_auto_record_can_be_activated_after_integrated_evidence(monkeypatch):
    import quant_evaluator.runtime.source_auto_evidence as registry

    entry = next(item for item in registry.SOURCE_AUTO_EVIDENCE
                 if item.evidence_id == "real_cos_f48_mixed_three_tile2")
    assert entry.evidence_status == "measured_source_ab"
    _activate_f48_candidate(monkeypatch, "measured_source_ab_pending_auto")
    assert _select() is None
    _activate_f48_candidate(monkeypatch)

    route = _select()
    assert route.evidence_id == "real_cos_f48_mixed_three_tile2"
    assert route.legacy_reason == "bounded_f48_mixed_three_gpu_tile2"
    assert route.effective_tile_width == 2
    assert route.evidence_status == "measured_source_ab"


def test_f48_one_shot_metric_iterator_is_routed_after_admission():
    assert _select(metrics=iter(F48_METRICS)).effective_tile_width == 2


@pytest.mark.parametrize("overrides", [
    {"shape": (2585, 5461, 48)}, {"shape": (2586, 5460, 48)},
    {"shape": (2586, 5461, 47)}, {"shape": (2586, 5461, 49)},
    {"shape": (2586, 5461, 63)}, {"shape": (2586, 5461, 64)},
    {"shape": (2586, 5461, 65)},
    {"source_dtype": "float32"}, {"label_dtype": "float32"},
    {"metrics": F48_METRICS[:-1]},
    {"metrics": (*F48_METRICS, "coverage")},
    {"metrics": (*F48_METRICS, F48_METRICS[0])},
    {"requested_tile_width": 1}, {"requested_tile_width": 4},
    {"requested_tile_width": 8},
])
def test_f48_activated_route_does_not_interpolate_beyond_measured_request(
        monkeypatch, overrides):
    _activate_f48_candidate(monkeypatch)
    assert _select(**overrides) is None
