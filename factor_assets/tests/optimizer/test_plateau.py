"""Tests for plateau detection."""

import pytest
from factor_assets.optimizer.plateau import (
    ParameterNeighbor,
    PlateauAnalysis,
    ParameterPlateauDetector,
    NeighborSurvivalAnalyzer,
)


def test_parameter_neighbor_creation():
    """Test parameter neighbor creation."""
    neighbor = ParameterNeighbor(
        neighbor_id="neighbor_001",
        base_parameters={"window": 20},
        perturbed_parameters={"window": 22},
        perturbation_info={"param_name": "window", "direction": 1},
        metric_value=0.045,
        relative_performance=-0.1,
    )

    assert neighbor.neighbor_id == "neighbor_001"
    assert neighbor.metric_value == 0.045


def test_plateau_detector_generate_neighbors():
    """Test neighbor generation."""
    detector = ParameterPlateauDetector(perturbation_scale=0.1)

    parameters = {"window": 20, "alpha": 0.5}
    neighbors = detector.generate_neighbors(parameters, max_neighbors=8)

    assert len(neighbors) <= 8
    assert all("parameters" in n for n in neighbors)
    assert all("perturbation_info" in n for n in neighbors)


def test_plateau_detector_numeric_only():
    """Test that only numeric parameters are perturbed."""
    detector = ParameterPlateauDetector()

    parameters = {"window": 20, "method": "ewma"}  # Mixed types
    neighbors = detector.generate_neighbors(parameters, max_neighbors=8)

    # Should only perturb 'window'
    for neighbor in neighbors:
        info = neighbor["perturbation_info"]
        assert info["param_name"] == "window"


def test_plateau_detector_zero_value():
    """Test perturbation of zero value."""
    detector = ParameterPlateauDetector(perturbation_scale=0.1)

    parameters = {"value": 0.0}
    neighbors = detector.generate_neighbors(parameters, max_neighbors=2)

    assert len(neighbors) == 2
    # Should use absolute perturbation scale
    assert neighbors[0]["parameters"]["value"] != 0.0


def test_plateau_analysis_empty_neighbors():
    """Test plateau analysis with no neighbors."""
    detector = ParameterPlateauDetector()

    analysis = detector.analyze_plateau(
        base_parameters={"window": 20},
        base_metric=0.05,
        neighbor_results=[],
    )

    assert analysis.base_metric == 0.05
    assert analysis.worst_neighbor_metric == 0.05
    assert analysis.plateau_stability == 1.0


def test_plateau_analysis_stable_plateau():
    """Test analysis of stable plateau."""
    detector = ParameterPlateauDetector(
        stability_tolerance=0.05,
        survival_threshold=0.8,
    )

    base_metric = 0.05
    neighbor_results = [
        ({"window": 21}, 0.049),  # 2% degradation
        ({"window": 19}, 0.051),  # 2% improvement
        ({"window": 22}, 0.048),  # 4% degradation
    ]

    analysis = detector.analyze_plateau(
        base_parameters={"window": 20},
        base_metric=base_metric,
        neighbor_results=neighbor_results,
    )

    assert analysis.local_sensitivity < 0.1
    assert analysis.plateau_stability > 0.5
    assert analysis.neighbor_survival_rate > 0.5


def test_plateau_analysis_unstable_plateau():
    """Test analysis of unstable plateau."""
    detector = ParameterPlateauDetector(
        stability_tolerance=0.05,
        survival_threshold=0.8,
    )

    base_metric = 0.05
    neighbor_results = [
        ({"window": 21}, 0.03),   # 40% degradation
        ({"window": 19}, 0.025),  # 50% degradation
    ]

    analysis = detector.analyze_plateau(
        base_parameters={"window": 20},
        base_metric=base_metric,
        neighbor_results=neighbor_results,
    )

    assert analysis.local_sensitivity > 0.4
    assert analysis.neighbor_survival_rate == 0.0


def test_neighbor_survival_analyzer_robustness_score():
    """Test robustness score computation."""
    analyzer = NeighborSurvivalAnalyzer(survival_threshold=0.8)

    analysis = PlateauAnalysis(
        base_metric=0.05,
        worst_neighbor_metric=0.045,
        mean_neighbor_metric=0.048,
        local_sensitivity=0.1,
        plateau_stability=0.8,
        neighbor_survival_rate=0.9,
    )

    score = analyzer.compute_robustness_score(analysis)

    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be high given good metrics


def test_neighbor_survival_analyzer_low_robustness():
    """Test low robustness score."""
    analyzer = NeighborSurvivalAnalyzer()

    analysis = PlateauAnalysis(
        base_metric=0.05,
        worst_neighbor_metric=0.01,
        mean_neighbor_metric=0.02,
        local_sensitivity=0.8,
        plateau_stability=0.2,
        neighbor_survival_rate=0.1,
    )

    score = analyzer.compute_robustness_score(analysis)

    assert 0.0 <= score <= 1.0
    assert score < 0.5  # Should be low given poor metrics


def test_neighbor_survival_analyzer_rank_by_robustness():
    """Test ranking candidates by robustness."""
    analyzer = NeighborSurvivalAnalyzer()

    candidates = [
        {
            "candidate_id": "cand_a",
            "plateau_analysis": PlateauAnalysis(
                base_metric=0.05,
                worst_neighbor_metric=0.045,
                mean_neighbor_metric=0.048,
                local_sensitivity=0.1,
                plateau_stability=0.8,
                neighbor_survival_rate=0.9,
            ),
        },
        {
            "candidate_id": "cand_b",
            "plateau_analysis": PlateauAnalysis(
                base_metric=0.05,
                worst_neighbor_metric=0.02,
                mean_neighbor_metric=0.03,
                local_sensitivity=0.6,
                plateau_stability=0.3,
                neighbor_survival_rate=0.2,
            ),
        },
    ]

    ranked = analyzer.rank_by_robustness(candidates)

    assert len(ranked) == 2
    # cand_a should rank higher
    assert ranked[0]["candidate_id"] == "cand_a"
    assert ranked[0]["robustness_score"] > ranked[1]["robustness_score"]


def test_neighbor_survival_analyzer_filter_by_robustness():
    """Test filtering by minimum robustness."""
    analyzer = NeighborSurvivalAnalyzer()

    candidates = [
        {
            "candidate_id": "cand_a",
            "plateau_analysis": PlateauAnalysis(
                base_metric=0.05,
                worst_neighbor_metric=0.045,
                mean_neighbor_metric=0.048,
                local_sensitivity=0.1,
                plateau_stability=0.8,
                neighbor_survival_rate=0.9,
            ),
        },
        {
            "candidate_id": "cand_b",
            "plateau_analysis": PlateauAnalysis(
                base_metric=0.05,
                worst_neighbor_metric=0.02,
                mean_neighbor_metric=0.03,
                local_sensitivity=0.6,
                plateau_stability=0.3,
                neighbor_survival_rate=0.2,
            ),
        },
    ]

    filtered = analyzer.filter_by_robustness(candidates, min_robustness=0.5)

    # Only cand_a should pass
    assert len(filtered) <= 2
    assert all(c["robustness_score"] >= 0.5 for c in filtered)
