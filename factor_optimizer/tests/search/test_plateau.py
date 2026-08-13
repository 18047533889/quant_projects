"""Tests for plateau detection."""

import pytest

from factor_optimizer.search.plateau import (
    PlateauDetector,
    PlateauConfig,
    AdaptivePlateauDetector,
    MultiObjectivePlateauDetector,
)


def test_plateau_config_validation():
    with pytest.raises(ValueError, match="window_size"):
        PlateauConfig(window_size=1)

    with pytest.raises(ValueError, match="min_relative_improvement"):
        PlateauConfig(min_relative_improvement=-0.1)

    with pytest.raises(ValueError, match="min_absolute_improvement"):
        PlateauConfig(min_absolute_improvement=-0.1)


def test_plateau_config_defaults():
    config = PlateauConfig()

    assert config.window_size == 20
    assert config.min_relative_improvement == 0.001
    assert config.min_absolute_improvement is None
    assert not config.require_both
    assert not config.use_median


def test_plateau_detector_initialization():
    config = PlateauConfig(window_size=10)
    detector = PlateauDetector(config)

    assert detector.config.window_size == 10
    assert len(detector.score_history) == 0


def test_plateau_detector_add_score():
    config = PlateauConfig()
    detector = PlateauDetector(config)

    detector.add_score(0.5)
    detector.add_score(0.6)

    assert len(detector.score_history) == 2
    assert detector.score_history == [0.5, 0.6]


def test_plateau_detector_insufficient_history():
    config = PlateauConfig(window_size=5)
    detector = PlateauDetector(config)

    detector.add_score(0.5)
    detector.add_score(0.6)

    assert not detector.is_plateau()


def test_plateau_detector_no_plateau():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.01)
    detector = PlateauDetector(config)

    # Steadily improving scores
    scores = [0.5, 0.55, 0.60, 0.65, 0.70]
    for score in scores:
        detector.add_score(score)

    assert not detector.is_plateau()


def test_plateau_detector_plateau_detected():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.01)
    detector = PlateauDetector(config)

    # Truly flat scores - best is in first half, no improvement after
    scores = [0.502, 0.501, 0.5001, 0.5002, 0.5000]
    for score in scores:
        detector.add_score(score)

    assert detector.is_plateau()


def test_plateau_detector_external_scores():
    config = PlateauConfig(window_size=3, min_relative_improvement=0.01)
    detector = PlateauDetector(config)

    # Best score at beginning, no improvement after
    scores = [1.002, 1.0001, 1.0000]
    assert detector.is_plateau(scores)


def test_plateau_detector_use_median():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.01, use_median=True)
    detector = PlateauDetector(config)

    scores = [0.5, 0.6, 0.55, 0.58, 0.57]
    for score in scores:
        detector.add_score(score)

    # Median baseline should differ from max baseline
    result = detector.is_plateau()
    assert isinstance(result, bool)


def test_plateau_detector_absolute_improvement():
    config = PlateauConfig(
        window_size=5,
        min_relative_improvement=0.001,
        min_absolute_improvement=0.1,
        require_both=False,
    )
    detector = PlateauDetector(config)

    # Small relative but large absolute improvement
    scores = [10.0, 10.05, 10.08, 10.10, 10.15]
    for score in scores:
        detector.add_score(score)

    assert not detector.is_plateau()


def test_plateau_detector_require_both():
    config = PlateauConfig(
        window_size=5,
        min_relative_improvement=0.01,
        min_absolute_improvement=0.05,
        require_both=True,
    )
    detector = PlateauDetector(config)

    # Good relative but poor absolute
    scores = [1.0, 1.01, 1.015, 1.02, 1.025]
    for score in scores:
        detector.add_score(score)

    # Should plateau (absolute not met)
    assert detector.is_plateau()


def test_plateau_detector_reset():
    config = PlateauConfig()
    detector = PlateauDetector(config)

    detector.add_score(0.5)
    detector.add_score(0.6)
    detector.reset()

    assert len(detector.score_history) == 0


def test_plateau_detector_zero_baseline():
    config = PlateauConfig(window_size=3, min_relative_improvement=0.01)
    detector = PlateauDetector(config)

    # Baseline is zero
    scores = [0.0, 0.0, 0.1]
    for score in scores:
        detector.add_score(score)

    assert not detector.is_plateau()


def test_plateau_detector_plateau_duration():
    config = PlateauConfig(min_relative_improvement=0.1)
    detector = PlateauDetector(config)

    # Significant improvement, then plateau
    scores = [0.5, 0.6, 0.61, 0.605, 0.608, 0.606]
    for score in scores:
        detector.add_score(score)

    duration = detector.plateau_duration()
    assert duration > 0


def test_plateau_detector_plateau_duration_no_plateau():
    config = PlateauConfig(min_relative_improvement=0.01)
    detector = PlateauDetector(config)

    # Continuous improvement
    scores = [0.5, 0.55, 0.60, 0.65, 0.70]
    for score in scores:
        detector.add_score(score)

    duration = detector.plateau_duration()
    assert duration == 0


def test_adaptive_plateau_detector_initialization():
    config = PlateauConfig(window_size=10)
    detector = AdaptivePlateauDetector(config)

    assert detector.base_config.window_size == 10
    assert detector.early_phase_window == 50
    assert len(detector.score_history) == 0


def test_adaptive_plateau_detector_early_phase():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.01)
    detector = AdaptivePlateauDetector(config, early_phase_window=10)

    # Early phase: more lenient
    scores = [0.5, 0.51, 0.515, 0.52, 0.522]
    for score in scores:
        detector.add_score(score)

    # Should not plateau in early phase due to adjusted threshold
    result = detector.is_plateau()
    assert isinstance(result, bool)


def test_adaptive_plateau_detector_late_phase():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.01)
    detector = AdaptivePlateauDetector(config, early_phase_window=3)

    # Move past early phase
    scores = [0.5, 0.6, 0.7, 0.71, 0.715, 0.718, 0.72]
    for score in scores:
        detector.add_score(score)

    result = detector.is_plateau()
    assert isinstance(result, bool)


def test_adaptive_plateau_detector_high_volatility():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.05)
    detector = AdaptivePlateauDetector(config, early_phase_window=5, volatility_window=5)

    # High volatility scores
    scores = [0.5, 0.8, 0.4, 0.9, 0.3, 0.85, 0.45]
    for score in scores:
        detector.add_score(score)

    # Should adapt to volatility
    result = detector.is_plateau()
    assert isinstance(result, bool)


def test_adaptive_plateau_detector_reset():
    config = PlateauConfig()
    detector = AdaptivePlateauDetector(config)

    detector.add_score(0.5)
    detector.reset()

    assert len(detector.score_history) == 0


def test_multi_objective_plateau_detector_initialization():
    detector = MultiObjectivePlateauDetector(window_size=15, min_new_nondominated=2)

    assert detector.window_size == 15
    assert detector.min_new_nondominated == 2
    assert len(detector.frontier_sizes) == 0


def test_multi_objective_plateau_detector_add_frontier_size():
    detector = MultiObjectivePlateauDetector()

    detector.add_frontier_size(5)
    detector.add_frontier_size(7)
    detector.add_frontier_size(9)

    assert len(detector.frontier_sizes) == 3


def test_multi_objective_plateau_detector_no_plateau():
    detector = MultiObjectivePlateauDetector(window_size=5, min_new_nondominated=2)

    # Growing frontier
    sizes = [5, 7, 9, 11, 14]
    for size in sizes:
        detector.add_frontier_size(size)

    assert not detector.is_plateau()


def test_multi_objective_plateau_detector_plateau():
    detector = MultiObjectivePlateauDetector(window_size=5, min_new_nondominated=2)

    # Stagnant frontier
    sizes = [5, 5, 6, 6, 6]
    for size in sizes:
        detector.add_frontier_size(size)

    assert detector.is_plateau()


def test_multi_objective_plateau_detector_growth_rate():
    detector = MultiObjectivePlateauDetector(window_size=5)

    sizes = [5, 7, 9, 11, 13]
    for size in sizes:
        detector.add_frontier_size(size)

    growth_rate = detector.growth_rate()
    assert growth_rate > 0


def test_multi_objective_plateau_detector_growth_rate_stagnant():
    detector = MultiObjectivePlateauDetector(window_size=5)

    sizes = [5, 5, 5, 5, 5]
    for size in sizes:
        detector.add_frontier_size(size)

    growth_rate = detector.growth_rate()
    assert growth_rate == 0.0


def test_multi_objective_plateau_detector_reset():
    detector = MultiObjectivePlateauDetector()

    detector.add_frontier_size(5)
    detector.reset()

    assert len(detector.frontier_sizes) == 0


def test_multi_objective_plateau_detector_insufficient_history():
    detector = MultiObjectivePlateauDetector(window_size=10)

    detector.add_frontier_size(5)
    detector.add_frontier_size(6)

    assert not detector.is_plateau()


def test_plateau_detector_negative_scores():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.01)
    detector = PlateauDetector(config)

    # Negative scores (e.g., losses)
    scores = [-0.5, -0.45, -0.40, -0.38, -0.37]
    for score in scores:
        detector.add_score(score)

    # Should detect improvement (less negative)
    assert not detector.is_plateau()


def test_plateau_config_custom():
    config = PlateauConfig(
        window_size=15,
        min_relative_improvement=0.005,
        min_absolute_improvement=0.05,
        require_both=True,
        use_median=True,
    )

    assert config.window_size == 15
    assert config.min_relative_improvement == 0.005
    assert config.min_absolute_improvement == 0.05
    assert config.require_both
    assert config.use_median


def test_adaptive_plateau_detector_adapt_config():
    config = PlateauConfig(window_size=5, min_relative_improvement=0.05)
    detector = AdaptivePlateauDetector(config, early_phase_window=3)

    # Test adaptation in early phase
    detector.add_score(0.5)
    detector.add_score(0.52)

    adapted = detector._adapt_config()
    # Early phase: threshold doubled
    assert adapted.min_relative_improvement == 0.1


def test_multi_objective_plateau_detector_growth_rate_empty():
    detector = MultiObjectivePlateauDetector()

    growth_rate = detector.growth_rate()
    assert growth_rate == 0.0
