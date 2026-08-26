"""RED-first tests for soft desirability floors and dimension aggregation.

These test the desirability mapping (soft floor, no hard cliff), the separate
catastrophic floor, and the bottleneck/geomean dimension aggregation that the
sibling-owned RobustBalancedUtility will call.
"""

import math

import pytest

from factor_optimizer.search.desirability import (
    Desirability,
    catastrophic_floor,
    desirability_for,
)
from factor_optimizer.search.dimensions import (
    aggregate_dimension,
    DimensionName,
    DimensionScores,
    geomean_floor,
    bottleneck_min,
)


RANK_IC_ANCHORS = [
    (0.010, 0.05),
    (0.020, 0.35),
    (0.030, 0.72),
    (0.040, 0.90),
    (0.050, 1.00),
]


def test_soft_floor_continuity_around_threshold():
    """0.0249 vs 0.0251 must NOT be a 0->1 cliff."""
    lo = desirability_for("increasing", RANK_IC_ANCHORS, 0.0249)
    hi = desirability_for("increasing", RANK_IC_ANCHORS, 0.0251)
    diff = abs(hi - lo)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    # Continuous, bounded small jump, not a binary flip.
    assert diff < 0.05, f"soft floor produced a cliff: lo={lo} hi={hi} diff={diff}"
    assert hi > lo


def test_monotone_direction():
    """Increasing target maps monotonically up; decreasing maps monotonically down."""
    inc_vals = [desirability_for("increasing", RANK_IC_ANCHORS, x) for x in (0.011, 0.015, 0.022, 0.035, 0.048)]
    assert all(b >= a for a, b in zip(inc_vals, inc_vals[1:]))

    turnover_anchors = [(0.1, 1.0), (0.2, 0.6), (0.3, 0.25), (0.5, 0.05)]
    dec_vals = [desirability_for("decreasing", turnover_anchors, x) for x in (0.12, 0.18, 0.25, 0.40, 0.48)]
    assert all(b <= a for a, b in zip(dec_vals, dec_vals[1:]))


def test_clamp_beyond_breakpoints():
    """Values beyond the anchor range clamp/plateau, no runaway."""
    far_above = desirability_for("increasing", RANK_IC_ANCHORS, 0.999)
    far_below = desirability_for("increasing", RANK_IC_ANCHORS, -1.0)
    assert 0.0 <= far_above <= 1.0
    assert 0.0 <= far_below <= 1.0
    # Plateau at the ends, not extrapolation.
    assert far_above == pytest.approx(1.0)
    assert far_below == pytest.approx(0.05)


def test_catastrophic_floor_separate_from_soft():
    """Catastrophic floor triggers only below the integrity threshold; soft range unaffected."""
    # A value that is bad but above the integrity threshold must pass through.
    val = 0.015
    assert val > 0.01, "test fixture must stay above the integrity threshold"
    soft = desirability_for("increasing", RANK_IC_ANCHORS, val)
    assert catastrophic_floor(val, catastrophic_threshold=0.01) == val
    # A truly unacceptable value trips the hard floor.
    assert catastrophic_floor(0.005, catastrophic_threshold=0.01) == 0.0
    # The soft mapping at the bad-but-acceptable value is still positive (soft, not hard).
    assert soft > 0.0
    # The hard floor is a distinct mechanism: a sub-threshold value collapses to 0 even
    # though its soft desirability would be small-but-nonzero.
    assert catastrophic_floor(0.005, catastrophic_threshold=0.01) < soft


def test_bottleneck_aggregation():
    """One terrible dimension drags the aggregate below a balanced set."""
    bad_set = aggregate_dimension([0.95, 0.82, 0.15])
    balanced_set = aggregate_dimension([0.82, 0.80, 0.78])
    assert bad_set < balanced_set, "bottleneck aggregation must punish the weak dimension"


def test_geomean_floor():
    """Geometric mean of a set with a weak member is far below the arithmetic mean."""
    vals = [0.95, 0.82, 0.15]
    gm = geomean_floor(vals)
    am = sum(vals) / len(vals)
    assert gm < am
    # The gap is material, not marginal.
    assert gm < am * 0.7


def test_bottleneck_min_matches_min():
    assert bottleneck_min([0.95, 0.82, 0.15]) == pytest.approx(0.15)


def test_dimension_scores_dataclass():
    ds = DimensionScores(
        dimension=DimensionName.ROBUSTNESS,
        desirability=0.6,
        raw_metrics={"rank_ic": 0.02},
    )
    assert ds.desirability == 0.6
    assert ds.raw_metrics["rank_ic"] == 0.02
    assert ds.dimension is DimensionName.ROBUSTNESS


def test_catalog_has_common_metrics():
    d = Desirability()
    # Every catalogued metric resolves to a bounded desirability across its range.
    for key in ("rank_ic", "icir", "turnover", "worst_slice", "cost_adjusted_alpha", "coverage"):
        assert 0.0 <= d.score(key, 0.5) <= 1.0
        assert 0.0 <= d.score(key, -0.5) <= 1.0
