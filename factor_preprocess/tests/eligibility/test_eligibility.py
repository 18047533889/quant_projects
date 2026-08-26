"""
Tests for the treatment eligibility engine.

Key rules:
- fundamental profile does NOT search short temporal smoothing.
- price-volume profile CAN search smoothing.
- binary profile restrictions (no default winsor/zscore).
- RAW always present.
- low_turnover trims smoothing budget.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from factor_preprocess.eligibility.engine import (
    TreatmentEligibilityEngine,
    PRICE_VOLUME,
    HIGH_TURNOVER,
    FUNDAMENTAL,
    SPARSE_UPDATE,
    EVENT,
    BINARY,
    DISCRETE,
    RAW_SEMANTIC_ID,
)
from factor_preprocess.contracts.factor_profile import FactorProfileArtifact
from factor_preprocess.contracts.treatment_lineage import ExistingTreatmentSignature


def _profile(family: str, **time_behavior) -> FactorProfileArtifact:
    return FactorProfileArtifact(
        factor_id="f1",
        factor_version="1.0.0",
        semantic_family=family,
        source_type="price",
        update_frequency="daily",
        natural_horizon=20,
        time_behavior=time_behavior,
    )


def test_fundamental_profile_does_not_search_short_smoothing():
    profile = _profile(FUNDAMENTAL)
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile)
    # No short temporal smoothing transforms.
    for name in ("ewma", "trailing_sma", "kama", "kalman_local_level"):
        assert not space.allows(name), f"{name} should be forbidden for fundamental"
    # Freshness-aware fill IS allowed.
    assert space.allows("freshness_aware_fill")


def test_price_volume_profile_can_search_smoothing():
    profile = _profile(PRICE_VOLUME)
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile)
    assert space.allows("ewma")
    assert space.allows("trailing_sma")
    assert space.allows("cs_winsor")
    assert space.allows("cs_rank")


def test_binary_profile_restrictions():
    profile = _profile(BINARY)
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile)
    # No default winsor/zscore for binary.
    assert not space.allows("cs_winsor")
    assert not space.allows("cs_zscore")
    # Rank is still allowed.
    assert space.allows("cs_rank")


def test_raw_always_present():
    for family in (PRICE_VOLUME, FUNDAMENTAL, EVENT, BINARY, DISCRETE, HIGH_TURNOVER):
        profile = _profile(family)
        engine = TreatmentEligibilityEngine()
        space = engine.build_search_space(profile)
        assert RAW_SEMANTIC_ID in space.allowed_transform_ids


def test_low_turnover_trims_smoothing_budget():
    profile = _profile(PRICE_VOLUME, raw_turnover=0.05)
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile)
    # Smoothing still present but budget trimmed (upper bounds halved).
    assert space.allows("ewma")
    lo, hi = space.allowed_transform_ids["ewma"]["halflife"]
    # Full budget upper bound is 60; trimmed is <= 30.
    assert hi <= 30.0


def test_event_allows_event_decay_forbids_long_smoothing():
    profile = _profile(EVENT)
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile)
    assert space.allows("event_decay")
    assert not space.allows("trailing_sma")
    assert not space.allows("ewma")


def test_sparse_update_allows_freshness_fill():
    profile = _profile(SPARSE_UPDATE)
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile)
    assert space.allows("freshness_aware_fill")
    assert not space.allows("ewma")
