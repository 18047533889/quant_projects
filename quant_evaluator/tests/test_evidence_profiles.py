"""
R61-FI-020 (plan section 12, workstream D D0+D1): named EvidenceProfile
registry on quant_evaluator.

This pins the QE-side metric-binding authority:

1. The five named profiles exist and are registered:
   CHEAP_SCREEN_CN_1D / SHAPE_DIAGNOSTIC_CN_1D / FULL_VALIDATION_CN_1D /
   EXPENSIVE_STATISTICAL_CN_1D / MODEL_FEATURE_DIAGNOSTIC_CN_1D.
2. Every metric id a profile binds exists in the single MetricRegistry
   authority (``quant_evaluator.registry.metrics``) — an unknown id fails
   closed with the typed ``UnsupportedMetricError`` at construction time, so
   a profile can never fabricate a metric.  All bound ids must also have a
   compute_fn (no catalog-only placeholder).
3. Profiles are frozen dataclasses carrying ``policy_id`` /
   ``policy_version``; the registry resolves them by ``profile_id``.
4. Cost/scope monotonicity: CHEAP_SCREEN_CN_1D (and SHAPE_DIAGNOSTIC_CN_1D)
   metric sets are proper subsets of FULL_VALIDATION_CN_1D, and the cost
   classes are ordered CHEAP < MODERATE < EXPENSIVE.
5. Unknown profile query raises the typed ``UnknownEvidenceProfileError``.
6. A profile's declared ``required_inputs`` is a closed vocabulary
   (artifact-type class names used by the registry) and never invents a new
   evidence status: missing-metric status vocabulary must come from the
   existing ``EvidenceStatus`` enum in
   ``quant_evaluator.contracts.evidence_status`` (never invented, never 0.0).

Run from the repo root:

    cd /home/sunhaiwei/quant_projects && .venv/bin/python -m pytest -q \
        quant_evaluator/tests/test_evidence_profiles.py --timeout=300
"""

from __future__ import annotations

import pytest

from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.contracts.evidence_status import EvidenceStatus
from quant_evaluator.registry import presets as presets_module
from quant_evaluator.registry.metrics import (
    list_metrics,
    list_metrics_by_status,
)
from quant_evaluator.registry.presets import (
    CHEAP_SCREEN_CN_1D,
    EXPENSIVE_STATISTICAL_CN_1D,
    FULL_VALIDATION_CN_1D,
    MODEL_FEATURE_DIAGNOSTIC_CN_1D,
    SHAPE_DIAGNOSTIC_CN_1D,
    CostClass,
    EvidenceProfile,
    GPUPreference,
    TargetFrequency,
    UnknownEvidenceProfileError,
    get_profile,
    get_profile_or_none,
    list_profile_versions,
    list_profiles,
    register_profile,
)

# ---------------------------------------------------------------------------
# 1. The five named profiles exist and are registered.
# ---------------------------------------------------------------------------

EXPECTED_PROFILE_IDS = [
    "CHEAP_SCREEN_CN_1D",
    "SHAPE_DIAGNOSTIC_CN_1D",
    "FULL_VALIDATION_CN_1D",
    "EXPENSIVE_STATISTICAL_CN_1D",
    "MODEL_FEATURE_DIAGNOSTIC_CN_1D",
]

ALL_FIVE = [
    CHEAP_SCREEN_CN_1D,
    SHAPE_DIAGNOSTIC_CN_1D,
    FULL_VALIDATION_CN_1D,
    EXPENSIVE_STATISTICAL_CN_1D,
    MODEL_FEATURE_DIAGNOSTIC_CN_1D,
]


def test_five_expected_profiles_registered():
    registered = set(list_profiles())
    assert registered == set(EXPECTED_PROFILE_IDS)
    for profile_id in EXPECTED_PROFILE_IDS:
        assert get_profile(profile_id).profile_id == profile_id


def test_each_profile_is_an_evidence_profile_instance():
    for profile in ALL_FIVE:
        assert isinstance(profile, EvidenceProfile)


def test_each_profile_is_nonempty_binding():
    for profile in ALL_FIVE:
        assert len(profile.metric_ids) >= 1
        assert len(profile.metric_ids) == len(set(profile.metric_ids))


def test_all_profiles_declare_cn_1d_target_and_policy_version():
    for profile in ALL_FIVE:
        assert profile.target is TargetFrequency.CN_1D
        assert profile.policy_id == "QE_EVIDENCE_PROFILE"
        assert isinstance(profile.policy_version, str)
        assert profile.policy_version.strip()


# ---------------------------------------------------------------------------
# 2. Bound metric ids are real registry entries (anti-fabrication).
# ---------------------------------------------------------------------------

def test_every_bound_metric_id_exists_in_metric_registry():
    registered = set(list_metrics())
    for profile in ALL_FIVE:
        unknown = [m for m in profile.metric_ids if m not in registered]
        assert not unknown, (
            f"{profile.profile_id} binds metric ids not in the MetricRegistry "
            f"authority: {sorted(unknown)}"
        )


def test_every_bound_metric_id_has_compute_fn():
    """No catalog-only placeholder (compute_fn=None) may be bound."""
    from quant_evaluator.registry.metrics import get_metric

    for profile in ALL_FIVE:
        placeholder = [
            m for m in profile.metric_ids if get_metric(m).compute_fn is None
        ]
        assert not placeholder, (
            f"{profile.profile_id} binds catalog-only metrics without an "
            f"implementation: {sorted(placeholder)}"
        )


def test_profile_construction_rejects_unregistered_metric_id():
    with pytest.raises(UnsupportedMetricError):
        EvidenceProfile(
            profile_id="__FABRICATED__",
            metric_ids=("rank_ic", "__not_a_real_metric__"),
            required_inputs=frozenset({"factor_batch", "label_bundle"}),
            cost_class=CostClass.CHEAP,
            target=TargetFrequency.CN_1D,
        )


def test_profile_rejects_duplicate_metric_id():
    with pytest.raises(ValueError):
        EvidenceProfile(
            profile_id="__DUP__",
            metric_ids=("rank_ic", "rank_ic"),
            required_inputs=frozenset({"factor_batch", "label_bundle"}),
            cost_class=CostClass.CHEAP,
            target=TargetFrequency.CN_1D,
        )


def test_profile_rejects_empty_metric_binding():
    with pytest.raises(ValueError):
        EvidenceProfile(
            profile_id="__EMPTY__",
            metric_ids=(),
            required_inputs=frozenset(),
            cost_class=CostClass.CHEAP,
            target=TargetFrequency.CN_1D,
        )


def test_duplicate_profile_registration_fails_closed():
    with pytest.raises(ValueError):
        register_profile(CHEAP_SCREEN_CN_1D)


# ---------------------------------------------------------------------------
# 3. Frozen dataclass + policy version.
# ---------------------------------------------------------------------------

def test_profile_is_frozen_and_hashable():
    for profile in ALL_FIVE:
        with pytest.raises(Exception):
            profile.metric_ids = ("rank_ic",)  # type: ignore[misc]
        assert isinstance(hash(profile), int)
        assert profile.metric_ids == profile.metric_ids  # stable read


def test_profile_policy_version_semver_shape():
    import re

    semver = re.compile(r"^\d+\.\d+\.\d+$")
    versions = list_profile_versions()
    for profile in ALL_FIVE:
        assert profile.policy_id in ("QE_EVIDENCE_PROFILE",)
        assert semver.match(profile.policy_version), (
            f"{profile.profile_id} policy_version {profile.policy_version!r} "
            "is not SemVer-shaped"
        )
        assert versions[profile.profile_id] == profile.policy_version


def test_profile_serialization_roundtrip():
    payload = CHEAP_SCREEN_CN_1D.to_dict()
    assert payload["profile_id"] == "CHEAP_SCREEN_CN_1D"
    assert isinstance(payload["metric_ids"], list)
    assert payload["policy_version"] == "1.0.0"
    assert payload["target"] == "cn_1d"
    assert payload["cost_class"] == "cheap"
    assert payload["parallelizable"] is True
    assert set(payload["required_inputs"]) == set(
        CHEAP_SCREEN_CN_1D.required_inputs
    )


# ---------------------------------------------------------------------------
# 4. Cost/scope monotonicity of the evaluation ladder.
# ---------------------------------------------------------------------------

def test_cheap_screen_metrics_subset_of_full_validation():
    cheap = set(CHEAP_SCREEN_CN_1D.metric_ids)
    full = set(FULL_VALIDATION_CN_1D.metric_ids)
    assert cheap <= full
    assert cheap < full, (
        "CHEAP_SCREEN must be a STRICT subset of FULL_VALIDATION (the full "
        "ladder adds portfolio-stats / stability metrics)"
    )


def test_shape_diagnostic_metrics_subset_of_full_validation():
    shape = set(SHAPE_DIAGNOSTIC_CN_1D.metric_ids)
    full = set(FULL_VALIDATION_CN_1D.metric_ids)
    assert shape <= full
    assert shape < full


def test_cost_classes_ordered_cheap_moderate_expensive():
    assert CostClass.CHEAP < CostClass.MODERATE < CostClass.EXPENSIVE
    assert CHEAP_SCREEN_CN_1D.cost_class is CostClass.CHEAP
    assert SHAPE_DIAGNOSTIC_CN_1D.cost_class is CostClass.CHEAP
    assert FULL_VALIDATION_CN_1D.cost_class is CostClass.MODERATE
    assert MODEL_FEATURE_DIAGNOSTIC_CN_1D.cost_class is CostClass.MODERATE
    assert EXPENSIVE_STATISTICAL_CN_1D.cost_class is CostClass.EXPENSIVE
    # The full-validation ladder is more expensive than the screen.
    assert CHEAP_SCREEN_CN_1D.cost_class < FULL_VALIDATION_CN_1D.cost_class


def test_screen_and_statistical_profiles_are_disjoint_by_design():
    """CHEAP_SCREEN (daily screening) and the expensive statistical bundle
    do not overlap: HAC/bootstrap/block-bootstrap are NOT part of the cheap
    screen (they are the expensive confirmation stage)."""
    cheap = set(CHEAP_SCREEN_CN_1D.metric_ids)
    expensive = set(EXPENSIVE_STATISTICAL_CN_1D.metric_ids)
    assert cheap.isdisjoint(expensive)


# ---------------------------------------------------------------------------
# 5. Unknown profile query -> typed error.
# ---------------------------------------------------------------------------

def test_unknown_profile_raises_typed_error():
    with pytest.raises(UnknownEvidenceProfileError) as excinfo:
        get_profile("__NO_SUCH_PROFILE__")
    assert excinfo.value.profile_id == "__NO_SUCH_PROFILE__"
    assert "__NO_SUCH_PROFILE__" in str(excinfo.value)
    # It is a KeyError subclass (historical get_preset contract preserved).
    assert isinstance(excinfo.value, KeyError)


def test_get_profile_or_none_returns_none_for_unknown():
    assert get_profile_or_none("__NO_SUCH_PROFILE__") is None
    assert get_profile_or_none("CHEAP_SCREEN_CN_1D") is CHEAP_SCREEN_CN_1D


# ---------------------------------------------------------------------------
# 6. required_inputs vocabulary + evidence-status vocabulary discipline.
# ---------------------------------------------------------------------------

KNOWN_INPUT_VOCABULARY = {
    "factor_batch",
    "label_bundle",
    "ICSeriesArtifact",
    "QuantileReturnArtifact",
    "p_values",
}


def test_profile_required_inputs_is_closed_vocabulary():
    for profile in ALL_FIVE:
        unknown = set(profile.required_inputs) - KNOWN_INPUT_VOCABULARY
        assert not unknown, (
            f"{profile.profile_id} declares required_inputs outside the "
            f"known artifact vocabulary: {sorted(unknown)}"
        )


def test_profile_required_inputs_consistent_with_bound_metric_specs():
    """The declared required_inputs must cover every artifact type the bound
    MetricSpecs declare in ``requires``."""
    from quant_evaluator.registry.metrics import get_metric

    for profile in ALL_FIVE:
        spec_requires = set()
        for metric_id in profile.metric_ids:
            spec_requires.update(get_metric(metric_id).requires or [])
        missing = spec_requires - set(profile.required_inputs)
        assert not missing, (
            f"{profile.profile_id} required_inputs does not cover the bound "
            f"metrics' spec requires: {sorted(missing)}"
        )


def test_missing_metric_status_vocabulary_is_existing_evidence_status():
    """R61-FI-020 must not invent new status vocabulary: every status a
    profile/evidence consumer may emit when a bound metric is unavailable is
    one of the existing EvidenceStatus enum values."""
    existing = {status.value for status in EvidenceStatus}
    assert {
        "computed",
        "not_computed",
        "unavailable",
        "unsupported",
        "insufficient_data",
        "label_not_mature",
        "invalid_evidence",
        "failed",
    } <= existing
    # And there is exactly the 8-token vocabulary (no near-duplicates like
    # STALE / UNKNOWN / NOT_APPLICABLE have been invented as statuses).
    assert len(EvidenceStatus) == 8
    assert "stale" not in existing
    assert "unknown" not in existing
    assert "not_applicable" not in existing
    # A missing metric is NEVER rendered as 0.0: the non-computed statuses
    # all exist and are distinct from a numeric fill.
    assert "not_computed" in existing


# ---------------------------------------------------------------------------
# Legacy MetricPreset backward compatibility (never break the 3 presets).
# ---------------------------------------------------------------------------

def test_legacy_presets_still_registered_unchanged():
    assert set(presets_module.list_presets()) == {
        "factor_core",
        "factor_extended",
        "production_daily",
    }
    core = presets_module.get_preset("factor_core")
    assert core.metric_names == [
        "mean_ic",
        "ic_std",
        "ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
    ]
    assert len(presets_module.get_preset("production_daily").metric_names) == 4
    # Legacy preset names never collide with the new profile ids.
    assert set(presets_module.list_presets()).isdisjoint(
        set(list_profiles())
    )
