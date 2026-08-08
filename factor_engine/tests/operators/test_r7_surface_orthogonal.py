# -*- coding: utf-8 -*-
"""WS-H review findings #309-#311: surface / production-tier orthogonality.

* #309 ``pandas_first_production_canonicals()`` is a live function (no
  import-time snapshot of the extended-only surface).
* #310 AuthoringTier / ProductionCertification / BackendCapability are three
  orthogonal dimensions; certification never derives from surface membership.
* #311 daily-factor migration is owned by a reviewed manifest
  (``REVIEWED_MIGRATION_MANIFEST``) seeded from the legacy manual frozenset.
"""
from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    ensure_cleaned_loaded()


# ---------------------------------------------------------------------------
# #309
# ---------------------------------------------------------------------------
def test_pandas_first_production_canonicals_is_live_function():
    import cleaned_operators.operator_surface as surface
    from cleaned_operators.production_tiers import (
        pandas_first_production_canonicals,
    )

    assert callable(pandas_first_production_canonicals)
    old_snapshot = frozenset(surface.extended_only_canonicals())
    dummy = "__ws_h_dummy_309__"
    try:
        surface.extend_extended_only([dummy])
        # The OLD import-time snapshot would never see a late extension ...
        assert dummy not in old_snapshot
        # ... but the live function always reflects the current surface.
        assert dummy in pandas_first_production_canonicals()
    finally:
        # Undo: restore the pre-test extended-only surface.
        surface.EXTENDED_ONLY_CANONICALS = old_snapshot
    assert dummy not in pandas_first_production_canonicals()


def test_pandas_first_live_view_backward_compat():
    import cleaned_operators.operator_surface as surface
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS

    old_snapshot = frozenset(surface.extended_only_canonicals())
    dummy = "__ws_h_dummy_309b__"
    try:
        surface.extend_extended_only([dummy])
        # Membership on the backward-compatible module-level name is LIVE too,
        # so consumers that were not migrated to the function are not stale.
        assert dummy in PANDAS_FIRST_PRODUCTION_CANONICALS
        assert len(PANDAS_FIRST_PRODUCTION_CANONICALS) == len(old_snapshot) + 1
    finally:
        surface.EXTENDED_ONLY_CANONICALS = old_snapshot


# ---------------------------------------------------------------------------
# #310
# ---------------------------------------------------------------------------
def test_authoring_tier_enum_matches_classify_canonical():
    from cleaned_operators.operator_surface import (
        AuthoringTier,
        authoring_tier,
        classify_canonical,
    )

    for canonical in (
        "ts_mean",  # daily
        "cs_sliced_wasserstein_copula_shift",  # extended
        "holder_concentration_change",  # research
        "constant",  # internal
        "cube",  # legacy
        "tan",  # unsafe
    ):
        assert authoring_tier(canonical) is AuthoringTier(classify_canonical(canonical))


def test_certification_is_orthogonal_to_authoring_tier():
    """An EXTENDED operator with production_certified=False must report
    tier=EXTENDED and cert=PENDING/DENIED — never conflated into 'certified'."""
    from cleaned_operators.operator_surface import (
        ProductionCertification,
        classify_canonical,
        production_certification,
    )
    from cleaned_operators.registry import OperatorRegistry

    # An operator that is authored on the extended surface and registered.
    canonical = "cs_sliced_wasserstein_copula_shift"
    assert classify_canonical(canonical) == "extended"
    catalog = OperatorRegistry._catalog.get(canonical, {})
    assert catalog.get("production_certified") is not True
    cert = production_certification(canonical)
    assert cert in (
        ProductionCertification.PENDING,
        ProductionCertification.DENIED,
    )
    # The certification never silently claims CERTIFIED for a non-certified op.
    assert cert is not ProductionCertification.CERTIFIED


def test_certification_not_driven_by_surface_membership():
    """An operator can be DAILY-authored yet still not evidence-certified in an
    evidence-absent environment; certification must be read from the catalog."""
    from cleaned_operators.operator_surface import (
        classify_canonical,
        production_certification,
    )
    from cleaned_operators.registry import OperatorRegistry

    for canonical in ("ts_mean", "abs"):
        assert classify_canonical(canonical) == "daily"
        if (OperatorRegistry._catalog.get(canonical, {}).get("production_certified")) is True:
            # When evidence is bound, daily ops are CERTIFIED.
            assert production_certification(canonical).value == "certified"
        else:
            # Otherwise they are PENDING (reviewed target awaiting evidence),
            # not conflated into CERTIFIED by surface membership alone.
            assert production_certification(canonical).value in ("pending", "denied")


def test_backend_capability_is_backend_granular():
    from cleaned_operators.operator_surface import backend_capability
    from cleaned_operators.registry import OperatorRegistry

    cap = backend_capability("ts_mean", "polars")
    assert cap.backend == "polars"
    assert cap.available == ("polars" in OperatorRegistry.backends_for("ts_mean"))
    assert isinstance(cap.production_certified, bool)
    assert cap.to_dict()["backend"] == "polars"


# ---------------------------------------------------------------------------
# #311
# ---------------------------------------------------------------------------
def test_reviewed_migration_manifest_structure_present():
    from cleaned_operators.operator_surface import (
        DAILY_FACTOR_MIGRATED,
        REVIEWED_MIGRATION_MANIFEST,
        daily_factor_migrated,
        register_daily_migration,
    )

    assert isinstance(REVIEWED_MIGRATION_MANIFEST, dict)
    assert len(REVIEWED_MIGRATION_MANIFEST) == len(DAILY_FACTOR_MIGRATED)
    for canonical, review in REVIEWED_MIGRATION_MANIFEST.items():
        assert set(review) == {"review_id", "semantic_hash", "approved_authoring_tier"}
        assert review["review_id"]
        assert review["approved_authoring_tier"] == "daily"
    # daily_factor_migrated() reads the manifest.
    assert daily_factor_migrated() == frozenset(REVIEWED_MIGRATION_MANIFEST)
    assert daily_factor_migrated() == frozenset(DAILY_FACTOR_MIGRATED)

    # Forward path: a new migration is recorded with a review_id.
    from cleaned_operators import operator_surface as surface_mod

    dummy = "__ws_h_dummy_311__"
    try:
        register_daily_migration(
            dummy,
            review_id="review-311-test",
            semantic_hash="abc123",
            approved_authoring_tier="daily",
        )
        assert dummy in daily_factor_migrated()
        assert REVIEWED_MIGRATION_MANIFEST[dummy]["review_id"] == "review-311-test"
        # The legacy frozenset is kept in sync at the MODULE level (a
        # ``from ... import DAILY_FACTOR_MIGRATED`` binding taken earlier stays
        # stale — the R5-50 gotcha — so re-read via the module).
        assert dummy in surface_mod.DAILY_FACTOR_MIGRATED
    finally:
        REVIEWED_MIGRATION_MANIFEST.pop(dummy, None)
        # Restore the legacy frozenset to its pre-test value.
        surface_mod.DAILY_FACTOR_MIGRATED = frozenset(
            c for c in surface_mod.DAILY_FACTOR_MIGRATED if c != dummy
        )


def test_daily_classification_consults_reviewed_manifest():
    from cleaned_operators.operator_surface import (
        classify_canonical,
        daily_factor_migrated,
    )

    # Every daily-migrated canonical classifies as daily (no regression).
    for canonical in daily_factor_migrated():
        assert classify_canonical(canonical) == "daily"
