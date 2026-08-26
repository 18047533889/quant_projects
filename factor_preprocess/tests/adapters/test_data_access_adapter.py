"""
Red-first tests for the DataAccess exposure adapter closure.

Covers:
- create_adapter() never raises ModuleNotFoundError (dataaccess present or absent)
- unknown exposure types fail closed
- exposure bundles without mandatory provenance are rejected (fail-closed)
- fetch_multi_exposure with an unknown key fails closed
"""
import numpy as np
import pytest

from factor_preprocess.adapters.data_access import (
    DataAccessAdapter,
    OptionalDependencyMissing,
    REQUIRED_PROVENANCE_KEYS,
    SUPPORTED_EXPOSURE_TYPES,
    _validate_exposure_bundle,
    check_data_access_available,
    create_adapter,
)


def _valid_bundle(overrides=None, metadata_overrides=None, drop_metadata_keys=None):
    """Build a provenance-complete exposure bundle."""
    metadata = {
        "knowledge_time": "2024-01-02T00:00:00",
        "effective_time": "2024-01-02",
        "snapshot_ref": "snap-123",
        "classification_version": "default",
        "market": "ashare",
        "universe_ref": "ashare_universe_daily",
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    if drop_metadata_keys:
        for k in drop_metadata_keys:
            metadata.pop(k, None)
    bundle = {
        "values": np.array([[1.0], [2.0]]),
        "dates": np.array(["2024-01-02", "2024-01-03"]),
        "assets": np.array(["000001.SZ", "000002.SZ"]),
        "metadata": metadata,
    }
    if overrides:
        bundle.update(overrides)
    return bundle


class _StubProvider:
    """Minimal ExposureProvider stub returning provenance-complete bundles."""

    def get_industry_exposure(self, *args, **kwargs):
        return _valid_bundle()

    def get_size_exposure(self, *args, **kwargs):
        return _valid_bundle()

    def get_sector_exposure(self, *args, **kwargs):
        return _valid_bundle()

    def get_beta_exposure(self, *args, **kwargs):
        return _valid_bundle()

    def get_custom_exposure(self, *args, **kwargs):
        return _valid_bundle()


class _NoProvenanceProvider:
    """Provider that returns bundles WITHOUT provenance metadata."""

    def get_industry_exposure(self, *args, **kwargs):
        return _valid_bundle(drop_metadata_keys=list(REQUIRED_PROVENANCE_KEYS))

    def get_size_exposure(self, *args, **kwargs):
        return _valid_bundle(drop_metadata_keys=list(REQUIRED_PROVENANCE_KEYS))

    def get_sector_exposure(self, *args, **kwargs):
        return _valid_bundle(drop_metadata_keys=list(REQUIRED_PROVENANCE_KEYS))

    def get_beta_exposure(self, *args, **kwargs):
        return _valid_bundle(drop_metadata_keys=list(REQUIRED_PROVENANCE_KEYS))

    def get_custom_exposure(self, *args, **kwargs):
        return _valid_bundle(drop_metadata_keys=list(REQUIRED_PROVENANCE_KEYS))


def test_unknown_exposure_fails_closed():
    """Requesting an exposure type outside the supported set raises, never garbage."""
    adapter = DataAccessAdapter(_StubProvider())
    with pytest.raises(ValueError):
        adapter.fetch_multi_exposure(market="ashare", exposure_types=["momentum"])
    # 'custom' is a supported type; an arbitrary name is not.
    assert "momentum" not in SUPPORTED_EXPOSURE_TYPES


def test_pit_exposure_safety():
    """A bundle without provenance is rejected (fail-closed)."""
    adapter = DataAccessAdapter(_NoProvenanceProvider())
    with pytest.raises(ValueError):
        adapter.fetch_industry_exposure(market="ashare")
    with pytest.raises(ValueError):
        adapter.fetch_size_exposure(market="ashare")
    with pytest.raises(ValueError):
        adapter.fetch_multi_exposure(market="ashare", exposure_types=["sector"])


def test_create_adapter_no_crash(monkeypatch):
    """create_adapter() returns an adapter OR raises OptionalDependencyMissing, never ModuleNotFoundError."""

    # Branch 1: dataaccess present -> returns a DataAccessAdapter.
    monkeypatch.setattr(
        "factor_preprocess.adapters.data_access.check_data_access_available",
        lambda: True,
    )
    adapter = create_adapter()
    assert isinstance(adapter, DataAccessAdapter)

    # Branch 2: dataaccess absent -> raises OptionalDependencyMissing.
    monkeypatch.setattr(
        "factor_preprocess.adapters.data_access.check_data_access_available",
        lambda: False,
    )
    with pytest.raises(OptionalDependencyMissing):
        create_adapter()

    # Branch 3: explicit provider -> never touches dataaccess at all.
    adapter = create_adapter(provider=_StubProvider())
    assert isinstance(adapter, DataAccessAdapter)


def test_multiexposure_unknown_key():
    """fetch_multi_exposure with an unknown exposure_type fails closed."""
    adapter = DataAccessAdapter(_StubProvider())
    with pytest.raises(ValueError):
        adapter.fetch_multi_exposure(market="ashare", exposure_types=["industry", "bogus"])


def test_validate_bundle_requires_provenance():
    """_validate_exposure_bundle rejects bundles missing any mandatory key."""
    with pytest.raises(ValueError):
        _validate_exposure_bundle(
            _valid_bundle(drop_metadata_keys=["market"]), "industry"
        )
    with pytest.raises(ValueError):
        _validate_exposure_bundle({"values": np.zeros(2)}, "industry")
    # A complete bundle passes through.
    out = _validate_exposure_bundle(_valid_bundle(), "industry")
    assert out["metadata"]["snapshot_ref"] == "snap-123"
