"""FA-P0-15: versioned codec contract for durable registry records.

Pins the CODEC_VERSION bump to v2 (validation_status/health_state on
asset payloads): writers emit v2, readers accept v1 (defaults fail-closed
to UNVALIDATED/ACTIVE) and v2, and unknown/future versions are rejected
with SchemaVersionError instead of being guessed at.
"""
import json

import pytest

from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import (
    HealthState,
    LifecycleState,
    StateEvent,
    ValidationStatus,
)
from factor_assets.errors import SchemaVersionError
from factor_assets.registry import serialization as ser


def _metadata():
    return AssetMetadata("F", "f", "hash", "daily", ("price",), "daily")


def _lineage():
    return LineageRef("F", ())


def _asset(**overrides):
    kwargs = dict(
        metadata=_metadata(),
        lineage=_lineage(),
        lifecycle_state=LifecycleState.EVALUATED,
        registered_at="2026-08-19T00:00:00Z",
    )
    kwargs.update(overrides)
    return FactorAsset(**kwargs)


def _event():
    return StateEvent(
        factor_id="F",
        from_state=LifecycleState.EVALUATED,
        to_state=LifecycleState.APPROVED,
        timestamp="2026-08-19T00:00:00Z",
        evidence_refs=("qe-bundle:b1",),
    )


class TestCodecVersion:
    def test_current_version_is_two(self):
        assert ser.CODEC_VERSION == 2

    def test_writers_emit_current_version(self):
        for text in (
            ser.metadata_to_json(_metadata()),
            ser.lineage_to_json(_lineage()),
            ser.asset_to_json(_asset()),
            ser.event_to_json(_event()),
        ):
            assert json.loads(text)["version"] == ser.CODEC_VERSION


class TestAssetRoundTrip:
    def test_validation_and_health_survive_round_trip(self):
        asset = _asset(
            validation_status=ValidationStatus.VALIDATED,
            health_state=HealthState.DEPRECATED,
        )
        restored = ser.asset_from_json(ser.asset_to_json(asset))
        assert restored.validation_status == ValidationStatus.VALIDATED
        assert restored.health_state == HealthState.DEPRECATED

    def test_defaults_round_trip(self):
        asset = _asset()
        restored = ser.asset_from_json(ser.asset_to_json(asset))
        assert restored.validation_status == ValidationStatus.UNVALIDATED
        assert restored.health_state == HealthState.ACTIVE

    def test_all_kinds_round_trip(self):
        metadata = ser.metadata_from_json(ser.metadata_to_json(_metadata()))
        assert metadata == _metadata()
        lineage = ser.lineage_from_json(ser.lineage_to_json(_lineage()))
        assert lineage == _lineage()
        event = ser.event_from_json(ser.event_to_json(_event()))
        assert event.to_state == LifecycleState.APPROVED
        assert ser.evidence_from_json(None) is None
        bundle = EvidenceBundleRef(
            "b", "run", ("F",), "2026-01-01T00:00:00Z", "qe",
            primary_metric="ic", primary_value=1.0,
        )
        assert ser.evidence_from_json(ser.evidence_to_json(bundle)) == bundle

    def test_deterministic(self):
        # Same asset → byte-identical JSON (revision-pinning in the
        # sqlite repository depends on canonical serialization).
        a = ser.asset_to_json(_asset())
        b = ser.asset_to_json(_asset())
        assert a == b


class TestVersionCompat:
    def _repack_as_v1(self, text):
        obj = json.loads(text)
        obj["version"] = 1
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def test_v1_asset_reads_with_fail_closed_defaults(self):
        # v1 payloads predate validation_status/health_state; reading
        # one must default to UNVALIDATED/ACTIVE (never VALIDATED), even
        # when every other field would justify the stronger status.
        v2 = json.loads(ser.asset_to_json(_asset(
            validation_status=ValidationStatus.VALIDATED,
            health_state=HealthState.RETIRED,
        )))
        v1 = {"codec": "asset", "version": 1, "value": v2["value"]}
        v1["value"] = {k: v for k, v in v1["value"].items()
                       if k not in ("validation_status", "health_state")}
        restored = ser.asset_from_json(json.dumps(v1))
        assert restored.validation_status == ValidationStatus.UNVALIDATED
        assert restored.health_state == HealthState.ACTIVE

    def test_v1_event_reads(self):
        restored = ser.event_from_json(self._repack_as_v1(ser.event_to_json(_event())))
        assert restored.to_state == LifecycleState.APPROVED

    def test_future_version_rejected(self):
        obj = json.loads(ser.asset_to_json(_asset()))
        obj["version"] = ser.CODEC_VERSION + 1
        future = json.dumps(obj)
        with pytest.raises(SchemaVersionError, match="codec version"):
            ser.asset_from_json(future)

    def test_unknown_version_zero_rejected(self):
        obj = json.loads(ser.metadata_to_json(_metadata()))
        obj["version"] = 0
        with pytest.raises(SchemaVersionError):
            ser.metadata_from_json(json.dumps(obj))

    def test_wrong_kind_rejected(self):
        # A metadata payload presented to the asset reader is invalid
        # regardless of version.
        text = ser.metadata_to_json(_metadata())
        with pytest.raises(ValueError, match="unsupported or invalid"):
            ser.asset_from_json(text)

    def test_missing_version_rejected(self):
        obj = json.loads(ser.asset_to_json(_asset()))
        del obj["version"]
        with pytest.raises(SchemaVersionError):
            ser.asset_from_json(json.dumps(obj))

    @pytest.mark.parametrize("bad_version", [True, 1.0, "1", None])
    def test_non_int_version_rejected(self, bad_version):
        # bool (True == 1) and float (1.0 == 1) must not satisfy the
        # v1 membership test — malformed versions fail closed.
        obj = json.loads(ser.asset_to_json(_asset()))
        obj["version"] = bad_version
        with pytest.raises(SchemaVersionError):
            ser.asset_from_json(json.dumps(obj))

    def test_version_error_is_not_value_error(self):
        # Pre-v2 a bad version raised ValueError; the typed
        # SchemaVersionError contract change is pinned here so an
        # external caller catching ValueError knows it no longer traps
        # version errors.
        obj = json.loads(ser.asset_to_json(_asset()))
        obj["version"] = 99
        with pytest.raises(SchemaVersionError) as excinfo:
            ser.asset_from_json(json.dumps(obj))
        assert not isinstance(excinfo.value, ValueError)
