"""R40 #105/#106/#107/#108: storage.catalog typed-JSON field remediation tests."""

from __future__ import annotations

import json

import pytest

from factor_engine.storage.catalog import CatalogJsonField, _canonical_checksum, _parse_json_field
from factor_engine.storage.exceptions import CatalogCorruptionError


# ---------------------------------------------------------------------------
# #105: future schema_version must be rejected
# ---------------------------------------------------------------------------


def test_catalog_json_field_rejects_future_schema():
    env = {"schema_version": 99, "value": {"a": 1}, "checksum": _canonical_checksum({"a": 1})}
    with pytest.raises(CatalogCorruptionError, match="future|高于|schema_version"):
        CatalogJsonField.loads(json.dumps(env), schema_version=1)


# ---------------------------------------------------------------------------
# #106: old checksum verified BEFORE migration
# ---------------------------------------------------------------------------


def test_catalog_migration_verifies_old_checksum_first():
    old_value = {"a": 1}
    raw = {
        "schema_version": 1,
        "value": old_value,
        "checksum": _canonical_checksum(old_value),
    }
    # tamper the value (checksum now stale) — pre-migration check must catch it
    raw["value"] = {"a": 999}
    with pytest.raises(CatalogCorruptionError, match="旧值|old|checksum"):
        CatalogJsonField.loads(
            json.dumps(raw),
            schema_version=3,
            migrate=lambda v: {**v, "migrated": True},
        )


def test_catalog_migration_ok_when_old_checksum_intact():
    old_value = {"a": 1}
    raw = {
        "schema_version": 1,
        "value": old_value,
        "checksum": _canonical_checksum(old_value),
    }
    loaded = CatalogJsonField.loads(
        json.dumps(raw),
        schema_version=3,
        migrate=lambda v: {**v, "migrated": True},
    )
    assert loaded.value == {"a": 1, "migrated": True}
    assert loaded.schema_version == 3


# ---------------------------------------------------------------------------
# #107: dumps always includes checksum
# ---------------------------------------------------------------------------


def test_catalog_dumps_always_includes_checksum():
    field = CatalogJsonField({"a": 1}, schema_version=2)
    raw = json.loads(field.dumps())
    assert "checksum" in raw
    assert raw["checksum"] == _canonical_checksum({"a": 1})


def test_catalog_dumps_legacy_omits_checksum():
    field = CatalogJsonField({"a": 1}, schema_version=2)
    raw = json.loads(field.dumps_legacy())
    assert "checksum" not in raw
    assert raw["schema_version"] == 2


# ---------------------------------------------------------------------------
# #108: non-dict/str/bytes raw types always raise (even non-strict)
# ---------------------------------------------------------------------------


def test_parse_json_field_rejects_scalar():
    with pytest.raises(CatalogCorruptionError):
        _parse_json_field(5, strict=None)
    with pytest.raises(CatalogCorruptionError):
        _parse_json_field(5, strict=False)  # even non-strict


def test_parse_json_field_rejects_list():
    with pytest.raises(CatalogCorruptionError):
        _parse_json_field([1, 2, 3], strict=None)
    with pytest.raises(CatalogCorruptionError):
        _parse_json_field([1, 2, 3], strict=False)


def test_parse_json_field_accepts_dict_and_json_string():
    assert _parse_json_field({"a": 1}, strict=None) == {"a": 1}
    assert _parse_json_field('{"a": 1}', strict=False) == {"a": 1}
