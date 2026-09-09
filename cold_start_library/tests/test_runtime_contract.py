from __future__ import annotations

import json

import pytest

from cold_start_library.runtime.loader import (
    default_catalog_path,
    load_cold_start_catalog,
    load_cold_start_for_training,
)
from cold_start_library.runtime.dsl import validate_factor_engine_dsl
from cold_start_library.runtime.paths import factor_engine_root


def test_default_catalog_is_shipped_authoritative_core_pool():
    path = default_catalog_path()
    assert path.name == "production_default_core_v9.json"
    entries = load_cold_start_catalog(
        path, market="A", surface="daily", availability_tier="core",
    )
    assert entries
    assert len({entry.factor_id for entry in entries}) == len(entries)
    assert len({entry.expr for entry in entries}) == len(entries)
    assert {entry.availability_tier for entry in entries} == {"core"}
    assert {entry.market for entry in entries} == {"A"}
    assert {entry.surface for entry in entries} == {"daily"}
    assert len({entry.catalog_ref for entry in entries}) == 1


def test_default_training_loader_is_deterministic_and_identity_bound():
    first = load_cold_start_for_training(sample_size=20, seed=7)
    second = load_cold_start_for_training(sample_size=20, seed=7)
    assert first == second
    assert len(first) == 20


def test_default_catalog_rejects_tier_drift_and_duplicates(tmp_path):
    source = json.loads(default_catalog_path().read_text(encoding="utf-8"))
    source[0]["v9_default_pool"] = "extended"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="core pool"):
        load_cold_start_catalog(changed)

    source = json.loads(default_catalog_path().read_text(encoding="utf-8"))
    source = [row for row in source if row["market"] == "A" and
              row["layer"] == "daily" and row["availability_tier"] == "core"]
    source.append(dict(source[0]))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_cold_start_catalog(
            duplicate, market="A", surface="daily", availability_tier="core",
        )


def test_dsl_uses_canonical_factor_engine_and_rejects_tree_override(monkeypatch, tmp_path):
    ok, message = validate_factor_engine_dsl("rank(close)")
    assert ok, message
    canonical = factor_engine_root()
    monkeypatch.setenv("FACTOR_ENGINE_ROOT", str(canonical))
    assert factor_engine_root() == canonical
    monkeypatch.setenv("FACTOR_ENGINE_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="different tree"):
        factor_engine_root()
