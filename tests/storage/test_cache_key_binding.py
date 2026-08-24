# -*- coding: utf-8 -*-
"""Focused tests for cache key digest binding and validation.

Audit: Verify that cache_key_digest prevents reading a payload written under
a different scoped key, even when the disk path happens to collide.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from factor_engine.storage.cache import PersistentPlanCache, _load_value, _save_value


def test_cache_key_digest_stored_on_write(tmp_path):
    """Verify _save_value stores cache_key_digest in meta."""
    path = tmp_path / "test.parquet"
    series = pd.Series([1.0, 2.0], index=["a", "b"])
    cache_key = "my_scoped_key"

    _save_value(path, series, cache_key=cache_key)

    meta_path = path.with_suffix(".meta.json")
    assert meta_path.exists(), "Meta file should exist"

    meta = json.loads(meta_path.read_text())
    expected_digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    assert meta.get("cache_key_digest") == expected_digest


def test_cache_key_digest_validated_on_read(tmp_path):
    """Verify _load_value rejects payload with mismatched cache_key_digest."""
    path = tmp_path / "test.parquet"
    series = pd.Series([1.0, 2.0], index=["a", "b"])
    cache_key_write = "original_key"

    _save_value(path, series, cache_key=cache_key_write)

    # Load with correct key succeeds
    hit_correct = _load_value(path, cache_key=cache_key_write)
    assert hit_correct is not None
    pd.testing.assert_series_equal(hit_correct, series, check_names=False)

    # Load with wrong key fails validation
    cache_key_wrong = "different_key"
    hit_wrong = _load_value(path, cache_key=cache_key_wrong)
    assert hit_wrong is None, "Mismatched cache_key_digest must reject payload"


def test_cache_key_digest_none_allows_legacy_read(tmp_path):
    """Verify cache_key=None bypasses digest check (legacy compatibility)."""
    path = tmp_path / "test.parquet"
    series = pd.Series([1.0, 2.0], index=["a", "b"])

    # Write with no cache_key (legacy)
    _save_value(path, series, cache_key=None)

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    assert "cache_key_digest" not in meta, "Legacy write should not store digest"

    # Read with cache_key=None succeeds
    hit = _load_value(path, cache_key=None)
    assert hit is not None
    pd.testing.assert_series_equal(hit, series, check_names=False)


def test_persistent_cache_end_to_end_digest_binding(tmp_path):
    """Verify PersistentPlanCache stores and validates digest end-to-end."""
    root = tmp_path / "cache_root"
    cache = PersistentPlanCache(root, data_scope="scope_a")

    series = pd.Series([10.0, 20.0], index=["x", "y"])
    cache.set("my_key", series)

    # Read back succeeds
    hit = cache.get("my_key")
    assert hit is not None
    pd.testing.assert_series_equal(hit, series, check_names=False)

    # Manually tamper with meta to simulate collision
    scoped = cache._scoped_key("my_key")
    path = cache._disk_path(scoped)
    meta_path = path.with_suffix(".meta.json")

    meta = json.loads(meta_path.read_text())
    original_digest = meta["cache_key_digest"]

    # Change digest to simulate different key wrote to same path
    meta["cache_key_digest"] = hashlib.sha256(b"wrong_key").hexdigest()
    meta_path.write_text(json.dumps(meta))

    # Read should fail validation
    cache2 = PersistentPlanCache(root, data_scope="scope_a")
    hit_tampered = cache2.get("my_key")
    assert hit_tampered is None, "Tampered digest must fail validation"

    # Restore correct digest
    meta["cache_key_digest"] = original_digest
    meta_path.write_text(json.dumps(meta))

    # Read succeeds again
    cache3 = PersistentPlanCache(root, data_scope="scope_a")
    hit_restored = cache3.get("my_key")
    assert hit_restored is not None
    pd.testing.assert_series_equal(hit_restored, series, check_names=False)


def test_series_index_name_roundtrip_preserves_none(tmp_path):
    """Verify Series with index.name=None preserves that after parquet round-trip."""
    path = tmp_path / "test.parquet"
    series = pd.Series([1.0, 2.0], index=["a", "b"])
    assert series.index.name is None, "Test precondition"

    _save_value(path, series, cache_key="test")
    loaded = _load_value(path, cache_key="test")

    assert loaded is not None
    assert loaded.index.name is None, "index.name=None must be preserved"
    pd.testing.assert_series_equal(loaded, series, check_names=False, check_index_type=True)
