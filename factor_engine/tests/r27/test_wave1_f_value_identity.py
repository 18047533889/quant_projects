# -*- coding: utf-8 -*-
"""Wave1-F Part A tests — factor-lake value identity (non-vacuous).

Covers:
  (a) identity round-trips: FactorValueIdentity <-> dict <-> FeatureSetIdentity,
      and writer->manifest->reader round-trip preserves identity_hash.
  (b) hash sensitivity: changing numeric_policy OR universe_snapshot_id (and
      every other component) changes the identity hash; reordering the member
      factor list changes the FeatureSetIdentity content hash.
  (c) rejection: a block whose manifest identity content-hash does not match the
      real block bytes is detected by the reader (verify_manifest_identities).
  (d) model-layer helper: feature_set_identity_for_backtest returns the ordered
      FeatureSetIdentity for a backtest factor set (never bare names).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from factor_engine.runtime.factor_value_identity import (
    UNKNOWN_SNAPSHOT,
    FactorValueIdentity,
    FeatureSetIdentity,
    build_factor_value_identity_from_block,
    compute_block_content_hash,
    feature_set_identity_for_backtest,
    verify_manifest_factors_against_content,
)
from factor_engine.runtime.feature_block import (
    FeatureBlockReader,
    FeatureBlockWriter,
    load_manifest,
)


def _identity(fid="F1", **overrides) -> FactorValueIdentity:
    fields_ = {
        "factor_id": fid,
        "factor_definition_hash": "def1",
        "factor_implementation_hash": "impl1",
        "treatment_preprocess_hash": "treat1",
        "market": "CN",
        "frequency": "1d",
        "partition_time": "2026-08-01",
        "universe_snapshot_id": "uni1",
        "ordered_instrument_hash": "inst1",
        "data_snapshot_id": "snap1",
        "calendar_snapshot_id": "cal1",
        "decision_time_policy": "pit",
        "numeric_policy": "float32",
        "dtype": "float32",
        "build_sha": "build-abc",
        "block_content_hash": "content-abc",
    }
    fields_.update(overrides)
    return FactorValueIdentity(**fields_)


def _writer_identity(fid: str) -> FactorValueIdentity:
    """Identity WITHOUT a content hash — the writer backfills the real block
    bytes so ``verify_manifest_identities`` recomputes and matches."""
    return _identity(fid, block_content_hash="")


# ---- (a) identity round-trips ------------------------------------------------

def test_factor_value_identity_roundtrip_dict():
    i = _identity()
    d = i.to_dict()
    # to_dict carries identity_hash but from_dict reconstructs the same semantic id.
    j = FactorValueIdentity.from_dict(d)
    assert j == i
    assert j.identity_hash() == i.identity_hash()
    assert d["identity_hash"] == i.identity_hash()


def test_feature_set_identity_roundtrip_dict():
    fs = FeatureSetIdentity.build("bt::v1", [_identity("F1"), _identity("F2")])
    d = fs.to_dict()
    g = FeatureSetIdentity.from_dict(d)
    assert g.feature_set_version_id == fs.feature_set_version_id
    assert g.feature_set_content_hash == fs.feature_set_content_hash
    assert [f.identity_hash() for f in g.factors] == [
        f.identity_hash() for f in fs.factors
    ]


def test_writer_manifest_reader_roundtrip_preserves_identity(tmp_path):
    w = FeatureBlockWriter(tmp_path / "feat", columns_per_block=64)
    for i in range(10):
        w.add(f"factor_{i:03d}", np.arange(100, dtype=np.float32) + i)
        w.set_identity(f"factor_{i:03d}", _writer_identity(f"factor_{i:03d}"))
    w.finish()

    r = FeatureBlockReader(tmp_path / "feat")
    ids = r.identities()
    assert len(ids) == 10
    # writer->reader identity round-trip: identity_hash preserved.
    for i in range(10):
        fid = f"factor_{i:03d}"
        assert ids[fid]["identity_hash"] == _identity(fid).identity_hash()
    # reader recomputes content hashes and finds no mismatch.
    assert r.verify_manifest_identities() == []


# ---- (b) hash sensitivity ---------------------------------------------------

@pytest.mark.parametrize(
    "field,value",
    [
        ("numeric_policy", "float64"),
        ("universe_snapshot_id", "uni2"),
        ("factor_definition_hash", "def2"),
        ("factor_implementation_hash", "impl2"),
        ("treatment_preprocess_hash", "treat2"),
        ("market", "HK"),
        ("frequency", "5m"),
        ("partition_time", "2026-09-01"),
        ("ordered_instrument_hash", "inst2"),
        ("data_snapshot_id", "snap2"),
        ("calendar_snapshot_id", "cal2"),
        ("decision_time_policy", "close"),
        ("dtype", "float64"),
        ("build_sha", "build-xyz"),
    ],
)
def test_any_component_change_changes_identity_hash(field, value):
    base = _identity()
    changed = _identity(**{field: value})
    assert base.identity_hash() != changed.identity_hash()


def test_block_content_hash_change_changes_feature_set_identity():
    a = _identity("F1", block_content_hash="h1")
    b = _identity("F1", block_content_hash="h2")
    # semantic identity hash ignores content hash (verification bound)
    assert a.identity_hash() == b.identity_hash()
    # ... but the FeatureSetIdentity content hash is bound to the block content.
    fs1 = FeatureSetIdentity.build("bt::v1", [a])
    fs2 = FeatureSetIdentity.build("bt::v1", [b])
    assert fs1.feature_set_content_hash != fs2.feature_set_content_hash


def test_universe_change_changes_feature_set_identity():
    u1 = _identity("F1", universe_snapshot_id="uniA")
    u2 = _identity("F1", universe_snapshot_id="uniB")
    fs1 = FeatureSetIdentity.build("bt::v1", [u1])
    fs2 = FeatureSetIdentity.build("bt::v1", [u2])
    assert fs1.feature_set_content_hash != fs2.feature_set_content_hash


def test_feature_set_member_order_changes_content_hash():
    a = _identity("F1")
    b = _identity("F2")
    fs_ab = FeatureSetIdentity.build("bt::v1", [a, b])
    fs_ba = FeatureSetIdentity.build("bt::v1", [b, a])
    assert fs_ab.feature_set_content_hash != fs_ba.feature_set_content_hash


# ---- (c) rejection: block whose identity doesn't match its content ----------

def test_reader_rejects_block_with_mismatched_identity_content(tmp_path):
    w = FeatureBlockWriter(tmp_path / "feat", columns_per_block=64)
    for i in range(10):
        w.add(f"factor_{i:03d}", np.arange(100, dtype=np.float32) + i)
        w.set_identity(f"factor_{i:03d}", _writer_identity(f"factor_{i:03d}"))
    w.finish()

    # Tamper with a block's value file so the stored bytes no longer match the
    # identity content hash the writer recorded.
    block_files = sorted((tmp_path / "feat").rglob("block_*.npy"))
    assert block_files
    arr = np.load(block_files[0], allow_pickle=False)
    arr[0, :] += 1.0  # mutate stored bytes
    np.save(block_files[0], arr)

    r = FeatureBlockReader(tmp_path / "feat")
    mismatches = r.verify_manifest_identities()
    assert mismatches, "reader must detect the tampered block"
    # At least the factors living in the first block are flagged.
    assert any(f.startswith("factor_") for f in mismatches)

    # verify_manifest_factors_against_content (module-level) agrees.
    mm = verify_manifest_factors_against_content(r.manifest, reader=r)
    assert set(mm) == set(mismatches)


def test_compute_block_content_hash_is_deterministic_and_sensitive():
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    h1 = compute_block_content_hash(arr, factor_ids=["a", "b", "c", "d", "e"])
    h2 = compute_block_content_hash(arr.copy(), factor_ids=["a", "b", "c", "d", "e"])
    assert h1 == h2
    arr2 = arr.copy()
    arr2[0, 0] += 1.0
    assert h1 != compute_block_content_hash(arr2, factor_ids=["a", "b", "c", "d", "e"])


# ---- (d) model-layer helper -------------------------------------------------

def test_feature_set_identity_for_backtest_returns_ordered_identity():
    ids = {
        "F1": _identity("F1"),
        "F2": _identity("F2"),
        "F3": _identity("F3"),
    }
    fs = feature_set_identity_for_backtest(
        backtest_id="bt-001",
        factor_ids=["F1", "F2", "F3"],
        identities=ids,
        universe="csi500",
    )
    assert fs.feature_set_version_id == "bt-001::v1"
    assert [f.factor_id for f in fs.factors] == ["F1", "F2", "F3"]
    assert fs.factors[0].identity_hash() == ids["F1"].identity_hash()
    # universe participates in the feature-set identity through the member's
    # universe_snapshot_id; pass different universes to the *helper synthesis*
    # for factors WITHOUT manifest identities.
    fs_other = feature_set_identity_for_backtest(
        backtest_id="bt-001",
        factor_ids=["F9"],
        identities={},
        universe="csi300",
    )
    fs_other2 = feature_set_identity_for_backtest(
        backtest_id="bt-001",
        factor_ids=["F9"],
        identities={},
        universe="csi500",
    )
    assert fs_other.feature_set_content_hash != fs_other2.feature_set_content_hash


def test_feature_set_identity_for_backtest_never_degrades_to_bare_name():
    # A factor with NO manifest identity is synthesized from context — never a
    # bare factor name in the identity.
    fs = feature_set_identity_for_backtest(
        backtest_id="bt-002",
        factor_ids=["F9"],
        identities={},
        universe="csi500",
        frequency="1d",
        build_sha="sha-x",
    )
    f = fs.factors[0]
    assert f.factor_id == "F9"
    assert f.identity_hash() != "F9"  # not a bare name
    assert f.universe_snapshot_id != UNKNOWN_SNAPSHOT  # universe baked in
    # Content hash of the feature set is non-trivial.
    assert len(fs.feature_set_content_hash) == 64


def test_feature_set_identity_for_backtest_from_manifest_dict():
    # identities from a manifest (dict form) also work.
    manifest_ids = {"F1": _identity("F1").to_dict()}
    fs = feature_set_identity_for_backtest(
        backtest_id="bt-003", factor_ids=["F1"], identities=manifest_ids
    )
    assert fs.factors[0].identity_hash() == _identity("F1").identity_hash()


def test_writer_feature_set_identity_lands_in_manifest(tmp_path):
    w = FeatureBlockWriter(tmp_path / "feat", columns_per_block=64)
    for i in range(5):
        w.add(f"factor_{i:03d}", np.arange(100, dtype=np.float32) + i)
    w.finish()
    fs = feature_set_identity_for_backtest(
        backtest_id="bt-x", factor_ids=[f"factor_{i:03d}" for i in range(5)],
        identities={},
    )
    # Wire the feature-set identity through the writer's manifest.
    w2 = FeatureBlockWriter(tmp_path / "feat2", columns_per_block=64)
    for i in range(5):
        w2.add(f"factor_{i:03d}", np.arange(100, dtype=np.float32) + i)
        w2.set_identity(f"factor_{i:03d}", _writer_identity(f"factor_{i:03d}"))
    w2.set_feature_set_identity(fs)
    w2.finish()

    r = FeatureBlockReader(tmp_path / "feat2")
    stored_fs = r.feature_set_identity()
    assert stored_fs is not None
    assert stored_fs["feature_set_version_id"] == fs.feature_set_version_id
    assert stored_fs["feature_set_content_hash"] == fs.feature_set_content_hash
    # Rebuilding from the manifest gives back the same identity.
    rebuilt = FeatureSetIdentity.from_dict(stored_fs)
    assert rebuilt.feature_set_content_hash == fs.feature_set_content_hash
    assert r.verify_manifest_identities() == []
