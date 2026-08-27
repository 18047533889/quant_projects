"""DailySnapshot + SnapshotManifest verification contract tests (spec §17/§18).

Cover:
(a) manifest with matching hashes → verification passes;
(b) a data_content_hash mismatch → fail closed;
(c) a feature-order change → schema hash mismatch → fail closed;
(d) missing manifest field → fail;
(e) the layout helper produces the right key;
(f) manifest_hash round-trips and is embedded for parquet-metadata proof;
(g) compute_schema_hash / compute_data_hash are deterministic and sensitive to
    order and content respectively.
"""

import pytest

from quant_platform.app.contracts.daily_snapshot import (
    DATA_PARQUET_NAME,
    MANIFEST_NAME,
    DailySnapshot,
    SnapshotFeatureRef,
    SnapshotManifest,
    SnapshotVerificationError,
    compute_data_hash,
    compute_schema_hash,
    manifest_hash,
    snapshot_layout_key,
    verify_snapshot,
    verify_snapshot_strict,
)

FS_VER = "FS_alpha_001"
FS_HASH = "abc123" * 4  # 24-char stub; real value would be a sha256
DATA_SNAP = "DS_20240101"
UNIVERSE = "UNI_HS300"


def _features():
    return (
        SnapshotFeatureRef(position=0, feature_name="f_momentum", factor_definition_id="FD_1", dtype="float64"),
        SnapshotFeatureRef(position=1, feature_name="f_vol", factor_definition_id="FD_2", dtype="float64"),
        SnapshotFeatureRef(position=2, feature_name="f_liquidity", factor_definition_id="FD_3", dtype="float64"),
    )


def _feature_cols():
    # Ordered feature columns (schema hash input — feature identity + dtype).
    return [("f_momentum", "float64"), ("f_vol", "float64"), ("f_liquidity", "float64")]


def _columns():
    # The full wide table: InstrumentID + Feature_001..N.
    return [("InstrumentID", "str")] + _feature_cols()


def _rows():
    return [
        ["600001", 0.1, 0.2, 0.3],
        ["600002", 0.4, 0.5, 0.6],
        ["600003", 0.7, 0.8, 0.9],
    ]


def _manifest(**over):
    d = dict(
        feature_set_version_id=FS_VER,
        feature_set_content_hash=FS_HASH,
        ordered_features=("f_momentum", "f_vol", "f_liquidity"),
        schema_hash=compute_schema_hash(_feature_cols()),
        data_content_hash=compute_data_hash(_rows(), _columns()),
        data_snapshot_id=DATA_SNAP,
        universe_id=UNIVERSE,
        trade_date="2024-01-31",
        row_count=len(_rows()),
        column_count=len(_columns()),
        created_at="2024-01-31T12:00:00Z",
    )
    d.update(over)
    return SnapshotManifest(**d)


def _meta():
    return {"data_snapshot_id": DATA_SNAP}


# ---- (a) matching hashes → pass ----
def test_matching_hashes_pass():
    m = _manifest()
    r = verify_snapshot(
        m,
        _meta(),
        expected_feature_set_version_id=FS_VER,
        expected_feature_set_content_hash=FS_HASH,
        expected_data_snapshot_id=DATA_SNAP,
        expected_feature_order=("f_momentum", "f_vol", "f_liquidity"),
        actual_schema_columns=_feature_cols(),
        actual_columns=_columns(),
        actual_rows=_rows(),
        actual_row_count=3,
        actual_column_count=4,
    )
    assert r.ok, r.summary()
    assert all(passed for passed, _ in r.checks.values())


def test_strict_matching_hashes_does_not_raise():
    m = _manifest()
    verify_snapshot_strict(
        m,
        _meta(),
        expected_feature_set_version_id=FS_VER,
        expected_feature_order=("f_momentum", "f_vol", "f_liquidity"),
        actual_schema_columns=_feature_cols(),
        actual_columns=_columns(),
        actual_rows=_rows(),
    )


# ---- (b) data content hash mismatch → fail closed ----
def test_data_hash_mismatch_fails_closed():
    m = _manifest()
    # A changed cell value (0.1 -> 9.9) changes the data content hash.
    bad_rows = [
        ["600001", 9.9, 0.2, 0.3],
        ["600002", 0.4, 0.5, 0.6],
        ["600003", 0.7, 0.8, 0.9],
    ]
    r = verify_snapshot(
        m, _meta(), actual_columns=_columns(), actual_rows=bad_rows
    )
    assert not r.ok
    assert r.checks["data_content_hash"][0] is False
    with pytest.raises(SnapshotVerificationError):
        r.raise_on_fail()


def test_data_hash_mismatch_strict_raises():
    m = _manifest()
    bad_rows = [
        ["600001", 9.9, 0.2, 0.3],
        ["600002", 0.4, 0.5, 0.6],
        ["600003", 0.7, 0.8, 0.9],
    ]
    with pytest.raises(SnapshotVerificationError):
        verify_snapshot_strict(m, _meta(), actual_columns=_columns(), actual_rows=bad_rows)


# ---- (c) feature-order change → schema hash mismatch → fail closed ----
def test_feature_order_change_fails_closed():
    m = _manifest()
    # Reordered columns change the schema hash AND the ordered manifest.
    reordered_feature_cols = [("f_vol", "float64"), ("f_momentum", "float64"), ("f_liquidity", "float64")]
    reordered_cols = [("InstrumentID", "str")] + reordered_feature_cols
    reordered_rows = [
        ["600001", 0.2, 0.1, 0.3],
        ["600002", 0.5, 0.4, 0.6],
        ["600003", 0.8, 0.7, 0.9],
    ]
    r = verify_snapshot(
        m,
        _meta(),
        expected_feature_set_version_id=FS_VER,
        actual_schema_columns=reordered_feature_cols,
        actual_columns=reordered_cols,
        actual_rows=reordered_rows,
    )
    # The reordered data columns change the schema hash → fail closed. The
    # manifest's own ordered_features still matches (the manifest is the source
    # of truth); the drift surfaces via the recomputed schema hash.
    assert not r.ok
    assert r.checks["schema_hash"][0] is False
    with pytest.raises(SnapshotVerificationError):
        r.raise_on_fail()


def test_expected_feature_order_mismatch_fails_closed():
    m = _manifest()
    r = verify_snapshot(
        m,
        _meta(),
        expected_feature_order=("f_vol", "f_momentum", "f_liquidity"),
    )
    assert not r.ok
    assert r.checks["feature_order"][0] is False
    with pytest.raises(SnapshotVerificationError):
        r.raise_on_fail()


def test_schema_hash_changes_with_order():
    a = compute_schema_hash([("x", "float64", 0), ("y", "float64", 1)])
    b = compute_schema_hash([("y", "float64", 0), ("x", "float64", 1)])
    assert a != b


# ---- (d) missing manifest field → fail ----
def test_missing_manifest_field_fails():
    d = {
        "feature_set_version_id": FS_VER,
        "feature_set_content_hash": FS_HASH,
        "ordered_features": ("f_momentum", "f_vol", "f_liquidity"),
        "schema_hash": compute_schema_hash(_columns()),
        "data_content_hash": "x",
        "data_snapshot_id": DATA_SNAP,
        "universe_id": UNIVERSE,
        "trade_date": "2024-01-31",
        "row_count": 3,
        "column_count": 3,
        "created_at": "2024-01-31T12:00:00Z",
    }
    del d["data_snapshot_id"]
    with pytest.raises(SnapshotVerificationError):
        SnapshotManifest.from_dict(d)


# ---- (e) layout helper ----
def test_layout_helper_key():
    keys = snapshot_layout_key(FS_VER, "2024-01-31")
    assert keys["directory"] == (
        f"daily_snapshot/feature_set_version_id={FS_VER}/TradeDate=2024-01-31"
    )
    assert keys["data_parquet"].endswith(f"/{DATA_PARQUET_NAME}")
    assert keys["manifest"].endswith(f"/{MANIFEST_NAME}")
    assert keys["manifest"] == f"{keys['directory']}/{MANIFEST_NAME}"


def test_layout_helper_rejects_bad_date():
    with pytest.raises(ValueError):
        snapshot_layout_key(FS_VER, "2024/01/31")


# ---- (f) manifest_hash embedding ----
def test_manifest_hash_roundtrip():
    m = _manifest()
    m2 = _manifest()
    assert m.manifest_hash  # auto-computed
    assert m.manifest_hash == m2.manifest_hash
    assert manifest_hash(m) == m.manifest_hash


def test_manifest_hash_sensitive_to_content():
    m1 = _manifest()
    m2 = _manifest(trade_date="2024-02-29")
    assert m1.manifest_hash != m2.manifest_hash


# ---- (g) hash determinism / sensitivity ----
def test_compute_data_hash_deterministic_and_sensitive():
    h1 = compute_data_hash(_rows(), _columns())
    h2 = compute_data_hash(_rows(), _columns())
    assert h1 == h2
    changed = [r[:] for r in _rows()]
    changed[1][2] = 0.51
    assert compute_data_hash(changed, _columns()) != h1


def test_compute_schema_hash_deterministic():
    a = compute_schema_hash(_columns())
    b = compute_schema_hash(_columns())
    assert a == b


# ---- DailySnapshot binding ----
def test_daily_snapshot_binds_identity():
    snap = DailySnapshot(
        feature_set_version_id=FS_VER,
        trade_date="2024-01-31",
        feature_set_content_hash=FS_HASH,
        ordered_features=_features(),
        data_snapshot_id=DATA_SNAP,
        universe_id=UNIVERSE,
        decision_time="2024-01-31T09:30:00Z",
        signal_available_time="2024-01-31T09:31:00Z",
        created_at="2024-01-31T12:00:00Z",
    )
    assert snap.feature_set_version_id == FS_VER
    assert snap.trade_date == "2024-01-31"
    assert snap.schema_hash  # auto-computed
    assert snap.schema_hash == compute_schema_hash(
        [("f_momentum", "float64", 0), ("f_vol", "float64", 1), ("f_liquidity", "float64", 2)]
    )
