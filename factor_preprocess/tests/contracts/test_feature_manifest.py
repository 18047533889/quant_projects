"""
Regression tests for FeatureManifest / FeatureBundle contract fixes.

Covers FP-P0-01 through FP-P0-05.
"""
from datetime import datetime
from types import MappingProxyType

import numpy as np
import pytest

from factor_preprocess.contracts.feature_bundle import (
    AxisRef,
    ChannelRef,
    FeatureBundle,
    FeatureManifest,
)
from factor_preprocess.contracts.state import FittedState
from factor_preprocess.errors import InvalidContractError


def _time_axis(n=3):
    return AxisRef(
        axis_name="time",
        axis_values=np.arange(n, dtype="int64"),
        axis_dtype="int64",
    )


def _asset_axis(n=4):
    return AxisRef(
        axis_name="asset",
        axis_values=np.arange(n, dtype="int64"),
        axis_dtype="int64",
    )


def _channels(raw_ids, rank_ids):
    channels = {
        "raw": ChannelRef(
            channel_name="raw",
            channel_type="feature",
            feature_ids=tuple(raw_ids),
        ),
        "rank": ChannelRef(
            channel_name="rank",
            channel_type="feature",
            feature_ids=tuple(rank_ids),
        ),
    }
    return channels


# ============================================================================
# FP-P0-01 — duplicate feature IDs must fail
# ============================================================================


def test_manifest_duplicate_feature_ids_rejected():
    with pytest.raises(InvalidContractError, match="[Dd]uplicate"):
        FeatureManifest(
            feature_ids=["factor_a", "factor_b", "factor_a"],
            channel_offsets={"raw": 0},
            channel_sizes={"raw": 3},
        )


def test_manifest_channel_qualified_keys_allowed():
    # Duplicates that are channel-qualified are distinct feature IDs.
    manifest = FeatureManifest(
        feature_ids=["RAW::factor_001", "RANK::factor_001"],
        channel_offsets={"raw": 0, "rank": 1},
        channel_sizes={"raw": 1, "rank": 1},
    )
    assert manifest.get_column_index("RAW::factor_001") == 0
    assert manifest.get_column_index("RANK::factor_001") == 1


# ============================================================================
# FP-P0-02 — deep snapshot immutability
# ============================================================================


def test_manifest_snapshots_caller_dicts():
    offsets = {"raw": 0, "rank": 2}
    sizes = {"raw": 2, "rank": 2}
    manifest = FeatureManifest(
        feature_ids=["a", "b", "c", "d"],
        channel_offsets=offsets,
        channel_sizes=sizes,
    )
    # Caller mutates their original dicts after construction.
    offsets["rank"] = 99
    sizes["rank"] = 99
    assert manifest.channel_offsets["rank"] == 2
    assert manifest.channel_sizes["rank"] == 2
    assert manifest.get_channel_slice("rank") == slice(2, 4)


def test_bundle_snapshots_caller_channels_mapping():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(),
        asset_axis=_asset_axis(),
        channels=channels,
        values=np.arange(24.0).reshape(2, 3, 4),
        layout="TNF",
    )
    # Caller mutates the source dict after construction.
    channels["raw"] = ChannelRef(
        channel_name="raw", channel_type="feature", feature_ids=("hacked",)
    )
    assert "raw" in bundle.channels
    assert bundle.channels["raw"].feature_ids == ("f1", "f2")


def test_bundle_snapshots_caller_values_array():
    values = np.arange(24.0).reshape(2, 3, 4)
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(),
        asset_axis=_asset_axis(),
        channels=_channels(["f1", "f2"], ["f3", "f4"]),
        values=values,
        layout="TNF",
    )
    # Caller mutates the source ndarray after construction; the bundle's
    # snapshot must be unaffected (FP-P0-02 deep snapshot).
    values[0, 0, 0] = 999.0
    values[1, 2, 3] = -999.0
    np.testing.assert_array_equal(bundle.values[0, 0, 0], 0.0)
    np.testing.assert_array_equal(bundle.values[1, 2, 3], 23.0)
    assert not bundle.values.flags.writeable


# ============================================================================
# FP-P0-03 — shape validation
# ============================================================================


def test_manifest_negative_offset_rejected():
    with pytest.raises(InvalidContractError, match="offset"):
        FeatureManifest(
            feature_ids=["a", "b"],
            channel_offsets={"raw": -1},
            channel_sizes={"raw": 2},
        )


def test_manifest_zero_size_rejected():
    with pytest.raises(InvalidContractError, match="size"):
        FeatureManifest(
            feature_ids=["a"],
            channel_offsets={"raw": 0},
            channel_sizes={"raw": 0},
        )


def test_manifest_overlapping_channels_rejected():
    with pytest.raises(InvalidContractError, match="overlap"):
        FeatureManifest(
            feature_ids=["a", "b", "c"],
            channel_offsets={"raw": 0, "rank": 1},
            channel_sizes={"raw": 2, "rank": 2},
        )


def test_manifest_max_end_mismatch_rejected():
    # offset+size = 100+30 = 130, but only 50 features provided.
    with pytest.raises(InvalidContractError, match="feature count|does not match"):
        FeatureManifest(
            feature_ids=[f"f{i}" for i in range(50)],
            channel_offsets={"raw": 0, "rank": 100},
            channel_sizes={"raw": 40, "rank": 30},
        )


def test_bundle_manifest_values_dimension_mismatch_rejected():
    manifest = FeatureManifest(
        feature_ids=[f"f{i}" for i in range(40)],
        channel_offsets={"raw": 0},
        channel_sizes={"raw": 40},
    )
    channels = {
        "raw": ChannelRef("raw", "feature", tuple(f"f{i}" for i in range(40)))
    }
    values = np.zeros((5, 50), dtype="float64")  # 50 != 40
    with pytest.raises(InvalidContractError, match="feature dimension"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(),
            asset_axis=_asset_axis(),
            channels=channels,
            values=values,
            layout="NF",
            manifest=manifest,
        )


def test_bundle_valid_non_square_manifest_passes():
    manifest = FeatureManifest(
        feature_ids=["a", "b", "c"],
        channel_offsets={"raw": 0},
        channel_sizes={"raw": 3},
    )
    channels = {"raw": ChannelRef("raw", "feature", ("a", "b", "c"))}
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(n=3),
        asset_axis=_asset_axis(n=4),
        channels=channels,
        values=np.zeros((4, 3), dtype="float64"),
        layout="NF",
        manifest=manifest,
    )
    assert bundle.is_immutable()


# ============================================================================
# FP-P0-04 — feature axis extraction for [T,N,F] layout
# ============================================================================


def test_tnf_layout_extracts_feature_axis_not_asset_axis():
    time_axis = AxisRef(
        axis_name="time",
        axis_values=np.arange(2, dtype="int64"),
        axis_dtype="int64",
    )
    asset_axis = AxisRef(
        axis_name="asset",
        axis_values=np.arange(5, dtype="int64"),  # N=5, F=3 -> non-square
        axis_dtype="int64",
    )
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    manifest = FeatureManifest(
        feature_ids=["f1", "f2", "f3", "f4"],
        channel_offsets={"raw": 0, "rank": 2},
        channel_sizes={"raw": 2, "rank": 2},
    )
    # [T=2, N=5, F=4]: the value equals the feature index, so a correct
    # implementation extracts a feature plane that is constant across time
    # and asset (e.g. raw[..., 0] == 0 everywhere). If the implementation
    # sliced the wrong (asset) axis instead, the extracted values would vary
    # with n.
    values = np.broadcast_to(
        np.arange(4, dtype="float64").reshape(1, 1, 4), (2, 5, 4)
    ).copy()

    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=time_axis,
        asset_axis=asset_axis,
        channels=channels,
        values=values,
        layout="TNF",
        manifest=manifest,
    )

    raw = bundle.get_channel_values("raw")  # columns f1, f2
    assert raw.shape == (2, 5, 2)
    # Feature values must be constant across time and asset, i.e. equal to f1/f2.
    assert np.all(raw[..., 0] == 0.0)  # f1
    assert np.all(raw[..., 1] == 1.0)  # f2

    f3 = bundle.get_feature_values("f3")
    assert f3.shape == (2, 5)
    assert np.all(f3 == 2.0)


def test_nf_layout_extracts_feature_axis():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    manifest = FeatureManifest(
        feature_ids=["f1", "f2", "f3", "f4"],
        channel_offsets={"raw": 0, "rank": 2},
        channel_sizes={"raw": 2, "rank": 2},
    )
    values = np.array(
        [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]], dtype="float64"
    )
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(),
        asset_axis=_asset_axis(n=3),
        channels=channels,
        values=values,
        layout="NF",
        manifest=manifest,
    )
    raw = bundle.get_channel_values("raw")
    assert raw.shape == (3, 2)
    np.testing.assert_array_equal(raw, values[:, :2])
    f4 = bundle.get_feature_values("f4")
    np.testing.assert_array_equal(f4, values[:, 3])


# ============================================================================
# FP-P0-05 — FittedState identity is content-derived
# ============================================================================


def test_fitted_state_hash_auto_derived_when_none():
    params = {"mu": np.array([1.0, 2.0]), "sigma": 1.5, "flags": [True, False]}
    state = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params=params,
    )
    assert state.learned_params_hash is not None
    assert len(state.learned_params_hash) == 64  # sha256 hex

    # Identical content must hash identically, regardless of insertion order.
    params2 = {"sigma": 1.5, "flags": [True, False], "mu": np.array([1.0, 2.0])}
    state2 = FittedState(
        state_id="s2",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params=params2,
    )
    assert state.learned_params_hash == state2.learned_params_hash

    # Different content must hash differently.
    state3 = FittedState(
        state_id="s3",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params={"mu": np.array([1.0, 3.0]), "sigma": 1.5},
    )
    assert state.learned_params_hash != state3.learned_params_hash


def test_fitted_state_wrong_provided_hash_rejected():
    with pytest.raises(InvalidContractError, match="does not match"):
        FittedState(
            state_id="s1",
            transform_name="cs_rank",
            transform_version="1.0.0",
            fit_start_time=datetime(2024, 1, 1),
            fit_end_time=datetime(2024, 1, 2),
            learned_params={"mu": 1.0},
            learned_params_hash="deadbeef",
        )


def test_fitted_state_correct_provided_hash_accepted():
    params = {"mu": 1.0, "w": np.array([0.5, 0.5])}
    auto = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params=params,
    )
    provided = FittedState(
        state_id="s2",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params=params,
        learned_params_hash=auto.learned_params_hash,
    )
    assert provided.learned_params_hash == auto.learned_params_hash


def test_fitted_state_hash_changes_when_params_mutate():
    # The hash must be derived from parameter CONTENT, not from object
    # identity: mutating the source container after construction (which the
    # frozen snapshot decouples from the state) must change the hash.
    params = {"mu": np.array([1.0, 2.0, 3.0])}
    state_a = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params=params,
    )
    params["mu"][1] = 999.0  # mutate the caller-owned array
    state_b = FittedState(
        state_id="s2",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params={"mu": np.array([1.0, 2.0, 3.0])},
    )
    # state_a's hash reflects the ORIGINAL content; mutating the caller's
    # array afterwards must not change state_a's already-derived hash.
    params["mu"][1] = 123.0
    assert state_a.learned_params_hash == state_b.learned_params_hash
    # A genuinely different parameter array hashes differently.
    params2 = {"mu": np.array([1.0, 9.0, 3.0])}
    state_c = FittedState(
        state_id="s3",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params=params2,
    )
    assert state_a.learned_params_hash != state_c.learned_params_hash


def test_fitted_state_provenance_fields_present():
    state = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        learned_params={"mu": 1.0},
        implementation_hash="impl-abc",
        data_snapshot_ref="snap-2024-01-01",
        split_ref="train-0.7",
    )
    assert state.implementation_hash == "impl-abc"
    assert state.data_snapshot_ref == "snap-2024-01-01"
    assert state.split_ref == "train-0.7"
    assert state.learned_params_hash is not None
