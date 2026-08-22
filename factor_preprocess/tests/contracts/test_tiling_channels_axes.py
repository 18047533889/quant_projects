"""
FP-P0-01/02/03/04/05 hardening tests.

Covers:
- FP-P0-01: exact-tiling (no holes) in FeatureManifest.
- FP-P0-02: auxiliary channels (missing/freshness/exposure) are physically
  addressable; has_missing/has_freshness/has_exposure agree with actual channels.
- FP-P0-03: AxisRef vs tensor shape validation (TNF/NF ndim, axis lengths,
  duplicate labels).
- FP-P0-04: FittedState content-derived state_id in production.
- FP-P0-05: stateful transform with empty feature contract fails closed.
"""
from datetime import datetime

import numpy as np
import pytest

from factor_preprocess.contracts.feature_bundle import (
    AxisRef,
    ChannelRef,
    FeatureBundle,
    FeatureManifest,
)
from factor_preprocess.contracts.state import FittedState, StateKind
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
    return {
        "raw": ChannelRef("raw", "feature", tuple(raw_ids)),
        "rank": ChannelRef("rank", "feature", tuple(rank_ids)),
    }


# ============================================================================
# FP-P0-01 — exact tiling (no holes)
# ============================================================================


def test_holed_tiling_rejected():
    # Columns {0, 2} occupied, column 1 is a hole. F=3. Must be rejected.
    with pytest.raises(InvalidContractError, match="no holes|tile"):
        FeatureManifest(
            feature_ids=["a", "b", "c"],
            channel_offsets={"raw": 0, "rank": 2},
            channel_sizes={"raw": 1, "rank": 1},
        )


def test_exact_tiling_accepted():
    manifest = FeatureManifest(
        feature_ids=["a", "b", "c", "d"],
        channel_offsets={"raw": 0, "rank": 2},
        channel_sizes={"raw": 2, "rank": 2},
    )
    assert manifest.total_features == 4
    assert manifest.get_channel_slice("raw") == slice(0, 2)
    assert manifest.get_channel_slice("rank") == slice(2, 4)


def test_contiguous_single_channel_accepted():
    manifest = FeatureManifest(
        feature_ids=["a", "b", "c"],
        channel_offsets={"raw": 0},
        channel_sizes={"raw": 3},
    )
    assert manifest.total_features == 3


def test_hole_at_start_rejected():
    # First offset != 0 -> hole at column 0.
    with pytest.raises(InvalidContractError, match="no holes|tile"):
        FeatureManifest(
            feature_ids=["a", "b", "c"],
            channel_offsets={"raw": 1, "rank": 2},
            channel_sizes={"raw": 1, "rank": 1},
        )


# ============================================================================
# FP-P0-02 — auxiliary channels physically addressable
# ============================================================================


def _aux_channels():
    return {
        "raw": ChannelRef("raw", "feature", ("f1", "f2")),
        "missing": ChannelRef("missing", "missing", ("f1", "f2")),
        "freshness": ChannelRef("freshness", "freshness", ("f1",)),
        "exposure": ChannelRef("exposure", "exposure", ("f1",)),
    }


def test_aux_channels_all_addressable():
    channels = _aux_channels()
    # width = 2 (raw) + 2 (missing) + 1 (freshness) + 1 (exposure) = 6
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(n=2),
        asset_axis=_asset_axis(n=3),
        channels=channels,
        values=np.zeros((2, 3, 6), dtype="float64"),
        layout="TNF",
        has_missing_channel=True,
        has_freshness_channel=True,
        has_exposure_channel=True,
    )
    assert bundle.manifest is not None
    assert bundle.manifest.total_width == 6
    # Every advertised channel is addressable.
    for name in channels:
        vals = bundle.get_channel_values(name)
        assert vals.shape == (2, 3, len(channels[name].feature_ids)), name
    # Feature extraction still works.
    assert bundle.get_feature_values("f1").shape == (2, 3)


def test_has_missing_channel_must_agree():
    channels = _aux_channels()
    with pytest.raises(InvalidContractError, match="has_missing_channel"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=3),
            channels=channels,
            values=np.zeros((2, 3, 6), dtype="float64"),
            layout="TNF",
            has_missing_channel=False,  # wrong: missing channel present
            has_freshness_channel=True,
            has_exposure_channel=True,
        )


def test_has_freshness_channel_must_agree():
    channels = _aux_channels()
    with pytest.raises(InvalidContractError, match="has_freshness_channel"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=3),
            channels=channels,
            values=np.zeros((2, 3, 6), dtype="float64"),
            layout="TNF",
            has_missing_channel=True,
            has_freshness_channel=False,  # wrong
            has_exposure_channel=True,
        )


def test_has_exposure_channel_must_agree():
    channels = _aux_channels()
    with pytest.raises(InvalidContractError, match="has_exposure_channel"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=3),
            channels=channels,
            values=np.zeros((2, 3, 6), dtype="float64"),
            layout="TNF",
            has_missing_channel=True,
            has_freshness_channel=True,
            has_exposure_channel=False,  # wrong
        )


def test_aux_channel_auto_manifest_width_matches_values():
    # Auto-generated manifest must account for auxiliary columns so the
    # feature-axis width matches the values array.
    channels = _aux_channels()
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(n=2),
        asset_axis=_asset_axis(n=3),
        channels=channels,
        values=np.zeros((2, 3, 6), dtype="float64"),
        layout="TNF",
        has_missing_channel=True,
        has_freshness_channel=True,
        has_exposure_channel=True,
    )
    assert bundle.manifest.total_width == 6
    assert bundle.manifest.total_features == 2  # only feature-type


# ============================================================================
# FP-P0-03 — AxisRef vs tensor shape validation
# ============================================================================


def test_tnf_wrong_ndim_rejected():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    with pytest.raises(InvalidContractError, match="ndim"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=3),
            channels=channels,
            values=np.zeros((2, 3, 4, 1), dtype="float64"),  # ndim 4
            layout="TNF",
        )


def test_nf_wrong_ndim_rejected():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    with pytest.raises(InvalidContractError, match="ndim"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=3),
            channels=channels,
            values=np.zeros((3, 4, 1), dtype="float64"),  # ndim 3
            layout="NF",
        )


def test_tnf_wrong_time_axis_length_rejected():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    with pytest.raises(InvalidContractError, match="time axis"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=5),  # T=5 but values T=2
            asset_axis=_asset_axis(n=3),
            channels=channels,
            values=np.zeros((2, 3, 4), dtype="float64"),
            layout="TNF",
        )


def test_tnf_wrong_asset_axis_length_rejected():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    with pytest.raises(InvalidContractError, match="asset axis"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=7),  # N=7 but values N=3
            channels=channels,
            values=np.zeros((2, 3, 4), dtype="float64"),
            layout="TNF",
        )


def test_nf_wrong_asset_axis_length_rejected():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    with pytest.raises(InvalidContractError, match="asset axis"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=_asset_axis(n=9),  # N=9 but values N=3
            channels=channels,
            values=np.zeros((3, 4), dtype="float64"),
            layout="NF",
        )


def test_duplicate_axis_labels_rejected():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    dup_asset = AxisRef(
        axis_name="asset",
        axis_values=np.array([0, 0, 1], dtype="int64"),  # duplicate 0
        axis_dtype="int64",
    )
    with pytest.raises(InvalidContractError, match="duplicate labels"):
        FeatureBundle(
            bundle_id="b1",
            time_axis=_time_axis(n=2),
            asset_axis=dup_asset,
            channels=channels,
            values=np.zeros((2, 3, 4), dtype="float64"),
            layout="TNF",
        )


def test_duplicate_axis_labels_allowed_when_explicit():
    channels = _channels(["f1", "f2"], ["f3", "f4"])
    dup_asset = AxisRef(
        axis_name="asset",
        axis_values=np.array([0, 0, 1], dtype="int64"),
        axis_dtype="int64",
    )
    bundle = FeatureBundle(
        bundle_id="b1",
        time_axis=_time_axis(n=2),
        asset_axis=dup_asset,
        channels=channels,
        values=np.zeros((2, 3, 4), dtype="float64"),
        layout="TNF",
        allow_duplicate_axis_labels=True,
    )
    assert bundle.is_immutable()


# ============================================================================
# FP-P0-04 — FittedState content-derived identity
# ============================================================================


def _prod_state(**overrides):
    base = dict(
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        state_kind=StateKind.FITTED,
        feature_ids=["f1", "f2"],
        feature_order=["f1", "f2"],
        learned_params={"mu": 1.0},
        implementation_hash="impl-abc",
        data_snapshot_ref="snap-2024-01-01",
        split_ref="train-0.7",
        universe_ref="univ-1",
        calendar_ref="cal-1",
        fit_coordinate_hash="coord-1",
        policy_hash="pol-1",
        production=True,
    )
    base.update(overrides)
    return FittedState(**base)


def test_production_state_id_content_derived():
    a = _prod_state()
    b = _prod_state()
    assert a.state_id == b.state_id
    assert len(a.state_id) == 64  # sha256 hex


def test_production_state_id_changes_when_params_mutate():
    a = _prod_state()
    b = _prod_state(learned_params={"mu": 2.0})
    assert a.state_id != b.state_id


def test_production_state_id_changes_when_snapshot_changes():
    a = _prod_state()
    b = _prod_state(data_snapshot_ref="snap-2024-02-01")
    assert a.state_id != b.state_id


def test_production_caller_state_id_rejected():
    with pytest.raises(InvalidContractError, match="content-derived"):
        _prod_state(state_id="caller-supplied")


def test_production_requires_provenance_fields():
    with pytest.raises(InvalidContractError, match="provenance"):
        _prod_state(policy_hash=None)


def test_non_production_state_id_required():
    with pytest.raises(Exception):
        FittedState(
            state_id="",
            transform_name="cs_rank",
            transform_version="1.0.0",
            fit_start_time=datetime(2024, 1, 1),
            fit_end_time=datetime(2024, 1, 2),
        )


# ============================================================================
# FP-P0-05 — stateful contract fail-closed
# ============================================================================


def test_fitted_state_empty_feature_contract_fails_closed():
    with pytest.raises(InvalidContractError, match="feature contract"):
        FittedState(
            state_id="s1",
            transform_name="cs_rank",
            transform_version="1.0.0",
            fit_start_time=datetime(2024, 1, 1),
            fit_end_time=datetime(2024, 1, 2),
            state_kind=StateKind.FITTED,
            feature_ids=[],
        )


def test_stateless_empty_feature_contract_ok():
    state = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        state_kind=StateKind.STATELESS,
        feature_ids=[],
    )
    assert state.is_compatible_with(["anything"]) is True


def test_fitted_is_compatible_with_empty_contract_raises():
    # A FITTED state with no feature contract must not fail-open.
    state = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        state_kind=StateKind.FITTED,
        feature_ids=["f1", "f2"],
        feature_order=["f1", "f2"],
    )
    # Force an empty contract to exercise the fail-closed path.
    object.__setattr__(state, "feature_ids", ())
    object.__setattr__(state, "feature_order", ())
    with pytest.raises(InvalidContractError, match="feature contract"):
        state.is_compatible_with(["f1"])


def test_fitted_compatible_with_matching_order():
    state = FittedState(
        state_id="s1",
        transform_name="cs_rank",
        transform_version="1.0.0",
        fit_start_time=datetime(2024, 1, 1),
        fit_end_time=datetime(2024, 1, 2),
        state_kind=StateKind.FITTED,
        feature_ids=["f1", "f2"],
        feature_order=["f1", "f2"],
    )
    assert state.is_compatible_with(["f1", "f2"]) is True
    assert state.is_compatible_with(["f2", "f1"]) is False
