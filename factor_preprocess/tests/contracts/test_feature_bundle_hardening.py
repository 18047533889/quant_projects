"""FP-P0-11: fail-closed FeatureBundle contract hardening tests.

The prior FeatureBundle only checked bundle_id/channels/layout. These tests
pin the new invariants: ChannelRef type/feature-id validation, channels-key
consistency, truthful has_missing/has_freshness flags, and wide-layout shape
agreement with axes and the feature channel.
"""
import numpy as np
import pytest
from datetime import datetime

from factor_preprocess.contracts import FeatureBundle, AxisRef, ChannelRef


def _axes(T=2, N=3):
    return (
        AxisRef("date", [datetime(2020, 1, i + 1) for i in range(T)], "datetime64[ns]"),
        AxisRef("asset_id", [f"A{i}" for i in range(N)], "object"),
    )


def _bundle(T=2, N=3, F=2, **overrides):
    time_axis, asset_axis = _axes(T, N)
    kwargs = dict(
        bundle_id="bundle_001",
        time_axis=time_axis,
        asset_axis=asset_axis,
        channels={
            "features": ChannelRef("features", "feature", [f"factor_{i}" for i in range(F)]),
        },
        values=np.random.default_rng(0).normal(size=(T, N, F)),
    )
    kwargs.update(overrides)
    return FeatureBundle(**kwargs)


class TestChannelRefValidation:
    def test_empty_channel_name_fails(self):
        with pytest.raises(ValueError, match="channel_name cannot be empty"):
            ChannelRef("", "feature", ["factor_1"])

    def test_invalid_channel_type_fails(self):
        with pytest.raises(ValueError, match="channel_type must be one of"):
            ChannelRef("features", "raw", ["factor_1"])

    @pytest.mark.parametrize("valid_type", ["feature", "missing", "freshness", "exposure"])
    def test_valid_channel_types_accepted(self, valid_type):
        ref = ChannelRef("ch", valid_type, ["factor_1"])
        assert ref.channel_type == valid_type

    def test_empty_feature_ids_fails(self):
        with pytest.raises(ValueError, match="at least one"):
            ChannelRef("features", "feature", [])

    def test_duplicate_feature_ids_fail(self):
        with pytest.raises(ValueError, match="duplicate feature_ids"):
            ChannelRef("features", "feature", ["factor_1", "factor_1"])


class TestBundleChannelConsistency:
    def test_channels_key_mismatch_fails(self):
        time_axis, asset_axis = _axes()
        with pytest.raises(ValueError, match="does not match ChannelRef"):
            FeatureBundle(
                bundle_id="bundle_001",
                time_axis=time_axis,
                asset_axis=asset_axis,
                channels={
                    "wrong_key": ChannelRef("features", "feature", ["factor_1"]),
                },
                values=None,
            )

    def test_keyed_channels_accepted(self):
        bundle = _bundle()
        assert bundle.has_channel("features")


class TestTruthfulFlags:
    def test_missing_flag_without_channel_fails(self):
        with pytest.raises(ValueError, match="has_missing_channel=True but no channel"):
            _bundle(has_missing_channel=True)

    def test_freshness_flag_without_channel_fails(self):
        with pytest.raises(ValueError, match="has_freshness_channel=True but no channel"):
            _bundle(has_freshness_channel=True)

    def test_missing_channel_with_flag_accepted(self):
        bundle = _bundle(
            channels={
                "features": ChannelRef("features", "feature", ["factor_1", "factor_2"]),
                "missing": ChannelRef("missing", "missing", ["factor_1", "factor_2"]),
            },
            has_missing_channel=True,
        )
        assert bundle.has_channel("missing")


class TestWideLayoutShapeContract:
    def test_wrong_feature_axis_fails(self):
        # Channel declares 3 features, values have 2 → mismatch.
        with pytest.raises(ValueError, match="feature axis size"):
            _bundle(
                channels={
                    "features": ChannelRef(
                        "features", "feature", ["f1", "f2", "f3"]
                    ),
                },
            )

    def test_wrong_time_axis_length_fails(self):
        with pytest.raises(ValueError, match="time_axis length"):
            _bundle(
                time_axis=AxisRef("date", [datetime(2020, 1, 1)], "datetime64[ns]"),
            )

    def test_wrong_asset_axis_length_fails(self):
        with pytest.raises(ValueError, match="asset_axis length"):
            _bundle(
                asset_axis=AxisRef("asset_id", ["A", "B", "C", "D"], "object"),
            )

    def test_2d_values_in_wide_layout_fail(self):
        with pytest.raises(ValueError, match="wide layout requires"):
            _bundle(values=np.zeros((2, 3)))

    def test_none_values_bypass_shape_check(self):
        # values=None (external reference) is still legal for any layout.
        bundle = _bundle(values=None)
        assert bundle.values is None

    def test_consistent_wide_bundle_accepted(self):
        bundle = _bundle()
        assert bundle.layout == "wide"
