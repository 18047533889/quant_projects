"""FP-CERT-IMMUTABLE: FeatureBundle deep immutability tests.

``@dataclass(frozen=True)`` alone does not freeze the mutable containers a
bundle holds (dict/list/numpy).  These tests pin that construction still
accepts plain dicts/lists while storage becomes immutable: mutating
``channels``, ``source_factor_ids``, ``fitted_state_refs``,
``ChannelRef.feature_ids`` raises TypeError, and mutating the axis numpy
array raises ValueError.
"""
from datetime import datetime

import numpy as np
import pytest

from factor_preprocess.contracts import AxisRef, ChannelRef, FeatureBundle


def _axes(T=2, N=3, as_numpy=False):
    if as_numpy:
        time_values = np.arange(T, dtype="int64")
    else:
        time_values = [datetime(2020, 1, i + 1) for i in range(T)]
    return (
        AxisRef("date", time_values, "int64" if as_numpy else "datetime64[ns]"),
        AxisRef("asset_id", [f"A{i}" for i in range(N)], "object"),
    )


def _bundle(T=2, N=3, F=2, numpy_time_axis=False, **overrides):
    time_axis, asset_axis = _axes(T, N, as_numpy=numpy_time_axis)
    kwargs = dict(
        bundle_id="bundle_001",
        time_axis=time_axis,
        asset_axis=asset_axis,
        channels={
            "features": ChannelRef(
                "features", "feature", [f"factor_{i}" for i in range(F)]
            ),
        },
        values=np.zeros((T, N, F)),
        source_factor_ids=["factor_0", "factor_1"],
        fitted_state_refs=["state_001"],
    )
    kwargs.update(overrides)
    return FeatureBundle(**kwargs)


class TestConstructorCompatibility:
    """Lists passed at construction are accepted and converted internally."""

    def test_construction_with_lists_and_dicts_still_works(self):
        bundle = _bundle()
        assert bundle.bundle_id == "bundle_001"
        assert bundle.has_channel("features")
        assert bundle.get_channel("features") is not None

    def test_source_factor_ids_converted_to_tuple(self):
        bundle = _bundle(source_factor_ids=["a", "b"])
        assert bundle.source_factor_ids == ("a", "b")
        assert isinstance(bundle.source_factor_ids, tuple)

    def test_fitted_state_refs_converted_to_tuple(self):
        bundle = _bundle(fitted_state_refs=["s1"])
        assert bundle.fitted_state_refs == ("s1",)
        assert isinstance(bundle.fitted_state_refs, tuple)

    def test_defaults_are_empty_tuples(self):
        bundle = _bundle(source_factor_ids=[], fitted_state_refs=[])
        assert bundle.source_factor_ids == ()
        assert bundle.fitted_state_refs == ()

    def test_channelref_feature_ids_converted_to_tuple(self):
        ref = ChannelRef("features", "feature", ["f1", "f2"])
        assert ref.feature_ids == ("f1", "f2")
        assert isinstance(ref.feature_ids, tuple)

    def test_axisref_list_values_converted_to_tuple(self):
        axis = AxisRef("asset_id", ["A", "B"], "object")
        assert isinstance(axis.axis_values, tuple)
        assert axis.axis_values == ("A", "B")

    def test_bundle_get_channel_still_works_on_frozen_mapping(self):
        bundle = _bundle()
        assert bundle.get_channel("features").channel_name == "features"
        assert bundle.get_channel("missing") is None
        assert bundle.has_channel("features")


class TestBundleMutationRejected:
    def test_mutating_channels_dict_raises(self):
        bundle = _bundle()
        with pytest.raises(TypeError):
            bundle.channels["extra"] = ChannelRef("extra", "feature", ["x"])

    def test_deleting_from_channels_raises(self):
        bundle = _bundle()
        with pytest.raises(TypeError):
            del bundle.channels["features"]

    def test_mutating_source_factor_ids_raises(self):
        bundle = _bundle()
        with pytest.raises((TypeError, AttributeError)):
            bundle.source_factor_ids.append("factor_9")

    def test_mutating_fitted_state_refs_raises(self):
        bundle = _bundle()
        with pytest.raises((TypeError, AttributeError)):
            bundle.fitted_state_refs.append("state_9")

    def test_mutating_channel_feature_ids_raises(self):
        bundle = _bundle()
        channel = bundle.get_channel("features")
        with pytest.raises((TypeError, AttributeError)):
            channel.feature_ids.append("factor_9")

    def test_reassigning_frozen_fields_raises(self):
        bundle = _bundle()
        with pytest.raises((AttributeError, TypeError)):
            bundle.source_factor_ids = ["other"]

    def test_constructor_list_arguments_not_aliased(self):
        # Mutating the caller's list after construction must not affect the
        # bundle (storage must not alias the constructor input).
        source = ["factor_0", "factor_1"]
        bundle = _bundle(source_factor_ids=source)
        source.append("factor_9")
        assert bundle.source_factor_ids == ("factor_0", "factor_1")

    def test_constructor_channels_dict_not_aliased(self):
        channels = {
            "features": ChannelRef(
                "features", "feature", ["factor_0", "factor_1"]
            ),
        }
        bundle = _bundle(channels=channels)
        channels["extra"] = ChannelRef("extra", "feature", ["x"])
        assert not bundle.has_channel("extra")


class TestAxisImmutability:
    def test_mutating_numpy_axis_values_raises(self):
        axis = AxisRef("date", np.arange(5), "int64")
        assert isinstance(axis.axis_values, np.ndarray)
        with pytest.raises(ValueError):
            axis.axis_values[0] = 99

    def test_mutating_bundle_time_axis_numpy_raises(self):
        bundle = _bundle(numpy_time_axis=True)
        with pytest.raises(ValueError):
            bundle.time_axis.axis_values[0] = 99

    def test_mutating_bundle_asset_axis_numpy_raises(self):
        axis = AxisRef("asset_id", np.array(["A", "B", "C"]), "object")
        bundle = _bundle(asset_axis=axis)
        with pytest.raises(ValueError):
            bundle.asset_axis.axis_values[0] = "Z"

    def test_numpy_axis_values_still_usable(self):
        axis = AxisRef("date", np.arange(5), "int64")
        assert len(axis.axis_values) == 5
        assert axis.axis_values.sum() == 10

    def test_tuple_axis_values_are_immutable(self):
        axis = AxisRef("asset_id", ["A", "B", "C"], "object")
        with pytest.raises((TypeError, AttributeError)):
            axis.axis_values.append("D")
