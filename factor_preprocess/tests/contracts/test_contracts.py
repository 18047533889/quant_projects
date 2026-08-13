"""
Test suite for contract classes.
"""
import pytest
from datetime import datetime
from factor_preprocess.contracts import (
    PreprocessingPolicy,
    TransformSpec,
    TransformKind,
    TransformMode,
    FittedState,
    FeatureBundle,
    AxisRef,
    ChannelRef,
)
from factor_preprocess.errors import TimingContractError, InvalidContractError


class TestTransformSpec:
    """Test TransformSpec contract."""

    def test_basic_construction(self):
        """Test basic spec construction."""
        spec = TransformSpec(
            name="zscore",
            kind=TransformKind.CROSS_SECTIONAL,
            mode=TransformMode.STATELESS,
            version="1.0.0",
            parameters={"ddof": 1},
        )
        assert spec.name == "zscore"
        assert spec.kind == TransformKind.CROSS_SECTIONAL
        assert spec.mode == TransformMode.STATELESS
        assert spec.version == "1.0.0"
        assert spec.parameters["ddof"] == 1

    def test_immutable(self):
        """Test that spec is immutable."""
        spec = TransformSpec(
            name="zscore",
            kind=TransformKind.CROSS_SECTIONAL,
            mode=TransformMode.STATELESS,
            version="1.0.0",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "rank"

    def test_empty_name_fails(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            TransformSpec(
                name="",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0.0",
            )

    def test_empty_version_fails(self):
        """Test that empty version is rejected."""
        with pytest.raises(ValueError, match="version cannot be empty"):
            TransformSpec(
                name="zscore",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="",
            )


class TestPreprocessingPolicy:
    """Test PreprocessingPolicy contract."""

    def test_basic_construction(self):
        """Test basic policy construction."""
        transforms = [
            TransformSpec(
                name="zscore",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0.0",
            ),
        ]
        policy = PreprocessingPolicy(
            policy_id="test_policy",
            transforms=transforms,
        )
        assert policy.policy_id == "test_policy"
        assert len(policy.transforms) == 1
        assert policy.transforms[0].name == "zscore"

    def test_empty_policy_id_fails(self):
        """Test that empty policy_id is rejected."""
        with pytest.raises(ValueError, match="policy_id cannot be empty"):
            PreprocessingPolicy(
                policy_id="",
                transforms=[
                    TransformSpec(
                        name="zscore",
                        kind=TransformKind.CROSS_SECTIONAL,
                        mode=TransformMode.STATELESS,
                        version="1.0.0",
                    ),
                ],
            )

    def test_empty_transforms_fails(self):
        """Test that empty transforms list is rejected."""
        with pytest.raises(ValueError, match="transforms cannot be empty"):
            PreprocessingPolicy(
                policy_id="test_policy",
                transforms=[],
            )

    def test_duplicate_transform_names_fails(self):
        """Test that duplicate transform names are rejected."""
        transforms = [
            TransformSpec(
                name="zscore",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0.0",
            ),
            TransformSpec(
                name="zscore",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0.0",
            ),
        ]
        with pytest.raises(ValueError, match="Duplicate transform names"):
            PreprocessingPolicy(
                policy_id="test_policy",
                transforms=transforms,
            )

    def test_has_fitted_transforms(self):
        """Test detection of fitted transforms."""
        # All stateless
        policy1 = PreprocessingPolicy(
            policy_id="test_policy",
            transforms=[
                TransformSpec(
                    name="zscore",
                    kind=TransformKind.CROSS_SECTIONAL,
                    mode=TransformMode.STATELESS,
                    version="1.0.0",
                ),
            ],
        )
        assert not policy1.has_fitted_transforms()

        # Mixed
        policy2 = PreprocessingPolicy(
            policy_id="test_policy",
            transforms=[
                TransformSpec(
                    name="zscore",
                    kind=TransformKind.CROSS_SECTIONAL,
                    mode=TransformMode.STATELESS,
                    version="1.0.0",
                ),
                TransformSpec(
                    name="rolling_mean",
                    kind=TransformKind.ROLLING,
                    mode=TransformMode.FITTED,
                    version="1.0.0",
                ),
            ],
        )
        assert policy2.has_fitted_transforms()


class TestFittedState:
    """Test FittedState contract."""

    def test_basic_construction(self):
        """Test basic state construction."""
        state = FittedState(
            state_id="state_001",
            transform_name="rolling_mean",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            feature_ids=["factor_1", "factor_2"],
            feature_order=["factor_1", "factor_2"],
        )
        assert state.state_id == "state_001"
        assert state.transform_name == "rolling_mean"
        assert len(state.feature_ids) == 2

    def test_fit_window_validation(self):
        """Test that fit window is validated."""
        with pytest.raises(TimingContractError, match="fit_start_time must be before fit_end_time"):
            FittedState(
                state_id="state_001",
                transform_name="rolling_mean",
                transform_version="1.0.0",
                fit_start_time=datetime(2020, 12, 31),
                fit_end_time=datetime(2020, 1, 1),
            )

    def test_feature_contract_validation(self):
        """Test that feature contract is validated."""
        # Missing feature_order when feature_ids provided
        with pytest.raises(InvalidContractError, match="feature_order required"):
            FittedState(
                state_id="state_001",
                transform_name="rolling_mean",
                transform_version="1.0.0",
                fit_start_time=datetime(2020, 1, 1),
                fit_end_time=datetime(2020, 12, 31),
                feature_ids=["factor_1", "factor_2"],
                feature_order=[],
            )

    def test_duplicate_feature_ids_fails(self):
        """Test that duplicate feature IDs are rejected."""
        with pytest.raises(InvalidContractError, match="Duplicate feature IDs"):
            FittedState(
                state_id="state_001",
                transform_name="rolling_mean",
                transform_version="1.0.0",
                fit_start_time=datetime(2020, 1, 1),
                fit_end_time=datetime(2020, 12, 31),
                feature_ids=["factor_1", "factor_1"],
                feature_order=["factor_1", "factor_1"],
            )

    def test_is_compatible_with(self):
        """Test compatibility check."""
        state = FittedState(
            state_id="state_001",
            transform_name="rolling_mean",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            feature_ids=["factor_1", "factor_2"],
            feature_order=["factor_1", "factor_2"],
        )

        # Compatible
        assert state.is_compatible_with(["factor_1", "factor_2"])
        assert state.is_compatible_with(["factor_2", "factor_1"])

        # Incompatible
        assert not state.is_compatible_with(["factor_1"])
        assert not state.is_compatible_with(["factor_1", "factor_2", "factor_3"])


class TestFeatureBundle:
    """Test FeatureBundle contract."""

    def test_basic_construction(self):
        """Test basic bundle construction."""
        import numpy as np

        bundle = FeatureBundle(
            bundle_id="bundle_001",
            time_axis=AxisRef(
                axis_name="date",
                axis_values=[datetime(2020, 1, 1), datetime(2020, 1, 2)],
                axis_dtype="datetime64[ns]",
            ),
            asset_axis=AxisRef(
                axis_name="asset_id",
                axis_values=["A", "B", "C"],
                axis_dtype="object",
            ),
            channels={
                "features": ChannelRef(
                    channel_name="features",
                    channel_type="feature",
                    feature_ids=["factor_1", "factor_2"],
                ),
            },
            values=np.random.randn(2, 3, 2),  # (time, assets, features)
        )
        assert bundle.bundle_id == "bundle_001"
        assert bundle.has_channel("features")
        assert not bundle.has_channel("missing")

    def test_empty_bundle_id_fails(self):
        """Test that empty bundle_id is rejected."""
        with pytest.raises(ValueError, match="bundle_id cannot be empty"):
            FeatureBundle(
                bundle_id="",
                time_axis=AxisRef("date", [], "datetime64[ns]"),
                asset_axis=AxisRef("asset_id", [], "object"),
                channels={},
                values=None,
            )

    def test_empty_channels_fails(self):
        """Test that empty channels dict is rejected."""
        with pytest.raises(ValueError, match="at least one channel"):
            FeatureBundle(
                bundle_id="bundle_001",
                time_axis=AxisRef("date", [], "datetime64[ns]"),
                asset_axis=AxisRef("asset_id", [], "object"),
                channels={},
                values=None,
            )

    def test_invalid_layout_fails(self):
        """Test that invalid layout is rejected."""
        with pytest.raises(ValueError, match="layout must be one of"):
            FeatureBundle(
                bundle_id="bundle_001",
                time_axis=AxisRef("date", [], "datetime64[ns]"),
                asset_axis=AxisRef("asset_id", [], "object"),
                channels={
                    "features": ChannelRef("features", "feature", ["factor_1"]),
                },
                values=None,
                layout="invalid",
            )

    def test_get_channel(self):
        """Test channel retrieval."""
        bundle = FeatureBundle(
            bundle_id="bundle_001",
            time_axis=AxisRef("date", [], "datetime64[ns]"),
            asset_axis=AxisRef("asset_id", [], "object"),
            channels={
                "features": ChannelRef("features", "feature", ["factor_1"]),
                "missing": ChannelRef("missing", "missing", ["factor_1"]),
            },
            values=None,
        )

        assert bundle.get_channel("features") is not None
        assert bundle.get_channel("features").channel_name == "features"
        assert bundle.get_channel("nonexistent") is None
