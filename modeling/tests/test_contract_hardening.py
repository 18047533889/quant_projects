from datetime import datetime

import numpy as np
import pytest

from modeling_adapters import TransformMode
from modeling_adapters.adapter import FactorPreprocessAdapter
from modeling_adapters.contracts import FitWindow, OutOfFoldSpec, PreprocessContract, SplitSpec
from modeling_adapters.errors import AdapterError, ContractViolation


def _folds():
    return [
        SplitSpec(
            split_id="f1",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 30),
            val_start=datetime(2020, 7, 1),
            val_end=datetime(2020, 12, 31),
        ),
        SplitSpec(
            split_id="f2",
            train_start=datetime(2020, 7, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 1),
            val_end=datetime(2021, 6, 30),
        ),
    ]


def test_oof_strategy_is_closed_set():
    with pytest.raises(Exception, match="strategy must be one of"):
        OutOfFoldSpec(oof_id="bad", folds=_folds(), strategy="random")


def test_oof_folds_are_not_mutable_through_input_list():
    folds = _folds()
    spec = OutOfFoldSpec(oof_id="ok", folds=folds)
    folds.clear()
    assert spec.num_folds() == 2
    with pytest.raises(TypeError):
        spec.folds[0] = spec.folds[1]


def test_mixed_transform_modes_are_rejected():
    with pytest.raises(ContractViolation, match="Mixed stateless and fitted"):
        PreprocessContract(
            contract_id="mixed",
            transforms=[
                {"name": "rank", "mode": "stateless"},
                {"name": "scale", "mode": "fitted"},
            ],
            mode=TransformMode.FITTED,
            fit_window=FitWindow(datetime(2020, 1, 1), datetime(2020, 12, 31)),
        )


def test_nested_contract_containers_are_deep_frozen():
    parameters = {"nested": {"winsor": [0.01, 0.99]}}
    contract = PreprocessContract(
        contract_id="frozen",
        transforms=[{"name": "rank", "mode": "stateless", "parameters": parameters}],
        mode=TransformMode.STATELESS,
    )
    parameters["nested"]["winsor"].append(1.0)
    assert len(contract.transforms[0]["parameters"]["nested"]["winsor"]) == 2
    with pytest.raises(TypeError):
        contract.transforms[0]["parameters"]["nested"]["winsor"] += (2.0,)


def test_fitted_adapter_requires_application_start():
    adapter = FactorPreprocessAdapter()
    contract = PreprocessContract(
        contract_id="fitted",
        transforms=[{"name": "scale", "kind": "cross_sectional", "mode": "fitted"}],
        mode=TransformMode.FITTED,
        fit_window=FitWindow(datetime(2020, 1, 1), datetime(2020, 12, 31)),
    )
    with pytest.raises(AdapterError, match="application_period_start"):
        adapter.translate_contract(contract)


def test_fit_snapshot_ref_is_propagated():
    adapter = FactorPreprocessAdapter()
    metadata = adapter.translate_fit_window(
        FitWindow(
            datetime(2020, 1, 1),
            datetime(2020, 12, 31),
            data_snapshot_ref="lake/gen-7",
        )
    )
    assert metadata["fit_snapshot_ref"] == "lake/gen-7"


def test_fitted_state_order_is_exact():
    adapter = FactorPreprocessAdapter()
    state = type("State", (), {"feature_order": ["a", "b"]})()
    adapter.validate_fitted_state_features(state, ["a", "b"])
    with pytest.raises(AdapterError, match="feature order mismatch"):
        adapter.validate_fitted_state_features(state, ["b", "a"])


def test_feature_bundle_values_and_channel_mapping_are_adapted():
    pytest.importorskip("factor_preprocess")
    from factor_preprocess.contracts.feature_bundle import AxisRef, ChannelRef, FeatureBundle

    bundle = FeatureBundle(
        bundle_id="b",
        time_axis=AxisRef("date", [np.datetime64("2020-01-01")], "datetime64[ns]"),
        asset_axis=AxisRef("asset", ["A1"], "object"),
        channels={"features": ChannelRef("features", "feature", ["a", "b"])},
        values=np.zeros((1, 1, 2)),
    )
    result = FactorPreprocessAdapter().wrap_output(
        bundle,
        PreprocessContract(contract_id="p"),
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
    )
    assert result.features.shape == (1, 1, 2)
    assert result.feature_names == ["a", "b"]


def test_model_ready_data_rejects_feature_name_mismatch():
    """Pre-fix: features vs feature_names length was never cross-checked."""
    from modeling_adapters.contracts import ModelReadyData
    from modeling_adapters.errors import ContractViolation

    with pytest.raises(ContractViolation, match="feature channels"):
        ModelReadyData(
            features=np.zeros((3, 2)),
            feature_names=["a", "b", "c"],
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2020, 1, 2),
            preprocess_contract_id="p",
        )


def test_model_ready_data_rejects_empty_feature_names():
    from modeling_adapters.contracts import ModelReadyData
    from modeling_adapters.errors import ContractViolation

    with pytest.raises(ContractViolation, match="feature_names"):
        ModelReadyData(
            features=np.zeros((3, 2)),
            feature_names=[],
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2020, 1, 2),
            preprocess_contract_id="p",
        )


def test_model_ready_data_accepts_3d_bundle_last_axis_channels():
    from modeling_adapters.contracts import ModelReadyData

    data = ModelReadyData(
        features=np.zeros((1, 1, 2)),
        feature_names=["a", "b"],
        data_start=datetime(2020, 1, 1),
        data_end=datetime(2020, 1, 2),
        preprocess_contract_id="p",
    )
    assert data.features.shape == (1, 1, 2)
