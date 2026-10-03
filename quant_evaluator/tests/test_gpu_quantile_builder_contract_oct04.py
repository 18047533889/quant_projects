import numpy as np
import pytest

from quant_evaluator.contracts.errors import InvalidContractError, UnsupportedMetricError
from quant_evaluator.runtime.gpu_quantile_builder_contract import (
    normalize_gpu_quantile_builder_parameters,
)


def _profile_metric():
    from quant_evaluator.runtime.gpu_quantile_shape_adapter import GPU_PROFILE_QUANTILE_METRICS

    return sorted(GPU_PROFILE_QUANTILE_METRICS)[0]


@pytest.mark.parametrize("parameters", [None, {}])
def test_empty_parameters_are_noop_without_importing_optional_adapter(parameters, monkeypatch):
    import sys

    sys.modules.pop("quant_evaluator.runtime.gpu_quantile_shape_adapter", None)
    assert normalize_gpu_quantile_builder_parameters(parameters, ()) == {}
    assert "quant_evaluator.runtime.gpu_quantile_shape_adapter" not in sys.modules


def test_nonempty_parameters_require_mapping():
    with pytest.raises(InvalidContractError, match="mapping"):
        normalize_gpu_quantile_builder_parameters([("n_quantiles", 10)], (_profile_metric(),))


def test_window_size_is_always_unsupported():
    with pytest.raises(UnsupportedMetricError, match="window_size"):
        normalize_gpu_quantile_builder_parameters({"window_size": 0}, (_profile_metric(),))


def test_unknown_key_is_invalid():
    with pytest.raises(InvalidContractError, match="Unknown"):
        normalize_gpu_quantile_builder_parameters({"bins": 10}, (_profile_metric(),))


@pytest.mark.parametrize("name", ["n_quantiles", "min_assets"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1, 1.5, "10", None])
def test_quantile_and_min_asset_values_must_be_integer_at_least_two(name, value):
    with pytest.raises(InvalidContractError):
        normalize_gpu_quantile_builder_parameters(
            {name: value}, (_profile_metric(),)
        )


def test_numpy_integer_values_normalize_to_builtin_ints():
    result = normalize_gpu_quantile_builder_parameters(
        {"n_quantiles": np.int64(10), "min_assets": np.int32(20)},
        (_profile_metric(), "unrelated_metric"),
    )
    assert result == {"n_quantiles": 10, "min_assets": 20}
    assert all(type(value) is int for value in result.values())


def test_nonempty_parameters_require_a_gpu_profile_quantile_metric():
    with pytest.raises(UnsupportedMetricError, match="GPU-profile quantile"):
        normalize_gpu_quantile_builder_parameters({"n_quantiles": 10}, ("unrelated_metric",))


def test_each_numeric_parameter_is_validated_independently():
    metric = _profile_metric()
    with pytest.raises(InvalidContractError, match="n_quantiles"):
        normalize_gpu_quantile_builder_parameters({"n_quantiles": 1}, (metric,))
    with pytest.raises(InvalidContractError, match="min_assets"):
        normalize_gpu_quantile_builder_parameters({"min_assets": 0}, (metric,))
