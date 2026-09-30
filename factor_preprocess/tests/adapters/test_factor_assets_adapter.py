from types import SimpleNamespace

import numpy as np
import pytest

from factor_preprocess.adapters.factor_assets import FactorAssetsAdapter


class _Provider:
    def __init__(self, result):
        self.result = result

    def validate_factor_set(self, factor_set):
        return True

    def get_factor_batch(self, **kwargs):
        self.requested_ids = kwargs["factor_ids"]
        return self.result


def _factor_set():
    return SimpleNamespace(
        set_id="set-1",
        factor_ids=("f1", "f2"),
        universe_ref="universe-1",
        name="test set",
        created_at="2026-09-30T00:00:00Z",
        frequency="daily",
        size=2,
    )


def _batch(**overrides):
    result = {
        "values": np.zeros((3, 4, 2)),
        "dates": np.array(["d1", "d2", "d3"]),
        "assets": np.array(["a1", "a2", "a3", "a4"]),
        "factor_ids": ["f1", "f2"],
        "metadata": {"f1": {"source": "catalog"}},
    }
    result.update(overrides)
    return result


def test_load_factor_set_preserves_batch_and_set_metadata():
    batch = _batch()
    result = FactorAssetsAdapter(_Provider(batch)).load_factor_set(_factor_set())

    assert result["factor_ids"] == ["f1", "f2"]
    assert result["values"] is batch["values"]
    assert result["metadata"] is batch["metadata"]
    assert result["set_metadata"] == {
        "set_id": "set-1",
        "set_name": "test set",
        "created_at": "2026-09-30T00:00:00Z",
        "frequency": "daily",
        "universe": "universe-1",
        "n_factors": 2,
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"factor_ids": ["f2", "f1"]}, "order does not match"),
        ({"factor_ids": ["f1", "f1"]}, "duplicate factor_ids"),
        ({"factor_ids": ["f1"]}, "missing=.*f2"),
        ({"factor_ids": ["f1", "f2", "f3"]}, "unexpected=.*f3"),
        ({"values": np.zeros((3, 4))}, r"shape \(n_times, n_assets, n_factors\)"),
        ({"values": np.zeros((3, 4, 1))}, "factor axis does not match"),
        ({"values": np.zeros((2, 4, 2))}, "axis 0 does not match dates"),
        ({"values": np.zeros((3, 3, 2))}, "axis 1 does not match assets"),
    ],
)
def test_load_factor_set_rejects_misaligned_provider_results(override, message):
    with pytest.raises(ValueError, match=message):
        FactorAssetsAdapter(_Provider(_batch(**override))).load_factor_set(_factor_set())
