import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters.repair_execution import compile_value_repair


def _plan(quantile, side):
    return compile_value_repair(
        "TAIL_SATURATION",
        {"saturation_quantile": quantile, "saturate": side},
        natural_time_scale=5,
        training_context_ref="train:tail-saturation-fe-v1",
    )


@pytest.mark.parametrize("side", ["top", "bottom", "both"])
def test_tail_saturation_matches_fp_reference_for_each_side(side):
    from factor_preprocess.transforms.repair_shapes import tail_saturation

    values = pd.DataFrame({
        "asset_id": list("ABCDEFG"),
        "date": [pd.Timestamp("2026-01-01")] * 5
                + [pd.Timestamp("2026-01-02")] * 2,
        "value": [-100., 0., 1., 2., 100., -5., 5.],
    }, index=[9, 2, 7, 4, 1, 8, 3])
    actual = _plan(.95, side).execute(values, allow_research=True)
    expected = tail_saturation(values, quantile=.95, side=side)
    pd.testing.assert_series_equal(actual, expected)


def test_tail_saturation_preserves_sparse_rows_order_ties_and_nonfinite_values(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_preprocess.transforms.repair_shapes import tail_saturation

    operator = OperatorRegistry.get("winsorize", backend="pandas_numpy", mode="any")
    assert operator is not None
    original = operator.calculate
    shapes = []

    def tracked(panel, *args, **kwargs):
        shapes.append(panel.shape)
        return original(panel, *args, **kwargs)

    monkeypatch.setattr(operator, "calculate", tracked)
    values = pd.DataFrame({
        "asset_id": ["A", "B", "C", "D", "A", "C", "D", "E"],
        "date": [pd.Timestamp("2026-01-02")] * 4
                + [pd.Timestamp("2026-01-01")] * 4,
        "value": [1., 1., 3., np.inf, -np.inf, 2., np.nan, 4.],
    }, index=[4, 4, 8, 0, 6, 2, 9, 1])
    actual = _plan(.95, "top").execute(values, allow_research=True)
    expected = tail_saturation(values, quantile=.95, side="top")
    pd.testing.assert_series_equal(actual, expected)
    assert actual.index.equals(values.index)
    categorical = pd.DataFrame({
        "asset_id": ["A", "B"],
        "date": pd.Categorical(["d1", "d1"], categories=["d1", "unused"]),
        "value": [1., 100.],
    })
    result = _plan(.95, "top").execute(categorical, allow_research=True)
    np.testing.assert_allclose(result, [1., 95.05])
    assert shapes == [(2, 4), (1, 2)]
    assert np.isnan(actual.iloc[3]) and np.isnan(actual.iloc[4])
    assert np.isnan(actual.iloc[6])


def test_tail_saturation_singleton_and_all_nonfinite_dates():
    values = pd.DataFrame({
        "asset_id": ["A", "A", "B", "C"],
        "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"),
                 pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")],
        "value": [7., np.nan, np.inf, -np.inf],
    })
    actual = _plan(.97, "both").execute(values, allow_research=True)
    np.testing.assert_equal(actual.to_numpy(), [7., np.nan, np.nan, np.nan])


def test_tail_saturation_rejects_q_half_for_both_sides():
    with pytest.raises(ValueError):
        _plan(.5, "both")



def test_tail_saturation_rejects_duplicate_date_asset_identity():
    missing_date = pd.DataFrame({"asset_id": ["A"], "date": [None], "value": [1.]})
    with pytest.raises(ValueError, match="cannot contain nulls"):
        _plan(.95, "top").execute(missing_date, allow_research=True)
    values = pd.DataFrame({
        "asset_id": ["A", "A"],
        "date": [pd.Timestamp("2026-01-01")] * 2,
        "value": [1., 2.],
    })
    with pytest.raises(ValueError, match="duplicate"):
        _plan(.95, "top").execute(values, allow_research=True)

