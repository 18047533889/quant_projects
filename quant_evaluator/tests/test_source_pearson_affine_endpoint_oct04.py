import numpy as np
import pandas as pd

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_pearson_oracle import reference_row_pearson, reference_source_pearson_chain


def test_affine_endpoint_stays_exact_across_masked_source_chain():
    rng = np.random.default_rng(19)
    T, N = 65, 100
    labels_values = np.tile(rng.permutation(N), (T, 1)).astype(np.float64)
    mask = rng.random((T, N)) > 0.2
    labels_values[~mask] = np.nan
    factors = np.stack((3 * labels_values + 7, 7 - 3 * labels_values), axis=2)
    factors[~mask] = np.nan
    times = pd.date_range("2024-01-01", periods=T, freq="B")
    time_axis = AxisRef("time", "datetime64[ns]", T, times.to_numpy())
    asset_axis = AxisRef("asset", "int64", N, np.arange(N, dtype=np.int64))
    batch = FactorBatch(("positive", "negative"), time_axis, asset_axis, factors)
    labels = LabelBundle("ret", labels_values, 1, decision_time=tuple(time_axis.values),
        label_start_time=tuple(times), label_end_time=tuple(times + pd.Timedelta(days=1)),
        asset_axis=asset_axis)

    class Source:
        factor_ids = batch.factor_ids
        time_axis, asset_axis = batch.time_axis, batch.asset_axis
        dtype, snapshot_id = "float64", "affine-endpoint"
        max_tile_size = admitted_max_tile_size = 2
        def read_tile(self, start, end):
            return FactorTile(start, end, FactorBatch(batch.factor_ids[start:end], time_axis,
                asset_axis, batch.values[:, :, start:end]), self.snapshot_id)
        def close(self):
            pass

    oracle = reference_source_pearson_chain(Source(), labels)
    np.testing.assert_array_equal(oracle.series_metrics["pearson_ic_series"][:, 0], 1.0)
    np.testing.assert_array_equal(oracle.series_metrics["pearson_ic_series"][:, 1], -1.0)
    np.testing.assert_array_equal(oracle.scalar_metrics["pearson_ic"], [1.0, -1.0])
    np.testing.assert_array_equal(oracle.scalar_metrics["pearson_ic_std"], [0.0, 0.0])
    assert np.isnan(oracle.scalar_metrics["pearson_ic_ir"]).all()


def test_one_ulp_non_affine_row_is_not_certified_as_endpoint(monkeypatch):
    import quant_evaluator.scripts.source_pearson_oracle as oracle

    x = np.arange(32, dtype=np.float64)
    y = x.copy()
    y[16] = np.nextafter(y[16], np.inf)
    candidate = 1.0 - 2.0 ** -52
    monkeypatch.setattr(oracle, "pearsonr", lambda *_args, **_kwargs:
                        type("Result", (), {"statistic": candidate})())
    assert reference_row_pearson(x, y) == candidate
