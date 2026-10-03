from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_pearson_oracle import (
    PEARSON_CHAIN, reference_row_pearson, reference_source_pearson_chain,
)


def _fixture(T=24, N=32, F=5):
    rng = np.random.default_rng(123)
    times = pd.date_range("2024-01-01", periods=T, freq="B")
    t = AxisRef("time", "datetime64[ns]", T, times.to_numpy())
    a = AxisRef("asset", "int64", N, np.arange(N, dtype=np.int64))
    x, y = rng.normal(size=(T, N, F)), rng.normal(size=(T, N))
    x[:, :, 0] = 1.0
    x[0, :15, 1] = np.nan
    if T > 1:
        x[1, :3, 2] = np.inf
    valid = np.ones(x.shape, dtype=bool)
    if T > 2:
        valid[2, :20, 3] = False
    lv = np.ones(y.shape, dtype=bool)
    if T > 3:
        lv[3, :15] = False
    batch = FactorBatch(tuple(f"f{i}" for i in range(F)), t, a, x, validity=valid)
    labels = LabelBundle("ret", y, 1, decision_time=tuple(t.values),
        label_start_time=tuple(times), label_end_time=tuple(times + pd.Timedelta(days=1)),
        asset_axis=a, validity=lv)

    class Source:
        factor_ids = batch.factor_ids
        time_axis, asset_axis = t, a
        dtype, snapshot_id = "float64", "oracle-test"
        max_tile_size, admitted_max_tile_size = 16, 2
        def __init__(self):
            self.reads = []
        def read_tile(self, start, end):
            assert end - start <= 2
            self.reads.append((start, end))
            b = FactorBatch(batch.factor_ids[start:end], t, a,
                batch.values[:, :, start:end], validity=batch.validity[:, :, start:end])
            return FactorTile(start, end, b, self.snapshot_id)
        def close(self):
            pytest.fail("oracle source close belongs to caller")
    return Source, labels


@pytest.mark.parametrize("T", [1, 19, 24])
def test_independent_reference_matches_public_cpu_masks_counts_values(T):
    Source, labels = _fixture(T=T)
    source = Source()
    oracle = reference_source_pearson_chain(source, labels, max_tile_size=16)
    actual = evaluate_factor_source_batch(Source(), labels,
        metrics=PEARSON_CHAIN, backend="cpu", max_tile_size=16)
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert oracle.metadata["source_request_fingerprint"] == actual.metadata["source_request_fingerprint"]
    assert oracle.metadata["coverage_scope"] == "every_time_factor_row"
    for name in PEARSON_CHAIN:
        field = "series_metrics" if name.endswith("_series") else "scalar_metrics"
        np.testing.assert_allclose(getattr(oracle, field)[name],
            getattr(actual, field)[name], rtol=0, atol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(oracle.observation_counts[name], actual.observation_counts[name])
    assert np.isnan(oracle.scalar_metrics["pearson_ic"][0])
    if T < 20:
        assert np.isnan(oracle.scalar_metrics["pearson_ic_std"]).all()


def test_huge_near_constant_reference_preserves_adjacent_float_distinctions():
    x = np.array([1e308 + i * 2e292 for i in range(32)])
    assert reference_row_pearson(x, x) == 1.0
    assert reference_row_pearson(x, -x) == -1.0
    assert np.isnan(reference_row_pearson(np.ones(32), x))


def test_budget_admission_happens_before_source_reads():
    Source, labels = _fixture()
    source = Source()
    with pytest.raises(MemoryError):
        reference_source_pearson_chain(source, labels, max_result_bytes=1)
    assert source.reads == []


def test_one_dimensional_label_rejected_like_public_source_api():
    Source, labels = _fixture()
    labels = replace(labels, values=labels.values[:, 0], validity=labels.validity[:, 0],
                     asset_axis=None)
    with pytest.raises(ValueError, match="label shape"):
        reference_source_pearson_chain(Source(), labels)


def test_float32_storage_uses_double_reference_arithmetic():
    rng = np.random.default_rng(15)
    x, y = rng.normal(size=(2, 32)).astype(np.float32)
    assert reference_row_pearson(x, y) == reference_row_pearson(
        x.astype(np.float64), y.astype(np.float64))


@pytest.mark.parametrize("mutation", ["nonfinite", "shape", "integer", "extended"])
def test_row_contract_fails_closed(mutation):
    x, y = np.arange(32, dtype=np.float64), np.arange(32, dtype=np.float64)
    if mutation == "nonfinite":
        x[0] = np.nan
    elif mutation == "shape":
        y = y[:-1]
    elif mutation == "integer":
        x = x.astype(np.int64)
    else:
        x = x.astype(np.longdouble)
    with pytest.raises((TypeError, ValueError)):
        reference_row_pearson(x, y)
