from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import spearmanr

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def fixture():
    rng = np.random.default_rng(871)
    x = np.repeat(np.round(rng.normal(size=(30, 40, 1)) * 10, 0), 2, axis=2)
    mask = np.ones(x.shape, dtype=bool)
    mask[:, :5, 0] = False
    mask[:, -5:, 1] = False
    x[7, :, 0] = 1  # Constant factor on one date, not a zero IC.
    y = np.round(rng.normal(size=(30, 40)) * 10, 0)
    y[:, 10] = np.nan
    batch = FactorBatch(("a", "b"), AxisRef("time", "int64", 30, np.arange(30)),
                        AxisRef("asset", "int64", 40, np.arange(40)), x, validity=mask)
    labels = LabelBundle("forward", y, 1, decision_time=tuple(range(30)),
                         label_start_time=tuple(range(1, 31)), label_end_time=tuple(range(2, 32)),
                         asset_axis=batch.asset_axis)
    return batch, labels


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_public_pairwise_rank_recomputes_both_ranks_after_each_factor_mask(backend):
    if backend == "gpu":
        pytest.importorskip("cupy")
    batch, labels = fixture()
    expected = np.full((30, 2), np.nan)
    for t in range(30):
        for f in range(2):
            common = batch.validity[t, :, f] & np.isfinite(labels.values[t])
            x, y = batch.values[t, common, f], labels.values[t, common]
            if min(len(np.unique(x)), len(np.unique(y))) >= 10:
                expected[t, f] = spearmanr(x, y).statistic
    actual = evaluate(batch, labels, metrics=("rank_ic_series",), backend=backend)
    np.testing.assert_allclose(actual.artifacts["rank_ic_series"].values, expected,
                               equal_nan=True, atol=1e-12)
    poisoned = batch.values.copy()
    poisoned[~batch.validity] = 1e100
    poison_result = evaluate(replace(batch, values=poisoned), labels,
                             metrics=("rank_ic_series",), backend=backend)
    np.testing.assert_allclose(poison_result.artifacts["rank_ic_series"].values, expected,
                               equal_nan=True, atol=1e-12)
    permutation = np.arange(40)[::-1]
    perm_axis = AxisRef("asset", "int64", 40, permutation)
    perm_batch = replace(batch, asset_axis=perm_axis, values=batch.values[:, permutation, ::-1],
                         validity=batch.validity[:, permutation, ::-1], factor_ids=("b", "a"))
    perm_labels = replace(labels, asset_axis=perm_axis, values=labels.values[:, permutation])
    perm_result = evaluate(perm_batch, perm_labels, metrics=("rank_ic_series",), backend=backend)
    np.testing.assert_allclose(perm_result.artifacts["rank_ic_series"].values, expected[:, ::-1],
                               equal_nan=True, atol=1e-12)
