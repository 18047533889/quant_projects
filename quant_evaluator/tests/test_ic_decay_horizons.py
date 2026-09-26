from dataclasses import replace
import numpy as np
import pytest

from quant_evaluator import compute_ic_decay, evaluate_horizons
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _inputs():
    t, n = 10, 6
    times = np.arange(t)
    assets = np.asarray([f"S{i}" for i in range(n)])
    asset_axis = AxisRef("asset", "str", n, assets)
    base = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 4.0])
    factor = np.stack([base + 0.1 * i * np.arange(n) for i in range(t)])
    batch = FactorBatch(
        ("f",), AxisRef("time", "int64", t, times), asset_axis,
        factor[:, :, None], value_hash="ic-decay-oracle-factor",
    )
    labels = {}
    for horizon, sign in ((1, 1.0), (3, -1.0)):
        values = sign * factor ** 3
        # The last H3 rows look strongly predictive but are not mature.
        if horizon == 3:
            values[-3:] = factor[-3:]
        valid = np.ones((t, n), dtype=bool)
        if horizon == 3:
            valid[2, 0] = False
        labels[horizon] = LabelBundle(
            f"h{horizon}", values, horizon, decision_time=tuple(times),
            observation_time=tuple(times), label_start_time=tuple(times),
            label_end_time=tuple(times + horizon), asset_axis=asset_axis,
            validity=valid, source_ref=f"oracle:h{horizon}",
        )
    return batch, labels, factor


@pytest.mark.parametrize("sample_policy", ["common", "per_horizon"])
def test_ic_decay_pearson_formula_and_maturity_oracle(sample_policy):
    batch, labels, factor = _inputs()
    bundle = evaluate_horizons(
        batch, labels, as_of=9, sample_policy=sample_policy,
        min_assets=3, min_periods=2, ic_method="pearson",
    )
    decay = compute_ic_decay(bundle)
    expected = []
    for horizon in bundle.horizons:
        daily = []
        for day in range(batch.num_times):
            cell_mask = bundle.evaluation_masks[horizon][day]
            if cell_mask.sum() < 3:
                continue
            daily.append(np.corrcoef(
                factor[day, cell_mask], labels[horizon].values[day, cell_mask],
            )[0, 1])
        expected.append(np.mean(daily))
    np.testing.assert_allclose(decay.values[:, 0], expected, rtol=0, atol=1e-12)
    assert decay.horizons == (1, 3)
    assert decay.factor_ids == ("f",)
    assert decay.maturity_counts == {1: 9, 3: 7}
    assert decay.label_content_refs == {h: labels[h].content_hash for h in (1, 3)}
    assert decay.provenance["ic_method"] == "pearson"
    assert decay.provenance["metric_id"] == "ic_decay"
    assert np.isnan(bundle.daily_ic_artifacts[3].values[7:]).all()
    with pytest.raises(ValueError):
        decay.values[0, 0] = 0.0


def test_ic_decay_rejects_spearman_and_untyped_input():
    batch, labels, _ = _inputs()
    rank_bundle = evaluate_horizons(
        batch, labels, as_of=9, min_assets=3, min_periods=2,
    )
    with pytest.raises(ValueError, match="Pearson"):
        compute_ic_decay(rank_bundle)
    with pytest.raises(TypeError, match="HorizonEvaluationBundle"):
        compute_ic_decay(np.ones((2, 1)))


def test_ic_decay_no_finite_day_is_nan_and_not_zero_filled():
    batch, labels, _ = _inputs()
    labels = dict(labels)
    labels[3] = replace(
        labels[3], validity=np.zeros(labels[3].values.shape, dtype=bool),
    )
    bundle = evaluate_horizons(
        batch, labels, as_of=9, sample_policy="per_horizon",
        min_assets=3, min_periods=2, ic_method="pearson",
    )
    decay = compute_ic_decay(bundle)
    assert np.isfinite(decay.values[0, 0])
    assert np.isnan(decay.values[1, 0])
    assert decay.sample_counts[3] == 0
