import numpy as np
import pytest


def _fixture():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    rng = np.random.default_rng(410)
    raw = rng.normal(size=(60, 30))
    candidate = raw.copy()
    candidate[7:10, :4] = np.nan
    ta = AxisRef("time", "int", 60, np.arange(60))
    aa = AxisRef("asset", "str", 30, np.array([f"a{i}" for i in range(30)]))
    batch = FactorBatch(("x",), ta, aa, raw[:, :, None])
    labels = LabelBundle("y", rng.normal(size=raw.shape).astype(np.longdouble), 1,
        decision_time=tuple(range(60)), label_start_time=tuple(range(60)),
        label_end_time=tuple(range(1, 61)), asset_axis=aa)
    return raw, candidate, batch, labels


def test_joint_scoring_reuses_exact_prepared_target_without_reslicing(monkeypatch):
    """Defect caught: every TRAIN candidate rebuilds/freezes/re-hashes the same label slice."""
    from factor_optimizer import research_batch, research_fitness

    raw, candidate, batch, labels = _fixture()
    indices = tuple(range(60))
    prepared = research_batch._prepare_pair_ic_split(batch, labels, indices)
    expected = research_fitness.paired_series(raw, candidate, batch, labels, indices)

    def forbidden_subset(*args, **kwargs):
        raise AssertionError("prepared TRAIN labels should be reused, not subsetted again")

    monkeypatch.setattr(research_batch, "_subset_labels", forbidden_subset)
    actual = research_fitness.paired_series(
        raw, candidate, batch, labels, indices, prepared_split=prepared)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("mismatch", ["labels", "batch", "indices", "wrong_type"])
def test_joint_scoring_rejects_mismatched_prepared_target(mismatch):
    """Defect caught: a prepared target from another request could silently score wrong rows."""
    from dataclasses import replace
    from factor_optimizer import research_batch, research_fitness
    from quant_evaluator.contracts.factor_batch import FactorBatch

    raw, candidate, batch, labels = _fixture()
    indices = tuple(range(60))
    prepared = research_batch._prepare_pair_ic_split(batch, labels, indices)
    if mismatch == "labels":
        labels = replace(labels, target_id="other")
    elif mismatch == "batch":
        batch = FactorBatch(batch.factor_ids, batch.time_axis, batch.asset_axis, batch.values.copy())
    elif mismatch == "wrong_type":
        prepared = object()
    else:
        indices = tuple(range(1, 60))
    with pytest.raises(ValueError, match="prepared|match"):
        research_fitness.paired_series(
            raw, candidate, batch, labels, indices, prepared_split=prepared)


def test_generic_joint_scoring_still_subsets_labels_and_preserves_longdouble(monkeypatch):
    """Defect caught: adding the fast path must not change ordinary exact-content callers."""
    from factor_optimizer import research_batch, research_fitness

    raw, candidate, batch, labels = _fixture()
    original = research_batch._subset_labels
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(research_batch, "_subset_labels", counted)
    result = research_fitness.paired_series(raw, candidate, batch, labels, tuple(range(60)))
    assert calls == [1]
    assert result[0].shape == (60, 3)
    assert np.isfinite(result[0][:, 0]).all()




def test_prepared_joint_scoring_raw_cache_remains_exact_across_candidate_masks():
    """Defect caught: reuse of prepared labels must not erase candidate-mask cache identity."""
    from factor_optimizer import research_batch, research_fitness

    raw, candidate, batch, labels = _fixture()
    prepared = research_batch._prepare_pair_ic_split(batch, labels, tuple(range(60)))
    alternate = candidate.copy()
    alternate[20:24, 5:11] = np.nan
    cache = research_fitness.RawSeriesCache()
    for values in (candidate, alternate, candidate):
        expected = research_fitness.paired_series(raw, values, batch, labels, tuple(range(60)))
        actual = research_fitness.paired_series(
            raw, values, batch, labels, tuple(range(60)), raw_cache=cache,
            prepared_split=prepared)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)

