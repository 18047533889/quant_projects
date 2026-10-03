import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _fixture(t=84, n=200):
    rng = np.random.default_rng(8104)
    time = AxisRef("time", "int", t, np.arange(t))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    values = rng.normal(size=(t, n))
    values[3::19, ::11] = np.nan
    values[9, :] = 0.0
    labels = rng.normal(size=(t, n))
    labels[::7, ::9] = np.nan
    validity = np.ones((t, n), dtype=bool)
    validity[5::13, ::8] = False
    batch = FactorBatch(("f",), time, assets, values[:, :, None],
                        validity=validity[:, :, None])
    bundle = LabelBundle(
        "decay-cache-test", labels, 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
        asset_axis=assets,
    )
    return batch, bundle, values, validity


def _legacy_cube(raw, idx, labels, batch, lags, min_assets):
    """Independent copy of the old per-lag FactorBatch/value aggregation path."""
    label_values = np.asarray(labels.values, dtype=float)
    if label_values.ndim == 3:
        label_values = label_values[:, :, 0]
    label_validity = labels.validity
    if label_validity is not None and np.asarray(label_validity).ndim == 3:
        label_validity = np.asarray(label_validity)[:, :, 0]
    cube = np.full((len(idx), 20, len(lags)), np.nan, dtype=np.float64)
    for li, lag in enumerate(lags):
        for out_t, score_t in enumerate(idx):
            source_t = int(score_t - lag)
            if source_t < 0:
                continue
            signal = np.asarray(raw[source_t], dtype=float)
            finite = np.isfinite(signal)
            sorted_signal = np.sort(signal[finite])
            if len(sorted_signal) < 20:
                continue
            boundaries = []
            for b in range(19):
                pos = (b + 1) / 20 * (len(sorted_signal) - 1)
                lo = int(pos)
                frac = pos - lo
                if lo >= len(sorted_signal) - 1:
                    boundary = sorted_signal[-1]
                elif frac < 1e-9:
                    boundary = sorted_signal[lo]
                elif frac > 1.0 - 1e-9:
                    boundary = sorted_signal[lo + 1]
                else:
                    left, right = float(sorted_signal[lo]), float(sorted_signal[lo + 1])
                    delta = right - left
                    boundary = (left + frac * delta if np.isfinite(delta)
                                else (1-frac)*left + frac*right)
                boundaries.append(boundary)
            assignment = np.full(len(signal), -1, dtype=np.int32)
            assignment[finite] = np.clip(np.searchsorted(
                boundaries, signal[finite], side="right"), 0, 19)
            y = np.asarray(label_values[score_t], dtype=float).copy()
            if label_validity is not None:
                y[~np.asarray(label_validity[score_t], dtype=bool)] = np.nan
            good = (assignment >= 0) & np.isfinite(y)
            counts = np.bincount(assignment[good], minlength=20)
            sums = np.bincount(assignment[good], weights=y[good], minlength=20)
            enough = counts >= min_assets
            cube[out_t, enough, li] = sums[enough] / counts[enough]
    return cube


def _legacy_decay_record(batch, labels, idx, config, min_assets):
    """Pre-cache diagnostic assembly, driven by the scalar `_legacy_cube`."""
    from factor_optimizer.research_fitness import portfolio_series
    from factor_optimizer.research_batch import _subset_labels

    lags = tuple(range(11)) + (15, 20)
    raw = np.asarray(batch.values[:, :, 0], dtype=float)
    if batch.validity is not None:
        raw = np.where(batch.validity[:, :, 0], raw, np.nan)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    record = dict(partition="TRAIN", status="unavailable", lags=list(lags),
                  layers=[], proposed_half_lives=[], full_notional_turnover=None,
                  high_turnover=False, common_days=0, coverage=0.,
                  turnover_policy=config.research_empty_leg_policy,
                  turnover_unavailable_reason="insufficient or unusable signal history",
                  reason="insufficient complete twenty-layer history")
    pnl, turnover = portfolio_series(raw[idx], np.zeros_like(raw[idx]), cost_rate=0.,
                                     empty_leg_policy=config.research_empty_leg_policy)
    if len(idx) and np.isfinite(pnl).all():
        record["full_notional_turnover"] = float(turnover.mean())
        record["high_turnover"] = record["full_notional_turnover"] > .5
        record["turnover_unavailable_reason"] = None
    if raw.shape[1] < 20 * min_assets or not len(idx):
        return record
    cube = _legacy_cube(raw, idx, labels, batch, lags, min_assets)
    good = np.isfinite(cube).all(axis=(1, 2))
    record.update(common_days=int(good.sum()), coverage=float(good.mean()))
    if good.sum() < config.minimum_train_days or good.mean() < config.minimum_coverage:
        return record
    excess = cube - cube.mean(axis=1, keepdims=True)
    profile = excess[good].mean(axis=0)
    fold_positions = np.array_split(np.arange(len(idx)), 3)
    enough_folds = all(good[p].sum() >= 10 for p in fold_positions)
    layers = []
    for layer in range(20):
        curve = profile[layer]
        direction = np.sign(curve[0])
        stable = enough_folds and direction != 0 and all(
            direction * excess[p[good[p]], layer, 0].mean() > 0 for p in fold_positions)
        crossings = [lag for lag, value in zip(lags[1:], curve[1:])
                     if direction * value <= .5 * abs(curve[0])]
        half = crossings[0] if stable and crossings else None
        layers.append(dict(layer=layer+1, mean_excess_returns=curve.tolist(),
                           stable_initial_direction=bool(stable), half_life_bars=half,
                           right_censored=bool(stable and half is None)))
    tails, proposals = (layers[0], layers[-1]), []
    if all(x["stable_initial_direction"] for x in tails):
        scale = float(np.median([x["half_life_bars"] or lags[-1] for x in tails]))
        proposals = sorted({h for h in (scale/2., scale) if 3. <= h <= 60.})
    record.update(status="available", layers=layers, proposed_half_lives=proposals,
                  reason=None,
                  scale_rule="median tail first-half-crossing; censored lower bound")
    return record


def test_unique_source_cache_matches_legacy_for_sparse_indices_and_boundary_lags():
    from factor_optimizer.research_decay import (
        _DECAY_ASSIGNMENT_LAGS, _cached_lagged_quantile_panels,
    )

    batch, labels, raw, validity = _fixture()
    effective = np.where(validity, raw, np.nan)
    # Explicit gaps exercise source-index mapping independently of label rows.
    idx = np.array([0, 2, 3, 8, 9, 17, 18, 25, 31, 38, 39, 48, 59, 60, 72, 83])
    expected = _legacy_cube(effective, idx, labels, batch,
                            _DECAY_ASSIGNMENT_LAGS, 2)
    actual = _cached_lagged_quantile_panels(
        effective, idx, labels, batch, _DECAY_ASSIGNMENT_LAGS,
        min_assets=2, budget_bytes=64 * 1024 * 1024,
    )
    np.testing.assert_array_equal(actual, expected)


def test_source_assignment_calls_scale_with_chunk_source_union_not_lags(monkeypatch):
    import quant_evaluator.metrics.quantile as qe_quantile
    from factor_optimizer.research_decay import (
        _DECAY_ASSIGNMENT_LAGS, _cached_lagged_quantile_panels,
        _source_rows_for_scores,
    )

    batch, labels, raw, validity = _fixture(t=84, n=200)
    effective = np.where(validity, raw, np.nan)
    idx = np.arange(24, 84, dtype=np.int64)
    calls = []
    original = qe_quantile.assign_quantiles_batch

    def counted(values, *args, **kwargs):
        calls.append(np.asarray(values).shape[0])
        return original(values, *args, **kwargs)

    monkeypatch.setattr(qe_quantile, "assign_quantiles_batch", counted)
    _cached_lagged_quantile_panels(
        effective, idx, labels, batch, _DECAY_ASSIGNMENT_LAGS,
        min_assets=2, budget_bytes=512 * 1024,
    )
    assigned_source_rows = sum(calls)
    global_unique = len(_source_rows_for_scores(idx, _DECAY_ASSIGNMENT_LAGS))
    assert assigned_source_rows >= global_unique
    assert assigned_source_rows < len(idx) * len(_DECAY_ASSIGNMENT_LAGS)
    assert max(calls) < len(idx)


def test_one_row_source_working_set_over_budget_fails_before_large_allocation():
    from factor_optimizer.research_decay import (
        _DECAY_ASSIGNMENT_LAGS, _cached_lagged_quantile_panels,
    )
    import pytest

    batch, labels, raw, validity = _fixture(t=30, n=200)
    with pytest.raises(MemoryError, match="result cube alone"):
        _cached_lagged_quantile_panels(
            np.where(validity, raw, np.nan), np.arange(0, 30), labels,
            batch, _DECAY_ASSIGNMENT_LAGS, min_assets=2, budget_bytes=1,
        )


def test_future_factor_and_label_poison_cannot_change_cached_train_panels():
    from dataclasses import replace
    from factor_optimizer.research_decay import (
        _DECAY_ASSIGNMENT_LAGS, _cached_lagged_quantile_panels,
    )
    batch, labels, raw, validity = _fixture(t=84, n=200)
    effective = np.where(validity, raw, np.nan)
    idx = np.arange(24, 72, dtype=np.int64)
    before = _cached_lagged_quantile_panels(
        effective, idx, labels, batch, _DECAY_ASSIGNMENT_LAGS,
        min_assets=2, budget_bytes=64 * 1024 * 1024,
    )
    poisoned_raw = effective.copy()
    poisoned_raw[72:] = 1e30
    poisoned_labels = labels.values.copy()
    poisoned_labels[72:] = -1e30
    after = _cached_lagged_quantile_panels(
        poisoned_raw, idx, replace(labels, values=poisoned_labels),
        batch, _DECAY_ASSIGNMENT_LAGS, min_assets=2,
        budget_bytes=64 * 1024 * 1024,
    )
    np.testing.assert_array_equal(after, before)


def test_diagnostic_rejects_boolean_and_nonpositive_cache_budgets():
    import pytest
    from factor_optimizer.research_batch import BatchOptimizationConfig
    from factor_optimizer.research_decay import diagnose_layer_decay

    batch, labels, _, _ = _fixture(t=84, n=200)
    split = type("TrainSplit", (), {"train_indices": np.arange(24, 84)})()
    for budget in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="assignment_cache_budget_bytes"):
            diagnose_layer_decay(batch, labels, split, BatchOptimizationConfig(), 0,
                                 minimum_assets_per_quantile=2,
                                 assignment_cache_budget_bytes=budget)


def test_decay_output_record_matches_pre_refactor_assembly_ignoring_cache_metadata():
    from factor_optimizer.research_batch import BatchOptimizationConfig
    from factor_optimizer.research_decay import diagnose_layer_decay

    batch, labels, _, _ = _fixture(t=84, n=200)
    idx = np.arange(84, dtype=np.int64)
    cfg = BatchOptimizationConfig(minimum_train_days=20, minimum_coverage=.1)
    expected = _legacy_decay_record(batch, labels, idx, cfg, 2)
    actual = diagnose_layer_decay(
        batch, labels, type("TrainSplit", (), {"train_indices": idx})(), cfg, 0,
        minimum_assets_per_quantile=2,
    )
    actual.pop("assignment_cache_status", None)
    assert expected["status"] == "available"
    assert actual == expected
