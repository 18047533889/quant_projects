"""TRAIN-only twenty-layer stale-signal decay, not holding-period PnL."""
from __future__ import annotations

import numpy as np


def diagnose_layer_decay(batch, labels, split, config, factor_index, *,
                         minimum_assets_per_quantile=10):
    """Compare x[t-lag] with the same y[t] and common TRAIN dates at each lag.

    Layers are rebuilt from the stale signal at each scoring date. These are
    predictive decay curves, not overlapping holding-return backtests.
    Half-life is the first sampled lag below half the signed lag-zero edge.
    A non-crossing curve is right-censored, never an estimated infinite life.
    """
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.metrics.quantile import compute_quantile_returns
    from factor_optimizer.research_batch import _subset_labels
    from factor_optimizer.research_fitness import portfolio_series

    idx = np.asarray(split.train_indices)
    lags = tuple(range(11)) + (15, 20)
    record = dict(partition="TRAIN", status="unavailable", lags=list(lags),
                  layers=[], proposed_half_lives=[], full_notional_turnover=None,
                  high_turnover=False, common_days=0, coverage=0.,
                  turnover_policy=config.research_empty_leg_policy,
                  turnover_unavailable_reason="insufficient or unusable signal history",
                  reason="insufficient complete twenty-layer history")
    raw = np.asarray(batch.values[:, :, factor_index], float)
    if batch.validity is not None:
        raw = np.where(batch.validity[:, :, factor_index], raw, np.nan)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    # Turnover depends on signals, never on which future returns happen to exist.
    current = raw[idx]
    signal_pnl, turnover = portfolio_series(current, np.zeros_like(current), cost_rate=0.,
                                           empty_leg_policy=config.research_empty_leg_policy)
    if len(current) and np.isfinite(signal_pnl).all():
        record["full_notional_turnover"] = float(turnover.mean())
        record["high_turnover"] = record["full_notional_turnover"] > .5
        record["turnover_unavailable_reason"] = None
    if raw.shape[1] < 20*minimum_assets_per_quantile or not len(idx):
        return record
    target = _subset_labels(labels, split.train_indices)
    axis = AxisRef("time", batch.time_axis.dtype, len(idx), batch.time_axis.values[idx])
    panels = []
    for lag in lags:
        source = idx-lag
        values = raw[np.maximum(source, 0)].copy()
        values[source < 0] = np.nan
        delayed = FactorBatch(("delayed",), axis, batch.asset_axis, values[:, :, None])
        quantiles, _ = compute_quantile_returns(delayed, target, n_quantiles=20,
                                               min_assets=minimum_assets_per_quantile)
        panels.append(quantiles[:, :, 0])
    # All lags and layers must use identical dates, including fold boundaries.
    cube = np.stack(panels, axis=-1)
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
            direction*excess[p[good[p]], layer, 0].mean() > 0 for p in fold_positions)
        crossings = [lag for lag, value in zip(lags[1:], curve[1:])
                     if direction*value <= .5*abs(curve[0])]
        half = crossings[0] if stable and crossings else None
        layers.append(dict(layer=layer+1, mean_excess_returns=curve.tolist(),
                           stable_initial_direction=bool(stable),
                           half_life_bars=half, right_censored=bool(stable and half is None)))
    # Two fixed, bounded scale proposals; no fitting on VALIDATION.
    tails = (layers[0], layers[-1])
    proposals = []
    if all(x["stable_initial_direction"] for x in tails):
        # Right-censoring contributes only the observed lower bound, not infinity.
        scale = float(np.median([x["half_life_bars"] or lags[-1] for x in tails]))
        proposals = sorted({h for h in (scale/2., scale) if 3. <= h <= 60.})
    record.update(status="available", layers=layers, proposed_half_lives=proposals,
                  reason=None, scale_rule="median tail first-half-crossing; censored lower bound")
    return record
