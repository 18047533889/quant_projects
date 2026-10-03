"""TRAIN-only twenty-layer stale-signal decay, not holding-period PnL."""
from __future__ import annotations

import numpy as np


_DECAY_ASSIGNMENT_CACHE_BUDGET = 64 * 1024 * 1024
_DECAY_ASSIGNMENT_Q = 20
_DECAY_ASSIGNMENT_LAGS = tuple(range(11)) + (15, 20)


def _source_rows_for_scores(score_rows, lags):
    """Return sorted unique valid source row IDs for this scoring block."""
    sources = np.concatenate([score_rows - lag for lag in lags])
    return np.unique(sources[sources >= 0])


def _assignment_work_bytes(source_count, score_count, assets, *, source_chunk,
                           score_cube_bytes, nq):
    """Conservative incremental peak, including QE sorter and owned copies.

    Per simultaneously sorted source cell budgets 40 bytes for gathered raw
    values, finite mask, fill/sort buffers, mutable QE IDs and boundary
    headroom. Owned immutable ID copies are counted in retained_ids. Per
    scoring cell counts mutable+owned gathered IDs (8 bytes), label subset,
    optional validity and normalized-label copy (up to 17 bytes), plus
    alignment headroom. Quantile boundary/lo/frac scratch is budgeted at 256
    bytes per source row per Q. This intentionally overestimates small-N panels.
    """
    retained_ids = 4 * source_count * assets
    gathered_and_labels = 32 * score_count * assets
    sorter_scratch = 40 * source_chunk * assets + 256 * source_chunk * nq
    return score_cube_bytes + retained_ids + gathered_and_labels + sorter_scratch


def _bounded_score_block(idx, start, max_rows, lags, assets, *,
                         score_cube_bytes, nq, budget_bytes):
    """Choose a chronological score block and source-chunk size within budget."""
    for rows in range(min(max_rows, len(idx) - start), 0, -1):
        score_rows = idx[start:start + rows]
        source_rows = _source_rows_for_scores(score_rows, lags)
        retained_and_gather = (
            score_cube_bytes + 4 * len(source_rows) * assets
            + 32 * rows * assets
        )
        scratch_per_source = 40 * assets + 256 * nq
        available = budget_bytes - retained_and_gather
        source_chunk = min(
            len(source_rows),
            max(0, available // max(1, scratch_per_source)),
        )
        if len(source_rows) == 0:
            source_chunk = 1
        estimate = _assignment_work_bytes(
            len(source_rows), rows, assets, source_chunk=source_chunk,
            score_cube_bytes=score_cube_bytes, nq=nq,
        )
        if source_chunk >= 1 and estimate <= budget_bytes:
            return rows, source_rows, source_chunk
    raise MemoryError(
        "TRAIN decay output cube plus one-score-row source working set exceeds "
        f"the {budget_bytes}-byte assignment cache budget"
    )


def _cached_lagged_quantile_panels(
    raw, idx, labels, batch, lags, *, min_assets, budget_bytes,
):
    """Reuse each distinct source-date assignment inside bounded score chunks.

    The retained membership arrays are at most the unique source dates needed
    by one score chunk, never a lag x full-TRAIN x asset cube. Chunk boundaries
    naturally include the max-lag history halo in their source-date union.
    """
    from quant_evaluator.contracts.factor_batch import AxisRef
    from quant_evaluator.contracts.quantile_assignments import QuantileAssignmentBatch
    from quant_evaluator.metrics.quantile import (
        assign_quantiles_batch,
        compute_quantile_returns_from_assignments,
    )
    from quant_evaluator.contracts.quantile_policy import QuantileTiePolicy
    from factor_optimizer.research_batch import _subset_labels

    T = len(idx)
    N = raw.shape[1]
    Q = _DECAY_ASSIGNMENT_Q
    L = len(lags)
    # Only the compact diagnostic result cube is retained across score blocks.
    # The full score-panel-by-lag-by-asset membership cube is never built.
    score_cube_bytes = T * Q * L * np.dtype(np.float64).itemsize
    if score_cube_bytes >= budget_bytes:
        raise MemoryError(
            "TRAIN decay 20-bin x lag result cube alone exceeds the cache budget"
        )
    cube = np.full((T, Q, L), np.nan, dtype=np.float64)
    score_start = 0
    max_score_rows = max(1, min(64, T))
    while score_start < T:
        rows, source_rows, source_chunk_rows = _bounded_score_block(
            idx, score_start, max_score_rows, lags, N,
            score_cube_bytes=score_cube_bytes, nq=Q,
            budget_bytes=budget_bytes,
        )
        score_rows = idx[score_start:score_start + rows]
        score_axis = AxisRef(
            "time", batch.time_axis.dtype, rows,
            batch.time_axis.values[score_rows],
        )
        source_chunks = []
        source_locations = {}
        for offset in range(0, len(source_rows), source_chunk_rows):
            stop = min(offset + source_chunk_rows, len(source_rows))
            source_chunk = source_rows[offset:stop]
            source_values = raw[source_chunk]
            source_ids = assign_quantiles_batch(
                source_values, n_quantiles=Q, method="max")
            source_axis = AxisRef(
                "time", batch.time_axis.dtype, len(source_chunk),
                batch.time_axis.values[source_chunk],
            )
            source_batch = QuantileAssignmentBatch(
                source_ids[:, :, None], source_axis, batch.asset_axis,
                ("delayed",), Q, QuantileTiePolicy.MAX,
            )
            del source_values, source_ids
            chunk_index = len(source_chunks)
            source_chunks.append(source_batch)
            # The list owns the chunk; avoid a stale alias across score blocks.
            del source_batch
            for local_row, source in enumerate(source_chunk):
                source_locations[int(source)] = (chunk_index, local_row)
        score_labels = _subset_labels(labels, score_rows)

        for lag_index, lag in enumerate(lags):
            source = score_rows - lag
            gathered = np.full((rows, N, 1), -1, dtype=np.int32)
            for output_row, source_row in enumerate(source):
                if source_row < 0:
                    continue
                chunk_index, source_offset = source_locations[int(source_row)]
                gathered[output_row, :, 0] = source_chunks[
                    chunk_index
                ].assignments[source_offset, :, 0]
            assignment_batch = QuantileAssignmentBatch(
                gathered, score_axis, batch.asset_axis,
                ("delayed",), Q, QuantileTiePolicy.MAX,
            )
            del gathered
            q_returns, _ = compute_quantile_returns_from_assignments(
                assignment_batch, score_labels, min_assets=min_assets,
            )
            cube[score_start:score_start + rows, :, lag_index] = q_returns[:, :, 0]
            # The immutable batch owns a copy of ``gathered``. Drop both it
            # and the previous return panel before the next lag allocates.
            del assignment_batch, q_returns
        del source_chunks, source_locations, score_labels
        score_start += rows
    return cube


def diagnose_layer_decay(batch, labels, split, config, factor_index, *,
                         minimum_assets_per_quantile=10,
                         assignment_cache_budget_bytes=_DECAY_ASSIGNMENT_CACHE_BUDGET):
    """Compare x[t-lag] with the same y[t] and common TRAIN dates at each lag.

    Layers are rebuilt from the stale signal at each scoring date. These are
    predictive decay curves, not overlapping holding-return backtests.
    Half-life is the first sampled lag below half the signed lag-zero edge.
    A non-crossing curve is right-censored, never an estimated infinite life.
    """
    if (isinstance(assignment_cache_budget_bytes, (bool, np.bool_))
            or not isinstance(assignment_cache_budget_bytes, (int, np.integer))
            or assignment_cache_budget_bytes < 1):
        raise ValueError("assignment_cache_budget_bytes must be a positive integer")
    from factor_optimizer.research_fitness import portfolio_series

    idx = np.asarray(split.train_indices)
    lags = _DECAY_ASSIGNMENT_LAGS
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
    if raw.shape[1] < 20 * minimum_assets_per_quantile or not len(idx):
        return record
    try:
        cube = _cached_lagged_quantile_panels(
            raw, idx, labels, batch, lags,
            min_assets=minimum_assets_per_quantile,
            budget_bytes=assignment_cache_budget_bytes,
        )
    except MemoryError as exc:
        record["reason"] = str(exc)
        record["assignment_cache_status"] = "budget_exceeded"
        return record
    # All lags and layers must use identical dates, including fold boundaries.
    good = np.isfinite(cube).all(axis=(1, 2))
    record.update(common_days=int(good.sum()), coverage=float(good.mean()),
                  assignment_cache_status="bounded_unique_sources")
    if good.sum() < config.minimum_train_days or good.mean() < config.minimum_coverage:
        return record
    from factor_optimizer.research_numeric import guarded_finite_mean
    with np.errstate(over="ignore", invalid="ignore"):
        excess = cube - guarded_finite_mean(cube, axis=1, keepdims=True)
    if not np.isfinite(excess[good]).all():
        record["reason"] = "centered quantile returns are not representable as finite Float64"
        return record
    profile = guarded_finite_mean(excess[good], axis=0)
    if not np.isfinite(profile).all():
        record["reason"] = "decay profile is not representable as finite Float64"
        return record
    fold_positions = np.array_split(np.arange(len(idx)), 3)
    enough_folds = all(good[p].sum() >= 10 for p in fold_positions)
    layers = []
    for layer in range(20):
        curve = profile[layer]
        direction = np.sign(curve[0])
        stable = enough_folds and direction != 0 and all(
            direction*guarded_finite_mean(excess[p[good[p]], layer, 0], axis=0) > 0
            for p in fold_positions)
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
        # Censoring contributes only an observed lower bound, not infinity.
        scale = float(np.median([x["half_life_bars"] or lags[-1] for x in tails]))
        proposals = sorted({h for h in (scale/2., scale) if 3. <= h <= 60.})
    record.update(status="available", layers=layers, proposed_half_lives=proposals,
                  reason=None, scale_rule="median tail first-half-crossing; censored lower bound")
    return record
