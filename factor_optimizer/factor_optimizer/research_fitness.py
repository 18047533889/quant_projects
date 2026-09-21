"""QE-owned research portfolio metrics and joint paired selection policy.

Daily, non-overlapping labels only. This is not production admission or an
execution simulator: no tradability, impact, borrow, or capacity certification.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


def portfolio_series(values, returns, *, cost_rate=.001, empty_leg_policy='signal_cash'):
    """Top/bottom quintiles, gross-one equal stock weights, signal-only membership.

    Full-notional turnover includes initial entry from cash. Missing selected
    labels invalidate that day's PnL. Under signal_cash, a sufficiently observed,
    nonconstant signal with an empty quantile leg follows QE's zero target weights
    (including liquidation costs). Missing/constant signals remain unavailable.
    unavailable reproduces the older strict empty-leg rejection.
    """
    from quant_evaluator.metrics.quantile import assign_quantiles_batch
    from quant_evaluator.metrics.portfolio_stats import equal_gross_weights, equal_gross_long_short_returns
    from quant_evaluator.metrics.turnover import compute_turnover_series
    values, returns = np.asarray(values, float), np.asarray(returns, float)
    if values.ndim != 2 or returns.shape != values.shape:
        raise ValueError('matching (time, asset) panels required')
    if isinstance(cost_rate, bool) or not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError('cost_rate must be finite and nonnegative')
    if empty_leg_policy not in ('signal_cash', 'unavailable'):
        raise ValueError('empty_leg_policy must be signal_cash or unavailable')
    bins = assign_quantiles_batch(values, n_quantiles=5)
    long, short = bins == 4, bins == 0
    active = long.any(axis=1) & short.any(axis=1)
    weights = equal_gross_weights(long, short)
    pnl = equal_gross_long_short_returns(long, short, returns, cost_rate=cost_rate,
                                         missing_return_policy='drop')
    # QE half-sum convention -> explicit full traded notional, including entry.
    turnover = 2 * compute_turnover_series(np.vstack((np.zeros((1, values.shape[1])), weights)))[1:]
    if empty_leg_policy == 'unavailable':
        pnl[~active] = np.nan
    else:
        finite = np.isfinite(values)
        low = np.min(np.where(finite, values, np.inf), axis=1)
        high = np.max(np.where(finite, values, -np.inf), axis=1)
        known_signal = (finite.sum(axis=1) >= 5) & (high > low)
        pnl[~known_signal] = np.nan
    return pnl, turnover


def summarize(series, *, periods_per_year=252):
    """Summarize aligned columns [RankIC, net portfolio return, full turnover]."""
    from quant_evaluator.metrics.ic_summary import compute_icir
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio, compute_maximum_drawdown
    a = np.asarray(series, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 20:
        raise ValueError('at least 20 aligned metric observations are required')
    if not np.isfinite(a).all():
        raise ValueError('missing or nonfinite joint metrics cannot be imputed')
    if np.any(a[:, 1] < -1) or np.any(a[:, 2] < 0):
        raise ValueError('invalid return capital scale or turnover')
    sharpe = lambda x: float(compute_sharpe_ratio(x, periods_per_year=periods_per_year,
                                                min_periods=2))
    metrics = {
        'rank_ic': float(a[:, 0].mean()),
        'rank_icir': float(compute_icir(a[:, :1], min_periods=20)[0]),
        'sharpe': sharpe(a[:, 1]),
        'max_drawdown': float(compute_maximum_drawdown(a[:, 1])[0]),
        'turnover': float(a[:, 2].mean()),
        'worst_block_sharpe': min(sharpe(x) for x in np.array_split(a[:, 1], 3)),
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('joint metrics unavailable, including zero-variance ratios')
    return metrics


def joint_utility(m):
    """Prespecified bounded research utility; never fitted to VALIDATION/TEST."""
    return float(.40*np.tanh(m['sharpe']/2) + .20*np.tanh(m['rank_icir']/.5)
                 + .15*np.tanh(m['rank_ic']/.05) - .10*m['max_drawdown']
                 + .10*np.tanh(m['worst_block_sharpe']/2) - .05*m['turnover']/2)


def passes_floors(raw, candidate):
    """Hard raw-relative guards prevent one metric buying material damage."""
    return (candidate['rank_ic'] >= raw['rank_ic']-.01
            and candidate['rank_icir'] >= raw['rank_icir']-.1
            and candidate['sharpe'] >= raw['sharpe']-.25
            and candidate['max_drawdown'] <= raw['max_drawdown']+.03
            and candidate['worst_block_sharpe'] >= raw['worst_block_sharpe']-.5
            and candidate['turnover'] <= raw['turnover']+.25)


class RawSeriesCache:
    """One-entry RAW evidence cache; returned arrays never alias stored state."""

    def __init__(self):
        self._key = None
        self._series = None

    def get(self, key):
        return self._series.copy() if key == self._key and self._series is not None else None

    def put(self, key, series):
        self._key, self._series = key, series.copy()


def paired_series(raw, candidate, batch, labels, indices, *, minimum_assets=20, cost_rate=.001,
                  raw_cache=None, empty_leg_policy='signal_cash', candidate_ic_cache=None):
    """QE metric inputs share signal availability, never ex-post label membership."""
    from dataclasses import replace
    from factor_optimizer.research_batch import _subset_labels, PairICCache, _candidate_ic_key
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.metrics.ic import compute_daily_ic
    if labels.horizon != 1:
        raise ValueError('joint portfolio selection requires single-bar labels')
    idx = np.asarray(indices)
    target = _subset_labels(labels, indices)
    if any(target.label_end_time[i] > target.label_start_time[i+1] for i in range(len(idx)-1)):
        raise ValueError('overlapping labels require cohort portfolio accounting')
    a, b = raw[idx], candidate[idx]
    common = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(common, a, np.nan), np.where(common, b, np.nan)
    pair = FactorBatch(('RAW', 'CANDIDATE'),
        AxisRef('time', batch.time_axis.dtype, len(idx), batch.time_axis.values[idx]),
        batch.asset_axis, np.stack((a, b), axis=-1))
    y = target.values if target.validity is None else np.where(target.validity, target.values, np.nan)
    cached, key = None, None
    if raw_cache is not None:
        if not isinstance(raw_cache, RawSeriesCache):
            raise TypeError('raw_cache must be RawSeriesCache')
        digest = hashlib.sha256()
        digest.update(repr((a.shape, minimum_assets, cost_rate, empty_leg_policy)).encode())
        for panel in (a, y):
            canonical = np.ascontiguousarray(panel)
            digest.update(canonical.dtype.str.encode())
            digest.update(canonical.tobytes())
        key = digest.digest()
        cached = raw_cache.get(key)
    candidate_ic = None
    if candidate_ic_cache is not None:
        if not isinstance(candidate_ic_cache, PairICCache):
            raise TypeError('candidate_ic_cache must be PairICCache')
        candidate_ic = candidate_ic_cache.get(_candidate_ic_key(b, target, minimum_assets))
    columns = ([] if cached is not None else [0]) + ([] if candidate_ic is not None else [1])
    ic = np.full((len(idx), 2), np.nan)
    if candidate_ic is not None:
        ic[:, 1] = candidate_ic
    if columns:
        pair = replace(pair, factor_ids=tuple(pair.factor_ids[k] for k in columns),
                       values=pair.values[:, :, columns])
        computed, _ = compute_daily_ic(pair, target, method='spearman', min_assets=minimum_assets)
        ic[:, columns] = computed
    out = []
    for k, values in enumerate((a, b)):
        if k == 0 and cached is not None:
            out.append(cached)
            continue
        pnl, turnover = portfolio_series(values, y, cost_rate=cost_rate,
                                        empty_leg_policy=empty_leg_policy)
        out.append(np.column_stack((ic[:, k], pnl, turnover)))
    if raw_cache is not None and cached is None:
        raw_cache.put(key, out[0])
    return tuple(out)


def compare_joint(raw_series, candidate_series, config):
    """Recompute all nonlinear metrics within each shared moving-block draw."""
    from factor_optimizer.search.paired_comparison import (
        PairedDraws, ComparisonThresholds, compare_paired_draws,
    )
    raw, candidate = np.asarray(raw_series, float), np.asarray(candidate_series, float)
    if raw.shape != candidate.shape:
        raise ValueError('paired series must have identical shapes')
    # Fail before resampling rather than dropping difficult dates or draws.
    summarize(raw); summarize(candidate)
    n, length = len(raw), config.block_length
    if n < 3*length:
        raise ValueError('insufficient observations for paired block comparison')
    rng = np.random.default_rng(config.seed)
    rd, cd = {}, {}
    for _ in range(config.bootstrap_draws):
        starts = rng.integers(0, n-length+1, size=math.ceil(n/length))
        idx = np.concatenate([np.arange(s, s+length) for s in starts])[:n]
        for dest, data in ((rd, raw), (cd, candidate)):
            for key, value in summarize(data[idx]).items():
                dest.setdefault(key, []).append(value)
    context = hashlib.sha256(raw.tobytes()+candidate.tobytes()).hexdigest()
    evidence = PairedDraws('candidate', 'raw', tuple(map(str, range(config.bootstrap_draws))),
                          cd, rd, context)
    return compare_paired_draws(evidence,
        ComparisonThresholds(config.minimum_improvement, .01, .01,
                             confidence_level=config.confidence_level),
        utility=joint_utility, expected_context_identity=context)
