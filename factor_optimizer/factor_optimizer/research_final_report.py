"""Authority-side final research reporting; never called by candidate search.

Uses the existing durable test broker. The trusted authority supplies labels
through its opaque store only after binding the frozen selections and profile.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np


def _digest(raw, result):
    if raw.factor_ids != result.optimized.factor_ids or tuple(result.factors) != raw.factor_ids:
        raise ValueError('frozen factor identities must match')
    h = hashlib.sha256()
    for batch in (raw, result.optimized):
        for axis in (batch.time_axis, batch.asset_axis):
            if axis.values is None:
                raise ValueError('explicit frozen axes required')
            h.update(json.dumps(axis.values.tolist(), default=str).encode())
        h.update(np.ascontiguousarray(batch.values, dtype=np.float64).tobytes())
        if batch.validity is not None:
            h.update(np.ascontiguousarray(batch.validity, dtype=bool).tobytes())
    if (not np.array_equal(raw.time_axis.values, result.optimized.time_axis.values)
            or not np.array_equal(raw.asset_axis.values, result.optimized.asset_axis.values)):
        raise ValueError('frozen raw and selected axes differ')
    for name, item in result.factors.items():
        if item.materialization_error is not None or item.status == 'materialization_failed':
            raise ValueError('resolve frozen plan materialization failure before final reporting')
        if item.plan_identity != item.plan.identity:
            raise ValueError('selected plan identity changed')
        h.update(json.dumps((name, item.plan_identity, item.selected_family, item.status)).encode())
    h.update(json.dumps(asdict(result.split), sort_keys=True).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class FrozenSelection:
    raw: object
    result: object
    dataset_identity: str
    selection_hash: str
    profile_hash: str
    split_identity: str


def _profile(result):
    if result.config is None:
        raise ValueError('final reporting requires the recorded search configuration')
    return hashlib.sha256(json.dumps({'search_config': asdict(result.config),
        'report_version': 'research-final.v1', 'periods_per_year': 252,
        'min_periods': 20}, sort_keys=True).encode()).hexdigest()


def freeze_selection(raw, result, *, dataset_identity):
    """Freeze identities without reading any label; detect later buffer changes."""
    if not isinstance(dataset_identity, str) or not dataset_identity.strip():
        raise ValueError('authoritative dataset_identity is required')
    if result.execution_mode != 'research_only' or result.test_evaluated:
        raise ValueError('only an untested research selection can be frozen')
    return FrozenSelection(raw, result, dataset_identity, _digest(raw, result),
                           _profile(result), result.split.identity)


def _partition(frozen, labels, indices, role):
    from factor_optimizer.research_batch import _subset_labels
    from factor_optimizer.research_fitness import portfolio_series
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic_value
    from quant_evaluator.metrics.ic_summary import compute_icir
    from quant_evaluator.metrics.quantile import compute_quantile_returns
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio, compute_maximum_drawdown

    idx = np.asarray(indices)
    target = _subset_labels(labels, indices)
    factors = {}
    single = labels.horizon == 1 and all(
        target.label_end_time[i] <= target.label_start_time[i+1] for i in range(len(idx)-1))
    number = lambda x: float(x) if np.isfinite(x) else None
    for k, name in enumerate(frozen.raw.factor_ids):
        scores = {}
        for key, batch in (('raw', frozen.raw), ('selected', frozen.result.optimized)):
            values = np.asarray(batch.values[idx, :, k], dtype=float)
            if batch.validity is not None:
                values = np.where(batch.validity[idx, :, k], values, np.nan)
            panel = FactorBatch((name,), AxisRef('time', batch.time_axis.dtype,
                len(idx), batch.time_axis.values[idx]), batch.asset_axis, values[:, :, None])
            ic, _ = compute_daily_ic(panel, target, method='spearman',
                                     min_assets=frozen.result.config.minimum_assets)
            quantiles, counts = compute_quantile_returns(panel, target, n_quantiles=20, min_assets=10)
            complete = np.isfinite(quantiles[:, :, 0]).all(axis=1)
            m = {'rank_ic': number(compute_mean_ic_value(ic, min_periods=20)[0]),
                 'rank_icir': number(compute_icir(ic, min_periods=20)[0]),
                 'ic_valid_days': int(np.isfinite(ic).sum()),
                 'finite_factor_fraction': float(np.isfinite(values).mean()),
                 'complete_20bin_days': int(complete.sum()),
                 'quantile_mean_returns': (quantiles[complete, :, 0].mean(axis=0).tolist()
                                          if complete.sum() >= 20 else None),
                 'sharpe': None, 'max_drawdown': None, 'turnover': None,
                 'worst_block_sharpe': None,
                 'portfolio_valid_days': 0, 'portfolio_unavailable_reason': None}
            if single:
                y = target.values if target.validity is None else np.where(target.validity, target.values, np.nan)
                pnl, turnover = portfolio_series(values, y, cost_rate=frozen.result.config.research_cost_rate)
                m['portfolio_valid_days'] = int(np.isfinite(pnl).sum())
                m['turnover'] = number(np.mean(turnover))
                if np.any(pnl < -1):
                    m['portfolio_unavailable_reason'] = 'returns below -100% require a capital contract'
                else:
                    m['sharpe'] = number(compute_sharpe_ratio(pnl, min_periods=20))
                    m['max_drawdown'] = number(compute_maximum_drawdown(pnl)[0])
                    if np.isfinite(pnl).all():
                        blocks = [float(compute_sharpe_ratio(block, min_periods=2))
                                  for block in np.array_split(pnl, 3)]
                        if np.isfinite(blocks).all():
                            m['worst_block_sharpe'] = min(blocks)
                    if not np.isfinite(pnl).all():
                        m['portfolio_unavailable_reason'] = 'missing valuations; full-period drawdown unknown'
            else:
                m['portfolio_unavailable_reason'] = 'multi-bar or overlapping labels require cohort accounting'
            scores[key] = m
        factors[name] = scores
    return {'role': role, 'days': len(idx), 'start': str(labels.decision_time[indices[0]]),
            'end': str(labels.decision_time[indices[-1]]), 'factors': factors,
            'portfolio_boundary': 'each reported segment starts from cash; entry cost included'}


def evaluate_frozen(frozen, broker):
    """Read the authority store once; persist TEST and separate description.

    Completed calls return the persisted report across process restarts. Any
    error after physical exposure burns the read; this function never retries
    or marks a statistical/data error as an infrastructure failure.
    """
    from factor_optimizer.data_capabilities import TestAuthorityBroker
    from factor_optimizer.research_batch import automatic_time_split
    from quant_evaluator.contracts.label_bundle import LabelBundle
    if not isinstance(broker, TestAuthorityBroker) or not broker.has_durable_authority:
        raise ValueError('durable independent test authority required')
    if (_digest(frozen.raw, frozen.result) != frozen.selection_hash
            or _profile(frozen.result) != frozen.profile_hash):
        raise ValueError('frozen selection or reporting profile changed')
    binding = broker.durable_binding()
    expected = {'dataset_identity': frozen.dataset_identity,
                'candidate_set_hash': frozen.selection_hash, 'split_id': frozen.split_identity,
                'profile_hash': frozen.profile_hash, 'purpose': 'research_final_report'}
    if any(binding[k] != v for k, v in expected.items()):
        raise ValueError('test authority does not match frozen selection binding')
    if binding['state'] == 'COMPLETED':
        return json.loads(binding['result_json'])
    mask = np.zeros(len(frozen.raw.time_axis.values), dtype=bool)
    mask[list(frozen.result.split.test_indices)] = True
    capability = broker.issue_test_capability(mask)
    provider = broker.create_test_provider(capability.attempt_token)
    labels = provider.resolve(capability)
    if not isinstance(labels, LabelBundle):
        raise TypeError('authority must provide a LabelBundle for final reporting')
    if (not np.array_equal(frozen.raw.time_axis.values, np.asarray(labels.decision_time))
            or labels.asset_axis is None or not np.array_equal(frozen.raw.asset_axis.values, labels.asset_axis.values)
            or labels.values.shape != frozen.raw.values.shape[:2]):
        raise ValueError('authority label axes do not match frozen factor axes')
    if automatic_time_split(labels, frozen.result.config).identity != frozen.split_identity:
        raise ValueError('authority label windows do not match frozen split')
    report = {'selection_hash': frozen.selection_hash, 'profile_hash': frozen.profile_hash,
              'dataset_identity': frozen.dataset_identity, 'test_evaluated': True,
              'selection_unchanged': True, 'research_only': True,
              'cost_rate': frozen.result.config.research_cost_rate,
              'test': _partition(frozen, labels, frozen.result.split.test_indices, 'held_out_final'),
              'full_sample': _partition(frozen, labels, tuple(range(len(labels.decision_time))), 'descriptive_only')}
    if _digest(frozen.raw, frozen.result) != frozen.selection_hash:
        raise ValueError('frozen selection changed during final reporting')
    broker.complete_test_attempt(capability.attempt_token, 'research-final:'+frozen.selection_hash, report)
    return report
