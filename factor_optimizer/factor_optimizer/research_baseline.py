"""Research baseline recipes and TRAIN-only substantial-degradation guard.

No data access, invented exposures, production admission, or TEST scoring.
Unknown lineage is explicitly unresolved, never interpreted as untreated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

from factor_preprocess.contracts.treatment_lineage import (
    TransformLineage, build_signature_from_lineage,
)


@dataclass(frozen=True)
class BaselinePlan:
    operations: tuple[str, ...]
    omissions: tuple[str, ...]
    exposure_columns: tuple[str, ...]
    training_context_ref: str
    minimum_assets: int = 20
    lower_quantile: float = .01
    upper_quantile: float = .99

    @property
    def identity(self):
        payload = (self.operations, self.omissions, self.exposure_columns,
                   self.training_context_ref, self.minimum_assets,
                   self.lower_quantile, self.upper_quantile, 'baseline.v1')
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def execute(self, values, *, exposures=None, allow_research=False):
        from factor_optimizer.adapters.repair_execution import _validate_frame
        from factor_preprocess.registry.transforms import get_default_registry
        from factor_preprocess.transforms.repair_shapes import cross_sectional_rank
        from factor_preprocess.neutralization.ols import ols_neutralize

        if allow_research is not True:
            raise ValueError('baseline is research-only; allow_research=True required')
        work = _validate_frame(values).copy()
        work['value'] = work['value'].astype(float).where(np.isfinite(work['value']))
        aligned_exposures = None
        if 'neutralize' in self.operations:
            required = {'date', 'asset_id', 'available_time', *self.exposure_columns}
            if exposures is None or not required.issubset(exposures.columns):
                raise ValueError('declared exposures and available_time are required')
            if not exposures.columns.is_unique or exposures.duplicated(['date', 'asset_id']).any():
                raise ValueError('exposure columns and date/asset keys must be unique')
            # Validate only rows requested by this execution, not future rows.
            selected = exposures.merge(work[['date', 'asset_id']],
                                       on=['date', 'asset_id'], how='inner', validate='one_to_one')
            if (selected['available_time'].isna().any()
                    or not (selected['available_time'] <= selected['date']).all()):
                raise ValueError('exposure must be available at the factor decision time')
            aligned_exposures = selected[['date', 'asset_id', *self.exposure_columns]]
        for op in self.operations:
            if op == 'winsor':
                result = get_default_registry().get_execution('cs_winsor')(
                    work, lower=self.lower_quantile, upper=self.upper_quantile)
            elif op == 'neutralize':
                result = ols_neutralize(work, aligned_exposures,
                                        min_observations=self.minimum_assets)
            elif op == 'cs_rank':
                result = cross_sectional_rank(work, method='average')
            else:
                raise ValueError(f'unknown baseline operation: {op}')
            # Positional assignment preserves duplicate caller index labels.
            work['value'] = np.asarray(result, dtype=float)
        return work['value']


@dataclass(frozen=True)
class BaselineRepairPlan:
    """Replay the TRAIN-frozen baseline followed by one frozen value repair.

    Neutralization requires callers to supply the declared aligned exposure
    panel again; no mutable exposure table is hidden inside the frozen plan.
    """
    baseline: BaselinePlan
    repair: object

    @property
    def family(self):
        return 'BASELINE' if self.repair.family == 'NO_OP_RAW' else self.repair.family

    @property
    def identity(self):
        return hashlib.sha256((self.baseline.identity + ':' + self.repair.identity).encode()).hexdigest()

    def execute(self, values, *, exposures=None, allow_research=False):
        transformed = self.baseline.execute(values, exposures=exposures, allow_research=allow_research)
        frame = values.copy()
        frame['value'] = np.asarray(transformed, dtype=float)
        return self.repair.execute(frame, allow_research=allow_research)


def compile_baseline(lineage, *, training_context_ref, exposure_columns=(), minimum_assets=20):
    """Frozen winsor -> OLS -> rank recipe; do not repeat historical CS rank.

    Exposure absence is explicit. Rank after OLS does not certify exact linear
    neutrality. Caller-supplied availability is checked, not PIT provenance.
    """
    if not isinstance(training_context_ref, str) or not training_context_ref.strip():
        raise ValueError('training_context_ref is required')
    if type(minimum_assets) is not int or minimum_assets < 3:
        raise ValueError('minimum_assets must be an integer >= 3')
    if isinstance(exposure_columns, (str, bytes)):
        raise ValueError('exposure_columns must be a sequence')
    columns = tuple(exposure_columns)
    if (len(set(columns)) != len(columns) or any(not isinstance(c, str) or not c
            or c in {'date', 'asset_id', 'available_time'} for c in columns)):
        raise ValueError('exposure_columns must be unique numeric exposure names')
    if lineage is not None and not isinstance(lineage, TransformLineage):
        raise TypeError('lineage must be TransformLineage or None')
    signature = None if lineage is None else build_signature_from_lineage(lineage)
    if signature is None or signature.is_unknown_or_incomplete:
        return BaselinePlan((), ('lineage_unknown',), columns, training_context_ref, minimum_assets)
    operations, omissions = [], []
    if signature.winsor:
        omissions.append('winsor_already_present')
    else:
        operations.append('winsor')
    if columns:
        operations.append('neutralize')
    else:
        omissions.append('neutralization_missing_exposures')
    if signature.cs_rank:
        omissions.append('cs_rank_already_present')
    else:
        operations.append('cs_rank')
    return BaselinePlan(tuple(operations), tuple(omissions), columns,
                        training_context_ref, minimum_assets)


def assess_baseline_training(raw, candidate, batch, labels, split, config, *, maximum_loss=.01):
    """Reject material TRAIN RankIC loss with a paired moving-block upper bound.

    This is a baseline loss guard, not evidence of superiority. Unknown or
    inadequate coverage fails closed. VALIDATION/TEST labels are not scored.
    """
    from factor_optimizer.research_batch import _pair_ic, _lower_bound
    if isinstance(maximum_loss, bool) or not math.isfinite(maximum_loss) or maximum_loss < 0:
        raise ValueError('maximum_loss must be finite and nonnegative')
    delta, good, coverage = _pair_ic(raw, candidate, batch, labels, split.train_indices, config)
    record = {'accepted': False, 'coverage': coverage, 'valid_days': int(good.sum()),
              'upper_bound': None, 'maximum_loss': maximum_loss,
              'reason': 'insufficient_training_evidence'}
    if coverage < config.minimum_coverage or good.sum() < config.minimum_train_days:
        return record
    bound = _lower_bound(-delta, replace(config, minimum_validation_days=config.minimum_train_days),
                         labels.horizon)
    if bound is None:
        return record
    upper = -bound
    accepted = upper >= -maximum_loss
    record.update(accepted=accepted, upper_bound=upper,
                  reason='no_substantial_training_degradation' if accepted
                  else 'substantial_training_degradation')
    return record
