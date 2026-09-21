"""Bounded research batch optimization with automatic chronological splitting.

No production admission, trading-profit claim, or sealed-test evaluation. QE
owns IC and portfolio metrics; FP/FE adapters own transformations. Only TRAIN chooses one plan
per factor. VALIDATION can accept that plan or fall back to RAW, never retry
the next candidate. TEST labels are never sent to an evaluator here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BatchOptimizationConfig:
    """Conservative defaults; callers need not choose calendar cutoffs."""
    train_fraction: float = .60
    validation_fraction: float = .20
    warmup_bars: int = 30
    embargo_bars: int = 1
    minimum_train_days: int = 60
    minimum_validation_days: int = 30
    minimum_test_days: int = 30
    minimum_assets: int = 20
    minimum_coverage: float = .90
    minimum_improvement: float = .01
    confidence_level: float = .95
    bootstrap_draws: int = 499
    block_length: int = 5
    seed: int = 20260921
    natural_time_scale: float = 10.
    families: tuple[str, ...] = ()
    maximum_candidates: int = 128
    compose_smoothing_sign: bool = True
    selection_objective: str = 'joint'
    research_cost_rate: float = .001
    research_empty_leg_policy: str = 'signal_cash'

    def __post_init__(self):
        if self.research_empty_leg_policy not in ('signal_cash', 'unavailable'):
            raise ValueError('research_empty_leg_policy must be signal_cash or unavailable')
        if self.selection_objective not in {'joint', 'rank_ic'}:
            raise ValueError('selection_objective must be joint or rank_ic')
        if (isinstance(self.research_cost_rate, bool) or not math.isfinite(self.research_cost_rate)
                or self.research_cost_rate < 0):
            raise ValueError('research_cost_rate must be finite and nonnegative')
        for name in ("train_fraction", "validation_fraction",
                     "confidence_level"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or not 0 < v < 1:
                raise ValueError(f"{name} must be in (0,1)")
        if (isinstance(self.minimum_coverage, bool) or not math.isfinite(self.minimum_coverage)
                or not 0 < self.minimum_coverage <= 1):
            raise ValueError("minimum_coverage must be in (0,1]")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("a separate TEST partition is required")
        for name in ("warmup_bars", "embargo_bars", "minimum_train_days",
                     "minimum_validation_days", "minimum_test_days", "minimum_assets",
                     "bootstrap_draws", "block_length", "maximum_candidates", "seed"):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name in {"warmup_bars", "embargo_bars", "seed"} else 1):
                raise ValueError(f"{name} must be a valid integer")
        if self.bootstrap_draws < 99:
            raise ValueError("bootstrap_draws must be at least 99")
        if self.minimum_assets < 3:
            raise ValueError("minimum_assets must be at least 3")
        if (isinstance(self.minimum_improvement, bool) or isinstance(self.natural_time_scale, bool)
                or not math.isfinite(self.minimum_improvement) or self.minimum_improvement < 0
                or not math.isfinite(self.natural_time_scale) or not 1 <= self.natural_time_scale <= 10000):
            raise ValueError("invalid gain or natural time scale")
        if type(self.compose_smoothing_sign) is not bool:
            raise ValueError("compose_smoothing_sign must be a strict bool")
        if isinstance(self.families, (str, bytes)) or any(
                not isinstance(name, str) or not name.strip() for name in self.families):
            raise ValueError("families must be a sequence of nonempty family names")
        object.__setattr__(self, "families", tuple(self.families))
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must be unique")


@dataclass(frozen=True)
class AutomaticTimeSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    validation_start: int
    test_start: int
    identity: str


@dataclass(frozen=True)
class FactorOptimizationResult:
    factor_id: str
    status: str
    selected_family: str
    plan_identity: str
    plan: Any
    train_gain: float | None
    validation_lower_bound: float | None
    candidates: tuple[Mapping[str, Any], ...]
    reason: str
    validation_candidate_identity: str | None = None
    validation_coverage: float | None = None
    training_diagnostics: Mapping[str, Any] | None = None
    baseline_diagnostics: Mapping[str, Any] | None = None
    joint_diagnostics: Mapping[str, Any] | None = None
    materialization_error: str | None = None


@dataclass(frozen=True)
class OrientedRepairPlan:
    """Frozen sign after a temporal repair; both decisions are chosen on TRAIN."""
    base: Any
    multiplier: int = -1

    def __post_init__(self):
        if type(self.multiplier) is not int or self.multiplier not in (-1, 1):
            raise ValueError("orientation multiplier must be +1 or -1")

    @property
    def family(self):
        return self.base.family

    @property
    def identity(self):
        return hashlib.sha256(f"oriented-repair.v1:{self.base.identity}:{self.multiplier}".encode()).hexdigest()

    def execute(self, values, *, allow_research=False):
        return self.multiplier * self.base.execute(values, allow_research=allow_research)


@dataclass(frozen=True)
class BatchOptimizationResult:
    optimized: Any
    factors: Mapping[str, FactorOptimizationResult]
    split: AutomaticTimeSplit
    execution_mode: str = "research_only"
    test_evaluated: bool = False
    config: BatchOptimizationConfig | None = None


def automatic_time_split(labels, config: BatchOptimizationConfig | None = None):
    """60/20/20 in time, warmup, and purge real label windows at boundaries."""
    config = config or BatchOptimizationConfig()
    n = len(labels.decision_time)
    v = int(n * config.train_fraction)
    t = int(n * (config.train_fraction + config.validation_fraction))
    if not 0 < v < t < n:
        raise ValueError("insufficient observations for chronological split")
    train_cutoff = labels.decision_time[max(0, v - config.embargo_bars)]
    validation_cutoff = labels.decision_time[max(0, t - config.embargo_bars)]
    train = tuple(i for i in range(config.warmup_bars, v)
                  if labels.label_end_time[i] < train_cutoff)
    valid = tuple(i for i in range(v, t) if labels.label_end_time[i] < validation_cutoff)
    test = tuple(range(t, n))
    if (len(train) < config.minimum_train_days or len(valid) < config.minimum_validation_days
            or len(test) < config.minimum_test_days):
        raise ValueError("insufficient observations after warmup and label-window purge")
    payload = (train, valid, test, tuple(map(str, labels.decision_time)),
               tuple(map(str, labels.label_end_time[:t])))
    identity = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    return AutomaticTimeSplit(train, valid, test, v, t, identity)


def _subset_labels(labels, indices):
    idx = np.asarray(indices)
    kwargs = {"values": labels.values[idx],
              "validity": None if labels.validity is None else labels.validity[idx]}
    for name in ("decision_time", "execution_time", "signal_available_time",
                 "label_start_time", "label_end_time", "observation_time"):
        seq = getattr(labels, name)
        kwargs[name] = tuple(seq[i] for i in indices) if seq else ()
    return replace(labels, **kwargs)


class PairICCache:
    """At most two IC references, private to one factor's TRAIN search.

    Effective value/mask/label contents and minimum assets define reuse.
    No mutable input or returned array aliases cached evidence.
    """

    def __init__(self):
        from collections import OrderedDict
        self._entries = OrderedDict()

    def get(self, key):
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key].copy()

    def put(self, key, value):
        self._entries[key] = value.copy()
        self._entries.move_to_end(key)
        while len(self._entries) > 2:
            self._entries.popitem(last=False)


def _candidate_ic_key(values, target, minimum_assets):
    """Exact effective Spearman inputs; shared by coverage and joint scoring."""
    valid = np.isfinite(values) & np.isfinite(target.values)
    if target.validity is not None:
        valid &= target.validity
    effective = np.where(valid, values, np.nan)
    digest = hashlib.sha256(b'candidate-ic.v1')
    digest.update(repr((effective.shape, minimum_assets)).encode())
    for panel in (effective, target.values):
        panel = np.ascontiguousarray(panel)
        digest.update(panel.dtype.str.encode())
        digest.update(panel.tobytes())
    digest.update(b'none' if target.validity is None
                  else np.ascontiguousarray(target.validity).tobytes())
    return digest.digest()


def _pair_ic(raw, candidate, batch, labels, indices, config, *, reference_cache=None,
             candidate_cache=None):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.runtime.evaluator import evaluate

    idx = np.asarray(indices)
    a, b = raw[idx], candidate[idx]
    target = _subset_labels(labels, indices)
    available = np.isfinite(a) & np.isfinite(target.values)
    if target.validity is not None:
        available &= target.validity
    common = available & np.isfinite(b)
    retention = float(common.sum() / max(1, available.sum()))
    time_axis = AxisRef("time", batch.time_axis.dtype, len(idx), batch.time_axis.values[idx])
    # Third series anchors valid-day coverage to RAW's original universe.
    # A candidate must not improve by silently dropping difficult/constant days.
    pair = np.stack((a, b, a), axis=-1)
    validity = np.stack((common, common, available), axis=-1)
    factors = FactorBatch(("RAW", "CANDIDATE", "RAW_FULL"), time_axis, batch.asset_axis, pair, validity=validity)
    columns, keys, pending = [0, 1, 2], {}, {}
    ic = np.full((len(idx), 3), np.nan)
    if reference_cache is not None:
        if not isinstance(reference_cache, PairICCache):
            raise TypeError("reference_cache must be PairICCache")
        digest = hashlib.sha256()
        digest.update(repr((a.shape, config.minimum_assets)).encode())
        # Identity must not round extended-precision labels into float64 ties.
        # Including invalid values is conservative: extra misses, never stale hits.
        y = np.ascontiguousarray(target.values)
        digest.update(y.dtype.str.encode())
        digest.update(y.tobytes())
        digest.update(b"none" if target.validity is None
                      else np.ascontiguousarray(target.validity).tobytes())
        columns = [1]
        for column in (0, 2):
            state = digest.copy()
            masked = np.ascontiguousarray(np.where(validity[:, :, column], a, np.nan))
            state.update(masked.dtype.str.encode())
            state.update(masked.tobytes())
            key = state.digest()
            keys[column] = key
            cached = reference_cache.get(key)
            if cached is not None:
                ic[:, column] = cached
            elif key not in pending:
                pending[key] = column
                columns.append(column)
        factors = replace(factors, factor_ids=tuple(factors.factor_ids[c] for c in columns),
                          values=pair[:, :, columns], validity=validity[:, :, columns])
    result = evaluate(factors, target, metrics=["rank_ic_series"], backend="cpu",
                      metric_parameters={"rank_ic_series": {"min_assets": config.minimum_assets}})
    computed = np.asarray(result.artifacts["rank_ic_series"].values)
    ic[:, columns] = computed
    if reference_cache is not None:
        for key, column in pending.items():
            reference_cache.put(key, ic[:, column])
        for column, key in keys.items():
            if key in pending:
                ic[:, column] = ic[:, pending[key]]
    if candidate_cache is not None:
        if not isinstance(candidate_cache, PairICCache):
            raise TypeError('candidate_cache must be PairICCache')
        key = _candidate_ic_key(np.where(common, b, np.nan), target, config.minimum_assets)
        candidate_cache.put(key, ic[:, 1])
    good = np.isfinite(ic[:, :2]).all(axis=1)
    day_retention = float(good.sum() / max(1, np.isfinite(ic[:, 2]).sum()))
    retention = min(retention, day_retention)
    # Retain the full timeline: invalid days remain NaN for block resampling.
    diff = ic[:, 1] - ic[:, 0]
    return diff, good, retention


def _specs(config):
    from factor_optimizer.policy.repair_registry import RepairFamilyRegistry
    registry = RepairFamilyRegistry.default()
    families = config.families or tuple(registry.family_names)
    specs = []
    for family in sorted(families):
        prior = registry.get(family).parameter_prior.to_dict()
        if family == "NO_OP_RAW":
            continue
        if family in {"U_SHAPE_REPAIR", "INVERTED_U_REPAIR"}:
            choices = [dict(prior, center=c, power=p, asymmetry=False)
                       for c in (.35, .5, .65) for p in (1., 2.)]
        elif family == "CAUSAL_SMOOTHING":
            # Compiled per factor with its TRAIN evidence reference below.
            choices = []
        elif family == "DECAY_REFINEMENT":
            choices = [dict(prior, decay=d, half_life_relative=r)
                       for d in (.5, .8) for r in (False, True)]
        elif family == "MISSINGNESS_FRESHNESS":
            choices = [dict(prior, mode="flag")]
            choices += [dict(prior, mode="fill", freshness_window=w) for w in (1, 3, 5)]
        elif family == "TAIL_HINGE":
            choices = [dict(prior, hinge=side) for side in ("top", "bottom")]
        elif family == "TAIL_SATURATION":
            choices = [dict(prior, saturate=side) for side in ("top", "bottom", "both")]
        elif family == "REPRESENTATION_RANK":
            choices = [prior] + [dict(prior, rank_axis="ts", tie_method=tie, window=window)
                                for tie in ("average", "min") for window in (5, 10, 20)]
        elif family == "REPRESENTATION_ZSCORE":
            choices = [prior] + [dict(prior, zscore_axis="ts", window=window) for window in (5, 10, 20)]
        else:
            choices = [prior]
        specs.extend((family, p) for p in choices)
    if len(specs) > config.maximum_candidates:
        raise ValueError("candidate budget is too small for declared families")
    return specs


def _lower_bound(differences, config, horizon):
    # Moving blocks preserve within-block dependence and missing-date positions.
    length = max(config.block_length, int(horizon))
    n = len(differences)
    if n < 3 * length:
        return None
    rng = np.random.default_rng(config.seed)
    draws = []
    for _ in range(config.bootstrap_draws):
        starts = rng.integers(0, n-length+1, size=math.ceil(n/length))
        values = np.concatenate([differences[s:s+length] for s in starts])[:n]
        values = values[np.isfinite(values)]
        if len(values) < config.minimum_validation_days:
            return None
        draws.append(float(values.mean()))
    return float(np.quantile(draws, (1-config.confidence_level)/2))


def optimize_factor_batch(batch, labels, *, config=None, allow_research=False,
                          lineages=None, exposures=None, exposure_columns=(),
                          maximum_baseline_loss=.01):
    """Optimize aligned QE contracts automatically, preserving every input ID.

    Defaults fit candidate choices on TRAIN, confirm only its winner on
    VALIDATION, and leave TEST labels unused. The output applies the frozen
    accepted plan to factor values (including later rows), not future labels.
    Missing DSL/exposures are explicit ineligible candidate records.
    Default joint.v1 compares costed Sharpe, ICIR, IC, drawdown, stability and
    turnover. Explicit selection_objective='rank_ic' reproduces legacy scoring.
    """
    if allow_research is not True:
        raise ValueError("explicit allow_research=True is required; not production admission")
    config = config or BatchOptimizationConfig()
    if (isinstance(maximum_baseline_loss, bool) or not math.isfinite(maximum_baseline_loss)
            or maximum_baseline_loss < 0):
        raise ValueError('maximum_baseline_loss must be finite and nonnegative')
    lineages = {} if lineages is None else lineages
    if not isinstance(lineages, Mapping) or set(lineages) - set(batch.factor_ids):
        raise ValueError('lineages must map input factor IDs to explicit treatment lineage')
    if (batch.time_axis.values is None or batch.asset_axis.values is None
            or labels.asset_axis is None or labels.asset_axis.values is None):
        raise ValueError("explicit time and asset axes are required")
    if (not np.array_equal(batch.time_axis.values, np.asarray(labels.decision_time))
            or not np.array_equal(batch.asset_axis.values, labels.asset_axis.values)
            or labels.values.shape != batch.values.shape[:2]):
        raise ValueError("factor and label axes must match exactly")
    split = automatic_time_split(labels, config)
    specs = _specs(config)
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    from factor_optimizer.research_baseline import (
        compile_baseline, assess_baseline_training, BaselineRepairPlan,
    )
    from factor_optimizer.research_fitness import (
        paired_series, summarize, joint_utility, passes_floors, compare_joint, RawSeriesCache,
    )
    from factor_optimizer.adapters.repair_execution import compile_value_repair
    from factor_optimizer.adapters.preprocessing import compile_admissible_smoothing_grid
    from quant_evaluator.contracts.factor_batch import FactorBatch
    import pandas as pd

    outputs, results = [], {}
    for k, factor_id in enumerate(batch.factor_ids):
        train_raw_cache = RawSeriesCache()
        train_ic_cache = PairICCache()
        train_candidate_ic_cache = PairICCache()
        raw = np.array(batch.values[:, :, k], dtype=float, copy=True)
        if batch.validity is not None:
            raw[~batch.validity[:, :, k]] = np.nan
        raw[~np.isfinite(raw)] = np.nan
        # Candidate transforms see TRAIN history only. VALIDATION is materialized
        # once, after its sole candidate has been frozen on TRAIN.
        prefix = raw[:split.validation_start]
        frame = pd.DataFrame({"date": np.repeat(batch.time_axis.values[:split.validation_start], raw.shape[1]),
                              "asset_id": np.tile(batch.asset_axis.values, split.validation_start),
                              "value": prefix.ravel()})
        train_hash = hashlib.sha256()
        train_hash.update(raw[:split.validation_start].tobytes())
        train_hash.update(labels.values[np.asarray(split.train_indices)].tobytes())
        if labels.validity is not None:
            train_hash.update(labels.validity[np.asarray(split.train_indices)].tobytes())
        train_hash.update(json.dumps(batch.asset_axis.values.tolist(), default=str).encode())
        train_hash.update((factor_id + split.identity).encode())
        train_hash.update(config.research_empty_leg_policy.encode())
        train_ref = train_hash.hexdigest()
        baseline_plan = compile_baseline(lineages.get(factor_id),
            training_context_ref=train_ref, exposure_columns=exposure_columns,
            minimum_assets=config.minimum_assets)
        baseline_record = {'operations': baseline_plan.operations,
                           'omissions': baseline_plan.omissions, 'accepted': False,
                           'reason': 'no_eligible_baseline_operations'}
        baseline_values, baseline_active = prefix, False
        neutralization_attempt = None
        for _ in range(2):
            if not baseline_plan.operations:
                break
            try:
                proposed = np.asarray(baseline_plan.execute(frame, exposures=exposures,
                                      allow_research=True), dtype=float).reshape(prefix.shape)
                baseline_record.update(assess_baseline_training(prefix, proposed, batch,
                    labels, split, config, maximum_loss=maximum_baseline_loss))
                if baseline_record['accepted']:
                    baseline_values, baseline_active = proposed, True
                    break
            except Exception as exc:
                baseline_record.update(reason=f'baseline_unavailable: {type(exc).__name__}: {exc}')
            if 'neutralize' not in baseline_plan.operations:
                break
            # TRAIN-only fallback: an unavailable/degraded OLS must not silently
            # discard independently usable winsor/rank. No VALIDATION retry.
            neutralization_attempt = dict(baseline_record)
            baseline_plan = replace(baseline_plan,
                operations=tuple(op for op in baseline_plan.operations if op != 'neutralize'),
                omissions=baseline_plan.omissions + ('neutralization_train_rejected',))
            baseline_record = {'operations': baseline_plan.operations,
                               'omissions': baseline_plan.omissions, 'accepted': False,
                               'reason': 'no_eligible_baseline_operations'}
        if neutralization_attempt is not None:
            baseline_record['neutralization_attempt'] = neutralization_attempt
        search_frame = frame.copy()
        search_frame['value'] = baseline_values.ravel()
        diagnostic_values = raw.copy()
        diagnostic_values[:split.validation_start] = baseline_values
        diagnostic_batch = replace(batch, factor_ids=(factor_id,),
            values=diagnostic_values[:, :, None], validity=np.isfinite(diagnostic_values[:, :, None]))
        diagnosis = diagnose_training_batch(diagnostic_batch, labels, config=config)[factor_id]
        diagnosis["input_stage"] = "accepted_baseline" if baseline_active else "raw"
        factor_specs = list(specs)
        if ('cs_rank_already_present' in baseline_plan.omissions
                or (baseline_active and 'cs_rank' in baseline_plan.operations)):
            factor_specs = [(f, p) for f, p in factor_specs
                            if not (f == 'REPRESENTATION_RANK' and p['rank_axis'] == 'cross_sectional')]
        shape = diagnosis["proposed_shape_family"]
        if shape and (not config.families or shape in config.families):
            # Up to two TRAIN-fitted proposals supplement the prespecified grid.
            # They count against the same budget and get no validation retries.
            for power in (1., 2.):
                params = {"center": diagnosis["proposed_center"], "power": power, "asymmetry": False}
                if (shape, params) not in factor_specs:
                    factor_specs.append((shape, params))
        proposals = [(family, params, None) for family, params in factor_specs]
        proposal_sources = {}
        if not config.families or "CAUSAL_SMOOTHING" in config.families:
            proposals += [(p.family, dict(p.parameters), p) for p in compile_admissible_smoothing_grid(
                natural_time_scale=config.natural_time_scale, training_context_ref=train_ref)]
            from factor_optimizer.adapters.preprocessing import compile_smoothing_repair
            present = {p.identity for _, _, p in proposals if p is not None}
            omissions = []
            for half_life in diagnosis["layer_decay"]["proposed_half_lives"]:
                try:
                    p = compile_smoothing_repair("CAUSAL_SMOOTHING",
                        {"method": "EWMA", "natural_time_scale_relative":
                         half_life/config.natural_time_scale},
                        natural_time_scale=config.natural_time_scale, training_context_ref=train_ref)
                except ValueError as exc:
                    omissions.append({"half_life": half_life, "reason": str(exc)})
                    continue
                proposal_sources[p.identity] = "TRAIN_layer_decay"
                if p.identity not in present:
                    proposals.append((p.family, dict(p.parameters), p))
                    present.add(p.identity)
            diagnosis["layer_decay"]["inadmissible_smoothing_scales"] = omissions
        if not config.families or "DECAY_REFINEMENT" in config.families:
            from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
            decay = diagnosis["layer_decay"]
            layers = decay["layers"]
            decay["inadmissible_layered_scales"] = []
            if (decay["status"] == "available" and len(layers) == 20
                    and sum(x["stable_initial_direction"] for x in layers) >= 10):
                # Censoring supplies an observed lower bound, not a fitted life.
                lives = tuple((x["half_life_bars"] or max(decay["lags"]))
                              if x["stable_initial_direction"] else config.natural_time_scale
                              for x in layers)
                present = set()
                for scale in (1., .5):
                    try:
                        p = LayeredDecayPlan(tuple(h*scale for h in lives), train_ref)
                    except ValueError as exc:
                        decay["inadmissible_layered_scales"].append(
                            {"scale": scale, "reason": str(exc)})
                        continue
                    if p.identity not in present:
                        proposals.append((p.family, dict(p.parameters), p))
                        proposal_sources[p.identity] = "TRAIN_layer_states"
                        present.add(p.identity)
        compose = config.compose_smoothing_sign and (
            not config.families or "SIGN_ORIENTATION" in config.families)
        # Adjacent signs reuse one materialization, without caching the grid.
        proposals = [(f, p, plan, sign)
                     for f, p, plan in proposals
                     for sign in ((1, -1) if compose and f in {
                         "CAUSAL_SMOOTHING", "DECAY_REFINEMENT"} else (1,))]
        if len(proposals) > config.maximum_candidates:
            raise ValueError("candidate budget is too small for admitted smoothing grid")
        raw_plan = compile_value_repair("NO_OP_RAW", {"keep_raw": True},
                                       natural_time_scale=config.natural_time_scale,
                                       training_context_ref=train_ref)
        chosen, train_gain, lower = raw_plan, None, None
        joint = config.selection_objective == 'joint'
        joint_record = {'objective': config.selection_objective,
                        'empty_leg_policy': config.research_empty_leg_policy,
                        'cost_rate': config.research_cost_rate if joint else None,
                        'policy': 'joint.v1' if joint else 'legacy_rank_ic'}
        validation_identity, validation_coverage = None, None
        status, reason, records = "raw_retained", "no robust TRAIN improvement", []
        materialization_error = None
        try:
            _, raw_good, _ = _pair_ic(prefix, prefix, batch, labels, split.train_indices, config,
                                      reference_cache=train_ic_cache)
            if raw_good.sum() < config.minimum_train_days:
                status, reason = "invalid_raw", "insufficient valid RAW training IC"
            else:
                best = None
                if baseline_active:
                    delta, _, _ = _pair_ic(prefix, baseline_values, batch, labels, split.train_indices, config,
                                           reference_cache=train_ic_cache, candidate_cache=train_candidate_ic_cache)
                    folds = [np.nanmean(x) for x in np.array_split(delta, 3)]
                    baseline_gain = float(np.mean(folds) - np.std(folds))
                    frozen_baseline = BaselineRepairPlan(baseline_plan, raw_plan)
                    best = (baseline_gain, frozen_baseline.identity, frozen_baseline, baseline_values)
                    if joint:
                        try:
                            ar, ac = paired_series(prefix, baseline_values, batch, labels, split.train_indices,
                                minimum_assets=config.minimum_assets, cost_rate=config.research_cost_rate,
                                empty_leg_policy=config.research_empty_leg_policy,
                                candidate_ic_cache=train_candidate_ic_cache,
                                raw_cache=train_raw_cache)
                            mr, mc = summarize(ar), summarize(ac)
                            baseline_gain = joint_utility(mc)-joint_utility(mr)
                            best = ((baseline_gain, frozen_baseline.identity, frozen_baseline, baseline_values)
                                    if passes_floors(mr, mc) else None)
                            joint_record['baseline_train_raw'] = mr
                            joint_record['baseline_train_candidate'] = mc
                        except ValueError as exc:
                            best = None
                            joint_record['baseline_unavailable'] = str(exc)
                last_base_identity, last_base_values = None, None
                for family, params, precompiled, orientation in proposals:
                    record = {"family": family, "parameters": dict(params), "orientation": orientation}
                    record["diagnosed_issues"] = [issue["code"] for issue in diagnosis["issues"]
                                                  if family in issue["families"] or (
                                                      orientation == -1 and "SIGN_ORIENTATION" in issue["families"])]
                    if precompiled is not None and precompiled.identity in proposal_sources:
                        record["proposal_source"] = proposal_sources[precompiled.identity]
                    try:
                        plan = precompiled or compile_value_repair(family, params,
                            natural_time_scale=config.natural_time_scale, training_context_ref=train_ref)
                        record["transform"] = plan.transform
                        if plan.identity != last_base_identity:
                            base_values = np.asarray(plan.execute(
                                search_frame, allow_research=True), dtype=float).reshape(prefix.shape)
                            last_base_identity, last_base_values = plan.identity, base_values
                        values = last_base_values if orientation == 1 else -last_base_values
                        if orientation == -1:
                            plan = OrientedRepairPlan(plan)
                        if baseline_active:
                            plan = BaselineRepairPlan(baseline_plan, plan)
                        record["plan_identity"] = plan.identity
                        delta, good, coverage = _pair_ic(prefix, values, batch, labels, split.train_indices, config,
                                                       reference_cache=train_ic_cache, candidate_cache=train_candidate_ic_cache)
                        record.update(coverage=coverage, valid_train_days=int(good.sum()))
                        if coverage < config.minimum_coverage or good.sum() < config.minimum_train_days:
                            raise ValueError("insufficient common coverage or training IC")
                        chunks = np.array_split(delta, 3)
                        if any(np.isfinite(x).sum() < 10 for x in chunks):
                            raise ValueError("insufficient chronological training folds")
                        fold_gains = np.array([np.nanmean(x) for x in chunks])
                        gain = float(fold_gains.mean() - fold_gains.std())
                        if joint:
                            ar, ac = paired_series(prefix, values, batch, labels, split.train_indices,
                                minimum_assets=config.minimum_assets, cost_rate=config.research_cost_rate,
                                empty_leg_policy=config.research_empty_leg_policy,
                                candidate_ic_cache=train_candidate_ic_cache,
                                raw_cache=train_raw_cache)
                            mr, mc = summarize(ar), summarize(ac)
                            record.update(train_raw_metrics=mr, train_candidate_metrics=mc,
                                          raw_rank_ic_fold_gain=gain)
                            if not passes_floors(mr, mc):
                                raise ValueError('joint raw-relative degradation floor failed')
                            gain = joint_utility(mc)-joint_utility(mr)
                        record.update(status="train_evaluated", train_gain=gain, plan_identity=plan.identity,
                                      coverage=coverage)
                        if gain > config.minimum_improvement:
                            entry = (gain, plan.identity, plan, values)
                            if best is None or (-gain, plan.identity) < (-best[0], best[1]):
                                best = entry
                    except Exception as exc:
                        record.update(status="ineligible", reason=f"{type(exc).__name__}: {exc}")
                    records.append(MappingProxyType(record))
                if best is not None:
                    gain, _, winner, values = best
                    validation_identity = winner.identity
                    train_gain = gain
                    validation_raw = raw[:split.test_start]
                    validation_frame = pd.DataFrame({
                        'date': np.repeat(batch.time_axis.values[:split.test_start], raw.shape[1]),
                        'asset_id': np.tile(batch.asset_axis.values, split.test_start),
                        'value': validation_raw.ravel()})
                    kwargs = {'exposures': exposures} if isinstance(winner, BaselineRepairPlan) else {}
                    validation_values = np.asarray(winner.execute(validation_frame,
                        allow_research=True, **kwargs), dtype=float).reshape(validation_raw.shape)
                    delta, good, coverage = _pair_ic(validation_raw, validation_values,
                        batch, labels, split.validation_indices, config)
                    validation_coverage = coverage
                    if coverage >= config.minimum_coverage and good.sum() >= config.minimum_validation_days:
                        if joint:
                            ar, ac = paired_series(validation_raw, validation_values, batch, labels,
                                split.validation_indices, minimum_assets=config.minimum_assets,
                                cost_rate=config.research_cost_rate,
                                empty_leg_policy=config.research_empty_leg_policy)
                            mr, mc = summarize(ar), summarize(ac)
                            joint_record.update(validation_raw=mr, validation_candidate=mc)
                            if passes_floors(mr, mc):
                                comparison = compare_joint(ar, ac, config)
                                joint_record['comparison_status'] = comparison.status.value
                                if comparison.difference_interval is not None:
                                    lower = comparison.difference_interval[0]
                            else:
                                joint_record['comparison_status'] = 'DEGRADATION_FLOOR_FAILED'
                        else:
                            lower = _lower_bound(delta, config, labels.horizon)
                    baseline_only = winner.family == 'BASELINE'
                    confirmed = lower is not None and (
                        lower >= -maximum_baseline_loss if baseline_only
                        else lower > 0 and lower >= config.minimum_improvement)
                    if confirmed:
                        chosen = winner
                        status = 'baseline_accepted' if baseline_only else 'improved'
                        reason = ('TRAIN baseline passed held-out noninferiority bound' if baseline_only
                                  else 'TRAIN winner passed held-out paired block bound')
                    else:
                        reason = "TRAIN winner not confirmed on VALIDATION; retained RAW without retry"
            if chosen.family == "NO_OP_RAW":
                out = raw
            else:
                full = pd.DataFrame({"date": np.repeat(batch.time_axis.values, raw.shape[1]),
                                     "asset_id": np.tile(batch.asset_axis.values, len(raw)), "value": raw.ravel()})
                kwargs = {'exposures': exposures} if isinstance(chosen, BaselineRepairPlan) else {}
                try:
                    out = np.asarray(chosen.execute(full, allow_research=True, **kwargs), dtype=float).reshape(raw.shape)
                except Exception as exc:
                    # Future input failures cannot rewrite the frozen selection
                    # or previously validated values. Do not splice RAW into an
                    # optimized series, and do not retry selection on TEST.
                    materialization_error = f"{type(exc).__name__}: {exc}"
                    out = np.full(raw.shape, np.nan)
                    out[:split.test_start] = validation_values
                    status = "materialization_failed"
                    reason = "frozen selection preserved; post-validation values unavailable"
        except Exception as exc:
            chosen, out, status = raw_plan, raw, "error_raw_retained"
            reason = f"{type(exc).__name__}: {exc}"
        outputs.append(out)
        results[factor_id] = FactorOptimizationResult(
            factor_id, status, chosen.family, chosen.identity, chosen, train_gain, lower,
            tuple(records), reason, validation_identity, validation_coverage, MappingProxyType(diagnosis),
            MappingProxyType(baseline_record), MappingProxyType(joint_record), materialization_error)
    values = np.stack(outputs, axis=-1)
    optimized = FactorBatch(batch.factor_ids, batch.time_axis, batch.asset_axis,
                            values, validity=np.isfinite(values),
                            context_refs={"optimization_mode": "research_only", "split": split.identity})
    return BatchOptimizationResult(optimized, MappingProxyType(results), split, config=config)


__all__ = ["BatchOptimizationConfig", "AutomaticTimeSplit", "FactorOptimizationResult",
           "BatchOptimizationResult", "OrientedRepairPlan", "automatic_time_split", "optimize_factor_batch"]
