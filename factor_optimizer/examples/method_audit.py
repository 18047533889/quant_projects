#!/usr/bin/env python3
"""Reproducible per-method TRAIN-only audit; no production writes or test scores."""
from __future__ import annotations
import json
import sys
from collections import Counter
import numpy as np
import pandas as pd

from factor_optimizer.adapters.preprocessing import compile_admissible_smoothing_grid
from factor_optimizer.adapters.repair_execution import compile_value_repair, IneligibleValueRepair
from factor_optimizer.policy.repair_registry import RepairFamilyRegistry
from factor_optimizer.research_batch import (
    BatchOptimizationConfig, automatic_time_split, _pair_ic, optimize_factor_batch,
)
from factor_optimizer.research_fitness import (
    paired_series, summarize, joint_utility, passes_floors,
)


def method_cases():
    """Exercise each family and each declared value-executable categorical branch."""
    registry = RepairFamilyRegistry.default()
    for family in registry.family_names:
        prior = registry.get(family).parameter_prior.to_dict()
        choices = [prior]
        if family == "CAUSAL_SMOOTHING":
            for plan in compile_admissible_smoothing_grid(
                    natural_time_scale=10., training_context_ref="prespecified-method-audit"):
                yield family, dict(plan.parameters), plan
            continue
        if family == "DECAY_REFINEMENT":
            from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
            layered = LayeredDecayPlan((3., 10.)*10, "prespecified-method-audit")
            yield family, dict(layered.parameters), layered
            choices = [dict(prior, half_life_relative=x) for x in (False, True)]
        elif family == "SIGN_ORIENTATION":
            choices = [dict(prior, direction=x) for x in ("keep", "flip")]
        elif family == "MISSINGNESS_FRESHNESS":
            choices = [dict(prior, mode=x) for x in ("flag", "fill", "drop")]
        elif family in {"U_SHAPE_REPAIR", "INVERTED_U_REPAIR"}:
            choices = [dict(prior, asymmetry=x, center=.35) for x in (False, True)]
        elif family == "TAIL_SATURATION":
            choices = [dict(prior, saturate=x) for x in ("top", "bottom", "both")]
        elif family == "TAIL_HINGE":
            choices = [dict(prior, hinge=x) for x in ("top", "bottom")]
        elif family == "ROBUST_SCALE":
            choices = [dict(prior, scale=s, center=c)
                       for s in ("mad", "iqr", "std") for c in ("mean", "median")]
        elif family == "REPRESENTATION_RANK":
            choices = [dict(prior, rank_axis=a, tie_method=m)
                       for a in ("cross_sectional", "ts") for m in ("average", "min")]
        elif family == "REPRESENTATION_ZSCORE":
            choices = [dict(prior, zscore_axis=a) for a in ("cross_sectional", "ts")]
        for params in choices:
            yield family, params, None


def audit_methods(batch, labels):
    """Evaluate every method, even when RAW has too few valid IC dates.

    Scores are exploratory TRAIN diagnostics only, not method acceptance.
    Each executable case checks prefix invariance, asset permutation and row
    alignment. Unsupported inputs remain explicit, not fake RAW execution.
    """
    config = BatchOptimizationConfig()
    split = automatic_time_split(labels, config)
    t, a = split.validation_start, batch.values.shape[1]
    rows = []
    for k, name in enumerate(batch.factor_ids):
        raw = np.array(batch.values[:t,:,k], dtype=float, copy=True)
        if batch.validity is not None:
            raw[~batch.validity[:t,:,k]] = np.nan
        raw[~np.isfinite(raw)] = np.nan
        frame = pd.DataFrame({"date": np.repeat(batch.time_axis.values[:t], a),
                              "asset_id": np.tile(batch.asset_axis.values, t),
                              "value": raw.ravel()})
        cut = (t // 2) * a
        order = np.arange(t*a).reshape(t,a)[:,::-1].ravel()
        for family, params, compiled in method_cases():
            row = {"factor": name, "family": family, "parameters": params}
            try:
                plan = compiled or compile_value_repair(family, params,
                    natural_time_scale=10., training_context_ref="prespecified-method-audit")
                row["transform"] = plan.transform
                output = plan.execute(frame, allow_research=True)
                assert output.index.equals(frame.index), "output row index changed"
                values = np.asarray(output, dtype=float)
                assert not np.isinf(values).any(), "infinite output"
                prefix = np.asarray(plan.execute(frame.iloc[:cut], allow_research=True), dtype=float)
                np.testing.assert_allclose(values[:cut], prefix, equal_nan=True)
                shuffled = frame.iloc[order]
                permuted = plan.execute(shuffled, allow_research=True)
                assert permuted.index.equals(shuffled.index), "permuted output row index changed"
                np.testing.assert_allclose(values, np.asarray(permuted)[np.argsort(order)], equal_nan=True)
                delta, good, retention = _pair_ic(
                    raw, values.reshape(raw.shape), batch, labels, split.train_indices, config)
                row.update(status="executed", prefix_invariant=True, asset_permutation_invariant=True,
                           finite_fraction=float(np.isfinite(values).mean()),
                           valid_ic_days=int(good.sum()), paired_retention=retention,
                           train_mean_ic_delta=float(np.nanmean(delta)) if good.any() else None)
                # Execution invariants and economic availability are separate:
                # missing PnL must not make a transform look broken or profitable.
                row["research_cost_rate"] = config.research_cost_rate
                row["empty_leg_policy"] = config.research_empty_leg_policy
                try:
                    ar, ac = paired_series(raw, values.reshape(raw.shape), batch,
                        labels, split.train_indices, minimum_assets=config.minimum_assets,
                        cost_rate=config.research_cost_rate,
                        empty_leg_policy=config.research_empty_leg_policy)
                    mr, mc = summarize(ar), summarize(ac)
                    row.update(joint_metrics_status="available",
                               train_raw_metrics=mr, train_candidate_metrics=mc,
                               joint_utility_delta=joint_utility(mc)-joint_utility(mr),
                               passes_raw_relative_floors=passes_floors(mr, mc))
                except ValueError as exc:
                    row.update(joint_metrics_status="unavailable", joint_metrics_reason=str(exc))
            except IneligibleValueRepair as exc:
                row.update(status="requires_additional_inputs_or_control", reason=str(exc))
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    return rows


def main():
    from real_batch_audit import load_sample
    batch, labels, inputs = load_sample()
    print("Loaded bounded real sample; auditing methods on TRAIN only", file=sys.stderr, flush=True)
    audit = audit_methods(batch, labels)
    print("Method audit complete; running automatic TRAIN/VALIDATION selection", file=sys.stderr, flush=True)
    result = optimize_factor_batch(batch, labels, allow_research=True)
    report = {
        "inputs": inputs, "test_evaluated": False, "method_audit_partition": "TRAIN only",
        "limits": "TRAIN execution and costed joint-metric audit; no profit claim or upstream PIT re-certification",
        "method_status_counts": dict(Counter(row["status"] for row in audit)),
        "methods": audit,
        "split": {"train": len(result.split.train_indices),
                  "validation": len(result.split.validation_indices),
                  "test_reserved": len(result.split.test_indices)},
        "automatic": {name: {
            "status": r.status, "selected_family": r.selected_family,
            "train_gain": r.train_gain, "validation_lower_bound": r.validation_lower_bound,
            "validation_candidate_identity": r.validation_candidate_identity,
            "validation_coverage": r.validation_coverage, "reason": r.reason,
            "candidates": [dict(c) for c in r.candidates],
        } for name,r in result.factors.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if any(r["status"] == "failed" for r in audit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
