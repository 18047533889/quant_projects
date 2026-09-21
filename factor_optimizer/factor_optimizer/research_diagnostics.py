"""TRAIN-only multi-dimensional research diagnosis using QE metric authorities."""
from __future__ import annotations

import numpy as np

from factor_optimizer.research_batch import (
    BatchOptimizationConfig, automatic_time_split, _subset_labels,
)


def _number(value):
    return float(value) if np.isfinite(value) else None


def _selection_portfolio(values, target, ic, config, single_bar):
    """Use precisely the selection portfolio and reuse the already computed IC."""
    from factor_optimizer.research_fitness import portfolio_series, summarize
    record = dict(status="unavailable", metrics=None, reason=None,
                  periods_per_year=252,
                  cost_rate=config.research_cost_rate,
                  empty_leg_policy=config.research_empty_leg_policy,
                  semantics="costed gross-one stock-equal top/bottom quintiles")
    if not single_bar:
        record["reason"] = "multi-bar or overlapping labels require cohort accounting"
        return record
    returns = target.values if target.validity is None else np.where(target.validity, target.values, np.nan)
    try:
        pnl, turnover = portfolio_series(values, returns, cost_rate=config.research_cost_rate,
                                        empty_leg_policy=config.research_empty_leg_policy)
        record["metrics"] = summarize(np.column_stack((ic, pnl, turnover)))
        record["status"] = "available"
    except ValueError as exc:
        record["reason"] = str(exc)
    return record


def diagnose_training_batch(batch, labels, *, config=None, periods_per_year=252,
                            minimum_assets_per_quantile=10):
    """Inspect 20 real quantile bins, IC/ICIR and gross-one spread risk on TRAIN.

    Shapes are proposals, not admissions. Every bin must have the requested
    number of valid labels; bins share the same dates for the profile. No
    unavailable bin is filled with zero. Portfolio statistics are descriptive
    gross-one top/bottom spreads, not executable or net-of-cost backtests.
    The separate selection_portfolio field uses the costed quintile selection
    contract; issues are descriptive hypotheses, never admissions.
    Overlapping/multi-bar labels are not compounded as daily returns.
    """
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic_value
    from quant_evaluator.metrics.ic_summary import compute_icir
    from quant_evaluator.metrics.quantile import compute_quantile_returns, compute_top_bottom_spread
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio, compute_maximum_drawdown

    for name, value in (("periods_per_year", periods_per_year),
                        ("minimum_assets_per_quantile", minimum_assets_per_quantile)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    config = config or BatchOptimizationConfig()
    if (batch.time_axis.values is None or batch.asset_axis.values is None or
            labels.asset_axis is None or labels.asset_axis.values is None or
            not np.array_equal(batch.time_axis.values, np.asarray(labels.decision_time)) or
            not np.array_equal(batch.asset_axis.values, labels.asset_axis.values) or
            labels.values.shape != batch.values.shape[:2]):
        raise ValueError("explicit, matching factor and label axes are required")
    split = automatic_time_split(labels, config)
    idx = np.asarray(split.train_indices)
    train = FactorBatch(
        batch.factor_ids, AxisRef("time", batch.time_axis.dtype, len(idx), batch.time_axis.values[idx]),
        batch.asset_axis, batch.values[idx],
        validity=None if batch.validity is None else batch.validity[idx])
    target = _subset_labels(labels, split.train_indices)
    # Trust actual label endpoints, not just the nominal horizon field.
    nonoverlapping = all(target.label_end_time[i] <= target.label_start_time[i+1]
                         for i in range(len(idx)-1))
    ic, _ = compute_daily_ic(train, target, method="spearman", min_assets=config.minimum_assets)
    mean_ic = compute_mean_ic_value(ic, min_periods=config.minimum_train_days)
    icir = compute_icir(ic, min_periods=config.minimum_train_days)
    quantiles, counts = compute_quantile_returns(
        train, target, n_quantiles=20, min_assets=minimum_assets_per_quantile)
    spreads = .5 * compute_top_bottom_spread(quantiles)
    results = {}
    for k, name in enumerate(batch.factor_ids):
        from factor_optimizer.research_decay import diagnose_layer_decay
        decay = diagnose_layer_decay(batch, labels, split, config, k,
                                     minimum_assets_per_quantile=minimum_assets_per_quantile)
        complete = np.isfinite(quantiles[:, :, k]).all(axis=1)
        enough = (complete.sum() >= config.minimum_train_days and
                  complete.mean() >= config.minimum_coverage)
        profile = np.mean(quantiles[complete, :, k], axis=0) if enough else None
        shape, center, contrast = None, None, None
        if profile is not None:
            # Locate the TRAIN valley/peak, not necessarily the median.
            # Three chronological blocks must agree on BOTH tail contrasts.
            for family, extremum, sign in (
                    ("U_SHAPE_REPAIR", int(np.argmin(profile)), 1),
                    ("INVERTED_U_REPAIR", int(np.argmax(profile)), -1)):
                if not 1 <= extremum <= 18:
                    continue
                reference = quantiles[:, extremum, k]
                left = np.where(complete, sign*(quantiles[:, 0, k]-reference), np.nan)
                right = np.where(complete, sign*(quantiles[:, -1, k]-reference), np.nan)
                blocks = list(zip(np.array_split(left, 3), np.array_split(right, 3)))
                if all(np.isfinite(l).sum() >= 10 for l, r in blocks):
                    means = np.array([(np.nanmean(l), np.nanmean(r)) for l, r in blocks])
                    if np.all(means > 0):
                        shape, center = family, (extremum + .5)/20
                        break
            contrast = float((profile[0]+profile[-1])/2 - profile[9:11].mean())
        spread = np.where(complete, spreads[:, k], np.nan)
        single_bar = labels.horizon == 1 and nonoverlapping
        capital_valid = not np.any(spread < -1.)
        sharpe = compute_sharpe_ratio(spread, periods_per_year=periods_per_year,
                                      min_periods=config.minimum_train_days) if enough and single_bar and capital_valid else np.nan
        drawdown = compute_maximum_drawdown(spread)[0] if enough and single_bar and capital_valid else np.nan
        fold_ic = [np.nanmean(f) if np.isfinite(f).sum() >= 10 else np.nan
                   for f in np.array_split(ic[:, k], 3)]
        signal = train.values[:, :, k]
        if train.validity is not None:
            signal = np.where(train.validity[:, :, k], signal, np.nan)
        selection = _selection_portfolio(signal, target, ic[:, k], config, single_bar)
        # These are descriptive TRAIN problem flags, not statistical admission
        # or proof that a named repair will improve the factor.
        issues = []
        if decay["high_turnover"]:
            issues.append(dict(code="high_turnover",
                families=["CAUSAL_SMOOTHING", "DECAY_REFINEMENT"],
                value=decay["full_notional_turnover"], threshold=.5))
        if np.isfinite(mean_ic[k]) and mean_ic[k] < 0:
            issues.append(dict(code="negative_rank_ic", families=["SIGN_ORIENTATION"],
                               value=float(mean_ic[k]), threshold=0.))
        if np.isfinite(fold_ic).all() and min(fold_ic) < 0 < max(fold_ic):
            issues.append(dict(code="unstable_ic_direction", families=[],
                               value=[float(x) for x in fold_ic]))
        if shape:
            issues.append(dict(code="nonmonotonic_twenty_layer_profile",
                               families=[shape], center=center))
        if selection["status"] == "unavailable":
            issues.append(dict(code="joint_metrics_unavailable", families=[],
                               reason=selection["reason"]))
        elif selection["metrics"]["worst_block_sharpe"] < 0:
            issues.append(dict(code="negative_worst_block", families=[],
                               value=selection["metrics"]["worst_block_sharpe"]))
        results[name] = {
            "selection_portfolio": selection,
            "issues": issues,
            "layer_decay": decay,
            "partition": "TRAIN", "train_days": len(idx), "periods_per_year": periods_per_year,
            "rank_ic": _number(mean_ic[k]), "rank_icir": _number(icir[k]),
            "rank_ic_valid_days": int(np.isfinite(ic[:, k]).sum()),
            "fold_rank_ic": [_number(v) for v in fold_ic],
            "complete_quantile_days": int(complete.sum()),
            "quantile_coverage": float(complete.mean()),
            "minimum_bin_count": int(counts[complete, :, k].min()) if complete.any() else 0,
            "quantile_mean_returns": profile.tolist() if profile is not None else None,
            "u_contrast": contrast, "proposed_shape_family": shape, "proposed_center": center,
            "long_short_sharpe": _number(sharpe), "long_short_max_drawdown": _number(drawdown),
            "portfolio_semantics": "gross-one top/bottom 5% diagnostic spread; no costs or tradability",
            "portfolio_unavailable_reason": (
                "multi-bar labels are not daily PnL" if labels.horizon != 1 else
                "overlapping label intervals are not daily PnL" if not nonoverlapping else
                "insufficient complete 20-bin days" if not enough else
                "returns below -100% require a capital contract" if not capital_valid else
                "missing dates make full-period drawdown unknown" if not complete.all() else None),
        }
    return results
