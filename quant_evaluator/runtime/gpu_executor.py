"""GPU batch executor (spec §3, §43, §44).

Runs a compiled metric plan against a :class:`DeviceEvaluationSession`,
reusing shared device intermediates (sort/rank, daily IC, quantile) across
metrics.  Returns a columnar :class:`BatchEvaluationBundle`.

Only the coordinator wires this into the public API; the executor itself is
read-only over the canonical kernels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.runtime.device_session import DeviceEvaluationSession


def _import_cp():
    import cupy as cp
    return cp


def _to_cpu(dev):
    cp = _import_cp()
    return cp.asnumpy(dev)


class GPUExecutor:
    """Executes a metric plan on a device session (spec §3)."""

    def __init__(self, session: DeviceEvaluationSession):
        self.session = session

    def run(
        self,
        factor_ids: Sequence[str],
        metrics: Sequence[str],
        label_id: str = "next_ret",
    ) -> BatchEvaluationBundle:
        cp = _import_cp()
        factors = self.session._staged_factors["__all__"]  # (T, F, N)
        labels = self.session._staged_labels[label_id]      # (T, N)
        T, F, N = factors.shape

        scalar: Dict[str, np.ndarray] = {}
        series: Dict[str, np.ndarray] = {}
        vector: Dict[str, np.ndarray] = {}

        # shared intermediates (spec §9)
        rank_ic = None
        pearson_ic = None
        quantile_ret = None
        turnover = None

        for m in metrics:
            if m == "coverage":
                # coverage = fraction of PAIRWISE-FINITE (factor & label) per (t,f)
                yb = labels[:, None, :]  # (T,1,N)
                finite = cp.isfinite(factors) & cp.isfinite(yb)
                cov = cp.mean(finite, axis=2)  # (T,F)
                scalar["coverage"] = _to_cpu(cp.mean(cov, axis=0))  # (F,)
            elif m in ("rank_ic", "ic_ir", "ic_std", "ic_median", "rank_ic_series"):
                if rank_ic is None:
                    from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic
                    rank_ic, _ = batched_spearman_ic(factors, labels, min_obs=20)
                if m == "rank_ic_series":
                    series["rank_ic_series"] = _to_cpu(rank_ic)
                elif m == "rank_ic":
                    scalar["rank_ic"] = _to_cpu(cp.nanmean(rank_ic, axis=0))
                elif m == "ic_ir":
                    mu = cp.nanmean(rank_ic, axis=0)
                    sd = cp.nanstd(rank_ic, axis=0, ddof=1)
                    scalar["ic_ir"] = _to_cpu(mu / cp.maximum(sd, 1e-12))
                elif m == "ic_std":
                    scalar["ic_std"] = _to_cpu(cp.nanstd(rank_ic, axis=0, ddof=1))
                elif m == "ic_median":
                    scalar["ic_median"] = _to_cpu(cp.nanmedian(rank_ic, axis=0))
            elif m in ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"):
                if pearson_ic is None:
                    from quant_evaluator.kernels.gpu.correlation import batched_pearson_ic
                    pearson_ic, _ = batched_pearson_ic(factors, labels, min_obs=20)
                if m == "pearson_ic_series":
                    series["pearson_ic_series"] = _to_cpu(pearson_ic)
                elif m == "pearson_ic":
                    scalar["pearson_ic"] = _to_cpu(cp.nanmean(pearson_ic, axis=0))
                elif m == "pearson_ic_std":
                    scalar["pearson_ic_std"] = _to_cpu(cp.nanstd(pearson_ic, axis=0, ddof=1))
                elif m == "pearson_ic_ir":
                    mu = cp.nanmean(pearson_ic, axis=0)
                    sd = cp.nanstd(pearson_ic, axis=0, ddof=1)
                    scalar["pearson_ic_ir"] = _to_cpu(mu / cp.maximum(sd, 1e-12))
            elif m in ("quantile_returns_full", "quantile_spread", "quantile_monotonicity"):
                if quantile_ret is None:
                    from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns
                    quantile_ret = batched_quantile_returns(factors, labels, n_quantiles=10)
                if m == "quantile_returns_full":
                    vector["quantile_returns"] = _to_cpu(quantile_ret)  # (T,Q,F)
                elif m == "quantile_spread":
                    # Qtop - Qbottom mean over time -> (F,)
                    vector["quantile_spread"] = _to_cpu(
                        cp.nanmean(quantile_ret[:, -1, :] - quantile_ret[:, 0, :], axis=0)
                    )
                elif m == "quantile_monotonicity":
                    # mean of adjacent positive diffs (rough monotonicity)
                    diffs = cp.diff(quantile_ret, axis=1)  # (T, Q-1, F)
                    vector["quantile_monotonicity"] = _to_cpu(cp.mean(diffs > 0, axis=(0, 1)))
            elif m in ("turnover", "factor_turnover_rate"):
                if turnover is None:
                    from quant_evaluator.kernels.gpu.turnover import batched_turnover
                    turnover = batched_turnover(factors, return_series=True)  # (T,F)
                if m == "turnover":
                    scalar["turnover"] = _to_cpu(cp.nanmean(turnover, axis=0))
                elif m == "factor_turnover_rate":
                    scalar["factor_turnover_rate"] = _to_cpu(cp.nanmean(turnover, axis=0))
            elif m in ("probe_ls_sharpe", "probe_ls_calmar", "probe_ls_max_drawdown",
                       "probe_ls_annual_return", "probe_ls_turnover"):
                # delegated to Agent D's portfolio kernel if present
                try:
                    from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
                    from quant_evaluator.kernels.gpu.drawdown import portfolio_metrics_batch
                except ImportError:
                    raise RuntimeError(
                        f"metric '{m}' requires probe portfolio GPU kernels (not yet built)"
                    )
                pnl = compute_cohort_pnl_batch_gpu(factors, labels, n_quantiles=10, holding=20)
                pm = portfolio_metrics_batch(pnl)  # dict of (F,) arrays
                if m == "probe_ls_sharpe":
                    scalar["probe_ls_sharpe"] = _to_cpu(pm["sharpe"])
                elif m == "probe_ls_calmar":
                    scalar["probe_ls_calmar"] = _to_cpu(pm["calmar"])
                elif m == "probe_ls_max_drawdown":
                    scalar["probe_ls_max_drawdown"] = _to_cpu(pm["max_drawdown"])
                elif m == "probe_ls_annual_return":
                    scalar["probe_ls_annual_return"] = _to_cpu(pm["annual_return"])
                elif m == "probe_ls_turnover":
                    scalar["probe_ls_turnover"] = _to_cpu(pm["turnover"])
            else:
                raise RuntimeError(f"GPUExecutor: unsupported metric '{m}'")

        return BatchEvaluationBundle(
            factor_ids=tuple(factor_ids),
            label_id=label_id,
            scalar_metrics=scalar,
            series_metrics=series,
            vector_metrics=vector,
            metadata=self.session.metadata(),
        )
