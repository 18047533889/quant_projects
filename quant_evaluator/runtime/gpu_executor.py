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

    SUPPORTED_METRICS = frozenset({
        "sharpe_ratio", "sortino_ratio", "win_rate", "max_drawdown", "calmar_ratio",
        "coverage", "rank_ic", "ic_ir", "ic_std", "ic_median", "rank_ic_series",
        "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
        "quantile_returns_full", "quantile_returns_daily", "quantile_spread",
        "quantile_monotonicity", "daily_quantile_monotonicity_series",
        "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
        "industry_exposure", "size_exposure", "beta_exposure",
        "liquidity_exposure", "volatility_exposure", "momentum_exposure",
        "max_absolute_style_exposure", "exposure_drift", "purity_ratio",
    })

    @classmethod
    def validate_metric_plan(cls, metrics):
        """Admission reflects implemented dispatch branches, before allocation."""
        unsupported = set(metrics) - cls.SUPPORTED_METRICS
        if unsupported:
            from quant_evaluator.contracts.errors import UnsupportedMetricError
            raise UnsupportedMetricError(f"CUDA has no implementation for {sorted(unsupported)}")

    def __init__(self, session: DeviceEvaluationSession):
        self.session = session
        self.metric_parameters = {}
        self.holding_return_id = None
        self.portfolio_spec = None
        self.trade_eligibility = None
        self.prebuilt_portfolio_pnl = None
        self.portfolio_factor_ids = ()
        self.risk_tiles_processed = 0
        self.probe_tiles_processed = 0
        self.shape_kernel_dispatches = 0
        self.exposure_panel_values = None
        self.exposure_regression_weights = None
        self.exposure_style_names = ()
        self.exposure_kernel_dispatches = 0
        self._portfolio_column_map = {}
        self._host_result_bytes = 0

    @property
    def portfolio_factor_ids(self):
        return self._portfolio_factor_ids

    @portfolio_factor_ids.setter
    def portfolio_factor_ids(self, ids):
        ids = tuple(ids)
        if len(set(ids)) != len(ids):
            raise ValueError("portfolio factor identities must be unique")
        self._portfolio_factor_ids = ids
        self._portfolio_column_map = {fid: i for i, fid in enumerate(ids)}

    def _reserve_host_result(self, nbytes):
        total = self._host_result_bytes + int(nbytes)
        limit = self.session.policy.max_host_result_bytes
        if total > limit:
            raise MemoryError(
                f"worker host result budget exceeded: {total} > {limit}; "
                "reduce the queued factor batch and persist each completed batch via DataAccess")
        self._host_result_bytes = total

    def build_probe_pnl_tiled(self, factor_batch, holding_returns, portfolio_spec, trade_eligibility=None):
        """Build the canonical research-probe trajectory on CUDA by factor tile."""
        cp = _import_cp()
        from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
        values = factor_batch.values
        T, N, F = values.shape
        self._reserve_host_result(T * F * np.dtype(np.float64).itemsize)
        tile_size = min(F, self.session.estimate_tile(("probe_pnl",), T, N, values.dtype.itemsize))
        holding_dev = self.session.stage_holding_returns(holding_returns)
        permission = (None if trade_eligibility is None
                      else self.session.stage_trade_eligibility(trade_eligibility))
        self.holding_return_id = holding_dev
        self.portfolio_spec = portfolio_spec
        self.trade_eligibility = permission
        out = np.empty((T, F), dtype=np.float64)
        options = {key: value for key, value in portfolio_spec.to_dict().items()
                   if key not in {"purpose", "missing_return_policy"}}
        start = 0
        while start < F:
            stop = min(start + tile_size, F)
            chunk = values[:, :, start:stop]
            if factor_batch.validity is not None:
                chunk = np.where(factor_batch.validity[:, :, start:stop], chunk, np.nan)
            try:
                staged = self.session.stage_factors(chunk, factor_batch.factor_ids[start:stop], layout="T,N,F")
                pnl = compute_cohort_pnl_batch_gpu(
                    staged, self.session._staged_holding_returns[holding_dev],
                    require_tradable=False, trade_eligibility=permission, **options)["pnl_net"]
                host = _to_cpu(pnl); out[:, start:stop] = host
                self.session._d2h_bytes += host.nbytes
                self.probe_tiles_processed += 1
            except cp.cuda.memory.OutOfMemoryError:
                if not self.session.policy.oom_retile or stop - start <= 1:
                    raise
                tile_size = self.session.retile_on_oom(stop - start)
                continue
            finally:
                staged = pnl = None
                self.session.release_factor_tile()
            start = stop
        return out

    def run_grouped_compounding(self, returns, row_groups, included_periods=None):
        """Strict CUDA grouped compounds from CPU-planned calendar rows."""
        cp = _import_cp()
        from quant_evaluator.kernels.gpu.drawdown import compute_grouped_compounded_returns_batch
        host = np.asarray(returns)
        self._reserve_host_result((2 * len(row_groups) + 2) * host.shape[1] * 8)
        tile = min(host.shape[1], self.session.estimate_tile(None, host.shape[0], 1, host.dtype.itemsize))
        outputs = [np.empty((len(row_groups), host.shape[1]), dtype=np.float64),
                   np.empty((len(row_groups), host.shape[1]), dtype=np.int64),
                   np.empty(host.shape[1], dtype=np.float64), np.empty(host.shape[1], dtype=np.int64)]
        for start in range(0, host.shape[1], tile):
            stop = min(start + tile, host.shape[1]); dev = cp.asarray(host[:, start:stop])
            self.session._h2d_bytes += dev.nbytes
            result = compute_grouped_compounded_returns_batch(dev, row_groups, included_periods=included_periods)
            for output, value in zip(outputs, result):
                converted = _to_cpu(value); output[..., start:stop] = converted
                self.session._d2h_bytes += converted.nbytes
            self.risk_tiles_processed += 1
        return tuple(outputs)

    def run_worst_rolling_compound(self, returns, window, min_periods=1):
        """Strict CUDA bounded-state rolling compound reduction."""
        cp = _import_cp()
        from quant_evaluator.kernels.gpu.drawdown import compute_worst_rolling_compounded_return_batch
        host = np.asarray(returns)
        self._reserve_host_result(2 * host.shape[1] * 8)
        tile = min(host.shape[1], self.session.estimate_tile(None, host.shape[0], 1, host.dtype.itemsize))
        host_values = np.empty(host.shape[1], dtype=np.float64)
        host_counts = np.empty(host.shape[1], dtype=np.int64)
        for start in range(0, host.shape[1], tile):
            stop = min(start + tile, host.shape[1]); dev = cp.asarray(host[:, start:stop])
            self.session._h2d_bytes += dev.nbytes
            values, counts = compute_worst_rolling_compounded_return_batch(
                dev, window=window, min_periods=min_periods)
            host_values[start:stop], host_counts[start:stop] = _to_cpu(values), _to_cpu(counts)
            self.session._d2h_bytes += host_values[start:stop].nbytes + host_counts[start:stop].nbytes
            self.risk_tiles_processed += 1
        return host_values, host_counts

    def run_tiled(self, factor_batch, label_bundle, metrics) -> BatchEvaluationBundle:
        """Upload bounded factor slices, retaining labels across the slices.

        Host results are preallocated once, so stitching does not retain a
        list of all tile bundles or allocate a second complete result array.
        """
        self.validate_metric_plan(metrics)
        cp = _import_cp()
        values = factor_batch.values
        T, N, F = values.shape
        tile = min(F, self.session.estimate_tile(metrics, T, N, values.dtype.itemsize))
        labels = label_bundle.values
        if label_bundle.validity is not None:
            labels = np.where(label_bundle.validity, labels, np.nan)
        self.session.stage_labels(labels, label_bundle.target_id)
        out = BatchEvaluationBundle(tuple(factor_batch.factor_ids), label_bundle.target_id)
        start = 0
        count = 0
        while start < F:
            stop = min(start + tile, F)
            ids = factor_batch.factor_ids[start:stop]
            self.session._final_tile = tile
            try:
                chunk = values[:, :, start:stop]
                if factor_batch.validity is not None:
                    chunk = np.where(factor_batch.validity[:, :, start:stop], chunk, np.nan)
                self.session.stage_factors(chunk, ids, layout="T,N,F")
                result = self.run(ids, metrics, label_bundle.target_id)
            except cp.cuda.memory.OutOfMemoryError:
                if not self.session.policy.oom_retile or stop - start <= 1:
                    raise
                tile = self.session.retile_on_oom(stop - start)
                continue
            finally:
                self.session.release_factor_tile()
            for field in ("scalar_metrics", "series_metrics", "vector_metrics", "observation_counts"):
                destination = getattr(out, field)
                for name, arr in getattr(result, field).items():
                    if arr.ndim == 0 or arr.shape[-1] != stop - start:
                        raise ValueError(f"GPU metric {name!r} has no trailing factor axis")
                    if name not in destination:
                        self._reserve_host_result(int(np.prod(arr.shape[:-1])) * F * arr.dtype.itemsize)
                        destination[name] = np.empty((*arr.shape[:-1], F), dtype=arr.dtype)
                    destination[name][..., start:stop] = arr
                    self.session._d2h_bytes += arr.nbytes
            start = stop
            count += 1
        out.metadata = self.session.metadata()
        out.metadata["factor_tiles_processed"] = count
        out.metadata["host_result_bytes_reserved"] = self._host_result_bytes
        out.metadata["host_result_budget_bytes"] = self.session.policy.max_host_result_bytes
        out.metadata["shape_kernel_dispatches"] = self.shape_kernel_dispatches
        out.metadata["shape_kernel_backend"] = "cuda_strict" if self.shape_kernel_dispatches else None
        out.metadata["shape_kernel_no_fallback"] = bool(self.shape_kernel_dispatches)
        out.metadata["exposure_kernel_dispatches"] = self.exposure_kernel_dispatches
        out.metadata["exposure_kernel_backend"] = "cuda_strict" if self.exposure_kernel_dispatches else None
        out.metadata["exposure_kernel_no_fallback"] = bool(self.exposure_kernel_dispatches)
        return out

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
        counts: Dict[str, np.ndarray] = {}

        # shared intermediates (spec §9)
        rank_cache = {}
        pearson_cache = {}
        quantile_cache = {}
        turnover = None
        portfolio_pnl = None

        exposure_cache = {}
        for m in metrics:
            import inspect
            from quant_evaluator.registry.metrics import get_metric
            try:
                spec = get_metric(m)
            except KeyError:
                raise RuntimeError(f"GPUExecutor: unsupported metric '{m}'") from None
            parameters = {
                name: p.default for name, p in inspect.signature(spec.compute_fn).parameters.items()
                if p.default is not inspect.Parameter.empty
            }
            overrides = dict(self.metric_parameters.get(m, {}))
            if set(overrides) - set(parameters):
                raise ValueError(f"Unsupported GPU parameters for {m}: {set(overrides) - set(parameters)}")
            parameters.update(overrides)
            min_assets = parameters.get("min_assets", 20 if "ICSeriesArtifact" in (spec.requires or []) else 10)
            min_periods = parameters.get("min_periods", spec.min_periods or 1)
            if m in {"ic_ir", "pearson_ic_ir"}:
                if isinstance(min_periods, (bool, np.bool_)) or not isinstance(
                    min_periods, (int, np.integer)
                ):
                    raise TypeError("min_periods must be an integer")
                if min_periods < 2:
                    raise ValueError(
                        "min_periods must be at least 2 for sample standard deviation"
                    )
            if m in {"sharpe_ratio", "sortino_ratio", "win_rate", "max_drawdown", "calmar_ratio"}:
                if self.prebuilt_portfolio_pnl is not None and portfolio_pnl is None:
                    columns = [self._portfolio_column_map[fid] for fid in factor_ids]
                    portfolio_pnl = cp.asarray(self.prebuilt_portfolio_pnl[:, columns])
                if portfolio_pnl is None and (self.holding_return_id is None or self.portfolio_spec is None):
                    raise ValueError("GPU portfolio metrics require independent HoldingReturnPanel and PortfolioSpec; labels are forbidden")
                if portfolio_pnl is None:
                    from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
                    options = {k: v for k, v in self.portfolio_spec.to_dict().items() if k not in {"purpose", "missing_return_policy"}}
                    portfolio_pnl = compute_cohort_pnl_batch_gpu(
                        factors, self.session._staged_holding_returns[self.holding_return_id],
                        require_tradable=False, trade_eligibility=self.trade_eligibility, **options)["pnl_net"]
                    self.probe_tiles_processed += 1
                from quant_evaluator.kernels.gpu.drawdown import (
                    compute_sharpe_batch, compute_sortino_batch, compute_win_rate_batch,
                    compute_max_drawdown_batch, compute_calmar_batch,
                )
                fn = {"sharpe_ratio": compute_sharpe_batch, "sortino_ratio": compute_sortino_batch,
                      "win_rate": compute_win_rate_batch, "max_drawdown": compute_max_drawdown_batch,
                      "calmar_ratio": compute_calmar_batch}[m]
                accepted = inspect.signature(fn).parameters
                scalar[m] = _to_cpu(fn(portfolio_pnl, **{k: v for k, v in parameters.items() if k in accepted and k != "returns"}))
                counts[m] = _to_cpu(cp.isfinite(portfolio_pnl).sum(axis=0))
            elif m == "coverage":
                # coverage = fraction of PAIRWISE-FINITE (factor & label) per (t,f)
                yb = labels[:, None, :]  # (T,1,N)
                finite = cp.isfinite(factors) & cp.isfinite(yb)
                cov = cp.mean(finite, axis=2)  # (T,F)
                scalar["coverage"] = _to_cpu(cp.mean(cov, axis=0))  # (F,)
                counts[m] = _to_cpu(cp.sum(finite, axis=(0, 2)))
            elif m in ("rank_ic", "ic_ir", "ic_std", "ic_median", "rank_ic_series"):
                if min_assets not in rank_cache:
                    from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic
                    rank_cache[min_assets], _ = batched_spearman_ic(factors, labels, min_obs=min_assets)
                rank_ic = rank_cache[min_assets]
                count = cp.sum(cp.isfinite(rank_ic), axis=0)
                counts[m] = _to_cpu(count)
                if m == "rank_ic_series":
                    series["rank_ic_series"] = _to_cpu(rank_ic)
                elif m == "rank_ic":
                    scalar["rank_ic"] = _to_cpu(cp.where(count >= min_periods, cp.nanmean(rank_ic, axis=0), cp.nan))
                elif m == "ic_ir":
                    finite_ic = cp.where(cp.isfinite(rank_ic), rank_ic, cp.nan)
                    mu = cp.nanmean(finite_ic, axis=0)
                    sd = cp.nanstd(finite_ic, axis=0, ddof=1)
                    constant = cp.nanmax(finite_ic, axis=0) == cp.nanmin(finite_ic, axis=0)
                    scalar["ic_ir"] = _to_cpu(cp.where((count >= min_periods) & ~constant, mu / sd, cp.nan))
                elif m == "ic_std":
                    scalar["ic_std"] = _to_cpu(cp.where(count >= min_periods, cp.nanstd(rank_ic, axis=0, ddof=1), cp.nan))
                elif m == "ic_median":
                    scalar["ic_median"] = _to_cpu(cp.where(count >= min_periods, cp.nanmedian(rank_ic, axis=0), cp.nan))
            elif m in ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"):
                if min_assets not in pearson_cache:
                    from quant_evaluator.kernels.gpu.correlation import batched_pearson_ic
                    pearson_cache[min_assets], _ = batched_pearson_ic(factors, labels, min_obs=min_assets)
                pearson_ic = pearson_cache[min_assets]
                count = cp.sum(cp.isfinite(pearson_ic), axis=0)
                counts[m] = _to_cpu(count)
                if m == "pearson_ic_series":
                    series["pearson_ic_series"] = _to_cpu(pearson_ic)
                elif m == "pearson_ic":
                    scalar["pearson_ic"] = _to_cpu(cp.where(count >= min_periods, cp.nanmean(pearson_ic, axis=0), cp.nan))
                elif m == "pearson_ic_std":
                    scalar["pearson_ic_std"] = _to_cpu(cp.where(count >= min_periods, cp.nanstd(pearson_ic, axis=0, ddof=1), cp.nan))
                elif m == "pearson_ic_ir":
                    finite_ic = cp.where(cp.isfinite(pearson_ic), pearson_ic, cp.nan)
                    mu = cp.nanmean(finite_ic, axis=0)
                    sd = cp.nanstd(finite_ic, axis=0, ddof=1)
                    constant = cp.nanmax(finite_ic, axis=0) == cp.nanmin(finite_ic, axis=0)
                    scalar["pearson_ic_ir"] = _to_cpu(cp.where((count >= min_periods) & ~constant, mu / sd, cp.nan))
            elif m in ("quantile_returns_full", "quantile_returns_daily", "quantile_spread", "quantile_monotonicity",
                       "daily_quantile_monotonicity_series", "daily_quantile_monotonicity_rate"):
                n_quantiles = parameters.get("n_quantiles", 5)
                quantile_key = (n_quantiles, parameters.get("min_assets", 10))
                if quantile_key not in quantile_cache:
                    from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns
                    quantile_cache[quantile_key] = batched_quantile_returns(factors, labels,
                        n_quantiles=n_quantiles, min_assets=quantile_key[1], return_counts=True,
                        workspace_bytes=self.session.workspace_budget(
                            output_bytes=int(factors.shape[0] * factors.shape[1] * n_quantiles * 16)))
                quantile_ret, quantile_count = quantile_cache[quantile_key]
                if m == "quantile_returns_daily":
                    vector[m] = _to_cpu(quantile_ret)
                    counts[m] = _to_cpu(quantile_count)
                elif m == "quantile_returns_full":
                    valid = cp.sum(cp.isfinite(quantile_ret), axis=0)
                    vector[m] = _to_cpu(cp.where(valid >= min_periods, cp.nanmean(quantile_ret, axis=0), cp.nan))
                elif m == "quantile_spread":
                    # Qtop - Qbottom mean over time -> (F,)
                    spread = quantile_ret[:, -1, :] - quantile_ret[:, 0, :]
                    count = cp.sum(cp.isfinite(spread), axis=0)
                    scalar[m] = _to_cpu(cp.where(count >= min_periods, cp.nanmean(spread, axis=0), cp.nan))
                    counts[m] = _to_cpu(count)
                elif m == "quantile_monotonicity":
                    days = cp.isfinite(quantile_ret).sum(axis=0)
                    profile = cp.where(days >= min_periods, cp.nanmean(quantile_ret, axis=0), cp.nan)
                    valid_pairs = cp.isfinite(profile[:-1]) & cp.isfinite(profile[1:])
                    count = valid_pairs.sum(axis=0)
                    from quant_evaluator.kernels.gpu.quantile_shape import quantile_monotonicity
                    monotonicity = quantile_monotonicity(profile, return_device=True)
                    scalar[m] = _to_cpu(monotonicity)
                    counts[m] = _to_cpu(count)
                    self.shape_kernel_dispatches += 1
                elif m in ("daily_quantile_monotonicity_series", "daily_quantile_monotonicity_rate"):
                    from quant_evaluator.kernels.gpu.quantile_shape import daily_quantile_monotonicity
                    daily_score = daily_quantile_monotonicity(quantile_ret, return_device=True)
                    valid_days = cp.sum(cp.isfinite(daily_score), axis=0)
                    if m == "daily_quantile_monotonicity_series":
                        series[m] = _to_cpu(daily_score)
                        counts[m] = _to_cpu(cp.isfinite(daily_score).astype(cp.int64))
                    else:
                        mean_score = cp.nansum(daily_score, axis=0) / cp.maximum(valid_days, 1)
                        scalar[m] = _to_cpu(cp.where(valid_days >= min_periods, mean_score, cp.nan))
                        counts[m] = _to_cpu(valid_days)
                    self.shape_kernel_dispatches += 1
            elif m in ("turnover", "factor_turnover_rate"):
                if m == "turnover":
                    if turnover is None:
                        from quant_evaluator.kernels.gpu.turnover import batched_turnover
                        turnover = batched_turnover(factors, return_series=True)
                    count = cp.isfinite(turnover).sum(axis=0)
                    scalar[m] = _to_cpu(cp.where(count >= min_periods - 1, cp.nanmean(turnover, axis=0), cp.nan))
                elif m == "factor_turnover_rate":
                    from quant_evaluator.kernels.gpu.turnover import batched_membership_turnover
                    membership = batched_membership_turnover(factors, quantile=parameters.get("quantile", .9))
                    count = cp.isfinite(membership).sum(axis=0)
                    scalar[m] = _to_cpu(cp.where(count >= min_periods, cp.nanmean(membership, axis=0), cp.nan))
                counts[m] = _to_cpu(count)
            elif m in {
                "industry_exposure", "size_exposure", "beta_exposure",
                "liquidity_exposure", "volatility_exposure", "momentum_exposure",
                "max_absolute_style_exposure", "exposure_drift", "purity_ratio",
            }:
                if self.exposure_panel_values is None:
                    raise ValueError("CUDA exposure metrics require a validated ExposurePanel")
                from quant_evaluator.kernels.gpu.exposure import batched_factor_loadings

                key=parameters.get("min_obs",10)
                if key not in exposure_cache:
                    risk = cp.asarray(self.exposure_panel_values, dtype=cp.float64)
                    self.session._h2d_bytes += risk.nbytes
                    weights=None if self.exposure_regression_weights is None else cp.asarray(self.exposure_regression_weights)
                    if weights is not None: self.session._h2d_bytes += weights.nbytes
                    exposure_cache[key] = batched_factor_loadings(
                        factors, risk, intercept=True, min_obs=key, weights=weights, standardized=True,return_diagnostics=True)
                    ll,rr,_,ee=exposure_cache[key]
                    vector[f"_exposure_{key}_values"]=_to_cpu(cp.moveaxis(ll[:,:,1:],1,-1))
                    vector[f"_exposure_{key}_r_squared"]=_to_cpu(rr)
                    for field,ev in ee.items():
                        if field=="raw_loadings": ev=cp.moveaxis(ev[:,:,1:],1,-1)
                        vector[f"_exposure_{key}_{field}"]=_to_cpu(ev)
                loadings, r_squared, _, _ = exposure_cache[key]
                panel = loadings[:, :, 1:]  # (T,F,K), identical to CPU builder
                finite = cp.isfinite(panel)
                style_counts = cp.sum(finite, axis=0)  # (F,K)
                safe = cp.where(finite, panel, 0.0)
                signed = cp.sum(safe, axis=0) / cp.maximum(style_counts, 1)
                signed = cp.where(style_counts > 0, signed, cp.nan)
                absolute = cp.sum(cp.where(finite, cp.abs(panel), 0.0), axis=0) / cp.maximum(style_counts, 1)
                absolute = cp.where(style_counts > 0, absolute, cp.nan)
                if m.endswith("_exposure") and m != "max_absolute_style_exposure":
                    style = m.removesuffix("_exposure")
                    if style not in self.exposure_style_names:
                        value = cp.full(F, cp.nan, dtype=cp.float64)
                        count = cp.zeros(F, dtype=cp.int64)
                    else:
                        index = self.exposure_style_names.index(style)
                        value, count = signed[:, index], style_counts[:, index]
                elif m == "max_absolute_style_exposure":
                    min_finite = parameters.get("min_finite", 5)
                    eligible = style_counts >= min_finite
                    any_eligible = cp.any(eligible, axis=1)
                    index = cp.argmax(cp.where(eligible, absolute, -cp.inf), axis=1)
                    rows = cp.arange(F)
                    value = cp.where(any_eligible, absolute[rows, index], cp.nan)
                    count = cp.where(any_eligible, style_counts[rows, index], 0)
                elif m == "exposure_drift":
                    if T < 2:
                        value = cp.full(F, cp.nan, dtype=cp.float64)
                        count = cp.zeros(F, dtype=cp.int64)
                    else:
                        joint = finite[1:] & finite[:-1]
                        per_date_count = cp.sum(joint, axis=2)
                        per_date_sum = cp.sum(cp.where(joint, cp.abs(panel[1:] - panel[:-1]), 0.0), axis=2)
                        valid_dates = per_date_count > 0
                        count = cp.sum(valid_dates, axis=0)
                        value = cp.sum(cp.where(valid_dates, per_date_sum / cp.maximum(per_date_count, 1), 0.0), axis=0) / cp.maximum(count, 1)
                        value = cp.where(count > 0, value, cp.nan)
                else:  # purity_ratio
                    min_finite = parameters.get("min_finite", 5)
                    count=cp.isfinite(r_squared).sum(axis=0)
                    value=cp.where(count>=min_finite,cp.nansum(1.-r_squared,axis=0)/cp.maximum(count,1),cp.nan)
                scalar[m] = _to_cpu(value)
                counts[m] = _to_cpu(count)
                self.exposure_kernel_dispatches += 1
            else:
                raise RuntimeError(f"GPUExecutor: unsupported metric '{m}'")

        return BatchEvaluationBundle(
            factor_ids=tuple(factor_ids),
            label_id=label_id,
            scalar_metrics=scalar,
            series_metrics=series,
            vector_metrics=vector,
            metadata=self.session.metadata(),
            observation_counts=counts,
        )
