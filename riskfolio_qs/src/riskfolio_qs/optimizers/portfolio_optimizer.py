"""
组合优化器：根据 BenchmarkSpec 和 resolved_params 执行权重求解。

支持两类后端：
1. cvxpy 凸优化后端（riskfolio_backend）：
   - minvar_enhance_index：最小化主动风险（min w.T Σ w 的主动形式）
   - meanvar_enhance_index：主动收益-风险权衡
   - meanvar_absolute_return：绝对收益 mean-variance
   每期独立求解，包含预算、长仓、个股上限、主动权重带宽、换手等约束。

2. 规则型后端（rule_backend）：
   - topn_long_only_equal_weight：topN 等权做多
   - topn_long_short_equal_weight：topN 等权多空
   不依赖 cvxpy，直接按 alpha 排序分配权重。

协方差估计支持两种风险口径：
- barra_factor：使用 F/F_ret/F_spec 构建 Σ = FΣ_fF.T + D（带对角收缩）
- historical_cov：使用行情收益的历史协方差

关联文档：
- riskfolio_qs_v0.2_优化器映射规范.md
- riskfolio_qs_v0.2_纯净数学定义.md
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from ..adapters.output_adapter import OutputAdapter
from ..core.contracts import InputBundle, OutputBundle, RunMetadata
from ..smoothers.signal_smoother import SignalSmoother
from ..constraints.constraint_builder import ConstraintBuilder
from .benchmark_router import BenchmarkSpec


class OptimizationError(RuntimeError):
    """Raised when a convex portfolio solve cannot produce a feasible result."""


@dataclass(slots=True)
class SolveResult:
    """Structured result for one date's convex optimization solve."""

    weights: np.ndarray | None
    status: str
    solver_status: str
    error: str | None
    fallback_used: bool
    solve_time_ms: float
    max_constraint_violation: float


@dataclass(slots=True)
class PrecomputedFactorRisk:
    """Point-in-time Barra B/F/D snapshot kept in scalable factor form."""

    exposure: np.ndarray          # B: asset x factor
    factor_cov: np.ndarray        # F: factor x factor
    specific_var: np.ndarray      # D diagonal: asset
    factor_names: tuple[str, ...]
    exposure_date: pd.Timestamp
    covariance_date: pd.Timestamp
    specific_risk_date: pd.Timestamp

    def marginal_risk(self, weights: np.ndarray) -> np.ndarray:
        factor_exposure = self.exposure.T @ weights
        return (
            self.exposure @ (self.factor_cov @ factor_exposure)
            + self.specific_var * weights
        )

    def variance(self, weights: np.ndarray) -> float:
        return float(weights @ self.marginal_risk(weights))


@dataclass(slots=True)
class PortfolioOptimizer:
    """组合优化器：支持 cvxpy 凸优化和 rule 规则两种后端。

    Attributes:
        smoother: 信号平滑器引用
        output_adapter: 输出适配器引用
        constraint_builder: 约束构建器引用（v0.1 兼容）
    """
    smoother: SignalSmoother
    output_adapter: OutputAdapter
    constraint_builder: ConstraintBuilder

    # ---- 规则型后端 ----

    def _select_topn(self, row: pd.Series, n: int, ascending: bool = False) -> list[str]:
        """从一行 alpha 中选取前 n 大（或前 n 小）的资产代码。

        Args:
            row: 单行 alpha 值
            n: 选取数量
            ascending: True 取最小（做空选股），False 取最大（做多选股）

        Returns:
            选中的资产代码列表。
        """
        clean = row.dropna()
        if clean.empty:
            return []
        n = min(n, len(clean))
        return list(clean.nsmallest(n).index if ascending else clean.nlargest(n).index)

    def _equal_weight_frame(self, alpha: pd.DataFrame, n: int, budget: float = 1.0) -> pd.DataFrame:
        """topN 等权做多：每期选 alpha 最大的 n 只股票，等权分配。

        Args:
            alpha: alpha 信号矩阵
            n: 做多股票数

        Returns:
            等权权重矩阵。
        """
        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        for dt, row in alpha.iterrows():
            chosen = self._select_topn(row, n, ascending=False)
            if not chosen:
                continue
            weights.loc[dt, chosen] = budget / len(chosen)  # 等权分配
        return weights

    def _long_short_equal_weight_frame(
        self, alpha: pd.DataFrame,
        long_n: int, short_n: int,
        long_weight: float, short_weight: float,
    ) -> pd.DataFrame:
        """topN 等权多空：每期选 alpha 最大的 long_n 只做多，最小的 short_n 只做空。

        做多和做空分别赋予固定权重（不做等权归一化，以满足总敞口和净敞口约束）。

        Args:
            alpha: alpha 信号矩阵
            long_n: 做多股票数
            short_n: 做空股票数
            long_weight: 每只做多个股权重
            short_weight: 每只做空个股权重（通常为负值）

        Returns:
            多空权重矩阵。
        """
        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        for dt, row in alpha.iterrows():
            if int(row.notna().sum()) < long_n + short_n:
                raise ValueError(
                    f"Not enough distinct assets for long/short selection on {dt}: "
                    f"need {long_n + short_n}"
                )
            longs = self._select_topn(row, long_n, ascending=False)
            shorts = self._select_topn(row.drop(index=longs), short_n, ascending=True)
            if longs:
                weights.loc[dt, longs] = long_weight
            if shorts:
                weights.loc[dt, shorts] = short_weight
        return weights

    def _inverse_volatility_frame(self, market: pd.DataFrame, alpha: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """逆波动率加权（v0.1 占位逻辑，保留用于降级方案）。

        按滚动窗口波动率的倒数分配权重。

        Args:
            market: 行情价格矩阵
            alpha: alpha 信号矩阵（用于对齐 index/columns）
            window: 滚动窗口大小

        Returns:
            逆波动率权重矩阵。
        """
        returns = market.pct_change().fillna(0.0)
        inv_vol = 1.0 / returns.rolling(window=window, min_periods=5).std(ddof=0).replace(0.0, np.nan)
        inv_vol = inv_vol.bfill().ffill().fillna(1.0)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
        weights = weights.reindex(index=alpha.index, columns=alpha.columns, fill_value=0.0)
        return weights

    @staticmethod
    def _lagged_window(
        frame: pd.DataFrame,
        dt: pd.Timestamp,
        lookback: int,
        lag_periods: int,
    ) -> pd.DataFrame:
        """Return a point-in-time window ending at dt minus lag_periods rows."""
        history = frame.loc[:dt]
        if lag_periods > 0:
            history = history.iloc[:-lag_periods] if len(history) > lag_periods else history.iloc[:0]
        return history.tail(lookback)

    @staticmethod
    def _nearest_psd(matrix: np.ndarray, eigenvalue_floor: float) -> np.ndarray:
        """Symmetrize and project a covariance matrix onto the PSD cone."""
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        projected = (eigenvectors * eigenvalues) @ eigenvectors.T
        return 0.5 * (projected + projected.T)

    @staticmethod
    def _ewma_covariance(values: pd.DataFrame, halflife: float) -> np.ndarray:
        """Pairwise EWMA covariance that preserves missing observations."""
        array = values.to_numpy(dtype=float)
        mask = np.isfinite(array)
        if len(array) == 0:
            return np.empty((array.shape[1], array.shape[1]))
        ages = np.arange(len(array) - 1, -1, -1, dtype=float)
        weights = np.exp(np.log(0.5) * ages / max(halflife, 1e-12))
        weights /= weights.sum()
        safe = np.where(mask, array, 0.0)
        denominators = (weights[:, None] * mask).sum(axis=0)
        means = np.divide(
            (weights[:, None] * safe).sum(axis=0),
            denominators,
            out=np.zeros(array.shape[1], dtype=float),
            where=denominators > 0,
        )
        centered = np.where(mask, array - means, 0.0)
        numerator = centered.T @ (weights[:, None] * centered)
        pair_weight = mask.T.astype(float) @ (weights[:, None] * mask.astype(float))
        return np.divide(
            numerator,
            pair_weight,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=pair_weight > 0,
        )

    # ---- 协方差估计 ----

    @staticmethod
    def _risk_scale_to_alpha_horizon(bundle: InputBundle) -> float:
        """Convert the declared covariance unit to the alpha horizon."""
        context = bundle.metadata
        units = context.risk_covariance_units
        target_days = float(context.alpha_horizon_days)
        if units == "daily_variance":
            source_days = 1.0
        elif units == "horizon_variance":
            source_days = float(context.risk_horizon_days)
        elif units == "annual_variance":
            source_days = float(context.annualization_factor)
        else:
            raise ValueError(f"Unsupported risk covariance units: {units}")
        if source_days <= 0 or target_days <= 0:
            raise ValueError("risk and alpha horizons must be positive")
        return target_days / source_days

    def _build_covariance(
        self, bundle: InputBundle, benchmark_spec: BenchmarkSpec,
        dt: pd.Timestamp, assets: list[str], params: dict[str, Any],
        risk_covered_row: Optional[pd.Series] = None,
    ) -> np.ndarray | PrecomputedFactorRisk:
        """根据 risk_mode 构建协方差矩阵。

        支持两种口径：
        - barra_factor：Σ = (1-λ) * (Σ_f + D) + λ * diag_avg * I
          其中 Σ_f 从 F_ret 的协方差估计，D 从 F_spec 的方差估计，
          λ 为 shrinkage 强度。
        - historical_cov：直接从行情收益的滚动协方差估计。

        Args:
            bundle: 输入数据包
            benchmark_spec: 优化器映射描述
            dt: 当前日期
            assets: 与优化变量严格同序的资产列表
            params: 合并后的参数字典

        Returns:
            n_assets × n_assets 的协方差矩阵（已做对称化和正则化）。
        """
        n_assets = len(assets)
        if (
            benchmark_spec.risk_mode != "barra_precomputed"
            and bundle.metadata.risk_covariance_units != "daily_variance"
        ):
            raise ValueError(
                f"{benchmark_spec.risk_mode} estimates covariance from daily "
                "returns and requires risk_covariance_units=daily_variance"
            )

        # ---- 预计算 Barra B/F/D 口径：保持因子形式，禁止展开 N x N ----
        if benchmark_spec.risk_mode == "barra_precomputed":
            if bundle.F is None or bundle.factor_cov is None or bundle.specific_var is None:
                raise ValueError(
                    "barra_precomputed requires F, factor_cov and specific_var"
                )
            risk_lag = int(bundle.metadata.market_data_lag_periods)
            exposure_lag = int(bundle.metadata.exposure_data_lag_periods)

            cov_dates = pd.DatetimeIndex(
                bundle.factor_cov.index.get_level_values(0).unique()
            ).sort_values()
            eligible_cov_dates = cov_dates[cov_dates <= dt]
            if len(eligible_cov_dates) <= risk_lag:
                raise ValueError(f"factor_cov has no point-in-time value available on {dt}")
            covariance_dt = eligible_cov_dates[-(risk_lag + 1)]
            factor_cov_frame = bundle.factor_cov.xs(covariance_dt, level=0)
            factors = list(factor_cov_frame.columns)
            factor_cov_frame = factor_cov_frame.reindex(
                index=factors, columns=factors
            )
            if factor_cov_frame.isna().any().any():
                raise ValueError(f"factor_cov is incomplete on {covariance_dt}")
            factor_cov = factor_cov_frame.to_numpy(dtype=float)
            symmetry_error = float(np.max(np.abs(factor_cov - factor_cov.T)))
            if symmetry_error > float(params.get("precomputed_symmetry_tolerance", 1e-10)):
                raise ValueError(
                    f"factor_cov symmetry error {symmetry_error} on {covariance_dt}"
                )
            risk_scale = self._risk_scale_to_alpha_horizon(bundle)
            factor_cov = self._nearest_psd(
                factor_cov * risk_scale,
                float(params.get("precomputed_factor_eigenvalue_floor", 1e-12)),
            )

            exposure_dates = pd.DatetimeIndex(
                bundle.F.index.get_level_values(0).unique()
            ).sort_values()
            eligible_exposure_dates = exposure_dates[exposure_dates <= dt]
            if len(eligible_exposure_dates) <= exposure_lag:
                raise ValueError(f"F has no point-in-time exposure available on {dt}")
            exposure_dt = eligible_exposure_dates[-(exposure_lag + 1)]
            exposure_frame = bundle.F.xs(exposure_dt, level=0).reindex(
                index=assets, columns=factors
            )

            specific_dates = bundle.specific_var.index[
                bundle.specific_var.index <= dt
            ]
            if len(specific_dates) <= risk_lag:
                raise ValueError(
                    f"specific_var has no point-in-time value available on {dt}"
                )
            specific_dt = specific_dates[-(risk_lag + 1)]
            specific = bundle.specific_var.loc[specific_dt].reindex(assets)

            actual_covered = (
                exposure_frame.notna().all(axis=1)
                & pd.Series(
                    np.isfinite(exposure_frame.to_numpy(dtype=float)).all(axis=1),
                    index=assets,
                )
                & specific.notna()
                & pd.Series(
                    np.isfinite(specific.to_numpy(dtype=float)),
                    index=assets,
                )
                & specific.gt(0)
            )
            covered = (
                actual_covered
                if risk_covered_row is None
                else risk_covered_row.reindex(assets).fillna(False).astype(bool)
            )
            incorrectly_covered = covered & ~actual_covered
            if incorrectly_covered.any():
                missing = list(incorrectly_covered.index[incorrectly_covered])
                raise ValueError(
                    "risk_covered marks assets without complete B/D data as covered "
                    f"on {dt}: {missing[:5]}"
                )
            # Missing rows remain missing in InputBundle.  Numerical placeholders
            # are introduced only inside the solver and only for assets that are
            # simultaneously hard-constrained from opening a position.
            exposure_frame = exposure_frame.where(
                covered, 0.0
            )
            specific_floor = float(
                params.get("precomputed_specific_variance_floor", 1e-8)
            )
            specific_values = specific.where(
                covered, specific_floor / max(risk_scale, 1e-30)
            ).to_numpy(dtype=float)
            specific_values = np.maximum(
                specific_values * risk_scale,
                specific_floor,
            )
            return PrecomputedFactorRisk(
                exposure=exposure_frame.to_numpy(dtype=float),
                factor_cov=factor_cov,
                specific_var=specific_values,
                factor_names=tuple(factors),
                exposure_date=pd.Timestamp(exposure_dt),
                covariance_date=pd.Timestamp(covariance_dt),
                specific_risk_date=pd.Timestamp(specific_dt),
            )

        # ---- Barra 因子风险口径 ----
        if benchmark_spec.risk_mode == "barra_factor":
            if bundle.F is None or bundle.F_ret is None or bundle.F_spec is None:
                raise ValueError("barra_factor requires F, F_ret and F_spec")
            lookback_f = int(params.get("barra_cov_lookback_days", 252))
            lookback_s = int(params.get("specific_risk_lookback_days", 252))
            shrinkage = float(params.get("cov_shrinkage", 0.05))

            # 因子收益协方差 Σ_f（K×K）
            risk_lag = int(bundle.metadata.market_data_lag_periods)
            exposure_lag = int(bundle.metadata.exposure_data_lag_periods)
            f_ret = self._lagged_window(bundle.F_ret, dt, lookback_f, risk_lag)
            factors = list(f_ret.columns)
            if len(f_ret) < 2:
                # 冷启动先验：仅在无法估计样本协方差时使用对角阵，
                # 不伪造 N=K 的资产因子映射。
                sigma_f = np.eye(len(factors), dtype=float) * 1e-4
            else:
                sigma_f = f_ret.cov().reindex(index=factors, columns=factors).to_numpy(dtype=float)
                if not np.isfinite(sigma_f).all():
                    raise ValueError(f"Factor covariance contains non-finite values on {dt}")

            # F 使用 MultiIndex(date, asset) x factor，严格实现
            # Σ_sys = F_t @ Σ_f @ F_t.T。
            exposure_dates = pd.DatetimeIndex(
                bundle.F.index.get_level_values(0).unique()
            ).sort_values()
            eligible_dates = exposure_dates[exposure_dates <= dt]
            if len(eligible_dates) <= exposure_lag:
                raise ValueError(f"F has no point-in-time exposure available on {dt}")
            exposure_dt = eligible_dates[-(exposure_lag + 1)]
            exposure = bundle.F.xs(exposure_dt, level=0).reindex(index=assets, columns=factors)
            if exposure.isna().any().any():
                raise ValueError(f"F has missing asset/factor exposures on {dt}")
            f_matrix = exposure.to_numpy(dtype=float)
            sigma_sys = f_matrix @ sigma_f @ f_matrix.T

            # 特质风险对角阵 D（从 F_spec 方差估计）
            f_spec = self._lagged_window(
                bundle.F_spec.reindex(columns=assets), dt, lookback_s, risk_lag
            )
            if len(f_spec) >= 1 and f_spec.isna().all(axis=0).any():
                raise ValueError(f"Specific returns are missing assets on {dt}")
            if len(f_spec) < 2:
                spec_var = np.full(n_assets, 1e-4, dtype=float)
            else:
                spec_var = np.nan_to_num(f_spec.var(axis=0).to_numpy(dtype=float), nan=1e-6)
            spec_var = np.where(np.isfinite(spec_var), np.maximum(spec_var, 1e-8), 1e-6)
            d = np.diag(spec_var[:n_assets])
            sigma = sigma_sys + d

            # Ledoit-Wolf 型收缩：Σ_shrunk = (1-λ)Σ + λ * tr(Σ)/n * I
            diag_avg = float(np.trace(sigma) / max(1, n_assets))
            sigma = (1.0 - shrinkage) * sigma + shrinkage * np.eye(n_assets) * max(diag_avg, 1e-6)

            # 严格对称化 + 正则化，确保通过 cvxpy 的 Hermitian 检查
            eigen_floor = float(params.get("covariance_eigenvalue_floor", 1e-8))
            sigma = self._nearest_psd(sigma, eigen_floor)
            return sigma * self._risk_scale_to_alpha_horizon(bundle)

        if benchmark_spec.risk_mode != "historical_cov":
            raise ValueError(f"Unsupported risk_mode for convex optimizer: {benchmark_spec.risk_mode}")

        # ---- 历史协方差口径 ----
        market = bundle.market.reindex(columns=assets)
        if market.isna().all(axis=0).any():
            missing = list(market.columns[market.isna().all(axis=0)])
            raise ValueError(f"market has no observations for assets: {missing[:5]}")
        if bundle.metadata.market_input_type == "return":
            returns = market
        elif bundle.metadata.market_input_type == "price":
            returns = market.pct_change(fill_method=None)
        else:
            raise ValueError(
                "historical_cov requires metadata.market_input_type to be "
                "'price' or 'return'"
            )
        lookback = int(params.get("hist_cov_lookback_days", 60))
        risk_lag = int(bundle.metadata.market_data_lag_periods)
        win = self._lagged_window(returns, dt, lookback, risk_lag)
        variance_floor = float(params.get("covariance_variance_floor", 1e-6))
        if len(win) < 2:
            sigma = np.eye(n_assets, dtype=float) * max(variance_floor, 1e-4)
        else:
            method = str(params.get("hist_cov_method", "ewma"))
            min_observations = int(params.get("hist_cov_min_observations", 20))
            effective_min = min(min_observations, max(2, len(win)))
            if method == "ewma":
                sigma = self._ewma_covariance(
                    win, float(params.get("hist_cov_halflife_days", 20))
                )
            elif method == "sample":
                sigma = win.cov(min_periods=effective_min).to_numpy(dtype=float)
            else:
                raise ValueError(f"Unsupported hist_cov_method: {method}")

            observation_counts = win.notna().sum(axis=0).to_numpy(dtype=int)
            diagonal = np.diag(sigma).copy()
            reliable = (
                (observation_counts >= effective_min)
                & np.isfinite(diagonal)
                & (diagonal > variance_floor)
            )
            fallback_variance = (
                float(np.median(diagonal[reliable]))
                if reliable.any()
                else max(variance_floor, 1e-4)
            )
            sigma = np.where(np.isfinite(sigma), sigma, 0.0)
            for i in range(n_assets):
                if not reliable[i]:
                    sigma[i, :] = 0.0
                    sigma[:, i] = 0.0
                    sigma[i, i] = fallback_variance
        if sigma.shape != (n_assets, n_assets):
            raise ValueError(f"Historical covariance shape mismatch on {dt}: {sigma.shape}")
        shrinkage = float(params.get("cov_shrinkage", 0.05))
        # 严格对称化 + 正则化
        diag_avg = float(np.trace(sigma) / max(1, n_assets))
        sigma = (1.0 - shrinkage) * sigma + shrinkage * np.eye(n_assets) * max(diag_avg, 1e-6)
        eigen_floor = float(params.get("covariance_eigenvalue_floor", 1e-8))
        sigma = self._nearest_psd(sigma, eigen_floor)
        return sigma * self._risk_scale_to_alpha_horizon(bundle)

    @staticmethod
    def _risk_expression(
        weights: cp.Expression,
        risk_model: np.ndarray | PrecomputedFactorRisk,
    ) -> cp.Expression:
        """Build a convex quadratic without expanding precomputed factor risk."""
        if isinstance(risk_model, PrecomputedFactorRisk):
            active_factors = risk_model.exposure.T @ weights
            return (
                cp.quad_form(active_factors, cp.psd_wrap(risk_model.factor_cov))
                + cp.sum(
                    cp.multiply(risk_model.specific_var, cp.square(weights))
                )
            )
        return cp.quad_form(weights, cp.psd_wrap(risk_model))

    @staticmethod
    def _risk_variance_and_marginal(
        risk_model: np.ndarray | PrecomputedFactorRisk,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        if isinstance(risk_model, PrecomputedFactorRisk):
            marginal = risk_model.marginal_risk(weights)
        else:
            marginal = risk_model @ weights
        return max(0.0, float(weights @ marginal)), marginal

    # ---- cvxpy 凸优化求解 ----

    @staticmethod
    def _expected_return_vector(alpha_row: pd.Series, params: dict[str, Any]) -> np.ndarray:
        """Convert a score or expected-return row into horizon return units."""
        raw = alpha_row.to_numpy(dtype=float)
        valid = np.isfinite(raw)
        input_type = str(params.get("alpha_input_type", "score"))
        if input_type == "expected_return":
            return np.where(valid, raw, 0.0)
        if input_type != "score":
            raise ValueError(f"Unsupported alpha_input_type: {input_type}")
        if not valid.any():
            return np.zeros_like(raw)
        mean = float(np.mean(raw[valid]))
        std = float(np.std(raw[valid]))
        standardized = np.zeros_like(raw)
        if std > 1e-12:
            standardized[valid] = (raw[valid] - mean) / std
        scale = float(params.get("score_return_scale_bps", 10.0)) * 1e-4
        return standardized * scale

    @staticmethod
    def _cost_vectors(
        assets: list[str],
        linear_cost_row: Optional[pd.Series],
        impact_cost_row: Optional[pd.Series],
        params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build non-negative per-asset linear and quadratic cost vectors."""
        default_bps = float(params.get("default_linear_cost_bps", 0.0))
        default_impact = float(params.get("default_impact_cost", 0.0))
        if linear_cost_row is None:
            linear = np.full(len(assets), default_bps, dtype=float)
        else:
            linear = linear_cost_row.reindex(assets).fillna(default_bps).to_numpy(dtype=float)
        if impact_cost_row is None:
            impact = np.full(len(assets), default_impact, dtype=float)
        else:
            impact = impact_cost_row.reindex(assets).fillna(default_impact).to_numpy(dtype=float)
        if (linear < 0).any() or (impact < 0).any():
            raise ValueError("transaction cost coefficients must be non-negative")
        return linear * 1e-4, impact

    def _solver_name(self, params: dict[str, Any], benchmark_spec: BenchmarkSpec) -> str:
        """获取求解器名称（大写，用于 cvxpy 的 getattr）。"""
        return str(params.get("solver_name", benchmark_spec.solver_default or "CLARABEL")).upper()

    @staticmethod
    def _solver_kwargs(solver_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Map shared solver settings to cvxpy solver-specific keyword names."""
        max_iters = int(params.get("max_iters", 5000))
        tolerance = float(params.get("solver_tol", 1e-6))
        if solver_name == "CLARABEL":
            return {
                "max_iter": max_iters,
                "tol_gap_abs": tolerance,
                "tol_gap_rel": tolerance,
                "tol_feas": tolerance,
            }
        if solver_name == "SCS":
            return {"max_iters": max_iters, "eps": tolerance}
        if solver_name == "OSQP":
            return {"max_iter": max_iters, "eps_abs": tolerance, "eps_rel": tolerance}
        if solver_name == "ECOS":
            return {
                "max_iters": max_iters,
                "abstol": tolerance,
                "reltol": tolerance,
                "feastol": tolerance,
            }
        return {}

    @staticmethod
    def _constraint_violation(
        weights: np.ndarray,
        previous: np.ndarray,
        *,
        budget: float,
        single_name_max: float,
        turnover_cap: float | None,
        enforce_turnover: bool,
    ) -> float:
        """Return the largest violation of the core long-only constraints."""
        violations = [
            abs(float(weights.sum()) - budget),
            max(0.0, -float(weights.min(initial=0.0))),
            max(0.0, float(weights.max(initial=0.0)) - single_name_max),
        ]
        if enforce_turnover and turnover_cap is not None:
            turnover = 0.5 * float(np.abs(weights - previous).sum())
            violations.append(max(0.0, turnover - turnover_cap))
        return max(violations)

    @staticmethod
    def _constraint_diagnostics(
        weights: np.ndarray,
        previous: np.ndarray,
        benchmark: np.ndarray,
        assets: list[str],
        params: dict[str, Any],
        risk_model: np.ndarray | PrecomputedFactorRisk | None = None,
        *,
        G_row: Optional[pd.Series] = None,
        F_mcap_row: Optional[pd.Series] = None,
        style_row: Optional[pd.Series] = None,
        tradable_row: Optional[pd.Series] = None,
        eligible_row: Optional[pd.Series] = None,
        enforce_turnover: bool = True,
    ) -> dict[str, float]:
        """Evaluate every enabled hard constraint on final, returned weights."""
        budget = float(params.get("budget", 1.0))
        cap = float(params.get("single_name_max", 1.0))
        values: dict[str, float] = {
            "budget_violation": abs(float(weights.sum()) - budget),
            "long_only_violation": max(0.0, -float(weights.min(initial=0.0))),
            "single_name_violation": max(0.0, float(weights.max(initial=0.0)) - cap),
        }
        gross_cap = params.get("gross_exposure_cap")
        values["gross_exposure_violation"] = (
            max(0.0, float(np.abs(weights).sum()) - float(gross_cap))
            if gross_cap is not None
            else 0.0
        )
        active_cap = params.get("active_weight_abs_max")
        values["active_weight_violation"] = (
            max(0.0, float(np.max(np.abs(weights - benchmark))) - float(active_cap))
            if active_cap is not None and benchmark.size
            else 0.0
        )
        turnover_cap = params.get("turnover_cap")
        turnover = 0.5 * float(np.abs(weights - previous).sum())
        values["turnover_violation"] = (
            max(0.0, turnover - float(turnover_cap))
            if enforce_turnover and turnover_cap is not None
            else 0.0
        )
        values["tracking_error_cap_violation"] = 0.0
        tracking_error_cap_slack = float("nan")
        if (
            bool(params.get("enforce_tracking_error_cap", False))
            and params.get("tracking_error_cap_annual") is not None
            and risk_model is not None
        ):
            cap_annual = float(params["tracking_error_cap_annual"])
            horizon_days = float(params.get("alpha_horizon_days", 1.0))
            annualization = float(params.get("annualization_factor", 252.0))
            active = weights - benchmark
            variance, _ = PortfolioOptimizer._risk_variance_and_marginal(
                risk_model, active
            )
            annual_te = float(np.sqrt(variance * annualization / horizon_days))
            values["tracking_error_cap_violation"] = max(
                0.0, annual_te - cap_annual
            )
            tracking_error_cap_slack = cap_annual - annual_te
        values["tradable_violation"] = 0.0
        if bool(params.get("enforce_tradable_mask", False)) and tradable_row is not None:
            tradable = tradable_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
            if (~tradable).any():
                values["tradable_violation"] = float(
                    np.max(np.abs((weights - previous)[~tradable]))
                )
        values["eligibility_violation"] = 0.0
        if eligible_row is not None:
            eligible = eligible_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
            tradable = (
                tradable_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
                if tradable_row is not None
                else np.ones(len(assets), dtype=bool)
            )
            forced_zero = tradable & ~eligible
            if forced_zero.any():
                values["eligibility_violation"] = float(
                    np.max(np.abs(weights[forced_zero]))
                )

        values["industry_violation"] = 0.0
        industry_mode = str(params.get("industry_neutral_mode", "off"))
        if industry_mode != "off" and G_row is not None:
            industries = G_row.reindex(assets).fillna(-1).to_numpy(dtype=float)
            exposures = []
            for industry in np.unique(industries[industries >= 0]):
                mask = industries == industry
                exposures.append(abs(float((weights - benchmark)[mask].sum())))
            max_exposure = max(exposures, default=0.0)
            band = 0.0 if industry_mode == "strict" else float(params.get("industry_band", 0.02))
            values["industry_violation"] = max(0.0, max_exposure - band)

        values["mcap_violation"] = 0.0
        mcap_mode = str(params.get("mcap_neutral_mode", "off"))
        if mcap_mode != "off" and F_mcap_row is not None:
            market_cap = F_mcap_row.reindex(assets).to_numpy(dtype=float)
            valid = np.isfinite(market_cap) & (market_cap > 0)
            if valid.any():
                market_cap = np.where(valid, market_cap, float(np.median(market_cap[valid])))
                exposure = np.log(market_cap)
                std = float(exposure.std())
                exposure = (exposure - exposure.mean()) / std if std > 0 else np.zeros_like(exposure)
                active = abs(float(exposure @ (weights - benchmark)))
                band = 0.0 if mcap_mode == "strict" else float(params.get("mcap_band", 0.10))
                values["mcap_violation"] = max(0.0, active - band)

        values["style_violation"] = 0.0
        style_mode = str(params.get("style_neutral_mode", "off"))
        if style_mode != "off" and style_row is not None and isinstance(style_row.index, pd.MultiIndex):
            active_exposures = []
            for factor in style_row.index.get_level_values(0).unique():
                exposure = style_row.xs(factor, level=0).reindex(assets).to_numpy(dtype=float)
                active_exposures.append(abs(float(exposure @ (weights - benchmark))))
            max_exposure = max(active_exposures, default=0.0)
            band = 0.0 if style_mode == "strict" else float(params.get("style_band", 0.10))
            values["style_violation"] = max(0.0, max_exposure - band)

        values["max_constraint_violation"] = max(values.values(), default=0.0)
        values["tracking_error_cap_slack"] = tracking_error_cap_slack
        values["long_only_min_slack"] = float(np.min(weights)) if len(weights) else float("nan")
        values["single_name_min_slack"] = (
            float(np.min(cap - weights)) if len(weights) else float("nan")
        )
        values["gross_exposure_slack"] = (
            float(gross_cap) - float(np.abs(weights).sum())
            if gross_cap is not None
            else float("nan")
        )
        values["active_weight_min_slack"] = (
            float(active_cap) - float(np.max(np.abs(weights - benchmark)))
            if active_cap is not None and benchmark.size
            else float("nan")
        )
        values["turnover_slack"] = (
            float(turnover_cap) - turnover
            if enforce_turnover and turnover_cap is not None
            else float("nan")
        )
        return values

    @staticmethod
    def _precheck_feasibility(
        benchmark: np.ndarray,
        previous: np.ndarray,
        assets: list[str],
        params: dict[str, Any],
        tradable_row: Optional[pd.Series],
        eligible_row: Optional[pd.Series] = None,
    ) -> None:
        """Fail early for deterministic box/budget/frozen-position conflicts."""
        budget = float(params.get("budget", 1.0))
        cap = float(params.get("single_name_max", 1.0))
        lower = np.zeros(len(assets), dtype=float)
        upper = np.full(len(assets), cap, dtype=float)
        active_cap = params.get("active_weight_abs_max")
        if active_cap is not None and benchmark.size:
            active_cap = float(active_cap)
            lower = np.maximum(lower, benchmark - active_cap)
            upper = np.minimum(upper, benchmark + active_cap)
        if bool(params.get("enforce_tradable_mask", False)) and tradable_row is not None:
            tradable = tradable_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
            lower[~tradable] = previous[~tradable]
            upper[~tradable] = previous[~tradable]
        else:
            tradable = np.ones(len(assets), dtype=bool)
        if eligible_row is not None:
            eligible = eligible_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
            forced_zero = tradable & ~eligible
            lower[forced_zero] = 0.0
            upper[forced_zero] = 0.0
        errors = []
        if (lower > upper + 1e-12).any():
            errors.append("per-asset lower bounds exceed upper bounds")
        if float(lower.sum()) > budget + 1e-12:
            errors.append(f"minimum feasible weight sum {lower.sum():.6f} exceeds budget {budget}")
        if float(upper.sum()) < budget - 1e-12:
            errors.append(f"maximum feasible weight sum {upper.sum():.6f} is below budget {budget}")
        if errors:
            raise ValueError("Pre-solve infeasible: " + "; ".join(errors))

    @staticmethod
    def _active_exposure_diagnostics(
        weights: np.ndarray,
        benchmark: np.ndarray,
        assets: list[str],
        *,
        G_row: Optional[pd.Series],
        F_mcap_row: Optional[pd.Series],
        style_row: Optional[pd.Series],
    ) -> dict[str, float]:
        """Return actual active exposures, independently of whether caps are enabled."""
        active = weights - benchmark
        result = {
            "max_abs_industry_active_exposure": float("nan"),
            "abs_mcap_active_exposure": float("nan"),
            "max_abs_style_active_exposure": float("nan"),
        }
        if G_row is not None:
            industries = pd.to_numeric(G_row.reindex(assets), errors="coerce").to_numpy()
            exposures = [
                abs(float(active[industries == industry].sum()))
                for industry in np.unique(industries[np.isfinite(industries) & (industries >= 0)])
            ]
            result["max_abs_industry_active_exposure"] = max(exposures, default=0.0)
        if F_mcap_row is not None:
            market_cap = F_mcap_row.reindex(assets).to_numpy(dtype=float)
            valid = np.isfinite(market_cap) & (market_cap > 0)
            if valid.any():
                market_cap = np.where(valid, market_cap, float(np.median(market_cap[valid])))
                exposure = np.log(market_cap)
                std = float(exposure.std())
                exposure = (exposure - exposure.mean()) / std if std > 0 else np.zeros_like(exposure)
                result["abs_mcap_active_exposure"] = abs(float(exposure @ active))
        if style_row is not None and isinstance(style_row.index, pd.MultiIndex):
            exposures = []
            for factor in style_row.index.get_level_values(0).unique():
                factor_values = style_row.xs(factor, level=0).reindex(assets).to_numpy(dtype=float)
                if np.isfinite(factor_values).all():
                    exposures.append(abs(float(factor_values @ active)))
            result["max_abs_style_active_exposure"] = max(exposures, default=0.0)
        return result

    def _solve_convex_result(
        self,
        alpha_row: pd.Series,
        benchmark_row: Optional[pd.Series],
        prev_row: Optional[pd.Series],
        sigma: np.ndarray | PrecomputedFactorRisk,
        objective_id: str,
        params: dict[str, Any],
        benchmark_spec: BenchmarkSpec,
        G_row: Optional[pd.Series] = None,
        F_mcap_row: Optional[pd.Series] = None,
        style_row: Optional[pd.Series] = None,
        tradable_row: Optional[pd.Series] = None,
        eligible_row: Optional[pd.Series] = None,
        linear_cost_row: Optional[pd.Series] = None,
        impact_cost_row: Optional[pd.Series] = None,
        *,
        enforce_turnover: bool = True,
    ) -> SolveResult:
        """Solve one date and return weights together with auditable diagnostics."""
        assets = list(alpha_row.index)
        n = len(assets)
        mu = self._expected_return_vector(alpha_row, params)
        b = (
            benchmark_row.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            if benchmark_row is not None
            else np.zeros(n, dtype=float)
        )
        previous = (
            prev_row.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            if prev_row is not None
            else np.zeros(n, dtype=float)
        )

        w = cp.Variable(n)
        budget = float(params.get("budget", 1.0))
        single_name_max = float(params.get("single_name_max", 0.05))
        turnover_cap_value = params.get("turnover_cap")
        turnover_cap = float(turnover_cap_value) if turnover_cap_value is not None else None
        turnover_penalty = float(params.get("turnover_penalty", 1.0))
        linear_cost_penalty = float(params.get("linear_cost_penalty", 1.0))
        impact_cost_penalty = float(params.get("impact_cost_penalty", 0.0))
        enable_impact_cost = bool(params.get("enable_impact_cost", False))
        soft_cost_scale = float(params.get("soft_cost_scale", 0.05))
        alpha_weight = float(params.get("alpha_weight", 1.0))
        risk_weight = float(params.get("risk_weight", 1.0))
        linear_cost, impact_cost = self._cost_vectors(
            assets, linear_cost_row, impact_cost_row, params
        )

        if not bool(params.get("long_only", True)):
            raise ValueError("long_only=false is not supported without explicit lower bounds")
        if budget <= 0 or single_name_max <= 0:
            raise ValueError("budget and single_name_max must be positive")
        if n * single_name_max + 1e-12 < budget:
            raise ValueError(
                f"Infeasible position cap: {n} assets * {single_name_max} < budget {budget}"
            )
        self._precheck_feasibility(
            b, previous, assets, params, tradable_row, eligible_row
        )

        constraints = [cp.sum(w) == budget, w >= 0, w <= single_name_max]
        gross_exposure_cap = params.get("gross_exposure_cap")
        if gross_exposure_cap is not None:
            gross_cap = float(gross_exposure_cap)
            if gross_cap + 1e-12 < budget:
                raise ValueError(f"gross_exposure_cap {gross_cap} is below budget {budget}")
            constraints.append(cp.norm1(w) <= gross_cap)
        active_weight_abs_max = params.get("active_weight_abs_max")
        if active_weight_abs_max is not None and benchmark_row is not None:
            cap = float(active_weight_abs_max)
            constraints.extend([w - b <= cap, w - b >= -cap])
        if enforce_turnover and turnover_cap is not None and prev_row is not None:
            constraints.append(0.5 * cp.norm1(w - previous) <= turnover_cap)

        if bool(params.get("enforce_tradable_mask", False)):
            if tradable_row is None:
                raise ValueError("enforce_tradable_mask=true but tradable data is missing for date")
            tradable = tradable_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
            frozen = (~tradable).astype(float)
            if frozen.any():
                constraints.append(cp.multiply(frozen, w - previous) == 0)
        else:
            tradable = np.ones(n, dtype=bool)
        if eligible_row is not None:
            eligible = eligible_row.reindex(assets).fillna(False).to_numpy(dtype=bool)
            forced_zero = (tradable & ~eligible).astype(float)
            if forced_zero.any():
                constraints.append(cp.multiply(forced_zero, w) == 0)

        industry_mode = str(params.get("industry_neutral_mode", "off"))
        if industry_mode != "off":
            if industry_mode not in {"strict", "band"}:
                raise ValueError(f"Unknown industry_neutral_mode: {industry_mode}")
            if G_row is None or benchmark_row is None:
                raise ValueError("industry neutrality is enabled but G/benchmark is missing")
            aligned_industries = pd.to_numeric(G_row.reindex(assets), errors="coerce")
            coverage = float((aligned_industries.notna() & aligned_industries.ge(0)).mean())
            min_coverage = float(params.get("min_exposure_coverage", 0.95))
            if coverage < min_coverage:
                raise ValueError(
                    f"industry exposure coverage {coverage:.2%} is below {min_coverage:.2%}"
                )
            industries = aligned_industries.fillna(-1).to_numpy(dtype=float)
            for industry in np.unique(industries[industries >= 0]):
                mask = (industries == industry).astype(float)
                if mask.sum() == 0:
                    continue
                active_exposure = mask @ (w - b)
                if industry_mode == "strict":
                    constraints.append(active_exposure == 0)
                elif industry_mode == "band":
                    band = float(params.get("industry_band", 0.02))
                    constraints.extend([active_exposure >= -band, active_exposure <= band])

        mcap_mode = str(params.get("mcap_neutral_mode", "off"))
        if mcap_mode != "off":
            if mcap_mode not in {"strict", "band"}:
                raise ValueError(f"Unknown mcap_neutral_mode: {mcap_mode}")
            if F_mcap_row is None or benchmark_row is None:
                raise ValueError("mcap neutrality is enabled but F_mcap/benchmark is missing")
            market_cap = F_mcap_row.reindex(assets).to_numpy(dtype=float)
            valid = np.isfinite(market_cap) & (market_cap > 0)
            coverage = float(valid.mean())
            min_coverage = float(params.get("min_exposure_coverage", 0.95))
            if coverage < min_coverage:
                raise ValueError(
                    f"market-cap exposure coverage {coverage:.2%} is below {min_coverage:.2%}"
                )
            if valid.any():
                fill_value = float(np.median(market_cap[valid]))
                market_cap = np.where(valid, market_cap, fill_value)
                exposure = np.log(market_cap)
                exposure_std = float(exposure.std())
                if exposure_std > 0:
                    exposure = (exposure - exposure.mean()) / exposure_std
                else:
                    exposure = np.zeros_like(exposure)
                active_mcap = exposure @ (w - b)
                if mcap_mode == "strict":
                    constraints.append(active_mcap == 0)
                elif mcap_mode == "band":
                    band = float(params.get("mcap_band", 0.10))
                    constraints.extend([active_mcap >= -band, active_mcap <= band])

        style_mode = str(params.get("style_neutral_mode", "off"))
        if style_mode != "off" and benchmark_row is not None:
            if style_row is None or not isinstance(style_row.index, pd.MultiIndex):
                raise ValueError("style neutrality is enabled but style exposure is missing or malformed")
            for factor in style_row.index.get_level_values(0).unique():
                exposure = style_row.xs(factor, level=0).reindex(assets)
                if exposure.isna().any():
                    raise ValueError(f"style factor {factor} has missing exposures")
                active_style = exposure.to_numpy(dtype=float) @ (w - b)
                if style_mode == "strict":
                    constraints.append(active_style == 0)
                elif style_mode == "band":
                    band = float(params.get("style_band", 0.10))
                    constraints.extend([active_style >= -band, active_style <= band])
                else:
                    raise ValueError(f"Unknown style_neutral_mode: {style_mode}")

        if bool(params.get("enforce_tracking_error_cap", False)):
            if benchmark_row is None or objective_id not in {
                "OBJ_MINVAR_ACTIVE",
                "OBJ_MEANVAR_ACTIVE",
            }:
                raise ValueError(
                    "tracking-error cap is only supported for active objectives "
                    "with a benchmark"
                )
            cap_value = params.get("tracking_error_cap_annual")
            if cap_value is None or float(cap_value) <= 0:
                raise ValueError(
                    "enforce_tracking_error_cap=true requires "
                    "tracking_error_cap_annual > 0"
                )
            horizon_days = float(params.get("alpha_horizon_days", 1.0))
            annualization = float(params.get("annualization_factor", 252.0))
            cap_horizon_variance = (
                float(cap_value) ** 2 * horizon_days / annualization
            )
            constraints.append(
                self._risk_expression(w - b, sigma) <= cap_horizon_variance
            )

        trade_delta = w - previous
        soft_cost = (
            soft_cost_scale * turnover_penalty * 0.5 * cp.norm1(trade_delta)
            + linear_cost_penalty * cp.sum(cp.multiply(linear_cost, cp.abs(trade_delta)))
        )
        if enable_impact_cost:
            if impact_cost_penalty <= 0:
                raise ValueError("enable_impact_cost=true requires impact_cost_penalty > 0")
            soft_cost += impact_cost_penalty * cp.sum(
                cp.multiply(impact_cost, cp.square(trade_delta))
            )

        if objective_id == "OBJ_MINVAR_ACTIVE":
            objective = cp.Minimize(
                self._risk_expression(w - b, sigma)
                + soft_cost
            )
        elif objective_id == "OBJ_MEANVAR_ACTIVE":
            objective = cp.Minimize(
                risk_weight * self._risk_expression(w - b, sigma)
                - alpha_weight * (mu @ w)
                + soft_cost
            )
        elif objective_id == "OBJ_MEANVAR_ABS":
            objective = cp.Minimize(
                risk_weight * self._risk_expression(w, sigma)
                - alpha_weight * (mu @ w)
                + soft_cost
            )
        else:
            raise ValueError(f"Unsupported objective_id: {objective_id}")

        problem = cp.Problem(objective, constraints)
        solver_name = self._solver_name(params, benchmark_spec)
        if (
            bool(params.get("enforce_tracking_error_cap", False))
            and solver_name == "OSQP"
        ):
            raise ValueError(
                "OSQP cannot solve the quadratic tracking-error constraint; "
                "use CLARABEL or SCS"
            )
        solver = getattr(cp, solver_name, None)
        if solver is None:
            return SolveResult(
                weights=None,
                status="failed",
                solver_status="unknown_solver",
                error=f"Unknown cvxpy solver: {solver_name}",
                fallback_used=False,
                solve_time_ms=0.0,
                max_constraint_violation=float("inf"),
            )

        started = time.perf_counter()
        try:
            problem.solve(
                solver=solver,
                verbose=False,
                **self._solver_kwargs(solver_name, params),
            )
        except Exception as exc:
            return SolveResult(
                weights=None,
                status="failed",
                solver_status="exception",
                error=f"{type(exc).__name__}: {exc}",
                fallback_used=False,
                solve_time_ms=(time.perf_counter() - started) * 1000,
                max_constraint_violation=float("inf"),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        solver_status = str(problem.status or "unknown")
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
            return SolveResult(
                weights=None,
                status="failed",
                solver_status=solver_status,
                error=f"Solver did not return an optimal solution: {solver_status}",
                fallback_used=False,
                solve_time_ms=elapsed_ms,
                max_constraint_violation=float("inf"),
            )

        raw = np.asarray(w.value, dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        # Keep the raw-solution sanity check consistent with the final
        # post-normalization constraint check below.  On broad universes,
        # tiny per-asset solver residuals can accumulate past 1e-6 in the
        # budget sum even though normalization produces a feasible portfolio.
        numeric_tolerance = max(
            1e-5, float(params.get("solver_tol", 1e-6)) * 10
        )
        if float(raw.min(initial=0.0)) < -numeric_tolerance:
            return SolveResult(
                weights=None,
                status="failed",
                solver_status=solver_status,
                error=f"Solution contains negative weights below tolerance: {raw.min()}",
                fallback_used=False,
                solve_time_ms=elapsed_ms,
                max_constraint_violation=abs(float(raw.min())),
            )
        raw[raw < 0] = 0.0
        total = float(raw.sum())
        if total <= 0 or abs(total - budget) > numeric_tolerance:
            return SolveResult(
                weights=None,
                status="failed",
                solver_status=solver_status,
                error=f"Solution budget mismatch: expected {budget}, got {total}",
                fallback_used=False,
                solve_time_ms=elapsed_ms,
                max_constraint_violation=abs(total - budget),
            )
        weights = raw * (budget / total)
        constraint_diagnostics = self._constraint_diagnostics(
            weights,
            previous,
            b,
            assets,
            params,
            sigma,
            G_row=G_row,
            F_mcap_row=F_mcap_row,
            style_row=style_row,
            tradable_row=tradable_row,
            eligible_row=eligible_row,
            enforce_turnover=enforce_turnover,
        )
        violation = constraint_diagnostics["max_constraint_violation"]
        # CLARABEL can leave roughly 1e-5 absolute residuals on the 5k-name
        # problem even with solver_tol=1e-7.  Accept up to 0.2 bp of portfolio
        # weight while retaining the exact diagnostic value for audit.
        constraint_tolerance = max(2e-5, numeric_tolerance)
        if violation > constraint_tolerance:
            violation_fields = {
                key: value
                for key, value in constraint_diagnostics.items()
                if key.endswith("_violation") and key != "max_constraint_violation"
            }
            worst_name, worst_value = max(
                violation_fields.items(), key=lambda item: item[1]
            )
            return SolveResult(
                weights=None,
                status="failed",
                solver_status=solver_status,
                error=(
                    f"Post-solve constraint violation: {violation} "
                    f"({worst_name}={worst_value})"
                ),
                fallback_used=False,
                solve_time_ms=elapsed_ms,
                max_constraint_violation=violation,
            )
        return SolveResult(
            weights=weights,
            status="optimal" if problem.status == cp.OPTIMAL else "optimal_inaccurate",
            solver_status=solver_status,
            error=None,
            fallback_used=False,
            solve_time_ms=elapsed_ms,
            max_constraint_violation=violation,
        )

    def _solve_convex_weights(
        self,
        alpha_row: pd.Series,
        benchmark_row: Optional[pd.Series],
        prev_row: Optional[pd.Series],
        sigma: np.ndarray | PrecomputedFactorRisk,
        objective_id: str,
        params: dict[str, Any],
        benchmark_spec: BenchmarkSpec,
        G_row: Optional[pd.Series] = None,
        F_mcap_row: Optional[pd.Series] = None,
        style_row: Optional[pd.Series] = None,
        tradable_row: Optional[pd.Series] = None,
        eligible_row: Optional[pd.Series] = None,
        linear_cost_row: Optional[pd.Series] = None,
        impact_cost_row: Optional[pd.Series] = None,
        *,
        enforce_turnover: bool = True,
    ) -> np.ndarray:
        """Compatibility wrapper that raises instead of silently returning equal weight."""
        result = self._solve_convex_result(
            alpha_row=alpha_row,
            benchmark_row=benchmark_row,
            prev_row=prev_row,
            sigma=sigma,
            objective_id=objective_id,
            params=params,
            benchmark_spec=benchmark_spec,
            G_row=G_row,
            F_mcap_row=F_mcap_row,
            style_row=style_row,
            tradable_row=tradable_row,
            eligible_row=eligible_row,
            linear_cost_row=linear_cost_row,
            impact_cost_row=impact_cost_row,
            enforce_turnover=enforce_turnover,
        )
        if result.weights is None:
            raise OptimizationError(result.error or "Convex optimization failed")
        return result.weights

    def _convex_optimize_frame(
        self, bundle: InputBundle, alpha: pd.DataFrame,
        benchmark_spec: BenchmarkSpec, params: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """对全时段逐日执行凸优化，返回权重矩阵。

        遍历 alpha 的每个日期行，构建当日协方差矩阵后调用 _solve_convex_weights。
        同时传入 G 和 F_mcap 用于行业/市值中性约束。

        Args:
            bundle: 输入数据包
            alpha: alpha 信号矩阵（可能已平滑）
            benchmark_spec: 优化器映射描述
            params: 合并后的参数字典

        Returns:
            权重矩阵（行=日期，列=资产代码）。
        """
        params = dict(params)
        params["alpha_input_type"] = bundle.metadata.alpha_input_type
        params["alpha_horizon_days"] = bundle.metadata.alpha_horizon_days
        params["annualization_factor"] = bundle.metadata.annualization_factor
        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        diagnostic_rows: list[dict[str, Any]] = []
        first_date = alpha.index[0]
        if bundle.prev_positions is not None:
            eligible_positions = bundle.prev_positions.index[
                bundle.prev_positions.index <= first_date
            ]
        else:
            eligible_positions = pd.DatetimeIndex([])
        if len(eligible_positions):
            initial_positions = bundle.prev_positions.loc[
                eligible_positions[-1]
            ].reindex(alpha.columns)
            coverage = float(initial_positions.notna().mean())
            min_coverage = float(params.get("min_prev_positions_coverage", 1.0))
            if coverage < min_coverage:
                raise ValueError(
                    f"initial position coverage is {coverage:.2%}, below {min_coverage:.2%}"
                )
            previous_weights = initial_positions.fillna(0.0)
        else:
            previous_weights = pd.Series(0.0, index=alpha.columns, dtype=float)

        for position, (dt, row) in enumerate(alpha.iterrows()):
            # 获取当日基准权重、上期持仓、行业暴露、市值暴露
            benchmark_row = bundle.benchmark.loc[dt] if bundle.benchmark is not None and dt in bundle.benchmark.index else None
            exposure_lag = int(bundle.metadata.exposure_data_lag_periods)

            def _asof_row(frame: Optional[pd.DataFrame]) -> Optional[pd.Series]:
                if frame is None:
                    return None
                eligible = frame.index[frame.index <= dt]
                if len(eligible) <= exposure_lag:
                    return None
                return frame.loc[eligible[-(exposure_lag + 1)]]

            G_row = _asof_row(bundle.G)
            F_mcap_row = _asof_row(bundle.F_mcap)
            style_row = _asof_row(bundle.style)
            tradable_row = bundle.tradable.loc[dt] if bundle.tradable is not None and dt in bundle.tradable.index else None
            linear_cost_row = (
                bundle.linear_cost_bps.loc[dt]
                if bundle.linear_cost_bps is not None and dt in bundle.linear_cost_bps.index
                else None
            )
            impact_cost_row = (
                bundle.impact_cost.loc[dt]
                if bundle.impact_cost is not None and dt in bundle.impact_cost.index
                else None
            )

            # 覆盖率必须基于原始 alpha；EMA 可能用历史值填上当前缺口，
            # 不能让平滑结果掩盖当日数据缺失。
            raw_alpha_row = bundle.alpha.loc[dt].reindex(alpha.columns)
            alpha_valid = raw_alpha_row.notna() & pd.Series(
                np.isfinite(raw_alpha_row.to_numpy(dtype=float)),
                index=alpha.columns,
            )
            alpha_coverage = float(alpha_valid.mean())
            min_alpha_coverage = float(params.get("min_alpha_coverage", 0.0))
            if alpha_coverage < min_alpha_coverage:
                raise ValueError(
                    f"alpha coverage on {dt} is {alpha_coverage:.2%}, below {min_alpha_coverage:.2%}"
                )

            if benchmark_spec.objective_id in {"OBJ_MINVAR_ACTIVE", "OBJ_MEANVAR_ACTIVE"}:
                if benchmark_row is None:
                    raise ValueError(f"benchmark is missing date {dt}")
                coverage = float(benchmark_row.reindex(alpha.columns).notna().mean())
                min_coverage = float(params.get("min_benchmark_coverage", 0.0))
                if coverage < min_coverage:
                    raise ValueError(
                        f"benchmark coverage on {dt} is {coverage:.2%}, below {min_coverage:.2%}"
                    )
                benchmark_sum = float(benchmark_row.reindex(alpha.columns).fillna(0.0).sum())
                budget = float(params.get("budget", 1.0))
                if abs(benchmark_sum - budget) > 1e-6:
                    raise ValueError(
                        f"benchmark weights on {dt} must sum to {budget}, got {benchmark_sum}"
                    )

            # 零仓位首日冷启动时不施加换手上限；后续严格串联昨日实际求解权重。
            risk_covered_row = None
            eligible_row = None
            if benchmark_spec.risk_mode == "barra_precomputed":
                if bundle.risk_covered is None or dt not in bundle.risk_covered.index:
                    # Backward compatibility for programmatically assembled,
                    # complete B/F/D bundles. The production Barra adapter
                    # always supplies an explicit point-in-time mask.
                    risk_covered_row = pd.Series(
                        True, index=alpha.columns, dtype=bool
                    )
                else:
                    risk_covered_row = (
                        bundle.risk_covered.loc[dt]
                        .reindex(alpha.columns)
                        .fillna(False)
                        .astype(bool)
                    )
                tradable_for_eligibility = (
                    tradable_row.reindex(alpha.columns).fillna(False).astype(bool)
                    if tradable_row is not None
                    else pd.Series(False, index=alpha.columns, dtype=bool)
                )
                eligible_row = (
                    alpha_valid
                    & tradable_for_eligibility
                    & risk_covered_row
                )

            cold_start = position == 0 and abs(float(previous_weights.sum())) <= 1e-12
            enforce_turnover = not cold_start

            # 构建当日协方差矩阵
            sigma = self._build_covariance(
                bundle,
                benchmark_spec,
                dt,
                list(alpha.columns),
                params,
                risk_covered_row=risk_covered_row,
            )

            # 凸优化求解
            result = self._solve_convex_result(
                alpha_row=row,
                benchmark_row=benchmark_row,
                prev_row=previous_weights,
                sigma=sigma,
                objective_id=benchmark_spec.objective_id,
                params=params,
                benchmark_spec=benchmark_spec,
                G_row=G_row,
                F_mcap_row=F_mcap_row,
                style_row=style_row,
                tradable_row=tradable_row,
                eligible_row=eligible_row,
                linear_cost_row=linear_cost_row,
                impact_cost_row=impact_cost_row,
                enforce_turnover=enforce_turnover,
            )
            fallback_mode = str(params.get("on_solve_failure", "fail_fast"))
            if result.weights is None:
                if fallback_mode == "carry_forward" and float(previous_weights.sum()) > 0:
                    result.weights = previous_weights.to_numpy(dtype=float)
                    result.status = "fallback_carry_forward"
                    result.fallback_used = True
                elif fallback_mode == "equal_weight":
                    if benchmark_spec.objective_id != "OBJ_MEANVAR_ABS":
                        raise OptimizationError(
                            "equal_weight fallback is only supported for absolute-return optimization"
                        )
                    result.weights = np.full(len(alpha.columns), 1.0 / len(alpha.columns), dtype=float)
                    result.status = "fallback_equal_weight"
                    result.fallback_used = True
                else:
                    raise OptimizationError(
                        f"Optimization failed on {dt}: {result.error or result.solver_status}"
                    )

                b_values = (
                    benchmark_row.reindex(alpha.columns).fillna(0.0).to_numpy(dtype=float)
                    if benchmark_row is not None
                    else np.zeros(len(alpha.columns), dtype=float)
                )
                fallback_constraints = self._constraint_diagnostics(
                    result.weights,
                    previous_weights.to_numpy(dtype=float),
                    b_values,
                    list(alpha.columns),
                    params,
                    sigma,
                    G_row=G_row,
                    F_mcap_row=F_mcap_row,
                    style_row=style_row,
                    tradable_row=tradable_row,
                    eligible_row=eligible_row,
                    enforce_turnover=enforce_turnover,
                )
                result.max_constraint_violation = fallback_constraints["max_constraint_violation"]
                if result.max_constraint_violation > 1e-5:
                    raise OptimizationError(
                        f"{result.status} violates constraints by "
                        f"{result.max_constraint_violation} on {dt}"
                    )

            weights.loc[dt] = result.weights
            b_values = (
                benchmark_row.reindex(alpha.columns).fillna(0.0).to_numpy(dtype=float)
                if benchmark_row is not None
                else np.zeros(len(alpha.columns), dtype=float)
            )
            previous_values = previous_weights.to_numpy(dtype=float)
            constraint_metrics = self._constraint_diagnostics(
                result.weights,
                previous_values,
                b_values,
                list(alpha.columns),
                params,
                sigma,
                G_row=G_row,
                F_mcap_row=F_mcap_row,
                style_row=style_row,
                tradable_row=tradable_row,
                eligible_row=eligible_row,
                enforce_turnover=enforce_turnover,
            )
            mu = self._expected_return_vector(row, params)
            linear_cost, impact_cost = self._cost_vectors(
                list(alpha.columns), linear_cost_row, impact_cost_row, params
            )
            delta = result.weights - previous_values
            active = result.weights - b_values
            portfolio_variance, _ = self._risk_variance_and_marginal(
                sigma, result.weights
            )
            tracking_variance, _ = self._risk_variance_and_marginal(sigma, active)
            tracking_error_horizon = float(np.sqrt(tracking_variance))
            tracking_error_annual = float(
                np.sqrt(
                    tracking_variance
                    * bundle.metadata.annualization_factor
                    / bundle.metadata.alpha_horizon_days
                )
            )
            risk_vector = active if benchmark_row is not None else result.weights
            risk_variance, marginal_risk = self._risk_variance_and_marginal(
                sigma, risk_vector
            )
            marginal_contributions = risk_vector * marginal_risk
            expected_linear_cost = float(
                params.get("linear_cost_penalty", 1.0)
                * np.sum(linear_cost * np.abs(delta))
            )
            expected_impact_cost = (
                float(
                    params.get("impact_cost_penalty", 0.0)
                    * np.sum(impact_cost * np.square(delta))
                )
                if bool(params.get("enable_impact_cost", False))
                else 0.0
            )
            exposure_metrics = self._active_exposure_diagnostics(
                result.weights,
                b_values,
                list(alpha.columns),
                G_row=G_row,
                F_mcap_row=F_mcap_row,
                style_row=style_row,
            )
            if isinstance(sigma, PrecomputedFactorRisk):
                factor_eigenvalues = np.linalg.eigvalsh(sigma.factor_cov)
                covariance_min_eigenvalue = float(factor_eigenvalues.min())
                covariance_condition_number = float(np.linalg.cond(sigma.factor_cov))
                risk_model_type = "barra_precomputed_factor_form"
                risk_factor_count = len(sigma.factor_names)
                specific_variance_min = float(sigma.specific_var.min())
                specific_variance_max = float(sigma.specific_var.max())
                risk_exposure_date = sigma.exposure_date
                risk_covariance_date = sigma.covariance_date
                specific_risk_date = sigma.specific_risk_date
            else:
                covariance_min_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
                covariance_condition_number = float(np.linalg.cond(sigma))
                risk_model_type = benchmark_spec.risk_mode
                risk_factor_count = 0
                specific_variance_min = float("nan")
                specific_variance_max = float("nan")
                risk_exposure_date = pd.NaT
                risk_covariance_date = pd.NaT
                specific_risk_date = pd.NaT
            if risk_covered_row is None:
                risk_coverage_weight = float("nan")
                uncovered_frozen_weight = 0.0
                risk_covered_asset_count = len(alpha.columns)
                eligible_asset_count = int(alpha_valid.sum())
            else:
                total_abs_weight = float(np.abs(result.weights).sum())
                covered_values = risk_covered_row.to_numpy(dtype=bool)
                risk_coverage_weight = (
                    float(np.abs(result.weights[covered_values]).sum())
                    / total_abs_weight
                    if total_abs_weight > 1e-12
                    else 1.0
                )
                tradable_values = (
                    tradable_row.reindex(alpha.columns)
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
                uncovered_frozen_weight = float(
                    np.abs(result.weights[~covered_values & ~tradable_values]).sum()
                )
                risk_covered_asset_count = int(covered_values.sum())
                eligible_asset_count = int(eligible_row.sum())
            previous_weights = weights.loc[dt].copy()
            diagnostic_rows.append(
                {
                    "solve_status": result.status,
                    "solver_status": result.solver_status,
                    "solve_time_ms": result.solve_time_ms,
                    "fallback_used": result.fallback_used,
                    "max_constraint_violation": result.max_constraint_violation,
                    "cold_start": cold_start,
                    "turnover_constraint_applied": enforce_turnover,
                    "solve_error": result.error,
                    "expected_return": float(mu @ result.weights),
                    "expected_linear_cost": expected_linear_cost,
                    "expected_impact_cost": expected_impact_cost,
                    "predicted_portfolio_volatility": float(np.sqrt(portfolio_variance)),
                    # Backward-compatible alias; explicitly horizon-based.
                    "predicted_tracking_error": tracking_error_horizon,
                    "predicted_tracking_error_horizon": tracking_error_horizon,
                    "predicted_tracking_error_annual": tracking_error_annual,
                    "tracking_error_cap_annual": (
                        float(params["tracking_error_cap_annual"])
                        if params.get("tracking_error_cap_annual") is not None
                        else float("nan")
                    ),
                    "predicted_objective_risk": float(np.sqrt(risk_variance)),
                    "active_share": 0.5 * float(np.abs(active).sum()),
                    "largest_abs_risk_contribution": float(
                        np.max(np.abs(marginal_contributions), initial=0.0)
                    ),
                    "risk_contribution_sum": float(marginal_contributions.sum()),
                    "covariance_min_eigenvalue": covariance_min_eigenvalue,
                    "covariance_condition_number": covariance_condition_number,
                    "risk_model_type": risk_model_type,
                    "risk_factor_count": risk_factor_count,
                    "specific_variance_min": specific_variance_min,
                    "specific_variance_max": specific_variance_max,
                    "risk_covariance_units": bundle.metadata.risk_covariance_units,
                    "risk_horizon_days": bundle.metadata.risk_horizon_days,
                    "alpha_horizon_days": bundle.metadata.alpha_horizon_days,
                    "annualization_factor": bundle.metadata.annualization_factor,
                    "risk_exposure_date": risk_exposure_date,
                    "risk_covariance_date": risk_covariance_date,
                    "specific_risk_date": specific_risk_date,
                    "risk_covered_asset_count": risk_covered_asset_count,
                    "eligible_asset_count": eligible_asset_count,
                    "risk_coverage_weight": risk_coverage_weight,
                    "uncovered_frozen_weight": uncovered_frozen_weight,
                    **exposure_metrics,
                    **constraint_metrics,
                }
            )
        solve_diagnostics = pd.DataFrame(diagnostic_rows, index=alpha.index)
        solve_diagnostics.index.name = alpha.index.name
        return weights, solve_diagnostics

    # ---- 主入口 ----

    def optimize(
        self,
        bundle: InputBundle,
        benchmark_spec: BenchmarkSpec,
        smoothed_alpha: Optional[pd.DataFrame] = None,
        resolved_params: dict[str, Any] | None = None,
        parameter_version: str = "",
        mapping_version: str = "",
        fallback_used: bool = False,
        requested_optimizer_name: str = "",
        fallback_reason: str = "",
        routing_reason: str = "",
        weak_missing_inputs: list[str] | None = None,
        applied_overrides: dict[str, Any] | None = None,
        data_version_hash: str = "",
        diagnostics: Optional[pd.DataFrame] = None,
    ) -> OutputBundle:
        """执行组合优化并返回 OutputBundle。

        根据 benchmark_spec.backend 选择求解路径：
        - rule_backend：调用规则型方法（_equal_weight_frame / _long_short_equal_weight_frame）
        - riskfolio_backend：调用凸优化方法（_convex_optimize_frame）

        构造包含完整审计字段的 RunMetadata，通过 OutputAdapter 打包返回。

        Args:
            bundle: 输入数据包
            benchmark_spec: 优化器映射描述
            smoothed_alpha: 平滑后的 alpha（None 则使用原始 alpha）
            resolved_params: 合并后的最终参数
            parameter_version: 参数版本号
            mapping_version: 映射版本号
            fallback_used: 是否使用了降级策略
            weak_missing_inputs: 弱依赖缺失字段列表
            applied_overrides: 本次应用的覆盖参数
            data_version_hash: 输入数据版本哈希
            diagnostics: 信号平滑诊断信息

        Returns:
            OutputBundle：目标持仓、交易明细、组合摘要和审计元数据。
        """
        params = dict(resolved_params or {})
        alpha = smoothed_alpha if smoothed_alpha is not None else bundle.alpha
        solve_diagnostics: pd.DataFrame

        # ---- 按后端类型路由 ----
        if benchmark_spec.backend == "rule_backend" and benchmark_spec.name == "topn_long_only_equal_weight":
            # 规则型：topN 等权做多
            topn_n = int(params.get("topn_n", 50))
            weights = self._equal_weight_frame(alpha, topn_n, float(params.get("budget", 1.0)))
            max_weight = float(weights.max(axis=1).max())
            single_name_max = float(params.get("single_name_max", 1.0))
            if max_weight > single_name_max + 1e-10:
                raise OptimizationError(
                    f"topN equal weight {max_weight:.6f} exceeds single_name_max={single_name_max}"
                )

        elif benchmark_spec.backend == "rule_backend" and benchmark_spec.name == "topn_long_short_equal_weight":
            # 规则型：topN 等权多空
            long_n = int(params.get("long_n", 10))
            short_n = int(params.get("short_n", 10))
            long_weight = float(params.get("long_weight", 0.05))
            short_weight = float(params.get("short_weight", -0.05))
            weights = self._long_short_equal_weight_frame(
                alpha, long_n=long_n, short_n=short_n,
                long_weight=long_weight, short_weight=short_weight,
            )

        elif benchmark_spec.backend == "riskfolio_backend":
            # 凸优化型：逐日 cvxpy 求解
            weights, solve_diagnostics = self._convex_optimize_frame(
                bundle, alpha, benchmark_spec, params
            )
        else:
            raise ValueError(
                f"Unsupported backend/name combination: {benchmark_spec.backend}/{benchmark_spec.name}"
            )

        if benchmark_spec.backend == "rule_backend":
            solve_diagnostics = pd.DataFrame(
                {
                    "solve_status": "rule",
                    "solver_status": "not_applicable",
                    "solve_time_ms": 0.0,
                    "fallback_used": False,
                    "max_constraint_violation": 0.0,
                    "cold_start": False,
                    "turnover_constraint_applied": False,
                    "solve_error": None,
                },
                index=alpha.index,
            )

        combined_diagnostics = solve_diagnostics
        if diagnostics is not None:
            combined_diagnostics = diagnostics.reindex(alpha.index).join(
                solve_diagnostics, how="left"
            )

        per_date_fallback = bool(solve_diagnostics["fallback_used"].fillna(False).any())
        total_solve_time_ms = float(solve_diagnostics["solve_time_ms"].fillna(0.0).sum())
        status_counts = {
            str(key): int(value)
            for key, value in solve_diagnostics["solve_status"].value_counts(dropna=False).items()
        }
        solver_name = (
            "none"
            if benchmark_spec.backend == "rule_backend"
            else str(params.get("solver_name", benchmark_spec.solver_default))
        )
        constraint_keys = {
            "budget", "single_name_max", "gross_exposure_cap", "active_weight_abs_max",
            "turnover_cap", "industry_neutral_mode", "industry_band",
            "style_neutral_mode", "style_band", "mcap_neutral_mode", "mcap_band",
            "enforce_tradable_mask", "long_only", "enforce_tracking_error_cap",
            "tracking_error_cap_annual",
        }
        resolved_constraints = {key: params[key] for key in constraint_keys if key in params}

        # ---- 构造审计元数据 ----
        metadata = RunMetadata(
            # v0.2 审计核心字段
            optimizer_name=benchmark_spec.name,
            objective_id=benchmark_spec.objective_id,
            hard_constraint_set=benchmark_spec.hard_constraint_set,
            soft_constraint_set=benchmark_spec.soft_constraint_set,
            risk_mode=benchmark_spec.risk_mode,
            parameter_version=parameter_version,
            mapping_version=mapping_version,
            data_version_hash=data_version_hash,
            # 兼容旧字段
            model_id=benchmark_spec.name,
            alpha_version=bundle.alpha.columns.name or "alpha_v0",
            constraint_profile=benchmark_spec.hard_constraint_set,
            solver=solver_name,
            solve_status=(
                "success_with_fallback"
                if fallback_used or per_date_fallback
                else "success"
            ),
            solve_time_ms=total_solve_time_ms,
            fallback_used=fallback_used or per_date_fallback,
            extras={
                "resolved_constraints": resolved_constraints,
                "backend": benchmark_spec.backend,
                "weak_missing_inputs": weak_missing_inputs or [],
                "applied_overrides": applied_overrides or {},
                "solve_status_counts": status_counts,
                "max_constraint_violation": float(
                    solve_diagnostics["max_constraint_violation"].fillna(0.0).max()
                ),
                "requested_optimizer_name": requested_optimizer_name or benchmark_spec.name,
                "routing_reason": routing_reason,
                "fallback_reason": fallback_reason,
                "timing": {
                    "calendar_name": bundle.metadata.calendar_name,
                    "decision_time": bundle.metadata.decision_time,
                    "execution_time": bundle.metadata.execution_time,
                    "market_data_lag_periods": bundle.metadata.market_data_lag_periods,
                    "exposure_data_lag_periods": bundle.metadata.exposure_data_lag_periods,
                    "risk_covariance_units": bundle.metadata.risk_covariance_units,
                    "risk_horizon_days": bundle.metadata.risk_horizon_days,
                    "annualization_factor": bundle.metadata.annualization_factor,
                },
            },
        )

        # ---- 打包输出 ----
        return self.output_adapter.build_output_bundle(
            target_positions=weights,
            prev_positions=bundle.prev_positions,
            metadata=metadata,
            diagnostics=combined_diagnostics,
        )
