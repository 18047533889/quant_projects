"""alphaprobe.evaluation.marginal_value —— L4 Marginal Information Value / ΔPoolUtility。

plan Task 9：防止新增大量 standalone IC 合格、但与现有池重复的信息进池。

核心输入
--------
- ``candidate``：单因子面板（DataFrame index=date columns=code，或同形状 ndarray）。
- ``pool_rep``：池表征（pool representation）。两种形态：
    1. DataFrame / (T,N,K) ndarray：池因子值矩阵（一列/一层一个池因子），
       ResidualEvaluator 直接对该矩阵做横截面正交化；
    2. DataFrame 集合（Sequence[DataFrame]）：每个元素 = 一个池因子的
       (T,N) 面板，等权对齐后转成 (T,N,K)。
- ``label_panel``：与 candidate 同轴的 vwap→vwap 远期收益面板（唯一收益口径）。
- ``price_panel``（可选）：VWAP 价格面板（index=date columns=code），
  incremental long-short（pool utility）需要它来构造真实 daily PnL。

核心输出（:class:`MarginalValueResult`）
----------------------------------------
- ``residual_rankic``：candidate 对池表征逐日横截面 OLS 正交化后，残差对
  label 的 RankIC **时间均值**。RankIC 通道一律消费
  :class:`QuantEvaluatorAdapter` 的 QE ``rank_ic`` 定义（Part E：#7 不在
  AlphaPROBE 内手写平行 RankIC 口径）；无 QE 时本字段显式 None（不降级手算）。
- ``incremental_value``：pool 组合 vs pool+[candidate] 组合的增量效用。
  走 QE ``probe_portfolio.compute_cohort_pnl`` 的 long_short_ret 序列后处理
  （组合 PnL 由 QE 构造；这里只做净变化/年化等纯 numpy 数学，不造第二套回测）。
- ``keep_recommendation``：``keep`` / ``reject`` / ``neutral``。相关性高 ≠ 自动删除：
  有明确 incremental value 的高相关候选输出 keep（Part G #22 可在 L4 生存）；
  空池 / 样本不足 / 面板过短 → neutral（不是满分也不是直接删）。
- ``model_delta_utility``：L4 **finalist-only** 的 ΔModelUtility。普通 L1/L2
  candidate 永不进入该通道（fail-closed None）；L4 finalist 且未注入
  ``model_fn`` 时同样 fail-closed None（不编造模型结果，留 TODO 交协调者接入
  真实模型增量通道，见 Part G：#7/#22）。

Part E 边界（不变量）
---------------------
1. RankIC 只来自 QE。本模块经 ``QuantEvaluatorAdapter._qe_registry_metrics``
   调 QE ``rank_ic``；QE 不可导入时 residual_rankic=None + diagnostics 记
   degraded_reason（绝不退化为本地手算 IC）。
2. incremental long-short 的 daily PnL 由 QE ``compute_cohort_pnl`` 构造；
   本模块只对 QE 产出的 ``long_short_ret`` / ``pnl_net`` 序列做算术后处理。
3. 相关矩阵（rho / 均相关）是纯描述性诊断（“因子间相关”不是 IC，不在禁止
   手写清单内），只进 diagnostics，不参与任何 RankIC/Sharpe/MDD 结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "MarginalValueEvaluator",
    "MarginalValueResult",
    "ResidualRankICResult",
]

EPS = 1e-12


# ---------------------------------------------------------------------------
# 结果契约
# ---------------------------------------------------------------------------


@dataclass
class ResidualRankICResult:
    """residual RankIC 子结果（QE 通道 + 诊断）。"""

    residual_rankic: float | None = None
    standalone_rankic: float | None = None
    n_dates_used: int = 0
    mean_abs_corr: float | None = None  # 候选与池表征的平均绝对相关（诊断）
    max_abs_corr: float | None = None   # 候选与任一池表征的最大绝对相关（诊断）
    degraded_reason: str = ""           # 非空 = QE 不可用 / 样本不足等降级说明


@dataclass
class MarginalValueResult:
    """MarginalValueEvaluator.evaluate 的完整输出。"""

    residual_rankic: float | None = None
    incremental_value: float | None = None
    keep_recommendation: str = "neutral"
    decision: str = "neutral"
    model_delta_utility: float | None = None
    l4_eligible: bool = False
    l4_finalist_only: bool = True
    pool_size: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_rankic": self.residual_rankic,
            "incremental_value": self.incremental_value,
            "keep_recommendation": self.keep_recommendation,
            "decision": self.decision,
            "model_delta_utility": self.model_delta_utility,
            "l4_eligible": self.l4_eligible,
            "l4_finalist_only": self.l4_finalist_only,
            "pool_size": self.pool_size,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# 阈值配置
# ---------------------------------------------------------------------------

#: 默认相关阈值：|spearman 相关| 超过该值视为「与池表征高相关」。
DEFAULT_HIGH_CORR_THRESHOLD = 0.80

#: 默认 residual RankIC 下界：低于该值 → 正交化后无可辨识增量预测力。
DEFAULT_RESIDUAL_IC_FLOOR = 0.0

#: 默认 incremental utility 下界：增量 <= 该值 → 池效用无提升。
DEFAULT_INCREMENTAL_FLOOR = 0.0

#: 默认最小历史行数：少于该行数不出非中性结论。
DEFAULT_MIN_HISTORY = 40

#: 默认最小横截面宽度：少于该宽度残差回归不可靠。
DEFAULT_MIN_CROSS_SECTION = 20

#: residual_rankic 依赖 QE 且该阈值只在 QE 通道可用时生效；
#: QE 不可用时 residual_rankic=None，keep/reject 仅由 incremental 通道裁决。


# ---------------------------------------------------------------------------
# MarginalValueEvaluator
# ---------------------------------------------------------------------------


class MarginalValueEvaluator:
    """L4 marginal information value / Δpool utility 评估器。

    Parameters
    ----------
    pool_rep : DataFrame | ndarray | Sequence[DataFrame] | None
        池表征（见模块 docstring）。None 或空 → 空池（结果中性）。
    high_corr_threshold : float
        候选与池表征的高相关判据（默认 0.80）。
    residual_ic_floor : float
        residual RankIC 下界（默认 0.0）。residual_rankic 可观测且
        < floor → 候选不携带正交预测信息（叠加到 incremental 裁决）。
    incremental_floor : float
        增量效用下界（默认 0.0）。incremental_value 可观测且 <= floor
        → 池效用无提升（叠加到 incremental 裁决）。
    residual_ic_weight / incremental_weight : float
        keep 裁决中两通道的权重（保留参数；当前决策表为阈值门，两通道权重
        用于未来组合分数裁决，未参与现有离散决策——不保留为死参数则删）.
    min_history : int
        面板最少行数（默认 40）；不足 → 结果全中性。
    min_cross_section : int
        每日最少可回归样本数（默认 20）。
    min_periods : int
        组合指标最小有效期数（透传 QE compute_metrics_from_cohort）。
    min_obs : int
        横截面 OLS 每日最少观测数（透传 QE compute_factor_loadings）。
    l4_finalist : bool
        该候选是否 L4 finalist（high-fitness/high-correlation，plan Task 9：
        只有 finalist 才进入 ΔModelUtility 通道）。普通 L1/L2 candidate
        恒 False → model_delta_utility 恒 None。
    model_fn : Callable | None
        可选的 ΔModelUtility 注入函数。None → 即使 finalist 也 fail-closed。
        真实模型增量通道接入留 TODO（本模块不手写模型增量语义）。
    adapter : QuantEvaluatorAdapter | None
        QE 通道适配器（label_days 建议 1：与合成/生产 forward label 面板直接
        对拍）。None → 自动构造默认 adapter。仅当 QE 可导入且 ``use_qe=True``
        时才真正调 QE（见 Part E 边界）；QE 不可导入时 residual_rankic=None。
    use_qe : bool
        是否允许走 QE rank_ic 通道（默认 True）。False → 强制不用 QE
        （residual_rankic=None + degraded_reason，避免测试环境依赖）。
    cost_scenario / n_quantiles / holding
        透传给 adapter / QE cohort 的组参数。
    """

    def __init__(
        self,
        *,
        pool_rep: Any = None,
        high_corr_threshold: float = DEFAULT_HIGH_CORR_THRESHOLD,
        residual_ic_floor: float = DEFAULT_RESIDUAL_IC_FLOOR,
        incremental_floor: float = DEFAULT_INCREMENTAL_FLOOR,
        residual_ic_weight: float = 0.5,
        incremental_weight: float = 0.5,
        min_history: int = DEFAULT_MIN_HISTORY,
        min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
        min_periods: int = 20,
        min_obs: int = 10,
        l4_finalist: bool = False,
        model_fn: Any = None,
        adapter: Any = None,
        use_qe: bool = True,
        cost_scenario: str | float = "gross",
        n_quantiles: int = 10,
        holding: int = 1,
    ) -> None:
        self.high_corr_threshold = float(high_corr_threshold)
        self.residual_ic_floor = float(residual_ic_floor)
        self.incremental_floor = float(incremental_floor)
        # residual_ic_weight / incremental_weight 为组合分数扩展保留：
        # 当前决策表是离散阈值门，两权重不参与；保留为公开配置供上层
        # 将来若改连续评分裁决时透传。若最终不需要可在收口时删除。
        self.residual_ic_weight = float(residual_ic_weight)
        self.incremental_weight = float(incremental_weight)
        self.min_history = int(min_history)
        self.min_cross_section = int(min_cross_section)
        self.min_periods = int(min_periods)
        self.min_obs = int(min_obs)
        self.l4_finalist = bool(l4_finalist)
        self.model_fn = model_fn
        self.use_qe = bool(use_qe)
        self.cost_scenario = cost_scenario
        self.n_quantiles = int(n_quantiles)
        self.holding = int(holding)

        self.pool_rep: Any = None
        self.pool_size: int = 0
        if pool_rep is not None:
            self._set_pool(pool_rep)

        self._adapter: Any = adapter
        if adapter is None and self.use_qe:
            # 延迟到首次真正需要 QE 时才 import（QE 不可导入时自然降级）。
            self._adapter = None

    # ------------------------------------------------------------------
    # pool 表征接入
    # ------------------------------------------------------------------

    def _set_pool(self, pool_rep: Any) -> None:
        """规范化 pool 表征为 (T,N,K) 浮点 ndarray；空/退化 → pool_size=0。"""
        arr = _coerce_pool_matrix(pool_rep)
        if arr is None:
            self.pool_rep = None
            self.pool_size = 0
            return
        T, N, K = arr.shape
        if K == 0 or T == 0 or N == 0:
            self.pool_rep = None
            self.pool_size = 0
            return
        self.pool_rep = arr
        self.pool_size = K

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: Any,
        label_panel: Any,
        price_panel: Any | None = None,
        *,
        factor_id: str = "candidate",
    ) -> MarginalValueResult:
        """评估单候选相对池表征的 marginal information value。

        Returns
        -------
        MarginalValueResult
            字段见 dataclass。任何情况下都不抛（除形状严重不合法）；数据不足/
            空池等统一走 neutral 输出。
        """
        cand = _as_2d(candidate, "candidate")
        labels = _as_2d(label_panel, "label_panel")
        T, N = cand.shape
        if labels.shape != (T, N):
            # 允许 label 面板行数与候选一致即可（列语义同轴由调用方保证）；
            # 形状完全不一致属严重错误。
            if labels.shape[0] != T:
                raise ValueError(
                    f"candidate {cand.shape} 与 label_panel {labels.shape} 行数不一致"
                )

        diagnostics: dict[str, Any] = {"factor_id": str(factor_id)}
        pool_arr = self.pool_rep

        # ---- 退化守卫：空池 / 样本不足 → 全中性 ----
        if pool_arr is None or self.pool_size == 0:
            return self._neutral_result(
                diagnostics=diagnostics, reason="empty_pool"
            )
        if T < self.min_history or N < self.min_cross_section:
            return self._neutral_result(
                diagnostics=diagnostics,
                reason=f"insufficient_samples: T={T}<{self.min_history} "
                f"or N={N}<{self.min_cross_section}",
            )
        pool_T = pool_arr.shape[0]
        if pool_T != T:
            # 池表征与候选不同行：尝试按日期对齐（由调用方保证；此处保守中性）。
            return self._neutral_result(
                diagnostics=diagnostics,
                reason=f"pool_T={pool_T} != candidate_T={T}",
            )

        # ---- 诊断相关（纯描述性，非 IC） ----
        corr_diag = _candidate_pool_corr(cand, pool_arr)
        diagnostics.update(corr_diag)
        high_corr = bool(
            corr_diag["max_abs_corr"] is not None
            and corr_diag["max_abs_corr"] >= self.high_corr_threshold
        )

        # ---- ① residual RankIC（QE 通道；QE 不可用 → None + 降级原因） ----
        resid: ResidualRankICResult = self._residual_rankic(
            cand, labels, pool_arr
        )
        diagnostics["residual"] = resid

        # ---- ② incremental long-short / pool utility（QE cohort PnL 后处理） ----
        incr: float | None = None
        if price_panel is not None:
            try:
                incr = self._incremental_utility(
                    cand, labels, pool_arr, price_panel
                )
            except Exception as exc:  # noqa: BLE001 - 后处理失败不致命，中性降级
                diagnostics["incremental_error"] = str(exc)
                incr = None

        # ---- ③ ΔModelUtility（L4 finalist-only；fail-closed） ----
        model_delta: float | None = None
        if self.l4_finalist and high_corr and self.model_fn is not None:
            try:
                model_delta = float(self.model_fn(candidate=candidate, pool_rep=self.pool_rep))
            except Exception as exc:  # noqa: BLE001 - fail-closed
                diagnostics["model_fn_error"] = str(exc)
                model_delta = None

        recommendation, decision = self._decide(
            residual_rankic=resid.residual_rankic,
            incremental_value=incr,
            high_corr=high_corr,
        )

        return MarginalValueResult(
            residual_rankic=resid.residual_rankic,
            incremental_value=incr,
            keep_recommendation=recommendation,
            decision=decision,
            model_delta_utility=model_delta,
            l4_eligible=self.l4_finalist and high_corr,
            l4_finalist_only=True,
            pool_size=self.pool_size,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # ① residual RankIC（QE 通道）
    # ------------------------------------------------------------------

    def _residual_rankic(
        self,
        cand: np.ndarray,
        labels: np.ndarray,
        pool_arr: np.ndarray,
    ) -> ResidualRankICResult:
        """逐日横截面 OLS 残差 → 残差对 label 的 QE rank_ic（时间均值）。

        正交化数学在本模块做（逐日解 XtX beta —— 这是“残差构造”，不是 IC）；
        RankIC 本身只由 QE ``rank_ic`` 定义。QE 不可导入或 evaluate 失败
        → degraded_reason + residual_rankic=None（fail-closed，不手算 IC）。
        """
        T, N = cand.shape
        out = ResidualRankICResult(
            n_dates_used=0,
            degraded_reason="",
        )

        # 构造 (T,N,K) 表征矩阵；K==0 视为空池。
        if pool_arr.shape[2] == 0:
            out.degraded_reason = "empty_pool"
            return out

        # 逐日 OLS：factor = alpha + Σ_k beta_k * pool_k + resid
        resid_panel = np.full_like(cand, np.nan)
        used = 0
        corr_by_date: list[float] = []
        X_all = pool_arr  # (T,N,K)
        for t in range(T):
            y = cand[t]
            X = X_all[t]
            finite_mask = (
                np.isfinite(y)
                & np.isfinite(X).all(axis=1)
            )
            if finite_mask.sum() < max(self.min_obs, 2):
                continue
            yv = y[finite_mask]
            Xv = X[finite_mask]
            # 表征矩阵退化（某池因子恒值/共线）→ 该日跳过
            # K >= N（池因子数不少于资产数）或列共线时 rank 不足 → 跳过
            # （无多余维度可正交化）。
            if Xv.shape[1] >= Xv.shape[0]:
                continue
            Xc = Xv - Xv.mean(axis=0)
            cov = Xc.T @ Xc
            if np.any(~np.isfinite(cov)) or np.linalg.matrix_rank(cov) < Xc.shape[1]:
                continue
            # 与 QE compute_factor_loadings 一致的含截距 OLS
            A = np.column_stack([np.ones(len(yv)), Xv])
            try:
                beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid_t = np.full(N, np.nan)
            A_full = np.column_stack([np.ones(N), X])
            resid_t[finite_mask] = yv - A_full[finite_mask] @ beta
            resid_panel[t] = resid_t
            used += 1
            # 描述性诊断：候选与该池表征的每日 spearman 绝对相关均值（非 IC）
            valid_c = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if valid_c.sum() >= 2:
                corr_by_date.append(
                    _spearman_abs_corr(y[valid_c], X[valid_c].mean(axis=1))
                )

        out.n_dates_used = used
        if used < max(2, self.min_obs):
            out.degraded_reason = f"insufficient_dates: used={used}"
            return out

        if corr_by_date:
            out.mean_abs_corr = float(np.mean(corr_by_date))
        out.max_abs_corr = float(corr_diag_max(cand, X_all)) if used else None

        # QE rank_ic 通道
        if not self.use_qe:
            out.degraded_reason = "qe_disabled_by_config"
            return out
        try:
            rri = self._qe_residual_rankic(resid_panel, labels)
        except Exception as exc:  # noqa: BLE001 - QE 通道失败 fail-closed
            out.degraded_reason = f"qe_unavailable: {exc}"
            return out
        if rri is None or not np.isfinite(rri):
            out.degraded_reason = "qe_rank_ic_None"
            return out
        out.residual_rankic = float(rri)
        return out

    def _qe_residual_rankic(
        self,
        resid_panel: np.ndarray,
        labels: np.ndarray,
    ) -> float | None:
        """把残差面板交给 QuantEvaluatorAdapter 的 QE rank_ic 通道。

        与 standalone 因子完全同一条 QE ``rank_ic`` 语义（Part E #7）：
        QE 的 rank_ic = daily Spearman IC 的时间均值。此处直接调 QE 的
        ``compute_daily_ic`` + ``compute_mean_ic_value``（IC 系列构造 +
        时间均值，均 QE 权威内核），而非通过 adapter 的 registry 通道
        （registry 通道把整个 (T,N) 面板当单个 factor，横截面 = N assets，
        不适合残差已含大量 NaN 的宽表）。

        时间轴：残差面板 T 行，label 面板 T 行 → 逐日横截面残差 vs 当日
        label（H=1 的 vwap→vwap forward 语义由调用方 label 面板保证）。
        """
        if not self.use_qe:
            return None
        try:
            from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
            from quant_evaluator.contracts.label_bundle import LabelBundle
            from quant_evaluator.metrics.ic import (
                compute_daily_ic,
                compute_mean_ic_value,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise RuntimeError(
                f"quant_evaluator 不可导入（fail-closed）：{exc}"
            ) from exc

        T, N = resid_panel.shape
        if T < 2:
            return None
        time_idx = _panel_index_from_shape(T)
        fb = FactorBatch(
            factor_ids=("residual",),
            time_axis=AxisRef("time", "str", T, values=list(time_idx)),
            asset_axis=AxisRef("asset", "str", N, values=list(range(N))),
            values=np.asarray(resid_panel, dtype=np.float64)[:, :, np.newaxis],
        )
        # label_end_time 用 i+1（H=1 语义；末尾 T-1 无未来 label → 末端 marker
        # 必须保 LabelBundle 因果链 start<end 与 decision_time 严格递增）。
        end_time = list(time_idx[1:]) + [time_idx[-1] + 1]
        lb = LabelBundle(
            target_id="vwap_to_vwap",
            values=np.asarray(labels, dtype=np.float64),
            horizon=1,
            decision_time=tuple(time_idx),
            label_start_time=tuple(time_idx),
            label_end_time=tuple(end_time),
        )
        try:
            ic_series, _valid_counts = compute_daily_ic(
                fb, lb, method="spearman", min_assets=2
            )
            mean_arr = compute_mean_ic_value(
                ic_series, valid_counts=None, min_periods=2
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise RuntimeError(
                f"QE compute_daily_ic/compute_mean_ic_value 失败：{exc}"
            ) from exc
        if mean_arr is None or mean_arr.shape[0] == 0:
            return None
        v = mean_arr[0]
        if v is None or not np.isfinite(v):
            return None
        return float(v)

    # ------------------------------------------------------------------
    # ② incremental long-short / pool utility（QE cohort PnL 后处理）
    # ------------------------------------------------------------------

    def _incremental_utility(
        self,
        cand: np.ndarray,
        labels: np.ndarray,
        pool_arr: np.ndarray,
        price_panel: Any,
    ) -> float | None:
        """pool 组合 vs pool+[candidate] 的增量效用。

        组合构造交给 QE ``probe_portfolio.compute_cohort_pnl``：
        1) pool 组合因子 = 池表征等权（K 个池因子横截面 rank 的均值）；
        2) augmented 组合因子 = pool 组合 + candidate 的横截面 rank 均值
           （候选的增量信号与池信号等权叠加——近似“组合池”的增量暴露）。
        3) 增量效用 = aug 的 long_short_ret 相对 pool 的净提升（年化差）。

        注意：这只是一个“pool utility 增量代理”。真实生产落库的
        ΔPoolUtility 由 QE/组合优化层权威定义；本模块的 long-short 代理只用于
        L4 决策的方向与量级估计，且全部 PnL 由 QE 构造（不手写回测）。
        若需组合优化权威的 ΔPoolUtility，走 factor_optimizer/quant_evaluator
        组合链（TODO 交协调者接线），不在此模块内重复实现。
        """
        pool_combo = _pool_composite(pool_arr)  # (T,N)
        # 把两个信号都转成横截面 percentile rank [0,1] 再等权 → 组合信号
        sig_pool = _cross_sectional_rank(pool_combo)
        sig_cand = _cross_sectional_rank(cand)

        from alphaprobe.evaluator_adapter import _as_2d as _adapter_as_2d

        p = _adapter_as_2d(price_panel, "price_panel")
        if p.shape != (labels.shape[0], labels.shape[1]):
            # 价格面板与 label 行数不一致时尝试截断对齐
            nrow = labels.shape[0]
            if p.shape[0] < nrow:
                return None
            p = p[:nrow, :]

        try:
            from quant_evaluator.metrics.probe_portfolio import compute_cohort_pnl
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise RuntimeError(
                f"quant_evaluator.probe_portfolio 不可导入（fail-closed）：{exc}"
            ) from exc

        # next_ret / next_vwap 与 evaluator_adapter._cohort_panel_prep 同口径：
        # r_t = P_t / P_{t-1} - 1；vwap_t = P_t。注意 cohort 的 long_short_ret
        # 在信号日 start 计算 = D10-D1(next_ret[entry=start+1])，即信号 start 的
        # label 由价格逐行构造出 next_ret（与 label 面板一致：label[start] =
        # vwap[start+1]/vwap[start]-1 = next_ret[start+1]）。
        with np.errstate(divide="ignore", invalid="ignore"):
            next_ret = np.divide(p[1:, :], p[:-1, :]) - 1.0
        next_vwap = p[1:, :]

        # QE 构造 cohort（long-short spread 的 daily PnL 来源）
        # 注意：compute_cohort_pnl 签名不含 min_periods（那是 compute_portfolio_metrics
        # 的参数）；这里只消费 cohort 的 daily long_short_ret 序列，指标统计在下面
        # 纯 numpy 后处理完成（组合构造仍由 QE 负责）。
        def _ls_sharpe(sig: np.ndarray) -> float | None:
            try:
                res = compute_cohort_pnl(
                    factor_values=sig[:-1, :],  # signal 日 → 次日起持有
                    next_ret=next_ret,
                    next_vwap=next_vwap,
                    n_quantiles=self.n_quantiles,
                    holding=self.holding,
                    per_side_cost=0.0,
                )
            except Exception:  # noqa: BLE001 - 单边构造失败按无增量处理
                return None
            lsr = np.asarray(res.get("long_short_ret", []), dtype=float)
            valid = lsr[np.isfinite(lsr)]
            if valid.size < max(10, self.min_periods):
                return None
            mean = float(np.mean(valid))
            std = float(np.std(valid, ddof=1))
            if not np.isfinite(std) or std <= EPS:
                return None
            # 年化：日频 spread 均值/std → 年化（纯序列后处理；不含持仓期假设）
            return mean / std * math.sqrt(252)

        # 对齐：long_short_ret 是信号日对齐 → 把 pool_combo 与 aug 截到与 next_ret
        # 同长的信号段（T-1 行）。
        if sig_pool.shape[0] <= 1 or sig_cand.shape[0] <= 1:
            return None
        sh_pool = _ls_sharpe(sig_pool)
        sh_cand = _ls_sharpe(sig_cand)
        # 增量 = aug（pool 等权 + 候选）相对 pool 的 spread Sharpe 变化：
        # aug 信号 = pool 与候选 rank 的 0.5/0.5 叠加（增量暴露近似）；
        # 池内新增重复候选 ≈ 平均回自身 → Δ≈0；独立候选带新暴露 → Δ 显著。
        sig_aug = 0.5 * sig_pool + 0.5 * sig_cand
        sh_aug = _ls_sharpe(sig_aug)
        if sh_pool is None or sh_cand is None or sh_aug is None:
            return None
        # 增量 = aug 年化 spread Sharpe − pool 年化 spread Sharpe
        return float(sh_aug - sh_pool)

    # ------------------------------------------------------------------
    # ③ keep/reject 裁决
    # ------------------------------------------------------------------

    def _decide(
        self,
        *,
        residual_rankic: float | None,
        incremental_value: float | None,
        high_corr: bool,
    ) -> tuple[str, str]:
        """输出 keep/reject/neutral 建议（不硬删；阈值全可配置）。

        裁决表：
        - 两通道都不可观测 → neutral（不猜）。
        - 高相关且 residual 低 且 增量不可观测/<=floor → reject
          （重复信息；无正交增量亦无组合增量）。
        - 高相关但 incremental > floor → keep（Part G #22：明显 incremental
          value 的高相关因子可保留）。
        - 低相关（有独立信息）且 residual 可观测 > floor → keep。
        - residual 通道不可用且增量 <= floor 且高相关 → reject（保守去重）。
        - 其余 → neutral。
        """
        residual_ok = (
            residual_rankic is not None and np.isfinite(residual_rankic)
        )
        incremental_ok = (
            incremental_value is not None and np.isfinite(incremental_value)
        )

        if not residual_ok and not incremental_ok:
            return "neutral", "neutral"

        residual_bad = residual_ok and residual_rankic < self.residual_ic_floor
        incremental_bad = incremental_ok and incremental_value <= self.incremental_floor

        # 高相关：相关高本身不足以保证 keep。要有「明显 incremental value」：
        # - 组合增量通道正向 → keep（Part G #22 可在 L4 生存）；
        # - 组合增量不可观测，仅凭 residual >= floor 不足以给 keep
        #   （同源候选的残差常带少量正数值噪声，见 docstring 语义说明）
        #   → 无增量证据时保守 reject；residual 也带正向但无 price 通道时
        #   neutral（留待 L4 组合通道裁决，不硬删也不满分）。
        if high_corr:
            if incremental_ok and incremental_value > self.incremental_floor:
                return "keep", "keep"  # 高相关但有明确增量 → keep
            if incremental_ok and incremental_value <= self.incremental_floor:
                return "reject", "reject"  # 高相关 + 有增量证据但 <=0 → reject
            # incremental 不可观测：residual 也无正向 → reject（保守去重）
            if residual_bad or not residual_ok:
                return "reject", "reject"
            # incremental 不可观测但 residual 正向 → neutral
            # （不凭高相关+残差噪声给满分 keep，也不凭未观测的增量硬删）
            return "neutral", "neutral"
        # 低相关 → 独立信息已确认（residual 高或增量高都算）
        if residual_ok and residual_rankic >= self.residual_ic_floor:
            return "keep", "keep"
        if incremental_ok and incremental_value > self.incremental_floor:
            return "keep", "keep"
        return "neutral", "neutral"

    # ------------------------------------------------------------------
    # 中性输出
    # ------------------------------------------------------------------

    def _neutral_result(
        self,
        *,
        diagnostics: dict[str, Any],
        reason: str,
    ) -> MarginalValueResult:
        diagnostics["neutral_reason"] = reason
        return MarginalValueResult(
            residual_rankic=None,
            incremental_value=None,
            keep_recommendation="neutral",
            decision="neutral",
            model_delta_utility=None,
            l4_eligible=False,
            l4_finalist_only=True,
            pool_size=self.pool_size,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _as_2d(panel: Any, name: str) -> np.ndarray:
    """DataFrame / ndarray → (T, N) float64。

    注意：永远返回副本（``np.asarray(..., copy)`` 的等价物），
    禁止返回可能共享内存的 2D 视图后被调用方原地覆写（会把 (1,N)
    广播回写 DataFrame/ndarray 源，产生 ``could not broadcast`` 错误）。
    """
    if panel is None:
        raise ValueError(f"{name} 不能为 None")
    if hasattr(panel, "to_numpy"):
        arr = np.array(panel.to_numpy(), dtype=np.float64, copy=True)
    else:
        arr = np.array(panel, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须为二维面板，got ndim={arr.ndim}")
    return arr


def _coerce_pool_matrix(pool_rep: Any) -> np.ndarray | None:
    """pool_rep → (T,N,K) 或 None。

    接受：
    - (T,N) DataFrame / ndarray（单池因子 → K=1）
    - (T,N,K) ndarray
    - Sequence[DataFrame]（每个 = 一池因子 (T,N) → K=len）
    """
    if pool_rep is None:
        return None
    if isinstance(pool_rep, (list, tuple)):
        frames = [p for p in pool_rep if p is not None]
        if not frames:
            return None
        arrs = [_as_2d(f, "pool_element") for f in frames]
        T = arrs[0].shape[0]
        N = arrs[0].shape[1]
        if any(a.shape != (T, N) for a in arrs):
            # 形状不一致时保守按 NaN 处理（不推断）
            stack = np.full((T, N, len(arrs)), np.nan)
            for i, a in enumerate(arrs):
                if a.shape == (T, N):
                    stack[:, :, i] = a
            return stack
        return np.stack(arrs, axis=2)
    if isinstance(pool_rep, np.ndarray):
        a = np.asarray(pool_rep, dtype=np.float64)
        if a.ndim == 2:
            # 显式拷贝：a[:, :, None] 是共享内存视图，调用方（逐日残差 OLS /
            # 组合矩阵）不会写它；但 np.asarray(dtype=float64) 在已是 float64 时
            # 返回原数组（若它本来就是 DataFrame 的 2D 视图，risk 相同）。
            # 这里直接转 (T,N,1)，且永远独立拷贝（防御外部 block 共享）。
            return np.array(a[:, :, np.newaxis], dtype=np.float64, copy=True)
        if a.ndim == 3:
            return a
        raise ValueError(f"pool_rep ndarray 必须为 2D/3D，got ndim={a.ndim}")
    # DataFrame / 类面板 → 单池因子 (K=1)
    arr = _as_2d(pool_rep, "pool_rep")
    return np.array(arr[:, :, np.newaxis], dtype=np.float64, copy=True)


def _panel_index_from_shape(n: int) -> list[int]:
    return list(range(n))


def _cross_sectional_rank(x: np.ndarray) -> np.ndarray:
    """逐日横截面 percentile rank → [0,1]（含 NaN 保留）。"""
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.full_like(x, np.nan)
    for t in range(T):
        row = x[t]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue
        order = np.argsort(np.argsort(row[finite]))
        out[t, finite] = order / max(int(finite.sum()) - 1, 1)
    return out


def _pool_composite(pool_arr: np.ndarray) -> np.ndarray:
    """池表征 → 单张池组合因子：(T,N,K)→(T,N)，横截面 rank 等权。"""
    T, N, K = pool_arr.shape
    if K == 0:
        raise ValueError("pool_arr K==0")
    ranks = np.zeros((T, N))
    n_valid = np.zeros((T, N))
    for k in range(K):
        rk = _cross_sectional_rank(pool_arr[:, :, k])
        finite_k = np.isfinite(rk)
        ranks[finite_k] += rk[finite_k]
        n_valid[finite_k] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(ranks, n_valid, out=np.full_like(ranks, np.nan))
    return out


def _spearman_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    """x/y 一维数组的 |spearman| 相关（诊断用；非 IC，非决策金融指标）。

    相关矩阵是纯描述性诊断（因子间相关不在 Part E 禁止手写清单内），
    只进 diagnostics；绝不参与 RankIC / Sharpe / MDD 结论。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[finite], y[finite]
    if xv.size < 3 or np.unique(xv).size < 2 or np.unique(yv).size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(xv))
    ry = np.argsort(np.argsort(yv))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= EPS:
        return float("nan")
    return float(abs((rx * ry).sum() / denom))


def _candidate_pool_corr(
    cand: np.ndarray, pool_arr: np.ndarray
) -> dict[str, float | None]:
    """候选与池表征的时间平均 |spearman| 相关诊断（逐层 + 合成池）。"""
    T = cand.shape[0]
    # 候选与池等权合成因子
    try:
        combo = _pool_composite(pool_arr)
    except Exception:  # noqa: BLE001
        combo = None
    daily: list[float] = []
    if combo is not None:
        for t in range(T):
            c = _spearman_abs_corr(cand[t], combo[t])
            if np.isfinite(c):
                daily.append(c)
    mean_abs = float(np.mean(daily)) if daily else None

    K = pool_arr.shape[2]
    per_layer: list[float] = []
    for k in range(K):
        dk: list[float] = []
        for t in range(T):
            c = _spearman_abs_corr(cand[t], pool_arr[t, :, k])
            if np.isfinite(c):
                dk.append(c)
        if dk:
            per_layer.append(float(np.mean(dk)))
    max_abs = float(max(per_layer)) if per_layer else None
    return {"mean_abs_corr": mean_abs, "max_abs_corr": max_abs}


def corr_diag_max(cand: np.ndarray, pool_arr: np.ndarray) -> float | None:
    """候选与逐池层的时间平均 |spearman| 相关最大值（诊断）。"""
    d = _candidate_pool_corr(cand, pool_arr)
    return d["max_abs_corr"]
