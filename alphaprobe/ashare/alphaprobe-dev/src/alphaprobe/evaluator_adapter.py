"""alphaprobe.evaluator_adapter（任务书 §31-§32）：统一评估层接入。

QuantEvaluatorAdapter 是 AlphaPROBE 评估链的**唯一评估入口**：输入
factor panel（DataFrame index=date columns=code）+ label panel + 可选价格面板，
输出标准 :class:`EvaluationBundle`（fitness/contracts.py 的强类型视图，字段名
与 QE registry / probe_portfolio 对齐，不造第二套）。

设计约束（V2-E）：
- 内部全部走 quant_evaluator 公共 API（runtime/evaluator.evaluate() /
  registry / metrics.probe_portfolio），**禁止手写** RankIC / ICIR / Sharpe /
  Sortino / MDD / turnover 计算。
- 20d label（label_days=20）时自动切换 cohort portfolio 路径：调 QE
  metrics/probe_portfolio 的 compute_metrics_from_cohort 得 LS 与 long-only-active
  双口径，把 Long-Short 维度所需的 9 项指标填进 EvaluationBundle；**禁止**用
  overlapping forward return 直接年化 Sharpe。
- fail-closed：QE 不可导入 / 评估抛错时抛明确异常（不是静默 0 分）。
- 成本情景参数化（gross/1x/2x/3x），默认 gross，adapter 接受 cost_scenario。
- 泄漏纪律：本模块只消费调用方传入的 train 段面板，不构造 test 段数据。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# QE registry 指标 → fitness EvaluationBundle 键（P/S/R 维度；L 维度走 cohort）。
# 只读 QE registry 的 compute_fn 结果，不重算。
_REGISTRY_METRICS: tuple[str, ...] = (
    "rank_ic",        # → rankic_valid
    "ic_ir",          # → rankicir
    "hac_tstat",      # → hac_tstat
    "coverage",       # → coverage
    "turnover",       # → turnover
)

# 允许的成本情景名（与 QE metrics/probe_portfolio.costs.COST_SCENARIOS 对齐）。
COST_SCENARIOS = ("gross", "1x", "2x", "3x")


class QuantEvaluatorError(RuntimeError):
    """QE 不可导入 / 评估失败时抛出的明确异常（fail-closed）。"""


class QuantEvaluatorAdapter:
    """统一评估入口：factor/label 面板 → 标准 EvaluationBundle。

    Parameters
    ----------
    label_days : int
        远期收益 label 的持有期（默认 20）。==20 时走 cohort portfolio 路径。
    cost_scenario : str | float
        "gross" | "1x" | "2x" | "3x"（或直接 per-side cost 比例）。默认 gross。
    n_quantiles : int
        cohort 分位数（默认 10：D1..D10）。
    holding : int
        cohort 持有期（默认 20，与 label_days 一致）。
    min_periods : int
        指标层最小有效期数（透传 compute_portfolio_metrics）。
    """

    def __init__(
        self,
        *,
        label_days: int = 20,
        cost_scenario: str | float = "gross",
        n_quantiles: int = 10,
        holding: int = 20,
        min_periods: int = 20,
    ) -> None:
        self.label_days = int(label_days)
        self.cost_scenario = cost_scenario
        self.n_quantiles = int(n_quantiles)
        self.holding = int(holding)
        self.min_periods = int(min_periods)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def evaluate(
        self,
        factor_panel: Any,
        label_panel: Any,
        price_panel: Any | None = None,
        *,
        factor_id: str = "f1",
    ) -> Any:
        """评估单因子 → 标准 EvaluationBundle。

        Parameters
        ----------
        factor_panel : DataFrame
            index=date, columns=code，因子值面板。
        label_panel : DataFrame
            与 factor_panel 同形状的远期收益面板（vwap→vwap，label_days 日）。
        price_panel : DataFrame | None
            可选，index=date, columns=code 的 VWAP 价格面板。label_days==20 的
            cohort 路径需要它来构造真实 daily PnL；缺失时抛 QuantEvaluatorError。
        factor_id : str
            因子标识（透传进 FactorBatch / 结果）。

        Returns
        -------
        EvaluationBundle
            fitness/contracts.py 的强类型视图（metrics 键与 QE registry /
            probe_portfolio 对齐）。
        """
        from alphaprobe.fitness.contracts import EvaluationBundle

        f = _as_2d(factor_panel, "factor_panel")
        y = _as_2d(label_panel, "label_panel")
        _check_same_shape(f, y, "factor_panel", "label_panel")
        dates = _panel_index(factor_panel)
        codes = _panel_columns(factor_panel)

        # 1) QE registry 指标（P/S/R 维度）
        registry_metrics = self._qe_registry_metrics(f, y, dates, codes, factor_id)

        # 2) 20d label → cohort portfolio 路径（L 维度）
        cohort_metrics: dict[str, Any] = {}
        if self.label_days == 20:
            if price_panel is None:
                raise QuantEvaluatorError(
                    "label_days=20 需要 price_panel（VWAP 价格面板）才能走 cohort "
                    "portfolio 路径；缺失时无法构造真实 daily PnL。"
                )
            cohort_metrics = self._cohort_metrics(f, price_panel, dates, codes)

        # 3) 合并为标准 EvaluationBundle
        metrics: dict[str, Any] = {}
        metrics.update(registry_metrics)
        metrics.update(cohort_metrics)
        return EvaluationBundle(metrics)

    # ------------------------------------------------------------------
    # QE registry 指标（P/S/R）
    # ------------------------------------------------------------------

    def _qe_registry_metrics(
        self,
        f: np.ndarray,
        y: np.ndarray,
        dates: Sequence[Any],
        codes: Sequence[Any],
        factor_id: str,
    ) -> dict[str, Any]:
        """走 QE runtime.evaluator.evaluate() 取 registry 指标，映射到 fitness 键。"""
        try:
            from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
            from quant_evaluator.contracts.label_bundle import LabelBundle
            from quant_evaluator.runtime.evaluator import evaluate as qe_evaluate
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise QuantEvaluatorError(
                f"quant_evaluator 不可导入（fail-closed）：{exc}"
            ) from exc

        T, N = f.shape
        try:
            fb = FactorBatch(
                factor_ids=(factor_id,),
                time_axis=AxisRef("time", "str", T, values=np.asarray(dates)),
                asset_axis=AxisRef("asset", "str", N, values=np.asarray(codes)),
                values=f[:, :, np.newaxis],
            )
            lb = LabelBundle(
                target_id="vwap_to_vwap",
                values=y,
                horizon=self.label_days,
                decision_time=tuple(dates),
                label_start_time=tuple(dates),
                label_end_time=tuple(_next_time(dates)),
            )
            result = qe_evaluate(fb, lb, metrics=_REGISTRY_METRICS)
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise QuantEvaluatorError(
                f"quant_evaluator.evaluate 失败（fail-closed）：{exc}"
            ) from exc

        mv = getattr(result, "metric_values", {}) or {}
        out: dict[str, Any] = {}
        out["rankic_valid"] = _metric_value(mv, "rank_ic")
        out["rankicir"] = _metric_value(mv, "ic_ir")
        out["ic_ir"] = _metric_value(mv, "ic_ir")
        out["hac_tstat"] = _metric_value(mv, "hac_tstat")
        out["coverage"] = _metric_value(mv, "coverage")
        out["turnover"] = _metric_value(mv, "turnover")
        # 数据质量：nan/inf 占比（简单面板统计，非金融指标，不属禁止手写项）
        out["nan_inf_ratio"] = float(np.mean(~np.isfinite(f)))
        return out

    # ------------------------------------------------------------------
    # 20d cohort portfolio 路径（L 维度）
    # ------------------------------------------------------------------

    def _cohort_metrics(
        self,
        f: np.ndarray,
        price_panel: Any,
        dates: Sequence[Any],
        codes: Sequence[Any],
    ) -> dict[str, Any]:
        """调 QE probe_portfolio.compute_metrics_from_cohort 得 LS + long-only-active。"""
        try:
            from quant_evaluator.metrics.probe_portfolio import (
                compute_metrics_from_cohort,
                cost_scenario,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise QuantEvaluatorError(
                f"quant_evaluator.metrics.probe_portfolio 不可导入（fail-closed）：{exc}"
            ) from exc

        p = _as_2d(price_panel, "price_panel")
        _check_same_shape(f, p, "factor_panel", "price_panel")
        # 真实 daily PnL：next_ret = vwap.pct_change()（当日持有收益），
        # next_vwap = 当日 VWAP 价（入场/出场）。与 label 面板同日历对齐。
        with np.errstate(divide="ignore", invalid="ignore"):
            next_ret = np.divide(p[1:, :], p[:-1, :]) - 1.0
        next_vwap = p[1:, :]
        f_aligned = f[1:, :]

        per_side_cost = cost_scenario(self.cost_scenario)
        try:
            res = compute_metrics_from_cohort(
                factor_values=f_aligned,
                next_ret=next_ret,
                next_vwap=next_vwap,
                n_quantiles=self.n_quantiles,
                holding=self.holding,
                per_side_cost=per_side_cost,
                min_periods=self.min_periods,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise QuantEvaluatorError(
                f"compute_metrics_from_cohort 失败（fail-closed）：{exc}"
            ) from exc

        ls = (res.get("metrics") or {}).get("long_short") or {}
        loa = (res.get("metrics") or {}).get("long_only_active") or {}
        return {
            # Long-Short 维度 9 项（fitness LWeights 消费）
            "net_sharpe": ls.get("sharpe"),
            "calmar_ratio": ls.get("calmar"),
            "sortino_ratio": ls.get("sortino"),
            "net_annualized_ls_return": ls.get("annualized_return"),
            "d10_long_only_active_return": loa.get("annualized_return"),
            "max_drawdown": ls.get("max_drawdown"),
            "drawdown_persistence": ls.get("drawdown_persistence"),
            "q20_rolling_sharpe": ls.get("rolling_sharpe_q20"),
            "positive_month_ratio": ls.get("positive_month_ratio"),
            # 补充（fitness _drawdown_persistence 消费）
            "max_dd_duration": ls.get("max_drawdown_duration"),
            "tuw": ls.get("time_underwater_pct"),
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _as_2d(panel: Any, name: str) -> np.ndarray:
    """DataFrame / ndarray → (T, N) float64 面板。"""
    if panel is None:
        raise QuantEvaluatorError(f"{name} 不能为 None")
    if hasattr(panel, "to_numpy"):
        arr = np.asarray(panel.to_numpy(), dtype=np.float64)
    else:
        arr = np.asarray(panel, dtype=np.float64)
    if arr.ndim != 2:
        raise QuantEvaluatorError(f"{name} 必须为二维面板，got ndim={arr.ndim}")
    return arr


def _check_same_shape(a: np.ndarray, b: np.ndarray, na: str, nb: str) -> None:
    if a.shape != b.shape:
        raise QuantEvaluatorError(
            f"{na} 与 {nb} 形状不一致：{a.shape} vs {b.shape}"
        )


def _panel_index(panel: Any) -> list[Any]:
    if hasattr(panel, "index"):
        return list(panel.index)
    return list(range(panel.shape[0]))


def _panel_columns(panel: Any) -> list[Any]:
    if hasattr(panel, "columns"):
        return list(panel.columns)
    return list(range(panel.shape[1]))


def _next_time(dates: Sequence[Any]) -> list[Any]:
    """label_end_time：decision_time 的下一时刻（严格递增、正 label 窗口）。

    对 pandas Timestamp / datetime 加一天；对 int 加 1；其余原样（尽力而为）。
    """
    out: list[Any] = []
    for d in dates:
        try:
            if hasattr(d, "to_pydatetime"):
                out.append(d + np.timedelta64(1, "D"))
            elif hasattr(d, "date") and hasattr(d, "time"):
                from datetime import timedelta

                out.append(d + timedelta(days=1))
            else:
                out.append(d + 1)
        except TypeError:
            out.append(d)
    return out


def _metric_value(mv: dict[str, Any], key: str) -> Any:
    """从 QE EvaluationBundle.metric_values 取标量值（None 安全）。"""
    item = mv.get(key)
    if item is None:
        return None
    v = getattr(item, "value", None)
    if v is None:
        return None
    try:
        return float(v) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "QuantEvaluatorAdapter",
    "QuantEvaluatorError",
    "COST_SCENARIOS",
]
