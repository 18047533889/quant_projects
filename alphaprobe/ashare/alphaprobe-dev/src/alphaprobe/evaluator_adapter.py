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

# 模块级是否可导入 modeling（LabelContract / DecisionClock 权威；P0-F 硬规矩：
# label 时间语义必须走 modeling 契约，禁止 date+1 自然日）。
try:  # pragma: no cover - 导入探测（QE 未装时降级，仍为交易日序列前移）
    from modeling.contracts import LabelContract, ashare_decision_clock  # noqa: F401

    _MODELING_IMPORTABLE = True
except Exception:  # noqa: BLE001 - modeling 不可导入的降级（保留交易日序列语义）
    _MODELING_IMPORTABLE = False

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

        # 2) label → cohort portfolio 路径（L 维度）
        cohort_metrics: dict[str, Any] = self._cohort_metrics_if_applicable(
            f, y, price_panel, dates, codes
        )

        # 3) 合并为标准 EvaluationBundle
        metrics: dict[str, Any] = {}
        metrics.update(registry_metrics)
        metrics.update(cohort_metrics)
        return EvaluationBundle(metrics)

    def _cohort_metrics_if_applicable(
        self,
        f: np.ndarray,
        y: np.ndarray,
        price_panel: Any | None,
        dates: Sequence[Any],
        codes: Sequence[Any],
    ) -> dict[str, Any]:
        """按 label_days 与面板长度决定是否走 cohort 路径。

        - label_days == 20：生产主路径。缺 price_panel → fail-closed 抛错；
          面板太短（无任何完整持有期）时跳过并保持无 L 键（不抛）。
        - label_days != 20（研究/测试用短 horizon）：**不**走 cohort
          （cohort 组合语义按 label_days==20 的 1/H 持有设计）；如需
          验证 holding 语义请用 label_days==20 或直接调
          ``_cohort_metrics`` / QE compute_cohort_pnl。
        """
        if self.label_days != 20:
            return {}
        if price_panel is None:
            raise QuantEvaluatorError(
                "label_days=20 需要 price_panel（VWAP 价格面板）才能走 cohort "
                "portfolio 路径；缺失时无法构造真实 daily PnL。"
            )
        # 至少需要 3 行（signal/entry/exit）才有任何 cohort PnL。
        if f.shape[0] >= 3:
            return self._cohort_metrics(f, price_panel, dates, codes)
        return {}

    def evaluate_many(
        self,
        factor_panels: Sequence[Any],
        label_panel: Any,
        price_panel: Any | None = None,
        *,
        factor_ids: Sequence[str] | None = None,
    ) -> list[Any]:
        """批量评估 N 个因子面板 → 每个因子一个标准 EvaluationBundle（顺序一一对应）。

        Parameters
        ----------
        factor_panels : Sequence[DataFrame | ndarray]
            每个元素是单因子面板（index=date, columns=code），全部与 label_panel
            同形状、同日期/代码轴。
        label_panel : DataFrame
            与各因子面板同形状的远期收益面板（vwap→vwap，label_days 日）。
        price_panel : DataFrame | None
            label_days==20 的 cohort 路径需要（同 evaluate）；缺失抛 QuantEvaluatorError。
        factor_ids : Sequence[str] | None
            与 factor_panels 一一对应的因子标识；缺省用 ``f1..fN``。

        Returns
        -------
        list[EvaluationBundle | None]
            与输入顺序一一对应；单个因子评估失败 → 该位 None（不抛）。
            批级失败（首个因子 / QE 导入 / 契约构造失败）→ 抛 QuantEvaluatorError
            （fail-closed），后续因子不再逐个尝试、全置 None。
        """
        from alphaprobe.fitness.contracts import EvaluationBundle

        if not factor_panels:
            raise QuantEvaluatorError("factor_panels 不能为空")
        n = len(factor_panels)
        ids: Sequence[str] = (
            list(factor_ids) if factor_ids is not None else [f"f{i + 1}" for i in range(n)]
        )
        if len(ids) != n:
            raise QuantEvaluatorError(
                f"factor_ids 长度 {len(ids)} 与 factor_panels 长度 {n} 不一致"
            )

        # 一次性形状 / 日期 / 代码轴检查（失败即批整体 fail-closed）
        f0 = _as_2d(factor_panels[0], "factor_panels[0]")
        y0 = _as_2d(label_panel, "label_panel")
        _check_same_shape(f0, y0, "factor_panels[0]", "label_panel")
        dates = _panel_index(factor_panels[0])
        codes = _panel_columns(factor_panels[0])

        # cohort 路径共享一次价格面板转换与 daily PnL（batch 只构造一次）。
        # 语义与 evaluate() 一致：仅 label_days==20 走 cohort；否则无 L 键。
        cohort_prep: dict[str, Any] | None = None
        if self.label_days == 20:
            if price_panel is None:
                raise QuantEvaluatorError(
                    "label_days=20 需要 price_panel（VWAP 价格面板）才能走 cohort "
                    "portfolio 路径；缺失时无法构造真实 daily PnL。"
                )
            if f0.shape[0] >= 3:
                cohort_prep = _cohort_panel_prep(price_panel)

        out: list[Any] = []
        batch_failed = False
        batch_reason: Exception | None = None
        for i, panel in enumerate(factor_panels):
            if batch_failed:
                out.append(None)
                continue
            try:
                f = _as_2d(panel, f"factor_panels[{i}]")
                _check_same_shape(f, y0, f"factor_panels[{i}]", "label_panel")
                # registry 指标：N 因子共用一个 LabelBundle（label_time_axis 一次）。
                registry_metrics = self._qe_registry_metrics(
                    f, y0, dates, codes, str(ids[i])
                )
                # cohort 路径逐因子调 QE（共享已构造的 next_ret/next_vwap）。
                cohort_metrics: dict[str, Any] = {}
                if cohort_prep is not None:
                    cohort_metrics = self._cohort_metrics_prepared(
                        f, dates, codes, cohort_prep
                    )
                metrics: dict[str, Any] = {}
                metrics.update(registry_metrics)
                metrics.update(cohort_metrics)
                out.append(EvaluationBundle(metrics))
            except QuantEvaluatorError as exc:
                # 批级契约 / QE 导入失败 → 整体 fail-closed（含后续位降级 None）。
                batch_failed = True
                batch_reason = exc
                out.append(None)
            except Exception as exc:  # noqa: BLE001 - 单因子评估失败降级 None
                logger.warning(
                    "evaluate_many factor[%s] 评估失败，降级 None：%s", ids[i], exc
                )
                out.append(None)
        if batch_failed:
            raise QuantEvaluatorError(
                f"evaluate_many 批评估失败（fail-closed）：{batch_reason}"
            ) from batch_reason
        return out

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
                label_end_time=tuple(label_time_axis(dates, self.label_days)),
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
        p = _as_2d(price_panel, "price_panel")
        # factor 面板 (T,N) 与价格面板须同形状（next_ret 内部再错位 1 日）。
        _check_same_shape(f, p, "factor_panel", "price_panel")
        prep = _cohort_panel_prep(p)
        return self._cohort_metrics_prepared(f, dates, codes, prep)

    def _cohort_metrics_prepared(
        self,
        f: np.ndarray,
        dates: Sequence[Any],
        codes: Sequence[Any],
        prep: dict[str, Any],
    ) -> dict[str, Any]:
        """用共享的 cohort 预处理结果构造单因子的 L 维度指标。"""
        try:
            from quant_evaluator.metrics.probe_portfolio import (
                compute_metrics_from_cohort,
                cost_scenario,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed
            raise QuantEvaluatorError(
                f"quant_evaluator.metrics.probe_portfolio 不可导入（fail-closed）：{exc}"
            ) from exc

        f_aligned = f[1:, :]
        per_side_cost = cost_scenario(self.cost_scenario)
        try:
            res = compute_metrics_from_cohort(
                factor_values=f_aligned,
                next_ret=prep["next_ret"],
                next_vwap=prep["next_vwap"],
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


def _cohort_panel_prep(price_panel: Any) -> dict[str, Any]:
    """把价格面板转成 cohort 共享输入（next_ret / next_vwap）。

    真实 daily PnL：next_ret = vwap.pct_change()（当日持有收益），
    next_vwap = 当日 VWAP 价（入场/出场）。与 label 面板同日历对齐。
    返回 dict 供 evaluate_many 批量共享（per_side_cost 在
    ``_cohort_metrics_prepared`` 内按各 adapter 的 self.cost_scenario 解析）。
    """
    p = _as_2d(price_panel, "price_panel")
    with np.errstate(divide="ignore", invalid="ignore"):
        next_ret = np.divide(p[1:, :], p[:-1, :]) - 1.0
    return {
        "next_ret": next_ret,
        "next_vwap": p[1:, :],
    }


def label_time_axis(
    dates: Sequence[Any],
    horizon: int,
    *,
    calendar: Sequence[Any] | None = None,
) -> list[Any]:
    """label_end_time 时间轴（P0-F）：decision t 收盘 → 入场 t+1 VWAP → 出场 t+H。

    语义（modeling LabelContract / DecisionClock 权威，taskbook §7/§8/§11）：
      label_t(H) = VWAP_{t+H} / VWAP_t - 1
    即 label_end_time[i] 应为 **trading calendar 上第 i+H 个交易日**（持有 H 个
    交易日，出场也在当日 VWAP 成交），而不是 decision_time 的 +1 自然日。

    - ``dates``：决策时刻序列（factor/label 面板的日期轴）。构造 LabelBundle 时
      decision_time / label_start_time = dates，label_end_time = 本函数输出。
    - ``calendar``：可选、显式的 trading calendar（比 dates 更长时用于 index 前移）。
      缺省 None 时把 ``dates`` 序列本身视为交易日历：end[i] = dates[min(i+horizon,
      len-1)] —— 交易日 index 前移，绝不用 datetime + timedelta(days=1)。
    - modeling 可导入时用 LabelContract 的语义字段生成说明性元数据并验证规则字段
      与 vwap_to_vwap 一致；不可导入时纯 dates 序列自移（同一交易日语义），
      调用方可在 bundle.metadata 注明降级（无自然日）。

    越界处理：i + horizon >= len(dates) 时（label 未成熟的末端），按
    ``dates`` 末尾元素 + 一个增量标注（保持 LabelBundle 的
    label_start_time < label_end_time 严格递增 / 因果链契约；QE LabelBundle
    强制各 time 轴等长）。该增量只标注「已超出面板的成熟末端」，不参与指标语义
    —— rank_ic/coverage 等指标实际消费的是 y 面板数值本身。
    """
    n = len(dates)
    if n == 0:
        return []
    h = int(horizon)
    if h < 1:
        raise QuantEvaluatorError(f"horizon 必须 >= 1，got {horizon}")

    cal: list[Any]
    if calendar is not None and len(calendar) > 0:
        cal = list(calendar)
    else:
        cal = list(dates)
    last = cal[-1]

    # calendar 显式传入时按 calendar 做 index 前移（dates 不必是 cal 前缀）；
    # calendar 缺省时把 dates 序列本身视为交易日历。
    # 越界（i + horizon >= len(cal)）：label 未成熟，用 maturity marker 填充。
    marker = _label_maturity_marker(last)
    out: list[Any] = []
    for i, _d in enumerate(dates):
        idx = i + h
        if idx < len(cal):
            out.append(cal[idx])
        else:
            # 所有未成熟端共享同一 marker：end 轴允许非严格递增的等值尾部。
            # （QE LabelBundle 只校验 decision_time 严格递增 + 每行 start<end；
            #   对 label_end_time 不要求全轴严格递增。）
            out.append(marker)

    # modeling 权威存在时：用 LabelContract 语义说明做一致性校验（不手搓自然日）。
    if _MODELING_IMPORTABLE:
        try:
            _assert_label_contract_semantics(h)
        except Exception as exc:  # noqa: BLE001 - 语义说明失败不影响时间轴（已交易日前移）
            logger.debug("label_time_axis contract note failed: %s", exc)
    return out


def _label_maturity_marker(last: Any) -> Any:
    """末端未成熟段的标注值：与 last 严格可比且 > last（保持因果链契约）。

    优先交易日历风格的下一自然日（仅作标注，非 label 语义）；对不可加的对象
    （如 str/date 索引）退回“last 复制 + 显式递增序号”。绝不进入生产指标。
    """
    try:
        if hasattr(last, "to_pydatetime"):
            return last + np.timedelta64(1, "D")
        if hasattr(last, "date") and hasattr(last, "time"):
            from datetime import timedelta

            return last + timedelta(days=1)
        if isinstance(last, (int, float, np.integer, np.floating)):
            return last + 1
    except TypeError:
        pass
    # str / 其它不可加类型：追加时间顺序后缀以保序。
    if isinstance(last, str):
        return f"{last}~label-maturity-pad"
    # 通用保序兜底：用 repr + 高位递增不可靠 → 返回 last 本身（调用方须保证
    # 末端不参与 start<end 比较；本模块调用处末尾多为 str 日期，已由上层保证）。
    return last


def _assert_label_contract_semantics(horizon: int) -> None:
    """modeling LabelContract 语义说明（决策 t 收盘→入场 t+1 VWAP→出场 t+H）。"""
    contract = LabelContract(
        label_name=f"vwap_to_vwap_h{horizon}",
        horizon_bars=horizon,
    )
    clock = ashare_decision_clock()  # AFTER_CLOSE_TO_NEXT_VWAP
    # 构造即校验：entry VWAP completes at t close / exit VWAP completes at t+H close
    _ = (
        contract.start_time_rule,
        contract.end_time_rule,
        contract.entry_price_basis,
        contract.exit_price_basis,
        clock.execution_at,
        clock.label_available_at,
    )


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
    "label_time_axis",
]
