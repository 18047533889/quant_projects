"""alphaprobe.pipeline（任务书 §41-§42）：默认主链编排器。

runner.py 的 **新默认主链**（--pipeline new）：SearchOrchestrator 生成候选
→ dedup 硬重复过滤 → FidelityFunnel L0/L1/L2 多保真评估 → SearchFitness 打分
→ ActivePool Pareto+QD 准入 → GlobalMemoryStore 实时写记忆。

设计约束：
- 所有依赖可注入且可降级：evaluator 缺省用 LegacyCompatEvaluator + fe_bridge
  真算 evaluate_fn（配 vwap→vwap label）；evaluator 不可用 → 纯静态降级；
- llm_fn=None 时用确定性 stub llm（规则化文本变异，不碰任何网络/模型）；
- **泄漏纪律**：本模块全程不接收、不构造 test 段数据；label 一律 vwap→vwap
  远期收益（Ref(vwap,-20)/vwap-1），数据接口只读 data_train 段；
- 模块级不 import torch / openai（runner 的 legacy 路径才需要）。
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

# P0-A：ExecutionMode 供 SearchPipeline dataclass 字段默认值使用。延迟 import
# 会破坏 dataclass 字段默认值求值，故模块级轻量导入（llm_client 不 import
# torch/openai，无副作用）。
from alphaprobe.llm_client import ExecutionMode

logger = logging.getLogger(__name__)

# 默认搜索轮预算：与 runner --search_time 语义一致（沿用 quota/stop 逻辑）。
DEFAULT_ROUNDS = 40
DEFAULT_BUDGET_PER_ROUND = 12
DEFAULT_TARGET = 1024
DEFAULT_MAX = 2048


# ---------------------------------------------------------------------------
# 配置 / 结果
# ---------------------------------------------------------------------------


#: PipelineConfig.structured_generation 的「用户未显式传」哨兵（P0-A）。
#: 生产语义默认 True；OFFLINE_TEST 下保持旧行为（False）以不破坏既有离线测试
#: ——用 dataclass field 对象哨兵区分「用户没传」与「显式传 False」。
UNSET = object()


def _offline_test_mode() -> Any:
    """SearchPipeline.mode 的 dataclass 默认工厂。

    顶层不 import llm_client（模块级不 import torch/openai 的纪律；llm_client
    本身轻量，但 class-body 引用 ExecutionMode 会让 dataclass 默认值在模块
    导入期求值）。改为惰性工厂，模块导入即安全。
    """
    from alphaprobe.llm_client import ExecutionMode

    return ExecutionMode.OFFLINE_TEST


@dataclass
class PipelineConfig:
    """§41 主链配置。所有阈值可注入；缺省即默认值。"""

    pool_target: int = DEFAULT_TARGET
    pool_max: int = DEFAULT_MAX
    budget_per_round: int = DEFAULT_BUDGET_PER_ROUND
    max_rounds: int = DEFAULT_ROUNDS
    # L0 静态检查（不跑市场回测）
    l1_gate: dict[str, float] = field(
        default_factory=lambda: {"rankic": 0.0, "coverage": 0.5, "nan_inf_ratio": 0.5}
    )
    l2_gate: dict[str, float] = field(default_factory=dict)
    # §58 导出 gate：refinement 未接外部引擎前不强制拒绝（仅记录 decision）
    export_gate_enforced: bool = False
    # 结构化 arm 每次生成目标数
    expected_num: int = 3
    # V2-H：结构化 generation 开关。生产语义默认 True（orchestrator 结构化路径，
    # parents 来自 parent_selector top-k DAG 采样、候选记录 lineage）。P0-A 兼容：
    # SearchPipeline.__post_init__ 在 mode==OFFLINE_TEST 且用户未显式传本字段时，
    # 保持旧行为（False）——见 _resolve_structured_generation。
    structured_generation: Any = UNSET


@dataclass
class RoundResult:
    """一轮主链的统计快照（RoundManager._update_memory 消费）。"""

    round_id: str
    candidates_generated: int = 0
    duplicates_filtered: int = 0
    evaluated: int = 0
    admitted: int = 0
    degraded: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    pool_snapshot: list[dict[str, Any]] = field(default_factory=list)
    memory_db_path: str = ""


# ---------------------------------------------------------------------------
# 确定性 stub llm_fn（§41：llm_fn=None 时保证无 LLM 额度也可端到端转）
# ---------------------------------------------------------------------------

STUB_FIELDS = ("close", "open", "high", "low", "volume", "vwap", "amount", "turnover")
STUB_WINDOWS = (5, 10, 20, 60)
_STUB_FAMILIES: tuple[tuple[str, str], ...] = (
    ("rank", "zscore"),
    ("ts_mean", "ts_std"),
    ("ts_mean", "ts_median"),
    ("ts_corr", "ts_cov"),
)


def _stub_replace(base: str, i: int) -> str:
    """确定性文本变异：替换 field / window / 算子（规则化，不调网络）。"""
    old, new = _STUB_FAMILIES[i % len(_STUB_FAMILIES)]
    if old in base:
        return base.replace(old, new, 1)
    field_old = STUB_FIELDS[i % len(STUB_FIELDS)]
    if field_old in base:
        return base.replace(field_old, STUB_FIELDS[(i + 1) % len(STUB_FIELDS)], 1)
    win_old = STUB_WINDOWS[i % len(STUB_WINDOWS)]
    return base.replace(str(win_old), str(STUB_WINDOWS[(i + 1) % len(STUB_WINDOWS)]), 1)


def make_stub_llm_fn(
    *,
    rng: Any | None = None,
    forced_candidates: Sequence[dict[str, Any]] | None = None,
    structured: bool = False,
) -> Callable[[str, str, str], str]:
    """构造确定性 stub llm_fn（返回 §31 JSON 文本）。

    forced_candidates 非空时：把固定 candidate JSON 序列化返回（测试用）；
    否则：从 parents 与 action 规则化生成变异候选（离线端到端默认路径）。
    绝不发起任何网络 / 模型调用。

    ``structured=True``（V2-H）：返回结构化 action JSON（dict 含 ``action`` 字段），
    供 ``normalize_llm_output`` 消费；默认 False = 返回旧文本（§31 candidates JSON），
    向后兼容（既有 pipeline 测试原样全绿）。
    """
    import random as _random

    rng = rng or _random.Random(0)

    def _llm_fn(system_prompt: str, user_prompt: str, model_class: str) -> str:
        if structured:
            from alphaprobe.generation.structured import make_structured_stub_llm_fn

            return make_structured_stub_llm_fn(
                rng=rng, forced_actions=forced_candidates
            )(system_prompt, user_prompt, model_class)
        if forced_candidates:
            return json.dumps({"candidates": list(forced_candidates)}, ensure_ascii=False)
        # 从 user_prompt 的 parent 行提取公式（"## Parent factor(s)" 之后 "- f :: desc"）。
        # prompt 行用 .rstrip() 生成（`- rank(close) ::`），因此按 "::" 切分（formula
        # 本身不含冒号），取前段 strip 后即公式。
        parents: list[dict[str, Any]] = []
        for line in user_prompt.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:]
            f = body.split("::", 1)[0].strip()
            if f and f != "?":
                parents.append({"formula": f})
        if not parents:
            return json.dumps({"candidates": []}, ensure_ascii=False)
        out: list[dict[str, Any]] = []
        for i, p in enumerate(parents):
            base = str(p.get("formula") or "")
            if not base:
                continue
            for k in range(3):
                mutated = _stub_replace(base, i * 3 + k)
                if mutated == base:
                    continue
                out.append(
                    {
                        "formula": mutated,
                        "explanation": f"stub-llm mutate #{k}",
                        "hypothesis": "确定性 stub：规则化文本变异",
                        "action_type": "REFINE",
                        "parent_ids": [],
                        "schema_tags": {},
                    }
                )
        return json.dumps({"candidates": out}, ensure_ascii=False)

    return _llm_fn


def _default_llm_fn() -> Callable[[str, str, str], str]:
    """pipeline 模块级默认 stub（模块不 import torch/openai，离线可跑）。

    .. note:: **legacy（仅 OFFLINE_TEST）**——P0-A 起 llm 解析走
       :func:`alphaprobe.llm_client.resolve_llm_fn`；本函数仅保留给未显式传
       mode 的默认构造路径（mode==OFFLINE_TEST）。PRODUCTION fail-closed。
    """
    return make_stub_llm_fn()


# ---------------------------------------------------------------------------
# 默认 evaluate_fn（fe_bridge 真算：vwap→vwap 20 日 label）
# ---------------------------------------------------------------------------


def make_fe_evaluate_fn(
    stock_data: Any,
    *,
    label_days: int = 20,
    segment: str = "train",
) -> Callable[[Sequence[str], str, Any], list[dict[str, float | None]]]:
    """fe_bridge 真算 evaluate_fn（LegacyCompatEvaluator 注入用）。

    公式列表 → FactorEngineStockData.evaluate_many 批算因子平面 → vwap→vwap
    远期收益 label（Ref(vwap,-label_days)/vwap-1）→ 每因子算 rank_ic/ic/icir/
    coverage/nan_inf_ratio，返回 metric bundle dict 列表（与 formula 一一对应）。

    数据无法构造时返回全 None bundle 并记录 degraded（不抛）。
    泄漏纪律：只读传入的 train 段 stock_data，不接收 test 段数据。

    .. note:: **legacy（仅 OFFLINE_TEST）**——生产评估只认 evaluator/QE 权威，
       ``make_fe_evaluate_fn`` 属本地 numpy/scipy 手算回退路径，PRODUCTION 模式
       由 SearchPipeline._evaluate 的 fail-closed 守卫禁止回落本函数。
    """
    import numpy as np

    def _evaluate_fn(
        formulas: Sequence[str],
        fidelity: str = "L2_full_train",
        context: Any = None,
    ) -> list[dict[str, float | None]]:
        bundles: list[dict[str, float | None]] = []
        if stock_data is None:
            return [None] * len(formulas)
        fmls = [str(f) for f in formulas if f]
        try:
            planes = stock_data.evaluate_many(fmls)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 数据不可用降级为全 None，不抛
            logger.warning("pipeline evaluate_many failed, degraded: %s", exc)
            return [None] * len(formulas)
        try:
            # vwap → 远期收益 label：Ref(vwap,-label_days)/vwap-1（后复权口径）
            names = list(stock_data._field_names())
            if "vwap" not in names:
                logger.warning("pipeline: no vwap field, degraded")
                return [None] * len(formulas)
            vwap = stock_data.data[:, :, names.index("vwap")]
            T = vwap.shape[0]
            if T <= label_days:
                logger.warning("pipeline: series too short for label_days, degraded")
                return [None] * len(formulas)
            label = vwap[label_days:, :] / vwap[:-label_days, :] - 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("pipeline vwap label build failed, degraded: %s", exc)
            return [None] * len(formulas)
        # 因子平面与 label 对齐：factor 也用 [label_days:, :] 段
        bundles: list[dict[str, float | None]] = []
        for plane in planes:
            try:
                f_arr = np.asarray(plane, dtype=float)
            except Exception:  # noqa: BLE001
                f_arr = plane
            if len(f_arr.shape) == 3:
                # evaluate_many 返回 (T,N,F) 时取最后一维（F=1）
                f_arr = f_arr[:, :, -1]
            try:
                f_arr = f_arr[-label.shape[0]:, :]
            except Exception:  # noqa: BLE001
                pass
            bundles.append(_bundle_from_plane(f_arr, label))
        return bundles

    return _evaluate_fn


def _bundle_from_plane(
    plane: Any,
    label: Any,
) -> dict[str, float | None]:
    """单因子平面 × label → 基础 metric bundle（离线 numpy/scipy，不跑 qlib）。

    .. note:: **legacy（仅 OFFLINE_TEST）**——同 :func:`make_fe_evaluate_fn`，
       PRODUCTION 评估不消费本函数（无手算回退）。
    """
    try:
        import numpy as np
        from scipy import stats as _scipy_stats

        f = np.asarray(plane, dtype=float)
        y = np.asarray(label, dtype=float)
        T, N = f.shape
        if T == 0 or N == 0:
            return _all_none()
        # 对齐到有效 label 段
        f = f[-y.shape[0]:, :]
        T = f.shape[0]
        valid = np.isfinite(f) & np.isfinite(y)
        f_clean = np.where(valid, f, np.nan)
        cov = np.mean(np.isfinite(f), axis=1)
        coverage = float(np.mean(cov >= 0.5))
        nan_ratio = float(np.mean(~np.isfinite(f)))
        rankics: list[float] = []
        ics: list[float] = []
        for t in range(T):
            ft = f_clean[t]
            yt = y[t]
            mask = np.isfinite(ft) & np.isfinite(yt)
            if mask.sum() < 30:
                continue
            r = _scipy_stats.spearmanr(ft[mask], yt[mask], nan_policy="omit")
            if r is not None and hasattr(r, "correlation") and r.correlation is not None and np.isfinite(r.correlation):
                rankics.append(float(r.correlation))
            c = _scipy_stats.pearsonr(ft[mask], yt[mask])
            if c is not None and hasattr(c, "statistic") and c.statistic is not None and np.isfinite(c.statistic):
                ics.append(float(c.statistic))
        rank_ic = float(np.mean(rankics)) if rankics else None
        ic = float(np.mean(ics)) if ics else None
        icir = float(np.mean(rankics) / (np.std(rankics) + 1e-6)) if len(rankics) > 1 else None
        return {
            "rankic": rank_ic,
            "ic": ic,
            "icir": icir,
            "coverage": coverage,
            "nan_inf_ratio": nan_ratio,
            "untradeable_ratio": None,
        }
    except Exception as exc:  # noqa: BLE001 - scipy/numpy 缺失时降级全 None
        logger.warning("pipeline metric computation degraded: %s", exc)
        return _all_none()


def make_qe_evaluate_fn(
    stock_data: Any,
    *,
    label_days: int = 20,
    segment: str = "train",
    adapter: Any | None = None,
) -> Callable[[Sequence[str], str, Any], list[dict[str, float | None] | None]]:
    """统一评估层 evaluate_fn（V2-E）：内部走 QuantEvaluatorAdapter。

    与 :func:`make_fe_evaluate_fn` 同签名（formulas, fidelity, context → bundle
    dict 列表），但每个 bundle 由 QuantEvaluatorAdapter 产出（QE registry 指标 +
    20d cohort portfolio 双口径），字段名与 fitness/contracts.py 的
    EvaluationBundle 对齐。数据无法构造时返回全 None 并记录 degraded（不抛）。

    泄漏纪律：只读传入的 train 段 stock_data，不接收 test 段数据。
    """
    import numpy as np
    import pandas as pd

    if adapter is None:
        from alphaprobe.evaluator_adapter import QuantEvaluatorAdapter

        adapter = QuantEvaluatorAdapter(label_days=label_days)

    def _evaluate_fn(
        formulas: Sequence[str],
        fidelity: str = "L2_full_train",
        context: Any = None,
    ) -> list[dict[str, float | None] | None]:
        if stock_data is None:
            return [None] * len(formulas)
        fmls = [str(f) for f in formulas if f]
        try:
            planes = stock_data.evaluate_many(fmls)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 数据不可用降级为全 None，不抛
            logger.warning("pipeline qe evaluate_many failed, degraded: %s", exc)
            return [None] * len(formulas)
        try:
            names = list(stock_data._field_names())
            if "vwap" not in names:
                logger.warning("pipeline qe: no vwap field, degraded")
                return [None] * len(formulas)
            vwap = stock_data.data[:, :, names.index("vwap")]
            T = vwap.shape[0]
            if T <= label_days:
                logger.warning("pipeline qe: series too short for label_days, degraded")
                return [None] * len(formulas)
            label = vwap[label_days:, :] / vwap[:-label_days, :] - 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("pipeline qe vwap label build failed, degraded: %s", exc)
            return [None] * len(formulas)
        dates = list(stock_data._dates)
        codes = list(stock_data._stock_ids)
        seg_dates = dates[label_days:]
        # 因子 / label / 价格面板对齐到同一 label 段（cohort 路径内部再错位 1 日）
        label_df = pd.DataFrame(label, index=seg_dates, columns=codes)
        price_df = pd.DataFrame(vwap[label_days:, :], index=seg_dates, columns=codes)
        bundles: list[dict[str, float | None] | None] = []
        for plane in planes:
            try:
                f_arr = np.asarray(plane, dtype=float)
            except Exception:  # noqa: BLE001
                f_arr = plane
            if len(f_arr.shape) == 3:
                f_arr = f_arr[:, :, -1]
            try:
                f_arr = f_arr[-label.shape[0]:, :]
            except Exception:  # noqa: BLE001
                pass
            factor_df = pd.DataFrame(f_arr, index=seg_dates, columns=codes)
            try:
                eb = adapter.evaluate(factor_df, label_df, price_df)
                bundles.append(eb.raw())
            except Exception as exc:  # noqa: BLE001 - 单因子评估失败降级，不抛
                logger.warning("pipeline qe adapter evaluate failed, degraded: %s", exc)
                bundles.append(None)
        return bundles

    return _evaluate_fn


def _all_none() -> dict[str, float | None]:
    return {
        "rankic": None,
        "ic": None,
        "icir": None,
        "coverage": None,
        "nan_inf_ratio": None,
        "untradeable_ratio": None,
    }


# ---------------------------------------------------------------------------
# SearchPipeline
# ---------------------------------------------------------------------------


@dataclass
class SearchPipeline:
    """§41-§42 默认主链。run_round = 一轮完整搜索（不触 test 段）。

    P0-A Train/Valid 分离：
    - ``data_train``（保留，向后兼容）= L1_scout / L2_full_train 段；
    - ``data_search_valid`` = L3_search_valid 段（valid 指标唯一来源；
      ``rankic_valid`` 等 valid 指标绝不能用 train 段数据填——关键正确性断言）；
    - ``data_audit_valid`` = L4_pool_audit 预留段（当前版本不消费，留接线）；
    - ``mode`` / ``llm_client``：见 :mod:`alphaprobe.llm_client`。PRODUCTION
      缺 llm → 构造即抛（fail-closed）；OFFLINE_TEST 缺 llm → stub。
    """

    experiment: Any
    data_train: Any
    config: PipelineConfig = field(default_factory=PipelineConfig)
    dedup_client: Any | None = None
    evaluator: Any | None = None
    memory_store: Any | None = None
    llm_fn: Callable[[str, str, str], str] | None = None
    orchestrator: Any | None = None
    calibrator: Any | None = None
    evaluator_factory: Callable[..., Any] | None = None
    # P0-A：search_valid / audit_valid 段数据与运行模式
    data_search_valid: Any = None
    data_audit_valid: Any = None
    mode: Any = field(default_factory=_offline_test_mode)
    llm_client: Any = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = PipelineConfig()
        # P0-A：mode 归一（str → ExecutionMode）。
        from alphaprobe.llm_client import resolve_llm_fn

        self.mode = (
            ExecutionMode(self.mode)
            if not isinstance(self.mode, ExecutionMode)
            else self.mode
        )
        # P0-A：structured_generation 解析——production 语义默认 True；
        # OFFLINE_TEST 且用户未显式传（config.structured_generation is UNSET）时
        # 保持旧行为（False），不破坏既有离线测试。
        self._structured_unset = self.config.structured_generation is UNSET
        if self.config.structured_generation is UNSET:
            # 用户未显式传：OFFLINE_TEST 保持 False（向后兼容既有离线测试）；
            # PRODUCTION / RESEARCH_DEGRADED 默认 True（生产结构化路径）。
            self.config.structured_generation = self.mode is not ExecutionMode.OFFLINE_TEST
        self.round_no = 0
        self.funnel: Any | None = None
        self.pool: Any | None = None
        self.degraded_reasons: list[str] = []
        # P0-A：llm_fn 处理（mode 感知）。
        # - llm_client 非 None：一律经 resolve_llm_fn 转成 llm_fn（stub 仅 OFFLINE_TEST）；
        # - PRODUCTION/RESEARCH_DEGRADED 且 llm_fn/llm_client 都缺：fail-closed 抛
        #   （resolve_llm_fn 的守卫，绝不静默切 stub）；
        # - OFFLINE_TEST 且 llm_fn 为 None：确定性 stub（现状不变）。
        if self.llm_client is not None:
            self.llm_fn = resolve_llm_fn(
                self.llm_client,
                self.mode,
                structured=self._structured_effective(),
            )
        elif self.llm_fn is None:
            # resolve_llm_fn 对 PRODUCTION 缺 client 抛 LLMConfigurationError
            # （fail-closed）；OFFLINE_TEST 落确定性 stub。
            self.llm_fn = resolve_llm_fn(
                None,
                self.mode,
                structured=self._structured_effective(),
            )
        if self.orchestrator is None:
            from alphaprobe.search.orchestrator import SearchOrchestrator

            self.orchestrator = SearchOrchestrator(
                scheduler=None,
                memory=self.memory_store,
                expected_num=self.config.expected_num,
                structured_generation=self._structured_effective(),
            )
        if self.calibrator is None:
            from alphaprobe.fitness import MetricCalibrator

            self.calibrator = MetricCalibrator()
        # dedup client：None → 内存版 GlobalSeenIndex（§41 降级）
        if self.dedup_client is None:
            from alphaprobe.dedup import GlobalSeenIndex

            seen = GlobalSeenIndex()
            try:
                from alphaprobe.dedup_client import DedupClient

                self.dedup_client = DedupClient(seen=seen)
            except Exception as exc:  # noqa: BLE001 - 无 dedup_client 包时退回 seen 直用
                logger.warning("DedupClient unavailable, using raw seen index: %s", exc)
                self.dedup_client = seen
        # 权威接线：引导 FE 路径（使 DedupClient FE-first 链可达）+ 校验 label 契约。
        # fail-closed：FE/modeling 不可用 → degraded_reasons 记录，不 throw（搜索仍可降级运行）。
        try:
            from alphaprobe.authority import ensure_authority_available, validate_label_contract_20d

            ensure_authority_available()
            validate_label_contract_20d()
        except Exception as exc:  # noqa: BLE001 - 权威缺失降级，明确记录
            self.degraded_reasons.append(f"authority unavailable: {exc}")
        # evaluator：None → LegacyCompatEvaluator + fe_bridge 真算 fn
        if self.evaluator is None:
            self.evaluator = self._build_default_evaluator()
        # 默认 evaluate_fn（供 funnel 消费的 metric bundle 真算）：train 段
        self._evaluate_fn = make_fe_evaluate_fn(self.data_train)
        # P0-A Train/Valid 分离：data_search_valid 非 None → valid 段 evaluate_fn
        # （L3_search_valid 专用；valid 指标唯一来源，绝不用 train 段数据填）。
        self._evaluate_fn_valid = None
        if self.data_search_valid is not None:
            self._evaluate_fn_valid = make_qe_evaluate_fn(
                self.data_search_valid, label_days=20, segment="search_valid"
            )
        # P0-A：audit_valid 段预留（L4_pool_audit；本版本不消费，留接线位）。
        self._evaluate_fn_audit = None
        if self.data_audit_valid is not None:
            self._evaluate_fn_audit = make_qe_evaluate_fn(
                self.data_audit_valid, label_days=20, segment="audit_valid"
            )

    # ------------------------------------------------------------------
    # P0-A：structured_generation 兼容解析
    # ------------------------------------------------------------------

    def _structured_effective(self) -> bool:
        """SearchPipeline 消费点：OFFLINE_TEST + UNSET → False（旧行为）。

        __post_init__ 已把 config.structured_generation 定值（默认 True），
        此处只针对「默认构造、未显式传、且 mode==OFFLINE_TEST」的既有测试
        路径回退 False（orchestrator / stub 用该值），生产语义仍默认 True。
        """
        if self.mode == ExecutionMode.OFFLINE_TEST and getattr(self, "_structured_unset", False):
            return False
        return bool(self.config.structured_generation)

    # ------------------------------------------------------------------
    # evaluator 构建（可注入 evaluator_factory 覆盖）
    # ------------------------------------------------------------------

    def _build_default_evaluator(self) -> Any | None:
        if self.evaluator_factory is not None:
            try:
                return self.evaluator_factory()
            except Exception as exc:  # noqa: BLE001
                logger.warning("evaluator_factory failed: %s", exc)
                self.degraded_reasons.append(f"evaluator_factory failed: {exc}")
        try:
            from alphaprobe.contracts import ExperimentContext, LabelSpec, ResearchSplitSpec
            from alphaprobe.evalcache import EvaluationCache
            from alphaprobe.integration.evaluator_client import LegacyCompatEvaluator

            # §5 上下文（train 段即可；不构造 test 段）
            split_spec = None
            mining_raw = (self.experiment.raw if self.experiment is not None else {}) or {}
            mining = mining_raw.get("mining") or {}
            train_period = mining.get("train_period", ["2016-01-01", "2021-12-31"])
            try:
                split_spec = ResearchSplitSpec(
                    train=DateRange(train_period[0], train_period[1]),
                    search_valid=DateRange(train_period[1], train_period[1]),
                    audit_valid=None,
                    sealed_test=DateRange(train_period[1], train_period[1]),
                )
            except Exception:  # noqa: BLE001
                split_spec = None
            ctx = ExperimentContext(
                run_id=f"run_{uuid.uuid4().hex[:8]}",
                round_id=f"round_{self.round_no}",
                campaign_id="pipeline",
                label_spec=LabelSpec(price_pair="vwap_to_vwap", horizon_trading_days=20),
                split_spec=split_spec,
            )
            return LegacyCompatEvaluator(
                evaluate_fn=make_fe_evaluate_fn(self.data_train),
                context=ctx,
                cache=EvaluationCache(),
                leakage_guard=None,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator 不可用 → 静态降级
            logger.warning("default evaluator build failed, static degrade: %s", exc)
            self.degraded_reasons.append(f"evaluator build failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # funnel / pool 惰性构建
    # ------------------------------------------------------------------

    def _build_funnel(self) -> Any:
        from alphaprobe.fitness.funnel import FidelityFunnel, L0StaticCheck

        static = L0StaticCheck(
            dsl_validator=_static_dsl_validator,
            dedup_client=self.dedup_client,
        )
        return FidelityFunnel(
            thresholds={},
            static=static,
            hard_gates=None,
            l5_access=None,
        )

    def _build_pool(self) -> Any:
        from alphaprobe.pool import ActivePool

        return ActivePool(
            target_size=self.config.pool_target,
            max_size=self.config.pool_max,
        )

    # ------------------------------------------------------------------
    # 搜索 + 评估 + 准入
    # ------------------------------------------------------------------

    def _generate(self, parents: list[dict[str, Any]]) -> list[Any]:
        """结构化变异臂生成 raw candidates（llm_fn 由 __post_init__ 兜底）。"""
        if not parents:
            return []
        out: list[Any] = []
        seen: set[str] = set()
        for p in parents[:3]:
            if self.orchestrator is None:
                continue
            step = self.orchestrator.step(
                dict(p),
                self.llm_fn,
                # 注：orchestrator 内部也会查 dedup（过滤硬重复）。这里保留所有
                # raw 候选到 pipeline 层统一过滤（避免 orchestrator 提前清空，
                # 让 pipeline 的 run_round 计数 duplicates_filtered 更真实）。
                dedup_client=None,
            )
            for cand in step.candidates:
                f = str(cand.formula or "").strip()
                if not f or f in seen:
                    continue
                seen.add(f)
                out.append(cand)
        return out

    def _identity_tuple(self, formula: str) -> tuple[str, str, str]:
        """返回 (canonical, signal_id, family_id)。

        §28 权威化：身份唯一来源是 ``DedupClient.get_identity``（其内部降级链
        FE identity → alphaprobe.identity → 文本 canonical，见 dedup_client.py），
        不再在本主链上并行调用 ``dedup.canonicalize_dsl`` 的正则文本化简。FE
        不可用时由 DedupClient 自身降级链兜底，绝不在此另起一套 regex 实现。
        """
        if self.dedup_client is not None and hasattr(self.dedup_client, "get_identity"):
            try:
                view = self.dedup_client.get_identity(formula)
                canonical = str(getattr(view, "canonical_formula", "") or formula)
                signal_id = str(getattr(view, "signal_equivalence_id", "") or "")
                family = str(getattr(view, "parameter_family_id", "") or "")
                if signal_id:
                    return canonical, signal_id, family
            except Exception as exc:  # noqa: BLE001
                logger.warning("get_identity failed for %r: %s", formula, exc)
        # 备用：FE 权威直接视图（fail-closed，不回落文本 canonicalize_dsl）。
        from alphaprobe.authority import FactorIdentityAuthorityError, build_identity_view

        try:
            view = build_identity_view(formula)
        except FactorIdentityAuthorityError as exc:
            logger.warning("authority identity unavailable for %r: %s", formula, exc)
            return str(formula or ""), "", ""
        return (
            str(view.get("canonical_formula", formula)),
            str(view.get("signal_equivalence_id", "") or ""),
            str(view.get("parameter_family_id", "") or ""),
        )

    def _evaluate(self, formulas: list[str]) -> list[dict[str, float | None] | None]:
        """评估 formulas → metric bundle 列表（once-compute）。

        P0-A 删手算回退语义：
        - OFFLINE_TEST：evaluator 结果全空/全 None → 回落本地 ``_evaluate_fn``
          （离线可跑，保证 stub/合成数据端到端）；
        - PRODUCTION / RESEARCH_DEGRADED：evaluator 失败/空 → 直接把该候选记
          EVALUATION_MISSING（返回 ``[None]*len``），**绝不手算**——生产评估
          只认 evaluator/QE 权威，不回落 fe_bridge 本地 numpy/scipy。
        """
        if not formulas:
            return []
        if self.evaluator is not None and hasattr(self.evaluator, "evaluate"):
            try:
                candidates = [_dict_as_factor_candidate(f) for f in formulas]
                records = self.evaluator.evaluate(
                    candidates,
                    profile="search",
                )
                out: list[dict[str, float | None] | None] = []
                useful = False
                for rec in records:
                    mb = getattr(rec, "metric_bundle", None) or {}
                    if mb and any(v is not None for v in mb.values()):
                        useful = True
                    out.append(dict(mb))
                    if self.memory_store is not None and hasattr(self.memory_store, "record_evaluation"):
                        try:
                            self.memory_store.record_evaluation(rec)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("record_evaluation failed: %s", exc)
                if useful:
                    return out
                if self.mode in (ExecutionMode.PRODUCTION, ExecutionMode.RESEARCH_DEGRADED):
                    # P0-A：生产 evaluator 空结果 → EVALUATION_MISSING，不手算。
                    logger.warning(
                        "evaluator returned empty bundles in %s mode → EVALUATION_MISSING",
                        self.mode.value,
                    )
                    return [None] * len(formulas)
                # OFFLINE_TEST：evaluator 无有效指标 → 回落本地真算
                logger.warning("evaluator returned empty bundles, falling back to local evaluate_fn")
            except Exception as exc:  # noqa: BLE001 - evaluator 异常 → 降级
                logger.warning("evaluator.evaluate failed, static degrade: %s", exc)
                self.degraded_reasons.append(f"evaluator.evaluate failed: {exc}")
                if self.mode in (ExecutionMode.PRODUCTION, ExecutionMode.RESEARCH_DEGRADED):
                    # P0-A：生产 evaluator 异常 → EVALUATION_MISSING，绝不手算。
                    return [None] * len(formulas)
        # 静态降级（OFFLINE_TEST）：跑本地 bundle 计算（纯 numpy/scipy，无网络无模型）
        bundles: list[dict[str, float | None] | None] = []
        try:
            bundles = self._evaluate_fn(formulas, "L2_full_train", None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("local evaluate_fn failed, degraded: %s", exc)
            self.degraded_reasons.append(f"local evaluate_fn failed: {exc}")
            bundles = [None] * len(formulas)
        return bundles

    def _evaluate_valid(self, formulas: list[str]) -> list[dict[str, float | None] | None]:
        """L3_search_valid 段评估（data_search_valid 提供）。

        valid 指标（rankic_valid 等）唯一来源是 search_valid 段 evaluate_fn；
        **绝不能用 train 段数据填 valid 指标**（关键正确性断言）。本方法不做
        evaluator 通路——L3 valid 段只消费注入的 valid evaluate_fn（QE 全窗）。
        data_search_valid 为 None 时返回全 None（调用方据 fidelity 决定是否拒）。
        """
        if not formulas or self._evaluate_fn_valid is None:
            return [None] * len(formulas)
        try:
            return list(self._evaluate_fn_valid(formulas, "L3_search_valid", None))
        except Exception as exc:  # noqa: BLE001 - valid 段评估失败不抛
            logger.warning("valid evaluate_fn failed, degraded: %s", exc)
            self.degraded_reasons.append(f"valid evaluate_fn failed: {exc}")
            return [None] * len(formulas)

    def _run_funnel(self, cand: Any) -> tuple[bool, dict[str, Any], Any | None]:
        """L0 → L1 scout → L2 full（train 段）→ L3_search_valid gate（P0-A）。

        泄漏纪律：L1/L2 只消费 data_train；L3_search_valid 只在 data_search_valid
        非 None 时评估，record.segment="search_valid"，valid 指标（rankic_valid 等）
        唯一来自 valid 段 evaluate_fn，绝不用 train 段数据填。
        """
        formula = str(getattr(cand, "formula", "") or "")
        canonical, signal_id, family_id = self._identity_tuple(formula)
        factor_id = f"ap_{self.round_id()}_{signal_id[:8]}"
        if self.funnel is None:
            self.funnel = self._build_funnel()
        # L0 静态（DSL 合法性 + 去重 + 复杂度）
        out = self.funnel.l0(formula, factor_id=factor_id)
        if not out.passed:
            return False, {"level": "L0", "rejections": [r.value for r in out.rejections]}, None
        # L1 scout：静态 bundle（低成本，train 段）
        mb = (self._evaluate([formula]) or [None])[0]
        if mb is None or all(v is None for v in mb.values()):
            return False, {"level": "L1", "rejections": ["EVALUATION_MISSING"]}, None
        record = _make_record(factor_id, formula, mb, segment="train", fidelity="L1_scout")
        reasons = self.funnel.gate("L1_scout", record)
        if reasons:
            return False, {"level": "L1", "rejections": [r.value for r in reasons]}, None
        # L2 full：train 段，funnel gate 消费同一 metric bundle（once-compute）
        record2 = _make_record(factor_id, formula, mb, segment="train", fidelity="L2_full_train")
        reasons2 = self.funnel.gate("L2_full_train", record2)
        if reasons2:
            return False, {"level": "L2", "rejections": [r.value for r in reasons2]}, None
        # P0-A L3_search_valid：L2 gate 之后、admit 之前，用 valid 段 evaluate_fn
        # 算一次 L3_search_valid bundle 并 gate。funnel 对未知 level 的 gate 返回 []
        # （不拒，见 funnel.gate 分发兜底）；data_search_valid 缺失时只记录 bundle
        # 不 gate（valid 指标无法构造 → 不强制拒绝，保持向后兼容）。
        record3 = None
        if self.data_search_valid is not None:
            mb_valid = (self._evaluate_valid([formula]) or [None])[0]
            if mb_valid is not None and any(v is not None for v in mb_valid.values()):
                # valid bundle 字段并入 train bundle（train 字段优先保留；valid 键
                # 显式覆盖——train bundle 绝不包含 valid 指标）。
                merged = dict(mb)
                for k, v in mb_valid.items():
                    if v is not None:
                        merged[k] = v
                record3 = _make_record(
                    factor_id, formula, merged, segment="search_valid", fidelity="L3_search_valid"
                )
                reasons3 = self.funnel.gate("L3_search_valid", record3)
                if reasons3:
                    return False, {"level": "L3", "rejections": [r.value for r in reasons3]}, record3
        # L2/L3 通过：返回带 valid 指标的 record（record3 有 valid 数据则用之）。
        return True, {"level": "L3" if record3 is not None else "L2", "rejections": []}, (
            record3 or record2
        )

    def _admit(self, cand: Any, record: Any, formula: str) -> bool:
        if self.pool is None:
            self.pool = self._build_pool()
        canonical, signal_id, family_id = self._identity_tuple(formula)
        factor_id = f"ap_{self.round_id()}_{signal_id[:8]}"
        try:
            from alphaprobe.pool.admission import PoolAdmission

            res = PoolAdmission().admit(
                None,
                record,
                self.pool,
                metric_bundle=record.metric_bundle if record is not None else None,
                factor_id=factor_id,
                canonical_formula=canonical,
                niche_key=(),
            )
            return bool(res.accepted)
        except Exception as exc:  # noqa: BLE001 - 准入失败不抛
            logger.warning("pool admit failed for %r: %s", formula, exc)
            return False

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run_round(
        self,
        *,
        round_id: str,
        parents: list[dict[str, Any]] | None = None,
    ) -> RoundResult:
        """执行一轮主链：generate → dedup → funnel → fitness → admit → memory。

        泄漏纪律：全程不接收/构造 test 段数据。
        """
        self.round_no += 1
        result = RoundResult(round_id=round_id)
        if self.memory_store is not None:
            result.memory_db_path = str(getattr(self.memory_store, "db_path", "") or "")
        parents = list(parents or [])
        if not parents:
            parents = self._fallback_parents()

        candidates = self._generate(parents)
        result.candidates_generated = len(candidates)
        if not candidates:
            result.degraded = bool(self.degraded_reasons)
            result.degraded_reasons = list(self.degraded_reasons)
            result.pool_snapshot = self._pool_snapshot()
            return result

        evaluated = 0
        admitted = 0
        for cand in candidates:
            formula = str(getattr(cand, "formula", "") or "")
            if not formula:
                continue
            # 硬重复过滤（第 2 轮同公式 → EXACT 拦截；此处只查不预留）
            verdict = None
            if self.dedup_client is not None and hasattr(self.dedup_client, "check_new_candidate"):
                try:
                    verdict = self.dedup_client.check_new_candidate(formula)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("check_new_candidate failed: %s", exc)
            if verdict is not None and getattr(verdict, "rejection_reason", None) is not None:
                result.duplicates_filtered += 1
                self._remember_attempt(formula, "duplicate", verdict)
                continue

            ok, info, record = self._run_funnel(cand)
            if ok and record is not None:
                # L0 通过后原子预留（NEW→RESERVED），下一轮同 formula 判 EXACT
                if self.dedup_client is not None and hasattr(self.dedup_client, "reserve"):
                    try:
                        self.dedup_client.reserve(formula)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("reserve failed: %s", exc)
                evaluated += 1
                self._remember_evaluation(record)
                if self._admit(cand, record, formula):
                    admitted += 1
            else:
                rej = info.get("rejections") or []
                self._remember_attempt(formula, "rejected", None, reasons=rej)

        result.evaluated = evaluated
        result.admitted = admitted
        result.degraded = bool(self.degraded_reasons)
        result.degraded_reasons = list(self.degraded_reasons)
        result.pool_snapshot = self._pool_snapshot()
        return result

    # ------------------------------------------------------------------
    # 记忆实时写（不做 round 结束一次性 upsert）
    # ------------------------------------------------------------------

    def _remember_attempt(
        self,
        formula: str,
        outcome: str,
        verdict: Any,
        *,
        reasons: list[str] | None = None,
    ) -> None:
        if self.memory_store is None:
            return
        canonical, signal_id, family_id = self._identity_tuple(formula)
        factor_id = f"ap_{self.round_id()}_{signal_id[:8]}"
        try:
            self.memory_store.upsert_factor_node(
                factor_id=factor_id,
                canonical_formula=canonical,
                canonical_ast_hash=canonical[:32] if len(canonical) <= 32 else canonical,
                signal_equivalence_id=signal_id,
                parameter_family_id=family_id,
                source_system="alphaprobe",
                source_snapshot=self.round_id(),
                source_type="MINED",
                exportable=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("upsert_factor_node failed: %s", exc)
        try:
            self.memory_store.record_attempt(
                attempt_id=f"at_{uuid.uuid4().hex[:12]}",
                action_id=f"act_{self.round_id()}",
                candidate_factor_id=factor_id,
                outcome=outcome,
                rejection_reason=(reasons[0] if reasons else None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_attempt failed: %s", exc)
        try:
            self.memory_store.bump_exploration(
                factor_id=factor_id,
                action_family="REFINE",
                success=(outcome == "admitted"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("bump_exploration failed: %s", exc)

    def _remember_evaluation(self, record: Any) -> None:
        if self.memory_store is None or record is None:
            return
        try:
            self.memory_store.record_evaluation(record)
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_evaluation failed: %s", exc)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def round_id(self) -> str:
        return f"r{self.round_no}"

    def _fallback_parents(self) -> list[dict[str, Any]]:
        """无 parents 时的默认 seed（简单可解析 DSL，不触 test 段）。"""
        return [
            {"formula": "rank(close)", "factor_id": "seed_rank_close", "fitness": 0.0},
            {"formula": "ts_mean(close, 20)", "factor_id": "seed_ts_mean_20", "fitness": 0.0},
            {"formula": "ts_std(close, 20)", "factor_id": "seed_ts_std_20", "fitness": 0.0},
        ]

    def _pool_snapshot(self) -> list[dict[str, Any]]:
        if self.pool is None:
            return []
        try:
            return self.pool.snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("pool snapshot failed: %s", exc)
            return []

    def pool_size(self) -> int:
        return len(self.pool) if self.pool is not None else 0


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _static_dsl_validator(formula: str) -> tuple[bool, str]:
    """纯静态 DSL 校验：非空 + 括号平衡（不调 factor_engine parser）。"""
    s = str(formula or "").strip()
    if not s:
        return False, "empty DSL"
    if s.count("(") != s.count(")"):
        return False, "unbalanced parentheses"
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False, "extra closing parenthesis"
    return True, "OK"


def _dict_as_factor_candidate(formula: str) -> Any:
    from alphaprobe.contracts import CandidateStatus, FactorCandidate, FactorIdentity

    identity = FactorIdentity(
        factor_id=f"cand_{abs(hash(formula)) & 0xFFFFFFFF:08x}",
        canonical_formula=formula,
        canonical_ast_hash=formula,
        signal_equivalence_id=formula,
    )
    return FactorCandidate(identity=identity, source="mined", status=CandidateStatus.NEW.value)


def _make_record(
    factor_id: str,
    formula: str,
    metric_bundle: dict[str, float | None] | None,
    *,
    segment: str,
    fidelity: str,
) -> Any:
    from alphaprobe.contracts import EvaluationRecord

    return EvaluationRecord(
        factor_id=factor_id,
        segment=segment,
        fidelity=fidelity,
        metric_bundle=dict(metric_bundle or {}),
        artifact_refs={},
        evaluator_version="alphaprobe.pipeline.v1",
        data_snapshot_id="auto",
        universe_snapshot_id="auto",
        label_spec_hash="vwap_to_vwap_h20",
        created_at=datetime.now(timezone.utc),
    )


__all__ = [
    "UNSET",
    "PipelineConfig",
    "RoundResult",
    "SearchPipeline",
    "make_stub_llm_fn",
    "make_fe_evaluate_fn",
    "make_qe_evaluate_fn",
    "DEFAULT_ROUNDS",
]
