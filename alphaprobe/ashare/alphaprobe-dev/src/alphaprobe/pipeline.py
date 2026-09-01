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

logger = logging.getLogger(__name__)

# 默认搜索轮预算：与 runner --search_time 语义一致（沿用 quota/stop 逻辑）。
DEFAULT_ROUNDS = 40
DEFAULT_BUDGET_PER_ROUND = 12
DEFAULT_TARGET = 1024
DEFAULT_MAX = 2048


# ---------------------------------------------------------------------------
# 配置 / 结果
# ---------------------------------------------------------------------------


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
) -> Callable[[str, str, str], str]:
    """构造确定性 stub llm_fn（返回 §31 JSON 文本）。

    forced_candidates 非空时：把固定 candidate JSON 序列化返回（测试用）；
    否则：从 parents 与 action 规则化生成变异候选（离线端到端默认路径）。
    绝不发起任何网络 / 模型调用。
    """
    import random as _random

    rng = rng or _random.Random(0)

    def _llm_fn(system_prompt: str, user_prompt: str, model_class: str) -> str:
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
    """pipeline 模块级默认 stub（模块不 import torch/openai，离线可跑）。"""
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
    """单因子平面 × label → 基础 metric bundle（离线 numpy/scipy，不跑 qlib）。"""
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
    """§41-§42 默认主链。run_round = 一轮完整搜索（不触 test 段）。"""

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

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = PipelineConfig()
        self.round_no = 0
        self.funnel: Any | None = None
        self.pool: Any | None = None
        self.degraded_reasons: list[str] = []
        if self.llm_fn is None:
            # 无 LLM 额度默认路径：确定性 stub，不发起任何网络/模型调用
            self.llm_fn = _default_llm_fn()
        if self.orchestrator is None:
            from alphaprobe.search.orchestrator import SearchOrchestrator

            self.orchestrator = SearchOrchestrator(
                scheduler=None,
                memory=self.memory_store,
                expected_num=self.config.expected_num,
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
        # evaluator：None → LegacyCompatEvaluator + fe_bridge 真算 fn
        if self.evaluator is None:
            self.evaluator = self._build_default_evaluator()
        # 默认 evaluate_fn（供 funnel 消费的 metric bundle 真算）
        self._evaluate_fn = make_fe_evaluate_fn(self.data_train)

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
        """返回 (canonical, signal_id, family_id)。dedup_client 优先。"""
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
        from alphaprobe.dedup import canonical_ast_hash, canonicalize_dsl, parameter_family_key, signal_equivalence_id

        canonical = canonicalize_dsl(formula)
        return canonical, signal_equivalence_id(canonical), parameter_family_key(canonical)

    def _evaluate(self, formulas: list[str]) -> list[dict[str, float | None] | None]:
        """评估 formulas → metric bundle 列表（once-compute；失败降级不抛）。

        优先 evaluator（记录缓存 + 泄漏 gate）；evaluator 结果全空/全 None 时
        回落到本地 _evaluate_fn（数据不可用等场景），保证可降级。
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
                # evaluator 结果无有效指标 → 回落本地真算
                logger.warning("evaluator returned empty bundles, falling back to local evaluate_fn")
            except Exception as exc:  # noqa: BLE001 - evaluator 异常 → 静态降级
                logger.warning("evaluator.evaluate failed, static degrade: %s", exc)
                self.degraded_reasons.append(f"evaluator.evaluate failed: {exc}")
        # 静态降级：跑本地 bundle 计算（纯 numpy/scipy，无网络无模型）
        bundles: list[dict[str, float | None] | None] = []
        try:
            bundles = self._evaluate_fn(formulas, "L2_full_train", None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("local evaluate_fn failed, degraded: %s", exc)
            self.degraded_reasons.append(f"local evaluate_fn failed: {exc}")
            bundles = [None] * len(formulas)
        return bundles

    def _run_funnel(self, cand: Any) -> tuple[bool, dict[str, Any], Any | None]:
        """L0 → L1 scout → fingerprint nearest confirm → L2 full（funnel gate）。"""
        formula = str(getattr(cand, "formula", "") or "")
        canonical, signal_id, family_id = self._identity_tuple(formula)
        factor_id = f"ap_{self.round_id()}_{signal_id[:8]}"
        if self.funnel is None:
            self.funnel = self._build_funnel()
        # L0 静态（DSL 合法性 + 去重 + 复杂度）
        out = self.funnel.l0(formula, factor_id=factor_id)
        if not out.passed:
            return False, {"level": "L0", "rejections": [r.value for r in out.rejections]}, None
        # L1 scout：静态 bundle（低成本）
        mb = (self._evaluate([formula]) or [None])[0]
        if mb is None or all(v is None for v in mb.values()):
            return False, {"level": "L1", "rejections": ["EVALUATION_MISSING"]}, None
        record = _make_record(factor_id, formula, mb, segment="train", fidelity="L1_scout")
        reasons = self.funnel.gate("L1_scout", record)
        if reasons:
            return False, {"level": "L1", "rejections": [r.value for r in reasons]}, None
        # L2 full：现有 funnel gate 消费同一 metric bundle（once-compute）
        record2 = _make_record(factor_id, formula, mb, segment="train", fidelity="L2_full_train")
        reasons2 = self.funnel.gate("L2_full_train", record2)
        if reasons2:
            return False, {"level": "L2", "rejections": [r.value for r in reasons2]}, None
        return True, {"level": "L2", "rejections": []}, record2

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
    "PipelineConfig",
    "RoundResult",
    "SearchPipeline",
    "make_stub_llm_fn",
    "make_fe_evaluate_fn",
    "DEFAULT_ROUNDS",
]
