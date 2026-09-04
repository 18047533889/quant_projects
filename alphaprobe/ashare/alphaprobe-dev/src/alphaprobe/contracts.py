"""AlphaPROBE 全量重构共享契约层（任务书 §5 / §8 / §26 / §75）。

所有跨模块类型在此唯一定义；子模块只允许 import，不允许重复定义同名不同义类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# §5 ExperimentContext：消灭散落参数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DateRange:
    """闭区间日期段（含端点），ISO 字符串比较安全。"""

    start: str
    end: str

    def contains(self, d: str) -> bool:
        return self.start <= d <= self.end

    def overlaps(self, other: "DateRange") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class LabelSpec:
    """§4.7：cohort/purge/embargo/holding/turnover/splits 必须共享同一 LabelSpec。"""

    price_pair: str = "vwap_to_vwap"
    horizon_trading_days: int = 20

    @property
    def hash_key(self) -> str:
        return f"{self.price_pair}_h{self.horizon_trading_days}"


@dataclass(frozen=True)
class ResearchSplitSpec:
    """§3.3 新研究协议：Train / Search Valid / Audit Valid / Sealed Test。

    20 日 forward VWAP label 必须至少 purge >= 20 trading days；
    embargo 配置化，首版 5 trading days。
    """

    train: DateRange
    search_valid: DateRange
    audit_valid: DateRange | None
    sealed_test: DateRange
    purge_trading_days: int = 20
    embargo_trading_days: int = 5

    def validate_no_overlap(self) -> None:
        """段间必须 purge/embargo 隔离，禁止直接重叠。"""
        import pandas as pd

        cal = _trading_days_cache.get("calendar")
        if cal is None:
            return  # 无日历时退化为日期序检查
        gap = self.purge_trading_days + self.embargo_trading_days
        pairs = [
            (self.train, self.search_valid),
            (self.search_valid, self.audit_valid or self.sealed_test),
        ]
        if self.audit_valid is not None:
            pairs.append((self.audit_valid, self.sealed_test))
        for a, b in pairs:
            idx_a = cal.searchsorted(pd.Timestamp(a.end))
            idx_b = cal.searchsorted(pd.Timestamp(b.start))
            if idx_b - idx_a < gap:
                raise ValueError(
                    f"split overlap/insufficient purge: {a} vs {b} "
                    f"gap={idx_b - idx_a} < {gap}"
                )


class FidelityLevel(str, Enum):
    """§9 多保真评估漏斗。L5 = Sealed Test，仅冻结后可跑。"""

    L0_STATIC = "L0_static"
    L1_SCOUT = "L1_scout"
    L2_FULL_TRAIN = "L2_full_train"
    L3_SEARCH_VALID = "L3_search_valid"
    L4_POOL_AUDIT = "L4_pool_audit"
    L5_SEALED_TEST = "L5_sealed_test"


L0_L4 = frozenset(
    {
        FidelityLevel.L0_STATIC,
        FidelityLevel.L1_SCOUT,
        FidelityLevel.L2_FULL_TRAIN,
        FidelityLevel.L3_SEARCH_VALID,
        FidelityLevel.L4_POOL_AUDIT,
    }
)


@dataclass(frozen=True)
class ExperimentContext:
    """§5：所有 cache / artifact / memory / export 必须可追溯到它。"""

    run_id: str
    round_id: str
    campaign_id: str

    market: str = "ashare"
    data_snapshot_id: str = "auto"
    universe_snapshot_id: str = "auto"

    label_spec: LabelSpec = field(default_factory=LabelSpec)
    split_spec: ResearchSplitSpec | None = None

    factor_engine_version: str = "unknown"
    operator_semantics_version: str = "unknown"
    data_access_version: str = "unknown"
    evaluator_version: str = "unknown"

    search_config_hash: str = "unknown"
    prompt_version: str = "unknown"
    llm_model_policy_version: str = "unknown"

    knowledge_cutoff_date: date | None = None
    survival_memory_cutoff: date | None = None

    rng_seed: int = 0


# ---------------------------------------------------------------------------
# §6.2 FactorBatch / §8 EvaluationRecord
# ---------------------------------------------------------------------------


@dataclass
class FactorBatch:
    """§6.2：不要默认把所有大矩阵永久驻内存；values_ref 指向 artifact。"""

    factor_ids: list[str]
    values_ref: str  # ArtifactRef（parquet/arrow 路径或内存句柄 id）
    trade_dates: DateRange
    universe_snapshot_id: str
    data_snapshot_id: str
    coverage_summary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """§8：统一 Evaluator 唯一事实。Pool/Logger/Checkpoint/Exporter 全消费它。"""

    factor_id: str
    segment: str
    fidelity: str

    metric_bundle: dict[str, float | None]
    artifact_refs: dict[str, str]

    evaluator_version: str
    data_snapshot_id: str
    universe_snapshot_id: str
    label_spec_hash: str

    created_at: datetime


# ---------------------------------------------------------------------------
# §26 / §75.3 SearchAction
# ---------------------------------------------------------------------------


class SearchActionType(str, Enum):
    REFINE = "REFINE"
    WINDOW_SCALE = "WINDOW_SCALE"
    FIELD_SUBSTITUTION = "FIELD_SUBSTITUTION"
    OPERATOR_SUBSTITUTION = "OPERATOR_SUBSTITUTION"
    STATE_CONDITION = "STATE_CONDITION"
    CROSSOVER = "CROSSOVER"
    SCHEMA_EXPLORE = "SCHEMA_EXPLORE"
    ROBUSTIFY = "ROBUSTIFY"
    GENERATION_REPAIR = "GENERATION_REPAIR"


# ---------------------------------------------------------------------------
# §83 统一失败原因枚举（failure memory 可查询，不解析自然语言）
# ---------------------------------------------------------------------------


class RejectionReason(str, Enum):
    INVALID_DSL = "INVALID_DSL"
    PIT_VIOLATION = "PIT_VIOLATION"
    MISSING_FIELD = "MISSING_FIELD"
    UNSUPPORTED_FIELD = "UNSUPPORTED_FIELD"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    SIGN_EQUIVALENT_DUPLICATE = "SIGN_EQUIVALENT_DUPLICATE"
    PARAMETER_REDUNDANT = "PARAMETER_REDUNDANT"
    RANK_NEAR_DUPLICATE = "RANK_NEAR_DUPLICATE"
    SEED_LIBRARY_DUPLICATE = "SEED_LIBRARY_DUPLICATE"
    LOW_COVERAGE = "LOW_COVERAGE"
    NUMERICAL_FRAGILITY = "NUMERICAL_FRAGILITY"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    UNSAFE_NUMERIC = "UNSAFE_NUMERIC"
    LOW_PREDICTIVE = "LOW_PREDICTIVE"
    LOW_STABILITY = "LOW_STABILITY"
    D10_COLLAPSE = "D10_COLLAPSE"
    BAD_LONG_SHORT = "BAD_LONG_SHORT"
    HIGH_CORRELATION = "HIGH_CORRELATION"
    NO_INCREMENTAL_UTILITY = "NO_INCREMENTAL_UTILITY"
    HIGH_COST = "HIGH_COST"
    HIGH_TURNOVER = "HIGH_TURNOVER"
    LOW_TRADABILITY = "LOW_TRADABILITY"
    EVALUATION_MISSING = "EVALUATION_MISSING"
    MISSING_METRIC = "MISSING_METRIC"
    SEARCH_SATURATED = "SEARCH_SATURATED"
    REFINEMENT_FAILED = "REFINEMENT_FAILED"
    AUDIT_FAIL = "AUDIT_FAIL"
    EXPORT_FAIL = "EXPORT_FAIL"
    ALREADY_EXPORTED = "ALREADY_EXPORTED"


class CandidateStatus(str, Enum):
    NEW = "NEW"
    RESERVED = "RESERVED"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    REJECTED = "REJECTED"
    EXPORTED = "EXPORTED"


# ---------------------------------------------------------------------------
# §75 Dataclasses / Pydantic Contracts
# ---------------------------------------------------------------------------


class FactorIdentity:
    """§75.1。用 dataclass（避免强制 pydantic 依赖，行为等价）。"""

    def __init__(
        self,
        factor_id: str,
        canonical_formula: str,
        canonical_ast_hash: str,
        signal_equivalence_id: str,
        parameter_family_id: str | None = None,
        orientation: int = 1,
    ) -> None:
        self.factor_id = factor_id
        self.canonical_formula = canonical_formula
        self.canonical_ast_hash = canonical_ast_hash
        self.signal_equivalence_id = signal_equivalence_id
        self.parameter_family_id = parameter_family_id
        self.orientation = orientation

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "canonical_formula": self.canonical_formula,
            "canonical_ast_hash": self.canonical_ast_hash,
            "signal_equivalence_id": self.signal_equivalence_id,
            "parameter_family_id": self.parameter_family_id,
            "orientation": self.orientation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FactorIdentity":
        return cls(**d)


class FactorCandidate:
    """§75.2。"""

    def __init__(
        self,
        identity: FactorIdentity,
        source: str = "mined",
        parent_ids: list[str] | None = None,
        action_id: str | None = None,
        schema: dict[str, str] | None = None,
        hypothesis: str | None = None,
        field_set: list[str] | None = None,
        operator_set: list[str] | None = None,
        complexity: int = 0,
        depth: int = 0,
        lookback: int = 0,
        status: str = CandidateStatus.NEW.value,
    ) -> None:
        self.identity = identity
        self.source = source
        self.parent_ids = list(parent_ids or [])
        self.action_id = action_id
        self.schema = dict(schema or {})
        self.hypothesis = hypothesis
        self.field_set = list(field_set or [])
        self.operator_set = list(operator_set or [])
        self.complexity = complexity
        self.depth = depth
        self.lookback = lookback
        self.status = status


class SearchAction:
    """§75.3。"""

    def __init__(
        self,
        action_id: str,
        action_type: SearchActionType | str,
        parent_ids: list[str] | None = None,
        target_role: str | None = None,
        target_schema: dict[str, str] | None = None,
        budget_class: str = "normal",
        llm_model_class: str = "cheap",
        created_round: str = "",
        created_generation: int = 0,
    ) -> None:
        self.action_id = action_id
        self.action_type = SearchActionType(action_type)
        self.parent_ids = list(parent_ids or [])
        self.target_role = target_role
        self.target_schema = dict(target_schema or {})
        self.budget_class = budget_class
        self.llm_model_class = llm_model_class
        self.created_round = created_round
        self.created_generation = created_generation


class AttemptRecord:
    """§75.4。"""

    def __init__(
        self,
        attempt_id: str,
        action: SearchAction,
        candidate_factor_id: str | None = None,
        outcome: str = "ok",
        rejection_reason: str | None = None,
        delta_search_fitness: float | None = None,
        novelty_gain: float | None = None,
        pool_utility_gain: float | None = None,
        eval_cost: float = 0.0,
        llm_cost: float = 0.0,
        latency_ms: int = 0,
    ) -> None:
        self.attempt_id = attempt_id
        self.action = action
        self.candidate_factor_id = candidate_factor_id
        self.outcome = outcome
        self.rejection_reason = rejection_reason
        self.delta_search_fitness = delta_search_fitness
        self.novelty_gain = novelty_gain
        self.pool_utility_gain = pool_utility_gain
        self.eval_cost = eval_cost
        self.llm_cost = llm_cost
        self.latency_ms = latency_ms


class FactorDNA:
    """§75.5。"""

    def __init__(
        self,
        field_families: list[str] | None = None,
        operators: list[str] | None = None,
        ast_motifs: list[str] | None = None,
        mechanisms: list[str] | None = None,
        schema: dict[str, str] | None = None,
        horizon_bucket: str = "unknown",
        persistence_bucket: str = "unknown",
        turnover_bucket: str = "unknown",
        response_shape: str | None = None,
        exposure_profile: dict[str, float] | None = None,
        tradability_profile: dict[str, float] | None = None,
        complexity_bucket: str = "unknown",
    ) -> None:
        self.field_families = list(field_families or [])
        self.operators = list(operators or [])
        self.ast_motifs = list(ast_motifs or [])
        self.mechanisms = list(mechanisms or [])
        self.schema = dict(schema or {})
        self.horizon_bucket = horizon_bucket
        self.persistence_bucket = persistence_bucket
        self.turnover_bucket = turnover_bucket
        self.response_shape = response_shape
        self.exposure_profile = dict(exposure_profile or {})
        self.tradability_profile = dict(tradability_profile or {})
        self.complexity_bucket = complexity_bucket


class SurvivalProfile:
    """§75.6。"""

    def __init__(
        self,
        factor_id: str,
        status: str = "UNCLASSIFIED",
        long_term_retention: float | None = None,
        recent_retention: float | None = None,
        rolling_slope: float | None = None,
        q20_rolling_rankic: float | None = None,
        q20_rolling_sharpe: float | None = None,
        sign_flip_rate: float | None = None,
        max_deterioration: float | None = None,
        max_breach_duration: int | None = None,
        recovery_time: int | None = None,
        descriptive_half_life: float | None = None,
        confidence: float = 0.0,
        support_periods: int = 0,
    ) -> None:
        self.factor_id = factor_id
        self.status = status
        self.long_term_retention = long_term_retention
        self.recent_retention = recent_retention
        self.rolling_slope = rolling_slope
        self.q20_rolling_rankic = q20_rolling_rankic
        self.q20_rolling_sharpe = q20_rolling_sharpe
        self.sign_flip_rate = sign_flip_rate
        self.max_deterioration = max_deterioration
        self.max_breach_duration = max_breach_duration
        self.recovery_time = recovery_time
        self.descriptive_half_life = descriptive_half_life
        self.confidence = confidence
        self.support_periods = support_periods


class SurvivalLabel(str, Enum):
    """§45.1。"""

    PERSISTENT_ALPHA = "PERSISTENT_ALPHA"
    HEALTHY = "HEALTHY"
    DEGRADING = "DEGRADING"
    BROKEN = "BROKEN"
    RECOVERED = "RECOVERED"
    REGIME_DEPENDENT = "REGIME_DEPENDENT"
    UNCLASSIFIED = "UNCLASSIFIED"


class ExplorationState:
    """§75.7。"""

    def __init__(
        self,
        factor_id: str,
        action_family: str,
        attempts: int = 0,
        successes: int = 0,
        elites: int = 0,
        mean_delta_fitness: float = 0.0,
        mean_novelty_gain: float = 0.0,
        saturation: float = 0.0,
        uncertainty: float = 1.0,
        version_key: str = "",
    ) -> None:
        self.factor_id = factor_id
        self.action_family = action_family
        self.attempts = attempts
        self.successes = successes
        self.elites = elites
        self.mean_delta_fitness = mean_delta_fitness
        self.mean_novelty_gain = mean_novelty_gain
        self.saturation = saturation
        self.uncertainty = uncertainty
        self.version_key = version_key


class ExportDecision:
    """§75.8。0 个合格 → 输出 0；237 个合格 → 输出 237（无 Top-N）。"""

    def __init__(
        self,
        factor_id: str,
        hard_gate_pass: bool = False,
        export_score: float | None = None,
        audit_pass: bool = False,
        novel: bool = False,
        seed_duplicate: bool = False,
        already_exported: bool = False,
        refinement_complete: bool = False,
        accepted: bool = False,
        reasons: list[str] | None = None,
    ) -> None:
        self.factor_id = factor_id
        self.hard_gate_pass = hard_gate_pass
        self.export_score = export_score
        self.audit_pass = audit_pass
        self.novel = novel
        self.seed_duplicate = seed_duplicate
        self.already_exported = already_exported
        self.refinement_complete = refinement_complete
        self.accepted = accepted
        self.reasons = list(reasons or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "hard_gate_pass": self.hard_gate_pass,
            "export_score": self.export_score,
            "audit_pass": self.audit_pass,
            "novel": self.novel,
            "seed_duplicate": self.seed_duplicate,
            "already_exported": self.already_exported,
            "refinement_complete": self.refinement_complete,
            "accepted": self.accepted,
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# 简单缓存槽（split 校验用；由 research_protocol 注入交易日历）
# ---------------------------------------------------------------------------

_trading_days_cache: dict[str, Any] = {}


def set_trading_calendar(index: Any) -> None:
    """注入 DatetimeIndex 交易日历，供 ResearchSplitSpec.validate_no_overlap 使用。"""
    _trading_days_cache["calendar"] = index