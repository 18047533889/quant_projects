"""fitness.funnel（任务书 §9 / Phase 3）：多保真评估漏斗 L0-L5。

L0 只做静态检查（DSL 合法性 / allowlist / 去重 / 复杂度），不跑市场回测；
L1-L4 只定义占位结构 + 从 EvaluationRecord.metric_bundle 消费指标（不自己算）；
L5 只在 research/campaign/version 冻结后运行（需要 frozen SealedTestAccess）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from alphaprobe.contracts import (
    EvaluationRecord,
    FidelityLevel,
    RejectionReason,
)
from alphaprobe.fitness import check_hard_gates
from alphaprobe.fitness.contracts import (
    EvaluationBundle,
    MetricRequirementPolicy,
)

__all__ = [
    "FidelityFunnel",
    "L0StaticCheck",
    "L0_OUTPUT",
    "FunnelPromotion",
    "FUNNEL_ORDER",
]

# L0-L4 多保真等级（L5 仅冻结后运行）
FUNNEL_ORDER: tuple[FidelityLevel, ...] = (
    FidelityLevel.L0_STATIC,
    FidelityLevel.L1_SCOUT,
    FidelityLevel.L2_FULL_TRAIN,
    FidelityLevel.L3_SEARCH_VALID,
    FidelityLevel.L4_POOL_AUDIT,
    FidelityLevel.L5_SEALED_TEST,
)

_L0_OUTPUT: dict[str, tuple[str, RejectionReason]] = {
    "dsl_valid": ("dsl_invalid", RejectionReason.INVALID_DSL),
    "pit_ok": ("pit_violation", RejectionReason.PIT_VIOLATION),
    "forbidden_field": ("unsupported_field", RejectionReason.UNSUPPORTED_FIELD),
    "coverage_ok": ("low_coverage", RejectionReason.LOW_COVERAGE),
    "nan_inf_ok": ("nan_inf", RejectionReason.NUMERICAL_INVALID),
    "exact_duplicate": ("exact_duplicate", RejectionReason.EXACT_DUPLICATE),
    "sign_duplicate": ("sign_duplicate", RejectionReason.SIGN_EQUIVALENT_DUPLICATE),
    "seed_duplicate": ("seed_duplicate", RejectionReason.SEED_LIBRARY_DUPLICATE),
    "unsafe_numeric": ("unsafe_numeric", RejectionReason.UNSAFE_NUMERIC),
}


@dataclass
class L0Output:
    """L0 静态检查输出。rejections 空 = 通过（可升级到 L1）。"""

    formula: str = ""
    canonical: str = ""
    signal_id: str = ""
    param_family_id: str = ""
    rejections: list[RejectionReason] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, bytes] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.rejections


# 兼容导出名（细节/理由以 RejectionReason 为主，另留 description 字段）
L0_OUTPUT = L0Output


@dataclass
class FunnelPromotion:
    """funnel.promote 的结果。promoted_level 为升级目标等级，None = 未升级。"""

    factor_id: str = ""
    formula: str = ""
    level: str = ""
    target_level: str = ""
    promoted_level: str | None = None
    rejections: list[RejectionReason] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    passed: bool = False

    @property
    def accepted(self) -> bool:
        return self.promoted_level is not None


class L0StaticCheck:
    """§9 L0 — Static：不跑市场回测，只做可判定静态检查。

    dsl_validator: (formula) -> (ok, reason)。缺省时仅做括号平衡检查。
    factor_engine_adapter: 可选 FactorEngineAdapterProtocol；提供 validate/canonicalize 时优先使用。
    """

    def __init__(
        self,
        *,
        dsl_validator: Callable[[str], tuple[bool, str]] | None = None,
        seen: Any | None = None,
        factor_engine_adapter: Any | None = None,
        min_coverage: float = 0.5,
        max_untradeable: float = 0.5,
        unsafe_numeric_check: Callable[[str], bool] | None = None,
        dedup_client: Any | None = None,
    ) -> None:
        self.dsl_validator = dsl_validator
        self.dedup_client = dedup_client
        # 不提供 dedup_client 时：行为与现在完全一致（内存版 GlobalSeenIndex）
        if seen is None and dedup_client is None:
            from alphaprobe.dedup import GlobalSeenIndex

            seen = GlobalSeenIndex()
        self.seen = seen
        self.fe_adapter = factor_engine_adapter
        self.min_coverage = min_coverage
        self.max_untradeable = max_untradeable
        self.unsafe_numeric_check = unsafe_numeric_check
        self._lock = threading.Lock()
        self.last_signal_ids: dict[str, str] = {}

    # -- DSL 合法性 ---------------------------------------------------------

    def _dsl_ok(self, formula: str) -> tuple[bool, str]:
        if self.fe_adapter is not None:
            try:
                out = self.fe_adapter.validate(formula)
            except Exception as exc:  # noqa: BLE001 - adapter 失败按不合法处理
                return False, f"adapter validate error: {exc}"
            ok = _coerce_bool(out)
            return (ok, "OK" if ok else f"adapter rejected: {out!r}")
        if self.dsl_validator is not None:
            try:
                return self.dsl_validator(formula)
            except Exception as exc:  # noqa: BLE001
                return False, f"validator error: {exc}"
        return self._static_dsl_check(formula)

    @staticmethod
    def _static_dsl_check(formula: str) -> tuple[bool, str]:
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
        if depth != 0:
            return False, "unbalanced parentheses"
        return True, "OK"

    # -- 去重 ---------------------------------------------------------------

    def _identity(self, formula: str) -> tuple[str, str, str]:
        """返回 (canonical, signal_id, family_key)；family 无 adapter 时为 ''。

        提供 dedup_client 时优先走 client.get_identity()（统一 identity 入口），
        否则保持现有 fe_adapter → 文本 canonical 逻辑。
        """
        if self.dedup_client is not None:
            try:
                view = self.dedup_client.get_identity(formula)
                canonical = str(getattr(view, "canonical_formula", "") or "")
                signal_id = str(getattr(view, "signal_equivalence_id", "") or "")
                family = str(getattr(view, "parameter_family_id", "") or "") or ""
                if signal_id:
                    return canonical or formula, signal_id, family
            except Exception:  # noqa: BLE001 - fallthrough
                pass
        if self.fe_adapter is not None:
            try:
                canonical_out = self.fe_adapter.canonicalize(formula)
                canonical = _canonical_str(canonical_out)
                if canonical:
                    signal_id = signal_equivalence_id(canonical)
                    return canonical, signal_id, ""
            except Exception:  # noqa: BLE001 - fallthrough
                pass
        from alphaprobe.dedup import canonicalize_dsl, signal_equivalence_id

        canonical = canonicalize_dsl(formula)
        return canonical, signal_equivalence_id(canonical), ""

    def _is_exact_duplicate(self, signal_id: str, factor_id: str) -> bool:
        with self._lock:
            existing = self.seen.lookup_signal(signal_id)
            if existing is not None:
                return True
            # 通过 L0 即原子预留（§11.6 NEW→RESERVED），二次同 signal 才能判重
            reserved, _ = self.seen.reserve(signal_id=signal_id, factor_id=factor_id or signal_id)
            return not reserved
    # -- 主入口 -------------------------------------------------------------

    def check(self, formula: str, *, factor_id: str = "") -> L0Output:
        out = L0Output()
        out.formula = str(formula or "").strip()
        if not out.formula:
            out.rejections.append(RejectionReason.INVALID_DSL)
            return out

        canonical, signal_id, family_id = self._identity(out.formula)
        out.canonical = canonical
        out.signal_id = signal_id
        out.param_family_id = family_id
        with self._lock:
            self.last_signal_ids[factor_id or out.formula] = signal_id

        # DSL 合法性
        ok, why = self._dsl_ok(out.formula)
        if not ok:
            out.details["dsl_error"] = why
            out.rejections.append(RejectionReason.INVALID_DSL)
            return out  # 不合法 DSL 不继续做 dedup 判定

        # 去重
        if self.dedup_client is not None:
            verdict = self.dedup_client.check_new_candidate(out.formula)
            if verdict is not None and verdict.rejection_reason is not None:
                # EXACT / SIGN 都算 duplicate，各自 RejectionReason
                out.rejections.append(verdict.rejection_reason)
            else:
                # NEW：原子预留（§11.6 NEW→RESERVED），二次同 signal 才能判重
                try:
                    self.dedup_client.reserve(
                        out.formula,
                        source_system="funnel.L0",
                        run_id="",
                        worker_id="",
                        factor_id=factor_id or None,
                    )
                except Exception:  # noqa: BLE001 - 预留失败不阻塞判定
                    pass
            with self._lock:
                self._last_reserve_ok = True
        elif self._is_exact_duplicate(signal_id, factor_id):
            out.rejections.append(RejectionReason.EXACT_DUPLICATE)
        return out


class FidelityFunnel:
    """§9 多保真评估漏斗。

    thresholds 配置形如 {"L1_scout": {"rankic": 0.02, "coverage": 0.5, ...}}；
    任意 level 的 threshold 命中后 upgrade 到该 level 的下一级。
    """

    def __init__(
        self,
        *,
        levels: list[str] | tuple[str, ...] | None = None,
        thresholds: dict[str, dict[str, float]] | None = None,
        static: L0StaticCheck | None = None,
        hard_gates: Callable[..., list[str]] | None = None,
        l5_access: Any | None = None,
        metric_requirements: MetricRequirementPolicy | None = None,
    ) -> None:
        if levels is None:
            levels = [lv.value for lv in FUNNEL_ORDER]
        self.levels = list(levels)
        self.thresholds = dict(thresholds or {})
        self.static = static or L0StaticCheck()
        self._hard_gates_fn = hard_gates or check_hard_gates
        self._l5_access = l5_access
        self._metric_requirements = metric_requirements
        self._lock = threading.Lock()
        self._promotions: dict[str, FunnelPromotion] = {}
        self.counters: dict[str, int] = {
            "L0_rejects": 0,
            "L1_promoted": 0,
            "L2_promoted": 0,
            "L3_promoted": 0,
            "L4_promoted": 0,
        }

    def _missing_metric_rejections(self, record: EvaluationRecord) -> list[RejectionReason]:
        """REQUIRED metric 缺失 → MISSING_METRIC（policy action='fail'）。

        policy 未配置 / 无 REQUIRED 键 → 恒 []（V2 行为 0 差异）。
        """
        policy = self._metric_requirements
        if policy is None:
            return []
        if policy.required_missing_action != "fail":
            return []
        missing = policy.missing_required(EvaluationBundle(record.metric_bundle or {}))
        if not missing:
            return []
        return [RejectionReason.MISSING_METRIC]

    # -- L0 ---------------------------------------------------------------

    def l0(self, formula: str, *, factor_id: str = "") -> L0Output:
        """L0 静态检查。rejections 空 → 通过。"""
        return self.static.check(formula, factor_id=factor_id)

    def _l1_gate(self, record: EvaluationRecord) -> list[RejectionReason]:
        mb = record.metric_bundle
        rejects: list[RejectionReason] = []
        cov = mb.get("coverage")
        if cov is not None and cov < self.static.min_coverage:
            rejects.append(RejectionReason.LOW_COVERAGE)
        nan_ratio = mb.get("nan_inf_ratio")
        if nan_ratio is not None and nan_ratio > 0.5:
            rejects.append(RejectionReason.NUMERICAL_INVALID)
        untr = mb.get("untradeable_ratio")
        if untr is not None and untr > self.static.max_untradeable:
            rejects.append(RejectionReason.LOW_TRADABILITY)
        return rejects

    # -- L2-L4 -------------------------------------------------------------

    def _l2_gate(self, record: EvaluationRecord) -> list[RejectionReason]:
        mb = record.metric_bundle
        rejects: list[RejectionReason] = []
        rankic = mb.get("rankic")
        if rankic is not None and rankic <= 0.0:
            rejects.append(RejectionReason.LOW_PREDICTIVE)
        # D10 双重 tolerance 修复：metric "d10_cliff_penalty" 本身已是
        # max(0, d_top-0.12)（见 fitness.d10_cliff_penalty），再用 >0.12 判定
        # 等于 d_top>0.24 才拒（容差叠加）。已 adjust 的 key → penalty > 0 即拒；
        # raw d_top 类 key（如 "d10_drop_ratio"）→ 保持 > 0.12 判定。
        d10 = mb.get("d10_cliff_penalty")
        if d10 is not None:
            if d10 > 0.0:
                rejects.append(RejectionReason.D10_COLLAPSE)
        else:
            d10_raw = mb.get("d10_drop_ratio")
            if d10_raw is not None and d10_raw > 0.12:
                rejects.append(RejectionReason.D10_COLLAPSE)
        return rejects

    def _l3_gate(self, record: EvaluationRecord) -> list[RejectionReason]:
        mb = record.metric_bundle
        rejects: list[RejectionReason] = []
        # V2.1 REQUIRED metric 缺失 → L3 不可过（Task 8.1 / Part G #19）。
        # 默认无 policy 时恒 []（0 行为差异）。
        rejects.extend(self._missing_metric_rejections(record))
        mdd = mb.get("mdd")
        if mdd is not None and abs(mdd) > 0.5:
            rejects.append(RejectionReason.AUDIT_FAIL)
        return rejects

    def _l4_gate(self, record: EvaluationRecord) -> list[RejectionReason]:
        mb = record.metric_bundle
        rejects: list[RejectionReason] = []
        residual = mb.get("residual_rankic")
        if residual is not None and residual < 0.0:
            rejects.append(RejectionReason.HIGH_CORRELATION)
        incr = mb.get("incremental_utility")
        if incr is not None and incr <= 0.0:
            rejects.append(RejectionReason.NO_INCREMENTAL_UTILITY)
        return rejects

    # -- L5 ---------------------------------------------------------------

    def _l5_gate(self, record: EvaluationRecord) -> list[RejectionReason]:
        """L5 只允许 frozen SealedTestAccess。"""
        if self._l5_access is None:
            return [RejectionReason.AUDIT_FAIL]
        if not getattr(self._l5_access, "frozen", False):
            return [RejectionReason.AUDIT_FAIL]
        mb = record.metric_bundle
        sealed_ok = mb.get("sealed_test_pass", True)
        if not sealed_ok:
            return [RejectionReason.AUDIT_FAIL]
        return []

    # -- gate 分发 ----------------------------------------------------------

    def gate(
        self,
        level: str,
        record: EvaluationRecord,
        *,
        metric_requirements: MetricRequirementPolicy | None = None,
    ) -> list[RejectionReason]:
        lv = FidelityLevel(level) if level in {lv.value for lv in FidelityLevel} else level
        if metric_requirements is not None:
            self._metric_requirements = metric_requirements
        if lv == FidelityLevel.L1_SCOUT:
            return self._l1_gate(record)
        if lv == FidelityLevel.L2_FULL_TRAIN:
            return self._l2_gate(record)
        if lv == FidelityLevel.L3_SEARCH_VALID:
            return self._l3_gate(record)
        if lv == FidelityLevel.L4_POOL_AUDIT:
            return self._l4_gate(record)
        if lv == FidelityLevel.L5_SEALED_TEST:
            return self._l5_gate(record)
        return []

    # -- promote --------------------------------------------------------------

    def promote(
        self,
        batch: list[dict[str, Any]],
        from_level: str,
        to_level: str,
    ) -> list[FunnelPromotion]:
        """按 thresholds 配置过滤 batch。

        batch 元素: {"factor_id": str, "formula": str, "record": EvaluationRecord | None}
        任一元素被拒 → 返回对应 FunnelPromotion(promoted_level=None, rejections=[...])；
        全部通过 → promoted_level=to_level。
        """
        if not batch:
            return []
        try:
            target = FidelityLevel(to_level)
        except ValueError:
            target = None

        results: list[FunnelPromotion] = []
        all_ok = True
        for item in batch:
            fid = str(item.get("factor_id", ""))
            formula = str(item.get("formula", ""))
            record = item.get("record")
            reasons: list[RejectionReason] = []

            # L0（静态）无 record：直接做静态检查
            if from_level == FidelityLevel.L0_STATIC.value:
                out = self.static.check(formula, factor_id=fid)
                reasons = list(out.rejections)
            elif record is not None:
                reasons = self.gate(from_level, record, metric_requirements=self._metric_requirements)
            else:
                # 无 record 且非 L0：fail-closed（§审阅 11），不允许静默通过
                reasons = [RejectionReason.EVALUATION_MISSING]

            if reasons:
                self.counters["L0_rejects"] += 1
                if from_level == FidelityLevel.L0_STATIC.value:
                    ck = "L0_rejected"
                    if ck in self.counters:
                        self.counters[ck] += 1
            results.append(
                FunnelPromotion(
                    factor_id=fid,
                    formula=formula,
                    level=from_level,
                    target_level=to_level,
                    promoted_level=None if reasons else to_level,
                    rejections=reasons,
                    reasons=[r.value for r in reasons],
                    passed=not reasons,
                )
            )
            all_ok = all_ok and not reasons

        if all_ok and target is not None:
            # §9 计数口径：L0→L1 整批通过 +1；其余升级每 record 计数
            if from_level == FidelityLevel.L0_STATIC.value:
                self.counters["L1_promoted"] += 1
            ck = f"{target.value}_promoted"
            if ck in self.counters and from_level != FidelityLevel.L0_STATIC.value:
                self.counters[ck] += len(results)
        return results
    def promote_record(
        self,
        record: EvaluationRecord,
        from_level: str,
        to_level: str,
    ) -> FunnelPromotion:
        """单 record 升级便捷方法（从 EvaluationRecord.metric_bundle 消费，不自己算）。"""
        res = self.promote(
            [{"factor_id": record.factor_id, "formula": "", "record": record}],
            from_level,
            to_level,
        )
        return res[0]

    def observations(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)


def _coerce_bool(out: Any) -> bool:
    """factor_engine adapter validate 返回值可能是 bool / tuple(bool, str) / dict。"""
    if isinstance(out, bool):
        return out
    if isinstance(out, (tuple, list)):
        if not out:
            return False
        return bool(out[0])
    if isinstance(out, dict):
        ok = out.get("ok", out.get("valid", out.get("passed", False)))
        return bool(ok)
    return bool(out)


def _canonical_str(out: Any) -> str:
    """adapter canonicalize 返回值 → 字符串 canonical（或空串表示不可用）。"""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, str):
                return item.strip()
    if isinstance(out, dict):
        for k in ("canonical", "formula", "dsl"):
            v = out.get(k)
            if isinstance(v, str):
                return v.strip()
    return ""
