"""DedupClient（任务书 §4 / §41）：AlphaPROBE 候选因子的去重统一入口。

AlphaPROBE 只调用，不自建 hash。提供三级降级：
- identity 降级链：FE identity → alphaprobe.identity → 文本 canonical
- seen 降级链：alphaprobe.seen 持久版 → alphaprobe.dedup 内存版
- action seen 降级链：alphaprobe.seen.ActionSeenIndex → 内存 sha256 key set

零 LLM、零模型、零网络。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any

from alphaprobe.contracts import RejectionReason

__all__ = [
    "DedupClient",
    "DedupConfig",
    "DedupVerdict",
    "IdentityView",
    "ReservationOutcome",
    "SignalConfirmResult",
]

# ---------------------------------------------------------------------------
# 配置 / 值类型
# ---------------------------------------------------------------------------


@dataclass
class DedupConfig:
    """§41 集中配置，阈值不散落写死。"""

    fingerprint_version: str = "v1"
    sample_date_set_id: str = "default"
    rank_exact_threshold: float = 0.995
    highly_correlated_threshold: float = 0.90
    nearest_k: int = 20
    identity_fallback_chain: bool = True


@dataclass
class IdentityView:
    """Identity 视图（简化版 FactorIdentity，不强制 contracts 依赖）。"""

    formula: str = ""
    canonical_formula: str = ""
    canonical_ast_hash: str = ""
    signal_equivalence_id: str = ""
    parameter_family_id: str | None = None
    factor_id: str = ""
    identity_version: str = "1"


@dataclass
class DedupVerdict:
    """check_new_candidate 的判定结果。"""

    status: str  # "EXACT" | "SIGN" | "NEW"
    rejection_reason: RejectionReason | None = None
    identity: IdentityView | None = None
    existing_factor_id: str | None = None


@dataclass
class ReservationOutcome:
    """reserve 的结果。"""

    reserved: bool = False
    existing_factor_id: str | None = None


@dataclass
class SignalConfirmResult:
    """confirm_signal 的结果。"""

    status: str  # "RANK_EQUIVALENT" | "HIGHLY_CORRELATED" | "NOVEL"
    nearest_corr: float = 0.0
    nearest_factor_id: str = ""
    neighbors: list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 内部：内存 ActionSeenIndex（降级用）
# ---------------------------------------------------------------------------


class _MemoryActionSeenIndex:
    """内存版 ActionSeenIndex（降级链路）。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def check_and_mark(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True


# ---------------------------------------------------------------------------
# DedupClient
# ---------------------------------------------------------------------------


class DedupClient:
    """§41 候选因子去重统一入口。

    Parameters
    ----------
    seen : optional
        外部 seen 索引（Phase B 持久版或 dedup.GlobalSeenIndex 内存版）。
        缺省时自动加载 alphaprobe.dedup.GlobalSeenIndex。
    identity_provider : optional
        外部 identity 提供者（如 FactorIdentityFactory）。
        缺省时自动加载 identity 降级链。
    config : DedupConfig, optional
        集中配置。缺省用全部默认值。
    """

    def __init__(
        self,
        *,
        seen: Any | None = None,
        identity_provider: Any | None = None,
        config: DedupConfig | None = None,
    ) -> None:
        self.config = config or DedupConfig()
        self.degraded: list[str] = []
        self._lock = threading.Lock()
        # signal_id -> canonical_formula（用于 EXACT vs SIGN 区分）
        self._canonical_by_signal: dict[str, str] = {}

        # --- seen 初始化 ---
        if seen is not None:
            self._seen = seen
        else:
            self._seen = self._auto_import_seen()
        # --- identity 初始化 ---
        if identity_provider is not None:
            self._identity_provider = identity_provider
        else:
            self._identity_provider = self._auto_import_identity()
        # --- action seen 初始化 ---
        self._action_seen = self._auto_import_action_seen()

    # ------------------------------------------------------------------
    # 降级导入
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_import_seen() -> Any:
        """自动加载 seen 索引：alphaprobe.seen.GlobalSeenIndex →
        alphaprobe.dedup.GlobalSeenIndex 内存版。"""
        try:
            import alphaprobe.seen as seen_module

            gsi = getattr(seen_module, "GlobalSeenIndex", None)
            if gsi is not None:
                return gsi()
        except Exception:
            pass
        from alphaprobe.dedup import GlobalSeenIndex

        return GlobalSeenIndex()

    def _auto_import_identity(self) -> Any:
        """自动加载 identity 提供者。"""
        # 优先尝试 factor_engine.identity.get_factor_identity
        try:
            import factor_engine.identity as fe_identity

            if hasattr(fe_identity, "get_factor_identity"):
                self.degraded.append("identity: using factor_engine.identity")
                return fe_identity
        except Exception:
            pass
        # 回落 alphaprobe.identity.FactorIdentityFactory
        try:
            from alphaprobe.identity import FactorIdentityFactory

            self.degraded.append("identity: using alphaprobe.identity.FactorIdentityFactory")
            return FactorIdentityFactory()
        except Exception:
            pass
        # 最终降级：None（get_identity 用文本 canonical）
        self.degraded.append("identity: fell back to text canonical (no identity provider)")
        return None

    @staticmethod
    def _auto_import_action_seen() -> Any:
        """自动加载 ActionSeenIndex。"""
        try:
            import alphaprobe.seen as seen_module

            asi = getattr(seen_module, "ActionSeenIndex", None)
            if asi is not None:
                return asi()
        except Exception:
            pass
        return _MemoryActionSeenIndex()

    # ------------------------------------------------------------------
    # 降级状态
    # ------------------------------------------------------------------

    @property
    def degradation_report(self) -> list[str]:
        """返回所有降级原因列表（外部 observability 可读取）。"""
        with self._lock:
            return list(self.degraded)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_identity(self, formula: str) -> IdentityView:
        """优先 factor_engine.identity.get_factor_identity（Phase A 产物），
        降级链：FE identity → alphaprobe.identity.FactorIdentityFactory →
        文本 canonical。
        """
        text = str(formula or "").strip()
        if not text:
            return IdentityView()

        provider = self._identity_provider

        # Level 1: factor_engine.identity.get_factor_identity
        try:
            if provider is not None and hasattr(provider, "get_factor_identity"):
                result = provider.get_factor_identity(formula)
                if result is not None:
                    view = self._fe_identity_to_view(formula, result)
                    if view.signal_equivalence_id or view.canonical_ast_hash:
                        return view
        except Exception:
            pass

        # Level 2: alphaprobe.identity.FactorIdentityFactory
        try:
            if provider is not None and hasattr(provider, "from_formula"):
                identity = provider.from_formula(formula)
                if identity is not None:
                    return IdentityView(
                        formula=text,
                        canonical_formula=getattr(identity, "canonical_formula", text),
                        canonical_ast_hash=getattr(identity, "canonical_ast_hash", ""),
                        signal_equivalence_id=getattr(identity, "signal_equivalence_id", ""),
                        parameter_family_id=getattr(identity, "parameter_family_id", None),
                        factor_id=getattr(identity, "factor_id", ""),
                    )
        except Exception:
            pass

        # Level 3: 文本 canonical（alphaprobe.dedup）
        return self._text_canonical_identity(text)

    @staticmethod
    def _fe_identity_to_view(formula: str, fe_result: Any) -> IdentityView:
        """将 factor_engine.identity 的返回值转为 IdentityView。"""
        text = str(formula or "").strip()
        if isinstance(fe_result, dict):
            return IdentityView(
                formula=text,
                canonical_formula=str(fe_result.get("canonical_formula", fe_result.get("canonical_dsl", text))),
                canonical_ast_hash=str(fe_result.get("canonical_ast_hash", fe_result.get("ast_hash", ""))),
                signal_equivalence_id=str(fe_result.get("signal_equivalence_id", fe_result.get("signal_id", ""))),
                parameter_family_id=str(fe_result.get("parameter_family_id", fe_result.get("family_id", ""))),
                factor_id=str(fe_result.get("factor_id", "")),
            )
        canonical = getattr(fe_result, "canonical_dsl", getattr(fe_result, "canonical_formula", None))
        canonical = str(canonical) if canonical else text
        signal_id = getattr(fe_result, "signal_equivalence_id", getattr(fe_result, "signal_id", ""))
        ast_hash = getattr(fe_result, "canonical_ast_hash", getattr(fe_result, "ast_hash", ""))
        family_id = getattr(fe_result, "parameter_family_id", None)
        if canonical or signal_id or ast_hash:
            return IdentityView(
                formula=text,
                # FE canonical 是 DSL-JSON；canonical_formula 语义应保留文本 canonical
                canonical_formula=_dsl_json_to_text(canonical) if _looks_like_dsl_json(canonical) else canonical,
                canonical_ast_hash=str(ast_hash or ""),
                signal_equivalence_id=str(signal_id or ""),
                parameter_family_id=str(family_id) if family_id else None,
                factor_id=str(getattr(fe_result, "factor_id", "") or "") or (str(ast_hash)[:12] if ast_hash else ""),
            )
        return IdentityView(formula=text)

    def _text_canonical_identity(self, formula: str) -> IdentityView:
        """纯文本 canonical 降级（alphaprobe.dedup）。"""
        from alphaprobe.dedup import (
            canonical_ast_hash,
            canonicalize_dsl,
            parameter_family_key,
            signal_equivalence_id,
            sign_normalized,
        )

        # canonicalize_dsl 仅做文本安全化简（不剥符号外壳）；结构化身份
        # canonical_formula 用 sign_normalized，与 signal_equivalence_id 同源
        _ = canonicalize_dsl  # 保留导入：canonicalize_dsl 仍用于参数校验/其他路径
        return IdentityView(
            formula=formula,
            # canonical_formula 语义 = 结构化身份（sign-normalized），
            # 否则 EXACT 与 SIGN 永远混淆：-(x) 与 x 的 sign_normalized 相同
            # （signal_equivalence_id 同源），但 canonicalize_dsl 文本不同，
            # 会把本应判 SIGN 的变体判成 NEW。
            canonical_formula=sign_normalized(formula),
            canonical_ast_hash=canonical_ast_hash(formula),
            signal_equivalence_id=signal_equivalence_id(formula),
            parameter_family_id=parameter_family_key(formula),
            factor_id=canonical_ast_hash(formula)[:12],
        )

    # ------------------------------------------------------------------
    # 去重判定
    # ------------------------------------------------------------------

    def check_new_candidate(self, formula: str) -> DedupVerdict:
        """§41 流程第一步：identity → seen.lookup_identity → 硬重复判定。

        EXACT: canonical_ast_hash 一致
        SIGN: signal_equivalence_id 一致但 canonical_ast_hash 不同
        NEW: 从未见过

        跨版本（identity_version 不同）由 Phase B 持久版 seen 负责判定；
        内存版/假 seen 按 signal_id 判定。
        """
        identity = self.get_identity(formula)
        if not identity.signal_equivalence_id:
            # 无法计算 identity → 视为 NEW（不阻塞）
            return DedupVerdict("NEW", identity=identity)

        # Phase B 持久版优先：lookup_identity(canonical_ast_hash) 区分 EXACT / SIGN
        seen = self._seen
        if hasattr(seen, "lookup_identity") and identity.canonical_ast_hash:
            try:
                lk = seen.lookup_identity(
                    _IdentityAdapter(identity),
                    identity_version=identity.identity_version,
                )
            except TypeError:
                lk = None
            if lk is not None:
                verdict_str = str(getattr(lk, "verdict", "") or "")
                if verdict_str in ("EXACT_DUPLICATE", "SIGN_EQUIVALENT_DUPLICATE"):
                    rr = (
                        RejectionReason.EXACT_DUPLICATE
                        if verdict_str == "EXACT_DUPLICATE"
                        else RejectionReason.SIGN_EQUIVALENT_DUPLICATE
                    )
                    return DedupVerdict(
                        "EXACT" if verdict_str == "EXACT_DUPLICATE" else "SIGN",
                        rr,
                        identity,
                        getattr(lk, "existing_factor_id", None) or "",
                    )
                if verdict_str == "VERSION_MISMATCH":
                    # 跨版本不互判 duplicate → NEW（Phase B 已各自建行）
                    return DedupVerdict("NEW", identity=identity)
                # NEW / FAMILY_SATURATED 均放行
                return DedupVerdict("NEW", identity=identity)

        existing = self._seen_lookup(identity.signal_equivalence_id)
        if existing is not None:
            # 已见过此 signal → 区分 EXACT / SIGN
            with self._lock:
                known_canonical = self._canonical_by_signal.get(identity.signal_equivalence_id)
            if known_canonical is not None and known_canonical == identity.canonical_formula:
                return DedupVerdict(
                    "EXACT",
                    RejectionReason.EXACT_DUPLICATE,
                    identity,
                    existing,
                )
            return DedupVerdict(
                "SIGN",
                RejectionReason.SIGN_EQUIVALENT_DUPLICATE,
                identity,
                existing,
            )

        return DedupVerdict("NEW", identity=identity)

    def _seen_lookup(self, signal_id: str) -> str | None:
        """查询 seen 索引，统一 duck-typing 接口。"""
        seen = self._seen
        # 优先 lookup_identity（Phase B 接口），其次 lookup_signal（内存版）
        if hasattr(seen, "lookup_identity"):
            return seen.lookup_identity(signal_id)
        if hasattr(seen, "lookup_signal"):
            return seen.lookup_signal(signal_id)
        return None

    # ------------------------------------------------------------------
    # 预留
    # ------------------------------------------------------------------

    def reserve(
        self,
        formula: str,
        *,
        source_system: str = "",
        run_id: str = "",
        worker_id: str = "",
        factor_id: str | None = None,
    ) -> ReservationOutcome:
        """原子预留：看到 NEW 才预留，否则返回冲突。

        source_system / run_id / worker_id 为审计元数据（内存版暂忽略）。

        优先 Phase B 持久版 seen.reserve（identity 对象签名），
        其次内存版/假 seen（signal_id 关键字签名）。
        """
        identity = self.get_identity(formula)
        fid = factor_id or identity.factor_id or identity.canonical_ast_hash[:12]
        if not identity.signal_equivalence_id:
            return ReservationOutcome(reserved=False)

        seen = self._seen
        if hasattr(seen, "reserve") and identity.canonical_ast_hash:
            # Phase B 持久版签名：reserve(identity, *, source_system, run_id, worker_id, ...)
            try:
                result = seen.reserve(
                    _IdentityAdapter(identity),
                    source_system=source_system or "alphaprobe",
                    run_id=run_id,
                    worker_id=worker_id,
                )
                if result is not None and hasattr(result, "acquired"):
                    if result.acquired:
                        with self._lock:
                            self._canonical_by_signal[identity.signal_equivalence_id] = (
                                identity.canonical_formula
                            )
                        return ReservationOutcome(reserved=True)
                    return ReservationOutcome(
                        reserved=False,
                        existing_factor_id=getattr(result, "existing_factor_id", None)
                        or getattr(result, "factor_id", None),
                    )
            except TypeError:
                pass  # 签名不匹配 → 降级 signal_id 路径
            except Exception:  # noqa: BLE001
                pass  # 持久版失败不阻塞 → 降级

        reserved, existing = self._seen_reserve(
            signal_id=identity.signal_equivalence_id,
            factor_id=fid,
            family_id=identity.parameter_family_id,
        )
        if reserved:
            with self._lock:
                self._canonical_by_signal[identity.signal_equivalence_id] = identity.canonical_formula
        return ReservationOutcome(reserved=reserved, existing_factor_id=existing)

    def _seen_reserve(self, signal_id: str, factor_id: str, family_id: str | None = None) -> tuple[bool, str | None]:
        """调用 seen 的 reserve，统一 duck-typing 接口。"""
        seen = self._seen
        if hasattr(seen, "reserve"):
            try:
                return seen.reserve(signal_id=signal_id, factor_id=factor_id, family_id=family_id)
            except TypeError:
                # Phase B 版本可能签名不同
                return seen.reserve(signal_id=signal_id, factor_id=factor_id)
        return True, None

    # ------------------------------------------------------------------
    # 信号确认
    # ------------------------------------------------------------------

    def confirm_signal(
        self,
        fingerprint: bytes,
        *,
        correlations: dict[str, float] | None = None,
    ) -> SignalConfirmResult:
        """nearest_by_fingerprint(k=20) → 由调用方提供 corr 或用缓存 corr → 判定。

        阈值从 config 读：rank_exact_threshold=0.995 / highly_correlated_threshold=0.90。

        Parameters
        ----------
        fingerprint : bytes
            待确认因子的 256-bit SimHash fingerprint。
        correlations : dict, optional
            {factor_id: correlation} 字典。未提供时用 hamming 距离的归一化值
            (1 - hamming/256) 作近似。
        """
        seen = self._seen
        neighbors: list[tuple[str, int]] = []
        if hasattr(seen, "topk_fingerprint_neighbors"):
            neighbors = seen.topk_fingerprint_neighbors(fingerprint, k=self.config.nearest_k)
        if not neighbors:
            return SignalConfirmResult("NOVEL", neighbors=[])

        best_corr = 0.0
        best_fid = ""
        result_neighbors: list[tuple[str, float]] = []
        for fid, hamming in neighbors:
            if correlations is not None:
                corr = correlations.get(fid, 0.0)
            else:
                # 无真实 corr 时用 hamming 近似
                corr = 1.0 - hamming / 256.0
            result_neighbors.append((fid, corr))
            if corr > best_corr:
                best_corr = corr
                best_fid = fid

        if best_corr >= self.config.rank_exact_threshold:
            status = "RANK_EQUIVALENT"
        elif best_corr >= self.config.highly_correlated_threshold:
            status = "HIGHLY_CORRELATED"
        else:
            status = "NOVEL"

        return SignalConfirmResult(
            status=status,
            nearest_corr=best_corr,
            nearest_factor_id=best_fid,
            neighbors=result_neighbors,
        )

    # ------------------------------------------------------------------
    # 记录评估
    # ------------------------------------------------------------------

    def record_evaluation(
        self,
        factor_id: str,
        *,
        metrics: dict[str, float] | None = None,
        fingerprint: bytes | None = None,
        subtree_hashes: tuple = (),
        passed_hard_gates: bool = True,
    ) -> None:
        """记录评估结果。内存版暂为 no-op（Phase B 持久版会写入评估表）。"""
        pass

    # ------------------------------------------------------------------
    # 记录 Action
    # ------------------------------------------------------------------

    def record_action(
        self,
        parent_signal_id: str,
        action_type: str,
        payload: dict,
        *,
        grammar_version: str = "",
    ) -> bool:
        """记录 Action 并返回是否首次见到（first_time）。

        Action 唯一性由 (parent_signal_id, action_type, payload, grammar_version)
        的 sha256 决定。
        """
        raw = f"{parent_signal_id}|{action_type}|{_sorted_json(payload)}|{grammar_version}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if hasattr(self._action_seen, "check_and_mark"):
            return self._action_seen.check_and_mark(key)
        return True


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _sorted_json(payload: dict) -> str:
    """字典的确定性 JSON 序列化（key 排序，不依赖 Python 3.7+ 插入序）。"""
    import json

    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


class _IdentityAdapter:
    """把 IdentityView 适配为 Phase B seen 的鸭子类型 identity 对象。"""

    def __init__(self, view: IdentityView) -> None:
        self.factor_id = view.factor_id or view.canonical_ast_hash[:12]
        self.canonical_formula = view.canonical_formula
        self.canonical_ast_hash = view.canonical_ast_hash
        self.signal_equivalence_id = view.signal_equivalence_id
        self.parameter_family_id = view.parameter_family_id
        self.orientation = 1
        self.identity_version = getattr(view, "identity_version", "1") or "1"


def _looks_like_dsl_json(canonical: str) -> bool:
    """判断 FE canonical 是否为 DSL-JSON（以 { 开头且含 op 键）。"""
    s = str(canonical).strip()
    return s.startswith("{") and '"op"' in s[:400]


def _dsl_json_to_text(canonical: str) -> str:
    """把 FE 的 DSL-JSON canonical 还原为近似文本公式（仅用于 IdentityView 展示）。

    不尝试完整 AST 打印：解析 op/args 递归生成 "op(arg1, arg2, ...)"。
    解析失败时原样返回。
    """
    import json

    try:
        obj = json.loads(canonical)
    except Exception:  # noqa: BLE001
        return str(canonical)
    return _dsl_node_to_text(obj)


def _dsl_node_to_text(node: Any) -> str:
    """递归把 DSL-JSON 节点转文本。field → source_name；literal → 值；call → op(args)。"""
    if isinstance(node, dict):
        kind = node.get("kind")
        if kind == "field":
            # 优先原始表名字段（source_name 常映射到 AdjClose 等复权别名），
            # 保证 IdentityView.canonical_formula 与 alphaprobe.dedup 文本 canonical 对齐
            return str(node.get("canonical_name") or node.get("source_name") or node.get("field_id") or "?")
        if kind == "literal":
            v = node.get("value")
            return repr(v) if isinstance(v, str) else str(v)
        if kind == "call":
            op = str(node.get("op") or "?")
            args = [str(_dsl_node_to_text(a)) for a in node.get("args") or []]
            return f"{op}({', '.join(args)})"
        return str(node.get("op") or "?")
    if isinstance(node, list):
        return ", ".join(_dsl_node_to_text(x) for x in node)
    return str(node)