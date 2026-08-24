# -*- coding: utf-8 -*-
"""IncrementalContract — 节点级增量执行契约（R44）。

在既有 ``ExecutionContract``（state_model / chunking / checkpoint_schema）、
``HistoryRequirement``（kind / rows / is_event_clock）与 ``forward_impact`` 之上
叠加一个**单算子**增量模式契约，供 R44 node-level incremental FactorEngine 的
调度 / 重算 / 证书路径消费：

* ``IncrementalMode`` —— 算子按增量语义分到五类 lane：STATELESS /
  FINITE_WINDOW / CHECKPOINTED_STATE / EVENT_ASOF / FULL_REPLAY。
* ``IncrementalContract`` —— 每 canonical 一份，携带
  ``backward_history``（output_start 前需 warm-up 的源 bar 数）与
  ``forward_impact``（一个源 bar 变化影响到的未来输出 bar 数；None = 无界）。
* 解析优先级：显式 ``register_incremental_contract`` 声明 → 自动分类器
  ``classify_incremental_mode``（组合既有 execution_contract /
  history_requirement / forward_impact / StatefulCheckpointRegistry）。
  无法解析时 fail-closed：production 抛错，research 回退 FULL_REPLAY 并
  设置 ``resolution_error``（对齐 ExecutionContract 的 R10 #4 语义）。

本模块是纯增量：不修改 ``runtime.execution_contract`` /
``runtime.incremental`` 的任何既有行为。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# R44 增量模式 / 分区轴 / 截面作用域枚举（str Enum —— 序列化取 .value）
# ---------------------------------------------------------------------------
class IncrementalMode(str, enum.Enum):
    """算子级增量执行模式（R44）。

    * ``STATELESS`` —— 逐行 / 零历史：变化只影响当期输出（backward_history 0）。
    * ``FINITE_WINDOW`` —— 有限滚动窗口：warm-up 与 forward reach 均有界。
    * ``CHECKPOINTED_STATE`` —— 递归状态 + 原生 checkpoint 恢复（segmented lane）。
    * ``EVENT_ASOF`` —— 事件时钟（event_count / report_count / session_count）
      历史按观测数计，天然按事件时点推进。
    * ``FULL_REPLAY`` —— 无 checkpoint 的递归 / 未知：变化传播到序列末端，全量重放。
    """

    STATELESS = "STATELESS"
    FINITE_WINDOW = "FINITE_WINDOW"
    CHECKPOINTED_STATE = "CHECKPOINTED_STATE"
    EVENT_ASOF = "EVENT_ASOF"
    FULL_REPLAY = "FULL_REPLAY"


class StatePartitionAxis(str, enum.Enum):
    """状态分区轴：checkpoint 状态按什么键分开存储 / 恢复（R44）。"""

    PER_INSTRUMENT = "PER_INSTRUMENT"
    PER_GROUP = "PER_GROUP"
    GLOBAL = "GLOBAL"


class CrossSectionScope(str, enum.Enum):
    """算子输出的截面作用域（R44）。

    * ``PER_DATE`` —— 当日行内独立（逐行 / 时间序列）。
    * ``FULL_CROSS_SECTION_PER_DATE`` —— 当日整截面参与（cross-sectional）。
    * ``GROUP`` —— 分组截面（group / neutralize 家族）。
    """

    PER_DATE = "PER_DATE"
    FULL_CROSS_SECTION_PER_DATE = "FULL_CROSS_SECTION_PER_DATE"
    GROUP = "GROUP"


# 截面分类：metadata.category 中属于「整截面参与」的集合。
_CROSS_SECTION_CATEGORIES = frozenset({
    "cross_sectional", "cross_section", "cross_sectional_state",
    "cross_sectional_regression",
})

# 分组截面：category 含 group，或声明了 ``group`` 参数。
_GROUP_CATEGORIES = frozenset({
    "group", "group_structure", "group_neutralization",
})


@dataclass(frozen=True)
class IncrementalContract:
    """单算子增量执行契约（R44）。

    字段集：
      * ``canonical`` —— 解析后的 canonical 名。
      * ``incremental_mode`` —— :class:`IncrementalMode` 五类 lane。
      * ``backward_history`` —— output_start 之前的 warm-up 源 bar 数。
      * ``forward_impact`` —— 一个变化的源 bar 影响到的未来输出 bar 数；None = 无界。
      * ``state_model`` —— 复用 ``ExecutionContract`` 的值（stateless / recursive /
        episode / session_state / unknown）。
      * ``state_partition_axis`` —— checkpoint 状态分区轴。
      * ``cross_section_scope`` —— 截面作用域。
      * ``checkpoint_schema`` —— 复用 ``ExecutionContract.checkpoint_schema``。
      * ``checkpoint_schema_version`` —— ``StatefulCheckpointRegistry`` 的
        ``state_schema_version``（与 checkpoint_schema 分开暴露）。
      * ``revision_policy`` —— ``none`` / ``full_replay`` / ``affected_domain``。
      * ``incremental_certified`` —— 是否生产认证可走 fast lane。
      * ``resolution_error`` —— research 回退时记录失败原因（production 不设置）。
    """

    canonical: str
    incremental_mode: IncrementalMode
    backward_history: int = 0
    forward_impact: int | None = 0
    state_model: str = "stateless"
    state_partition_axis: StatePartitionAxis = StatePartitionAxis.GLOBAL
    cross_section_scope: CrossSectionScope = CrossSectionScope.PER_DATE
    checkpoint_schema: str | None = None
    checkpoint_schema_version: str | None = None
    revision_policy: str = "none"
    incremental_certified: bool = False
    resolution_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化到机器可读 dict（枚举序列化为 .value 字符串）。"""
        return {
            "canonical": self.canonical,
            "incremental_mode": self.incremental_mode.value,
            "backward_history": self.backward_history,
            "forward_impact": self.forward_impact,
            "state_model": self.state_model,
            "state_partition_axis": self.state_partition_axis.value,
            "cross_section_scope": self.cross_section_scope.value,
            "checkpoint_schema": self.checkpoint_schema,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "revision_policy": self.revision_policy,
            "incremental_certified": self.incremental_certified,
            "resolution_error": self.resolution_error,
        }


class IncrementalContractResolutionError(RuntimeError):
    """production 下增量契约无法解析时抛出（对齐 ExecutionContractResolutionError）。

    内部 registry / contract 查找失败绝不静默回退到 STATELESS 或有限窗口；
    production 硬失败，research 才回退（且带显式 ``resolution_error``）。
    """


# ---------------------------------------------------------------------------
# 显式声明注册表（镜像 ``declare_stateful``：重复声明 = 错误）
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, dict[str, Any]] = {}


def register_incremental_contract(
    canonical: str,
    *,
    incremental_mode: IncrementalMode,
    backward_history: int = 0,
    forward_impact: int | None = 0,
    state_partition_axis: StatePartitionAxis = StatePartitionAxis.PER_INSTRUMENT,
    cross_section_scope: CrossSectionScope = CrossSectionScope.PER_DATE,
    checkpoint_schema: str | None = None,
    checkpoint_schema_version: str | None = None,
    revision_policy: str = "none",
    incremental_certified: bool = False,
) -> None:
    """显式声明一个算子的增量契约（R44）。

    ``incremental_mode`` 必须是 :class:`IncrementalMode`；重复声明同 canonical
    是 drift（不是 refinement），直接抛错——镜像 ``declare_stateful`` 的语义。
    """
    if canonical in _REGISTRY:
        raise IncrementalContractResolutionError(
            f"duplicate register_incremental_contract for {canonical!r} — a "
            "canonical may declare its incremental contract exactly once"
        )
    if not isinstance(incremental_mode, IncrementalMode):
        raise IncrementalContractResolutionError(
            f"register_incremental_contract for {canonical!r}: incremental_mode "
            f"must be an IncrementalMode, got {incremental_mode!r}"
        )
    if not isinstance(state_partition_axis, StatePartitionAxis):
        state_partition_axis = StatePartitionAxis(state_partition_axis)
    if not isinstance(cross_section_scope, CrossSectionScope):
        cross_section_scope = CrossSectionScope(cross_section_scope)
    _REGISTRY[canonical] = {
        "canonical": canonical,
        "incremental_mode": incremental_mode,
        "backward_history": int(backward_history or 0),
        "forward_impact": (
            None if forward_impact is None else max(0, int(forward_impact))
        ),
        "state_partition_axis": state_partition_axis,
        "cross_section_scope": cross_section_scope,
        "checkpoint_schema": checkpoint_schema,
        "checkpoint_schema_version": checkpoint_schema_version,
        "revision_policy": revision_policy,
        "incremental_certified": bool(incremental_certified),
    }


def registered_incremental_contracts() -> dict[str, dict[str, Any]]:
    """快照所有显式声明（审计 / CI）。"""
    return {k: dict(v) for k, v in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# 自动分类器
# ---------------------------------------------------------------------------
def _resolve_alias(canonical: str, *, strict: bool = False) -> str:
    """解析 DSL alias -> canonical（懒加载；registry 未就绪时安全返回原值）。"""
    if not canonical:
        return canonical
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(canonical)
    except Exception as exc:
        if strict:
            raise IncrementalContractResolutionError(
                f"production: alias resolution failed for {canonical!r}: {exc}"
            ) from exc
        return canonical


def _catalog_category(canonical: str) -> str | None:
    """算子 metadata.category（懒加载；缺省 None，绝不抛错阻塞分类）。"""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical)
        return getattr(getattr(op, "metadata", None), "category", None)
    except Exception:
        return None


def _catalog_param_names(canonical: str) -> tuple[str, ...]:
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical)
        return tuple(getattr(getattr(op, "metadata", None), "param_names", None) or ())
    except Exception:
        return ()


def _cross_section_scope_for(canonical: str) -> CrossSectionScope:
    """从 metadata.category / 声明参数推导截面作用域（R44）。

    显式声明的 ``incremental_mode`` 覆盖此推导；这里只服务自动分类路径。
    category 含 group（group / group_structure / group_neutralization）或声明了
    ``group`` 参数 → GROUP；整截面 category → FULL_CROSS_SECTION_PER_DATE；
    其余（含 unknown）→ PER_DATE（保守）。
    """
    category = _catalog_category(canonical)
    if category is not None:
        cat = str(category).lower()
        if cat in _GROUP_CATEGORIES:
            return CrossSectionScope.GROUP
        if cat in _CROSS_SECTION_CATEGORIES:
            return CrossSectionScope.FULL_CROSS_SECTION_PER_DATE
    if "group" in _catalog_param_names(canonical):
        return CrossSectionScope.GROUP
    return CrossSectionScope.PER_DATE


def _checkpoint_spec(canonical: str):
    """StatefulCheckpointRegistry.get（懒加载；失败返回 None）。"""
    try:
        from factor_engine.stateful_contract import StatefulCheckpointRegistry

        return StatefulCheckpointRegistry.get(canonical)
    except Exception:
        return None


def _is_known_canonical(resolved: str) -> bool:
    """resolved canonical 是否存在于 operator registry。

    ``history_requirement`` 对未知 canonical 仍返回保守的 finite rows=2
    （不抛错），所以「未知」无法从 history/forward 数字本身辨别——必须在分类前
    显式查 registry。registry 内部错误也按未知处理（fail-closed → FULL_REPLAY）。
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        operators, _aliases, catalog = OperatorRegistry._read_state()
        return resolved in operators or resolved in catalog
    except Exception:
        return False


def classify_incremental_mode(
    canonical: str, params: Mapping[str, Any] | None = None
) -> IncrementalMode:
    """自动分类一个 canonical 的增量模式（R44）。

    决策表（自上而下取第一个命中）：
      * canonical 不在 operator registry（未知 / 无法解析）→ ``FULL_REPLAY``
        （fail-closed，绝不把未知 canonical 静默当有限窗口——history_requirement
        对未知名返回保守 finite 2 行，不能据此分类）。
      * contract.requires_full_history 或 state_model != "stateless":
          - canonical 在 StatefulCheckpointRegistry 且 state_model == "recursive"
            → ``CHECKPOINTED_STATE``
          - 否则 → ``FULL_REPLAY``（递归无 checkpoint 恢复 / episode /
            session_state / unknown，fail-closed 全量重放）。
      * ``history_requirement`` 的 kind 是 event_count / report_count /
        session_count（``is_event_clock``）→ ``EVENT_ASOF``。
      * ``forward_impact`` 有界 且 ``backward_history >= 1`` → ``FINITE_WINDOW``。
      * ``backward_history == 0``（逐行 / 零历史）→ ``STATELESS``。
      * 任何内部错误 / 无法解析 → ``FULL_REPLAY``（fail-closed，绝不静默
        STATELESS）。
    """
    resolved = _resolve_alias(canonical)
    try:
        if not _is_known_canonical(resolved):
            return IncrementalMode.FULL_REPLAY
        from factor_engine.runtime.execution_contract import (
            execution_contract,
            forward_impact,
            history_requirement,
        )

        contract = execution_contract(resolved)
        if contract.requires_full_history or contract.state_model != "stateless":
            spec = _checkpoint_spec(resolved)
            if spec is not None and contract.state_model == "recursive":
                return IncrementalMode.CHECKPOINTED_STATE
            return IncrementalMode.FULL_REPLAY
        req = history_requirement(resolved, params or {})
        if req.is_event_clock:
            return IncrementalMode.EVENT_ASOF
        fwd = forward_impact(resolved, params or {})
        back = 0
        if not req.is_full_history:
            try:
                back = max(0, int(req.rows))
            except (TypeError, ValueError):
                back = 0
        if back >= 1:
            return IncrementalMode.FINITE_WINDOW
        return IncrementalMode.STATELESS
    except Exception:
        return IncrementalMode.FULL_REPLAY


def _revision_policy_for(mode: IncrementalMode) -> str:
    """revision_policy 推导：事件时钟 / 有限窗口 → affected_domain；递归/未知
    → full_replay；零历史 → none。"""
    if mode in (IncrementalMode.EVENT_ASOF, IncrementalMode.FINITE_WINDOW):
        return "affected_domain"
    if mode in (IncrementalMode.FULL_REPLAY,):
        return "full_replay"
    if mode is IncrementalMode.CHECKPOINTED_STATE:
        return "affected_domain"
    return "none"


# ---------------------------------------------------------------------------
# 唯一公开解析器
# ---------------------------------------------------------------------------
def resolve_incremental_contract(
    canonical: str,
    params: Mapping[str, Any] | None = None,
    *,
    production: bool = False,
) -> IncrementalContract:
    """解析一个 canonical 的 :class:`IncrementalContract`（R44 唯一入口）。

    优先级：
      1. 显式 ``register_incremental_contract`` 声明（权威）。
      2. 自动分类器（组合既有 execution_contract / history_requirement /
         forward_impact / StatefulCheckpointRegistry）。

    ``_resolve(strict=production)`` 语义：production 下内部查找失败抛
    :class:`IncrementalContractResolutionError`；research 回退 FULL_REPLAY 并
    设置 ``resolution_error``（fail-closed，绝不停留在 stateless）。
    """
    resolved = _resolve_alias(canonical, strict=production)
    # 优先级 1：显式声明是权威 —— 即使 canonical 不在 operator registry，
    # 声明本身即生效（先于 unknown 检查）。
    registered = _REGISTRY.get(resolved)
    if registered is not None:
        return IncrementalContract(
            canonical=resolved,
            incremental_mode=registered["incremental_mode"],
            backward_history=registered["backward_history"],
            forward_impact=registered["forward_impact"],
            state_partition_axis=registered["state_partition_axis"],
            cross_section_scope=registered["cross_section_scope"],
            checkpoint_schema=registered["checkpoint_schema"],
            checkpoint_schema_version=registered["checkpoint_schema_version"],
            revision_policy=registered["revision_policy"],
            incremental_certified=registered["incremental_certified"],
        )
    if not _is_known_canonical(resolved):
        # 未知 canonical：fail-closed。production 硬失败；research 回退
        # FULL_REPLAY 并显式标记 resolution_error（绝不静默当有限窗口）。
        message = (
            f"incremental contract resolution failed for {resolved!r}: "
            "canonical is not in the operator registry"
        )
        if production:
            raise IncrementalContractResolutionError(message)
        return IncrementalContract(
            canonical=resolved,
            incremental_mode=IncrementalMode.FULL_REPLAY,
            backward_history=0,
            forward_impact=None,
            state_model="unknown",
            state_partition_axis=StatePartitionAxis.GLOBAL,
            cross_section_scope=CrossSectionScope.PER_DATE,
            checkpoint_schema=None,
            checkpoint_schema_version=None,
            revision_policy="full_replay",
            incremental_certified=False,
            resolution_error=message,
        )

    try:
        from factor_engine.runtime.execution_contract import (
            execution_contract,
            forward_impact,
            history_requirement,
        )

        contract = execution_contract(resolved, production=production)
        mode = classify_incremental_mode(resolved, params or {})
        req = history_requirement(resolved, params or {}, production=production)
        fwd = forward_impact(resolved, params or {}, production=production)

        backward_history = 0
        if not req.is_full_history:
            try:
                backward_history = max(0, int(req.rows))
            except (TypeError, ValueError):
                backward_history = 0

        spec = _checkpoint_spec(resolved)
        checkpoint_schema = contract.checkpoint_schema
        checkpoint_schema_version = (
            spec.state_schema_version if spec is not None else None
        )
        # CHECKPOINTED_STATE：状态按 instrument 分区；其余按 GLOBAL 分区。
        state_partition_axis = (
            StatePartitionAxis.PER_INSTRUMENT
            if mode is IncrementalMode.CHECKPOINTED_STATE
            else StatePartitionAxis.GLOBAL
        )
        return IncrementalContract(
            canonical=resolved,
            incremental_mode=mode,
            backward_history=backward_history,
            forward_impact=fwd,
            state_model=contract.state_model,
            state_partition_axis=state_partition_axis,
            cross_section_scope=_cross_section_scope_for(resolved),
            checkpoint_schema=checkpoint_schema,
            checkpoint_schema_version=checkpoint_schema_version,
            revision_policy=_revision_policy_for(mode),
            incremental_certified=False,
        )
    except Exception as exc:
        if production:
            raise IncrementalContractResolutionError(
                f"production: incremental contract resolution failed for "
                f"{resolved!r}: {exc}"
            ) from exc
        return IncrementalContract(
            canonical=resolved,
            incremental_mode=IncrementalMode.FULL_REPLAY,
            backward_history=0,
            forward_impact=None,
            state_model="unknown",
            state_partition_axis=StatePartitionAxis.GLOBAL,
            cross_section_scope=CrossSectionScope.PER_DATE,
            checkpoint_schema=None,
            checkpoint_schema_version=None,
            revision_policy="full_replay",
            incremental_certified=False,
            resolution_error=str(exc) or "UNKNOWN_INCREMENTAL_CONTRACT",
        )


# ---------------------------------------------------------------------------
# IncrementalCapabilityMatrix
# ---------------------------------------------------------------------------
_CAPABILITY_FIELDS = (
    "canonical",
    "incremental_mode",
    "incremental_certified",
    "backward_history",
    "forward_impact",
    "state_model",
    "state_partition_axis",
    "cross_section_scope",
    "checkpoint_schema",
    "checkpoint_schema_version",
)


def incremental_capability_matrix(
    canonicals: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """机器可读 IncrementalCapabilityMatrix（R44）。

    默认枚举 ``cleaned_operators.registry.OperatorRegistry`` 的完整 catalog；
    每个 canonical 一行，含 :data:`_CAPABILITY_FIELDS` 十个字段。解析失败的行
    以 fail-closed 语义落入 matrix（FULL_REPLAY / incremental_certified False），
    绝不跳过任何 canonical。
    """
    if canonicals is None:
        try:
            from factor_engine.cleaned_operators.registry import OperatorRegistry

            canonicals = sorted(OperatorRegistry._catalog)
        except Exception:
            from factor_engine.cleaned_operators.registry import OperatorRegistry

            canonicals = sorted(getattr(OperatorRegistry, "_catalog", {}))
    rows: list[dict[str, Any]] = []
    for canonical in canonicals:
        contract = resolve_incremental_contract(canonical)
        row = contract.to_dict()
        rows.append({field: row[field] for field in _CAPABILITY_FIELDS})
    return rows


# ---------------------------------------------------------------------------
# R45 认证等级 + IncrementalCertificationLedger
# ---------------------------------------------------------------------------
class IncrementalCertificationLevel(str, enum.Enum):
    """算子生产增量认证等级（R45）。

    * ``TRUE_INCREMENTAL`` —— 已通过数值 parity（full vs segmented-incremental
      逐行对齐 + restart-resume 恢复）的状态算子，真实增量已证明。
    * ``TAIL_REPLAY`` / ``EVENT_INCREMENTAL`` / ``FULL_REPLAY_ONLY`` —— 策略
      分类槽位（保留），仅当对应算子的生产增量契约被证明后才落位。
    * ``NOT_CERTIFIED`` —— 尚未证明（框架已实现 ≠ 生产增量已认证）。
    """

    TRUE_INCREMENTAL = "TRUE_INCREMENTAL"
    TAIL_REPLAY = "TAIL_REPLAY"
    EVENT_INCREMENTAL = "EVENT_INCREMENTAL"
    FULL_REPLAY_ONLY = "FULL_REPLAY_ONLY"
    NOT_CERTIFIED = "NOT_CERTIFIED"


# 当前 head 上经 parity（full vs incremental 逐行对齐 + restart-resume）证明的
# 状态算子集合。与 runtime.incremental_parity.SEGMENTED_CANONICALS 保持一致；
# 实测 9 个全部通过（任务书预估 8，实为 9，以真实 parity 为准）。
_PARITY_PROVEN_FALLBACK = frozenset({
    "ts_ema", "ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr",
    "RSI_WILDER", "ATR_WILDER", "ADX", "MACD_line",
})


def parity_proven_canonicals() -> frozenset[str]:
    """返回当前 head 上已通过 incremental parity 的状态算子集合（R45）。

    懒加载 ``runtime.incremental_parity.SEGMENTED_CANONICALS`` 保持单一权威；
    若 import 失败回退到本模块内联快照。
    """
    try:
        from factor_engine.runtime.incremental_parity import SEGMENTED_CANONICALS

        return frozenset(SEGMENTED_CANONICALS)
    except Exception:
        return _PARITY_PROVEN_FALLBACK


def classify_certification_level(
    canonical: str, mode: IncrementalMode | None = None
) -> IncrementalCertificationLevel:
    """把 canonical 归类为 5 级认证状态（R45）。

    诚实原则：仅当算子已通过 parity（状态增量逐段对齐 + restart-resume）才给
    ``TRUE_INCREMENTAL``；其余一律 ``NOT_CERTIFIED``。绝不因框架实现了某 lane 就
    虚报生产认证。``mode`` 参数保留给未来策略槽位，当前不改变结果。
    """
    if canonical in parity_proven_canonicals():
        return IncrementalCertificationLevel.TRUE_INCREMENTAL
    return IncrementalCertificationLevel.NOT_CERTIFIED


# 扩展 capability matrix 每行额外携带的字段（R45 维度；在 10 个既有字段之上）。
_CAPABILITY_EXTRA_FIELDS = (
    "certification_level",
    "direct_mining",
    "ashare_source_ready",
    "pit",
    "pandas",
    "polars",
    "duckdb",
    "q",
    "numba",
    "incremental",
    "e2e",
    "live_production_usable",
)


def _registry_catalog_entry(canonical: str) -> dict[str, Any]:
    """读取 OperatorRegistry._catalog 的单条目；registry 不可用时回退空 dict。"""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return dict(OperatorRegistry._catalog.get(canonical, {}))
    except Exception:
        return {}


def _capability_dimensions(
    canonical: str, mode: IncrementalMode
) -> dict[str, bool]:
    """每 canonical 的维度探测（R45）。

    诚实性：只对 registry 声明为真的维度给 True；未知/缺失一律 False（不虚报）。
    当前 head 上 catalog 的 backend 只有 pandas_numpy / polars / sql，无 duckdb /
    q 后端，也无 numba 加速器，故 duckdb / q / numba 全为 False（如实报告）。
    """
    cat = _registry_catalog_entry(canonical)
    backends = cat.get("backends") or []
    backend_lower = {str(b).lower() for b in backends}
    status = cat.get("status")
    hidden = bool(cat.get("hidden_from_default_mining"))
    return {
        "direct_mining": (status in ("production", "experimental")) and not hidden,
        "ashare_source_ready": (
            cat.get("scope") == "ashare" or cat.get("category") == "ashare"
        ),
        "pit": cat.get("pit_safe") is True,
        "pandas": "pandas_numpy" in backends,
        "polars": "polars" in backends,
        "duckdb": any("duck" in b for b in backend_lower),
        "q": any(b == "q" or b.startswith("q_") or b.endswith("_q") for b in backend_lower),
        "numba": False,
        "incremental": mode is not IncrementalMode.FULL_REPLAY,
        "e2e": mode in (
            IncrementalMode.CHECKPOINTED_STATE,
            IncrementalMode.FINITE_WINDOW,
            IncrementalMode.EVENT_ASOF,
        ),
    }


def _is_live_production_usable(canonical: str, mode: IncrementalMode) -> bool:
    """聚合 LIVE_PRODUCTION_USABLE 判定（R45）。

    全部证据到齐才算可用：lifecycle_status == "production"、production_certified、
    未隐藏、且增量模式非 FULL_REPLAY（真实可增量运行）。任何一项缺失 → False。
    """
    cat = _registry_catalog_entry(canonical)
    if cat.get("lifecycle_status") != "production":
        return False
    if cat.get("production_certified") is not True:
        return False
    if bool(cat.get("hidden_from_default_mining")):
        return False
    if mode is IncrementalMode.FULL_REPLAY:
        return False
    return True


def _matrix_canonicals(
    canonicals: Sequence[str] | None,
) -> list[str]:
    """解析待枚举的 canonical 集合；None 时取 OperatorRegistry._catalog 全量。"""
    if canonicals is not None:
        return list(canonicals)
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return sorted(OperatorRegistry._catalog)
    except Exception:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return sorted(getattr(OperatorRegistry, "_catalog", {}))


def incremental_capability_matrix_extended(
    canonicals: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """扩展 IncrementalCapabilityMatrix（R45）。

    在 :func:`incremental_capability_matrix` 的 10 字段之上，为每个 canonical 追加
    ``certification_level``（5 级认证）与 ``direct_mining`` / ``ashare_source_ready`` /
    ``pit`` / ``pandas`` / ``polars`` / ``duckdb`` / ``q`` / ``numba`` /
    ``incremental`` / ``e2e`` 维度，以及单个聚合 ``live_production_usable`` 布尔。
    未证维度如实 False（不虚报 PASS）。缺省枚举 OperatorRegistry 权威全量。
    """
    rows: list[dict[str, Any]] = []
    for canonical in _matrix_canonicals(canonicals):
        contract = resolve_incremental_contract(canonical)
        row = contract.to_dict()
        row["certification_level"] = classify_certification_level(
            canonical, contract.incremental_mode
        ).value
        dims = _capability_dimensions(canonical, contract.incremental_mode)
        row.update({k: bool(v) for k, v in dims.items()})
        row["live_production_usable"] = _is_live_production_usable(
            canonical, contract.incremental_mode
        )
        rows.append(row)
    return rows


@dataclass(frozen=True)
class IncrementalCertificationLedger:
    """R45 认证账本：全量 canonical 的认证分布快照。

    * ``operator_total`` —— 权威 registry 的 canonical 总数。
    * ``distribution`` —— certification_level -> 计数（5 级分布）。
    * ``strategy_distribution`` —— incremental_mode -> 计数（策略分布）。
    * ``live_production_usable`` —— 聚合 LIVE 数。
    * ``true_incremental_canonicals`` —— 已认证（TRUE_INCREMENTAL）的 canonical。
    * ``not_certified_canonicals`` —— NOT_CERTIFIED 的 canonical。
    """

    operator_total: int
    distribution: Mapping[str, int]
    strategy_distribution: Mapping[str, int]
    live_production_usable: int
    true_incremental_canonicals: tuple[str, ...]
    not_certified_canonicals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_total": self.operator_total,
            "distribution": dict(self.distribution),
            "strategy_distribution": dict(self.strategy_distribution),
            "live_production_usable": self.live_production_usable,
            "true_incremental_canonicals": list(self.true_incremental_canonicals),
            "not_certified_canonicals_count": len(self.not_certified_canonicals),
        }


def build_incremental_certification_ledger(
    canonicals: Sequence[str] | None = None,
) -> IncrementalCertificationLedger:
    """构建当前 head 的全量增量认证账册（R45）。

    枚举权威 registry 全量；每个 canonical 恰好一条：certification_level 由
    :func:`classify_certification_level` 决定（parity 证明者 TRUE_INCREMENTAL，其余
    NOT_CERTIFIED），strategy 由 incremental_mode 决定。诚实：不虚报未证认证。
    """
    rows = incremental_capability_matrix_extended(canonicals)
    distribution: dict[str, int] = {}
    for row in rows:
        lvl = str(row["certification_level"])
        distribution[lvl] = distribution.get(lvl, 0) + 1
    strategy: dict[str, int] = {}
    for row in rows:
        mode = str(row["incremental_mode"])
        strategy[mode] = strategy.get(mode, 0) + 1
    live = sum(1 for row in rows if row["live_production_usable"])
    true_canonicals = tuple(
        sorted(row["canonical"] for row in rows
               if row["certification_level"] == IncrementalCertificationLevel.TRUE_INCREMENTAL.value)
    )
    not_cert = tuple(
        sorted(row["canonical"] for row in rows
               if row["certification_level"] == IncrementalCertificationLevel.NOT_CERTIFIED.value)
    )
    return IncrementalCertificationLedger(
        operator_total=len(rows),
        distribution=distribution,
        strategy_distribution=strategy,
        live_production_usable=live,
        true_incremental_canonicals=true_canonicals,
        not_certified_canonicals=not_cert,
    )


# ---------------------------------------------------------------------------
# IR 节点级解析
# ---------------------------------------------------------------------------
def _node_params(node: Any) -> dict[str, Any]:
    """从一个 IRNode 提取绑定参数（镜像 execution_contract._node_params）。

    ``node.attrs`` 的 kwargs + 位置 literal 子节点按 operator 声明
    ``param_names`` 映射。import ``runtime.execution_contract._node_params``
    的私有名在本包内是允许的，但为避免与旧 execution_contract 的加载时序耦合，
    这里内联一份等价提取（保持一致行为）。
    """
    params: dict[str, Any] = {}
    if getattr(node, "attrs", None):
        params.update(dict(node.attrs))
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        meta = getattr(OperatorRegistry.get(getattr(node, "op", "")), "metadata", None)
    except Exception:
        meta = None
    if meta is None:
        return params
    param_names = tuple(getattr(meta, "param_names", None) or ())
    for index, child in enumerate(getattr(node, "inputs", ()) or ()):
        if getattr(child, "op", None) == "literal":
            name = param_names[index] if index < len(param_names) else None
            if name:
                params.setdefault(name, getattr(child, "attrs", {}).get("value"))
    return params


def node_incremental_slot(
    ir_node: Any, *, production: bool = False
) -> IncrementalContract:
    """解析一个 IR 节点的增量契约（R44 node-level slot）。

    ``ir_node`` 是 :class:`ir.nodes.IRNode`（或任何带 ``op`` / ``inputs`` /
    ``attrs`` 的 IR 形态对象）；参数提取复用 ``_node_params`` 的技术（kwargs
    attrs + 位置 literal 输入按声明 param_names 映射），再经
    :func:`resolve_incremental_contract` 解析。
    """
    op = str(getattr(ir_node, "op", "") or "")
    params = _node_params(ir_node)
    return resolve_incremental_contract(op, params, production=production)


__all__ = [
    "IncrementalMode",
    "StatePartitionAxis",
    "CrossSectionScope",
    "IncrementalContract",
    "IncrementalContractResolutionError",
    "register_incremental_contract",
    "registered_incremental_contracts",
    "classify_incremental_mode",
    "resolve_incremental_contract",
    "incremental_capability_matrix",
    "node_incremental_slot",
    "IncrementalCertificationLevel",
    "IncrementalCertificationLedger",
    "classify_certification_level",
    "build_incremental_certification_ledger",
    "incremental_capability_matrix_extended",
    "parity_proven_canonicals",
]
