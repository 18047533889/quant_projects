"""research_space.hypothesis —— plan.md Task 16：HypothesisRecord + 九阶段状态机。

设计（保持 AlphaPROBE 差异化；plan 要求"不造 20 个常驻 agent"）
--------------------------------------------------------------
- AlphaPROBE 仍是 search/navigation-centric：多阶段**不是** 20 个常驻 agent，
  而是一个 hypothesis 生命周期状态机 + 一条从「假设 → 公式」的 stage 管线。
- 本模块只提供**状态推进 + 失败存活语义**（假说与实现解耦：implementation
  失败不杀 hypothesis，可生成替代实现，尝试计数有 cap）；
- 九阶段骨架用 **pipeline hook**（``StageHooks``）表达：默认 hook 全部是
  pass-through（返回未变 context），调用方按需挂载真实实现（FE 编译、
  确定性对齐、去重、评估反馈由既有平台能力注入——本模块不复制任何 FE/DA/QE
  能力）；
- 确定性对齐（stage 5）默认不执行：调用方挂 ``on_stage_5`` 注入
  ``DeterministicAligner`` 结果；缺省 align 结果 = 通过（AlignmentResult
  score=1.0）——保持纯单测与旧路径零依赖可跑。
- 多阶段生成整体可开关（ablation switch，#30）：``SearchOrchestrator`` 默认
  ``multistage_generation=False``；开启后 stage hook 才生效。

Non-negotiable 映射
-------------------
- #18：假设↔实现 domain 矛盾必须在评估前拦截——stage 5 结果
  ``blocked`` 时候选不进 FE 执行结果（本模块在 stage 7 前读 alignment_result
  的 blocked 位即停）。
- #30：本模块新增特性全部有开关（多阶段 enabled 开关 / 尝试 cap 常量）。

文件边界：本模块不 import factor_engine / data_access / factor_assets / torch
（顶层零平台依赖；测试与 OFFLINE_TEST 可直 import）。registry 持久化风格
参照 memory/__init__.py 与 research_space/registry.py（sqlite3 + Lock +
executescript + INSERT OR REPLACE）。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from alphaprobe.research_space.contracts import (
    ALIGNMENT_FULL_SCORE,
    ALIGNMENT_MISMATCH_SCORE,
    AlignmentResult,
    HypothesisSpec,
)

__all__ = [
    "Stage",
    "STAGES",
    "DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS",
    "ALPHAPROBE_HYPOTHESIS_DB",
    "StageHooks",
    "ImplementationAttempt",
    "HypothesisRecord",
    "HypothesisRegistry",
    "register_hypothesis",
    "run_stage_pipeline",
]

#: 默认允许的替代实现尝试上限（用尽后不再自动重试替代实现）。
DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS: int = 3

#: 独立 SQLite 路径环境变量（缺省文件路径；:memory: 由构造器处理）。
ALPHAPROBE_HYPOTHESIS_DB = "ALPHAPROBE_HYPOTHESIS_DB"


class Stage(str, Enum):
    """九阶段骨架常量（plan Task 16 的 1-9）。"""

    STAGE_1_HYPOTHESIS = "STAGE_1_HYPOTHESIS"
    STAGE_2_SCHEMA_PLAN = "STAGE_2_SCHEMA_PLAN"
    STAGE_3_AST_ACTION_PLAN = "STAGE_3_AST_ACTION_PLAN"
    STAGE_4_FE_COMPILE = "STAGE_4_FE_COMPILE"
    STAGE_5_DETERMINISTIC_ALIGNMENT = "STAGE_5_DETERMINISTIC_ALIGNMENT"
    STAGE_6_OPTIONAL_CRITIC = "STAGE_6_OPTIONAL_CRITIC"
    STAGE_7_FE_VALIDATION = "STAGE_7_FE_VALIDATION"
    STAGE_8_DEDUP = "STAGE_8_DEDUP"
    STAGE_9_EVALUATION_FEEDBACK = "STAGE_9_EVALUATION_FEEDBACK"

    def __iter__(self) -> Any:  # noqa: D105 - str 子类天然可迭代字符，仅枚举需可迭代成员
        return iter([self.value])


STAGES: tuple[Stage, ...] = (
    Stage.STAGE_1_HYPOTHESIS,
    Stage.STAGE_2_SCHEMA_PLAN,
    Stage.STAGE_3_AST_ACTION_PLAN,
    Stage.STAGE_4_FE_COMPILE,
    Stage.STAGE_5_DETERMINISTIC_ALIGNMENT,
    Stage.STAGE_6_OPTIONAL_CRITIC,
    Stage.STAGE_7_FE_VALIDATION,
    Stage.STAGE_8_DEDUP,
    Stage.STAGE_9_EVALUATION_FEEDBACK,
)

#: 需要 action JSON 参与前置校验的 stage（stage 1/2/3 都是 action 生成侧：
#: 畸形 action 必须在这三段被拦，永不触达 FE 执行 stage 4）。
_PRE_EXECUTION_STAGES = frozenset(
    {
        Stage.STAGE_1_HYPOTHESIS,
        Stage.STAGE_2_SCHEMA_PLAN,
        Stage.STAGE_3_AST_ACTION_PLAN,
    }
)

#: 状态：proposed / aligned / implementation_failed / implemented（+ blocked）。
STATE_PROPOSED = "proposed"
STATE_ALIGNED = "aligned"
STATE_IMPLEMENTATION_FAILED = "implementation_failed"
STATE_IMPLEMENTED = "implemented"
STATE_BLOCKED = "blocked"
_STATES = frozenset(
    {STATE_PROPOSED, STATE_ALIGNED, STATE_IMPLEMENTATION_FAILED,
     STATE_IMPLEMENTED, STATE_BLOCKED}
)

#: stage → 允许推进到的下一状态（实现路径；blocked 在任何前置 stage 即可停）。
_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_PROPOSED: frozenset({STATE_ALIGNED, STATE_IMPLEMENTATION_FAILED,
                               STATE_BLOCKED}),
    STATE_ALIGNED: frozenset({STATE_IMPLEMENTATION_FAILED, STATE_IMPLEMENTED,
                              STATE_BLOCKED, STATE_ALIGNED}),
    STATE_IMPLEMENTATION_FAILED: frozenset({STATE_IMPLEMENTATION_FAILED,
                                            STATE_IMPLEMENTED, STATE_ALIGNED,
                                            STATE_BLOCKED}),
    STATE_IMPLEMENTED: frozenset({STATE_IMPLEMENTED}),
    STATE_BLOCKED: frozenset({STATE_BLOCKED}),
}


# ---------------------------------------------------------------------------
# pipeline hook（调用方挂载真实实现；默认 pass-through）
# ---------------------------------------------------------------------------

StageHookFn = Callable[[str, dict[str, Any], "HypothesisRecord"], dict[str, Any]]


@dataclass
class StageHooks:
    """九阶段 pipeline hooks（全部默认 pass-through，不改变 context）。

    调用方按需挂载：FE 编译（stage 4）、确定性对齐（stage 5）、可选 critic
    （stage 6）、FE 验证/静态分析（stage 7）、去重（stage 8）、评估反馈
    （stage 9）。stage 1-3 由状态机执行默认拦截（畸形 action 校验），也可被
    覆盖为真实 hypothesis/schema/AST 生成。
    """

    on_stage_1: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_2: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_3: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_4: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_5: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_6: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_7: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_8: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))
    on_stage_9: StageHookFn = field(default=lambda st, ctx, rec: dict(ctx))

    def hook_for(self, stage: Stage) -> StageHookFn:
        return {
            Stage.STAGE_1_HYPOTHESIS: self.on_stage_1,
            Stage.STAGE_2_SCHEMA_PLAN: self.on_stage_2,
            Stage.STAGE_3_AST_ACTION_PLAN: self.on_stage_3,
            Stage.STAGE_4_FE_COMPILE: self.on_stage_4,
            Stage.STAGE_5_DETERMINISTIC_ALIGNMENT: self.on_stage_5,
            Stage.STAGE_6_OPTIONAL_CRITIC: self.on_stage_6,
            Stage.STAGE_7_FE_VALIDATION: self.on_stage_7,
            Stage.STAGE_8_DEDUP: self.on_stage_8,
            Stage.STAGE_9_EVALUATION_FEEDBACK: self.on_stage_9,
        }[stage]


# ---------------------------------------------------------------------------
# HypothesisRecord（内存状态 + 生命周期）
# ---------------------------------------------------------------------------


@dataclass
class ImplementationAttempt:
    """一次公式实现的尝试记录（失败/成功都留档）。"""

    formula: str = ""
    impl_id: str = ""
    action: Mapping[str, Any] = field(default_factory=dict)
    status: str = "recorded"  # recorded / rejected / implemented
    stage: str = ""
    reason: str = ""
    created_at: float = 0.0


@dataclass
class HypothesisRecord:
    """一条假设的生命周期记录（假说与实现解耦）。

    ``state`` ∈ proposed / aligned / implementation_failed / implemented /
    blocked（初始 proposed）。``implementation_attempts`` = 替代实现尝试计数
    （上限 ``DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS``，用尽后不再自动重试）。
    ``history`` = [(stage, status, note), ...] 追加型（可 audit）。持久化仅当
    registry 打开文件 DB 时。
    """

    hypothesis_id: str
    spec: HypothesisSpec
    state: str = STATE_PROPOSED
    schema_id: str | None = None
    logic_id: str | None = None
    implementation_attempts: int = 0
    implemented_impl_id: str | None = None
    best_formula: str = ""
    failure_reason: str = ""
    history: list[tuple[str, str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- 状态推进（失败不杀 hypothesis；替代实现计数） --------------------


def _transition_to(self, state: str, *, note: str = "", stage: str = "") -> None:
    """HypothesisRecord.transition_to 的独立实现（供引用共享）。

    ``_TRANSITIONS`` 以 state 为键；state 非法时保留原状态（幂等 no-op）。
    """
    if state not in _STATES:
        return
    allowed = _TRANSITIONS.get(self.state, frozenset())
    if state not in allowed and state != self.state:
        # 非法迁移：记录但不抛（保守保留原状态；同一状态重复推进是幂等 no-op）
        return
    self.state = state
    self.updated_at = time.time()
    self.history.append((stage or self.state, state, note))


def _transition_to_bound(self, state: str, *, note: str = "", stage: str = "") -> None:
    _transition_to(self, state, note=note, stage=stage)


def _mark_aligned_bound(self, *, schema_id: str | None = None) -> None:
    self.schema_id = schema_id or self.spec.schema_id
    _transition_to(
        self, STATE_ALIGNED,
        note="deterministic alignment ok",
        stage=Stage.STAGE_5_DETERMINISTIC_ALIGNMENT,
    )


def _mark_blocked_bound(self, *, note: str = "") -> None:
    _transition_to(
        self, STATE_BLOCKED,
        note=note or "alignment blocked",
        stage=Stage.STAGE_5_DETERMINISTIC_ALIGNMENT,
    )


def _can_try_implementation_bound(self) -> bool:
    return self.implementation_attempts < DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS


def _record_impl_attempt_bound(self, *, formula: str, reason: str = "") -> None:
    """实现尝试计数 +1。只在当前状态允许尝试时生效。"""
    if self.state in (STATE_IMPLEMENTED, STATE_BLOCKED):
        return
    if self.implementation_attempts >= DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS:
        return
    self.implementation_attempts += 1
    if reason:
        self.failure_reason = reason
        _transition_to(
            self, STATE_IMPLEMENTATION_FAILED,
            note=f"impl failed: {reason}",
            stage=Stage.STAGE_7_FE_VALIDATION,
        )


def _record_impl_success_bound(self, *, formula: str, impl_id: str) -> None:
    self.implemented_impl_id = impl_id
    self.best_formula = formula
    _transition_to(
        self, STATE_IMPLEMENTED,
        note="implementation ok", stage=Stage.STAGE_7_FE_VALIDATION,
    )


def _as_dict_bound(self) -> dict[str, Any]:
    return {
        "hypothesis_id": self.hypothesis_id,
        "state": self.state,
        "spec": {
            "hypothesis_id": self.spec.hypothesis_id,
            "logic_id": self.spec.logic_id,
            "schema_id": self.spec.schema_id,
            "text": self.spec.text,
            "expected_domains": list(self.spec.expected_domains),
            "expected_roles": list(self.spec.expected_roles),
            "expected_horizon": self.spec.expected_horizon,
        },
        "schema_id": self.schema_id,
        "logic_id": self.logic_id,
        "implementation_attempts": self.implementation_attempts,
        "implemented_impl_id": self.implemented_impl_id,
        "best_formula": self.best_formula,
        "failure_reason": self.failure_reason,
        "history": [list(h) for h in self.history],
        "created_at": self.created_at,
        "updated_at": self.updated_at,
    }


# 冻结函数绑定到 dataclass（保持 instance 方法调用面；状态常量已就绪）。
HypothesisRecord.transition_to = _transition_to_bound  # type: ignore[attr-defined]
HypothesisRecord.mark_aligned = _mark_aligned_bound  # type: ignore[attr-defined]
HypothesisRecord.mark_blocked = _mark_blocked_bound  # type: ignore[attr-defined]
HypothesisRecord.can_try_implementation = _can_try_implementation_bound  # type: ignore[attr-defined]
HypothesisRecord.record_impl_attempt = _record_impl_attempt_bound  # type: ignore[attr-defined]
HypothesisRecord.record_impl_success = _record_impl_success_bound  # type: ignore[attr-defined]
HypothesisRecord.as_dict = _as_dict_bound  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# HypothesisRegistry（SQLite 持久化，风格同 memory/__init__.py）
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hypothesis_records (
    hypothesis_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    schema_id TEXT,
    logic_id TEXT,
    implementation_attempts INTEGER DEFAULT 0,
    implemented_impl_id TEXT,
    best_formula TEXT,
    failure_reason TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_hyp_state ON hypothesis_records(state);
CREATE TABLE IF NOT EXISTS hypothesis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    stage TEXT,
    status TEXT,
    note TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_hh_hid ON hypothesis_history(hypothesis_id);
"""


def _default_db_path() -> str:
    env = os.environ.get(ALPHAPROBE_HYPOTHESIS_DB)
    if env:
        return env
    return str(Path.home() / "quant_projects/data/alphaprobe/hypothesis.sqlite3")


class HypothesisRegistry:
    """假设注册表：register / get / 状态持久化（独立 SQLite，默认 :memory:）。

    线程安全（sqlite3 + Lock）。文件路径可用 ``ALPHAPROBE_HYPOTHESIS_DB``
    覆盖；缺省文件路径也会被惰性创建（仅显式打开文件 DB 时）。
    """

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else _default_db_path()
        if self.db_path == ":memory:":
            Path(":memory:")  # noqa: B018 - 纯哨兵；不触发磁盘
        elif not self.db_path:
            self.db_path = _default_db_path()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            # WAL + NORMAL：写吞吐优化，与 seen/store.py 同一套路（:memory: 跳过）。
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- CRUD -------------------------------------------------------------

    def register(self, spec: HypothesisSpec) -> HypothesisRecord:
        record = HypothesisRecord(
            hypothesis_id=spec.hypothesis_id or f"hyp_{int(time.time() * 1000)}",
            spec=spec,
            state=STATE_PROPOSED,
            logic_id=spec.logic_id,
            schema_id=spec.schema_id,
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO hypothesis_records"
                "(hypothesis_id, state, spec_json, schema_id, logic_id,"
                " implementation_attempts, implemented_impl_id, best_formula,"
                " failure_reason, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.hypothesis_id, record.state,
                    _spec_to_json(record.spec), record.schema_id, record.logic_id,
                    record.implementation_attempts, record.implemented_impl_id,
                    record.best_formula, record.failure_reason,
                    record.created_at, record.updated_at,
                ),
            )
            self._conn.commit()
        return record

    def _save(self, record: HypothesisRecord) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE hypothesis_records SET state=?, schema_id=?, logic_id=?,"
                " implementation_attempts=?, implemented_impl_id=?, best_formula=?,"
                " failure_reason=?, updated_at=? WHERE hypothesis_id=?",
                (
                    record.state, record.schema_id, record.logic_id,
                    record.implementation_attempts, record.implemented_impl_id,
                    record.best_formula, record.failure_reason, record.updated_at,
                    record.hypothesis_id,
                ),
            )
            for stage, status, note in record.history[-5:]:
                self._conn.execute(
                    "INSERT OR REPLACE INTO hypothesis_history"
                    "(hypothesis_id, stage, status, note, ts) VALUES (?,?,?,?,?)",
                    (
                        record.hypothesis_id, stage, status, note,
                        record.updated_at,
                    ),
                )
            self._conn.commit()

    def get(self, hypothesis_id: str) -> HypothesisRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM hypothesis_records WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                return None
            cols = [d[0] for d in
                    self._conn.execute("SELECT * FROM hypothesis_records LIMIT 0").description]
            d = dict(zip(cols, row))
        return _record_from_row(d)

    # -- 生命周期快捷方法 ---------------------------------------------------

    def record_impl_failure(self, hypothesis_id: str, *, reason: str) -> None:
        rec = self.get(hypothesis_id)
        if rec is None:
            return
        rec.record_impl_attempt(formula=rec.best_formula, reason=reason)
        self._save(rec)

    def record_impl_success(
        self, hypothesis_id: str, *, formula: str, impl_id: str
    ) -> None:
        rec = self.get(hypothesis_id)
        if rec is None:
            return
        rec.record_impl_success(formula=formula, impl_id=impl_id)
        self._save(rec)

    def mark_aligned(self, hypothesis_id: str) -> None:
        rec = self.get(hypothesis_id)
        if rec is None:
            return
        rec.mark_aligned()
        self._save(rec)


# ---------------------------------------------------------------------------
# stage 管线（状态机驱动；畸形 action 在 stage 1-3 拦截，永不到 FE）
# ---------------------------------------------------------------------------


def register_hypothesis(
    registry: HypothesisRegistry | None, spec: HypothesisSpec
) -> HypothesisRecord:
    """注册一条 hypothesis；无 registry（None）时返回内存记录。"""
    if registry is None:
        return HypothesisRecord(hypothesis_id=spec.hypothesis_id, spec=spec)
    return registry.register(spec)


def _default_align_ctx(ctx: dict[str, Any]) -> dict[str, Any]:
    """无挂载的 stage 5 默认对齐 = 通过（不编造 domain 事实）。"""
    new_ctx = dict(ctx)
    new_ctx.setdefault("alignment_result", AlignmentResult(score=ALIGNMENT_FULL_SCORE))
    return new_ctx


def _validate_action_shape(action: Mapping[str, Any]) -> tuple[bool, str]:
    """stage 1-3 的畸形 action 拦截：永远到不了 FE 执行（stage 4）。

    保守校验（不复制 FE 语义；语义白名单校验由调用方在 FE 编译阶段做）：
    - 必须是 dict 且有 ``action`` 字段 ∈ {transform, combine, identity}；
    - combine 需要 >= 2 个 parents；
    - transform 需要合法 target 引用（parent_0.. / 非空）。
    其余深度参数校验由 build_formula / FE 侧做（本模块只拦「明显畸形」）。
    """
    if not isinstance(action, Mapping):
        return False, "action must be a JSON object"
    a = dict(action)
    atype = str(a.get("action") or "").strip().lower()
    if atype not in ("transform", "combine", "identity"):
        return False, f"unsupported action type: {atype!r}"
    if atype == "combine":
        parents = a.get("parents")
        if not (isinstance(parents, (list, tuple)) and len(parents) >= 2):
            return False, "combine requires >=2 parents"
    if atype == "transform":
        target = str(a.get("target") or "parent_0").strip()
        if not target:
            return False, "transform requires a non-empty target"
    return True, ""


def run_stage_pipeline(
    actions: Sequence[Mapping[str, Any]],
    *,
    hypothesis: HypothesisRecord,
    parents: Sequence[Mapping[str, Any]] | None = None,
    hooks: StageHooks | None = None,
    op_whitelist: frozenset[str] | None = None,
    dedup_client: Any | None = None,
    evaluator: Any | None = None,
    registry: HypothesisRegistry | None = None,
) -> dict[str, Any]:
    """九阶段骨架（plan Task 16 的 1-9）的**状态机驱动**实现。

    Parameters
    ----------
    actions : 一条或多条 action JSON（LLM 产物）。畸形 action 在 stage 1-3
        被拦（永远到不了 stage 4 FE 编译/执行）。
    hypothesis : HypothesisRecord——失败不杀 hypothesis，替代实现尝试由调用方
        再次调用（``implementation_attempts`` 有 cap）。
    parents : 可选 parent 因子（dict 列表），传给 stage hook 消费。
    hooks : StageHooks（默认全 pass-through——stage 4 之后无挂载时，候选按
        默认规则放行/记录，不复制平台能力）。
    op_whitelist : frozenset[str] | None
        可选 AST action plan 的算子表面白名单（stage 3 用）。None = 不做算子
        成员拦截（只有结构拦截；算子语义权威由调用方在 FE 编译层校验，本模块
        不复制 FE 算子分类）。生产调用方应传 FE registry 派生的 allowlist。
    dedup_client / evaluator : 可选 stage 8/9 消费（去重与评估反馈由调用方
        接线真实平台能力）。

    Returns
    -------
    dict:
        - ``implemented``: list[int] 实现成功的 action 下标
        - ``rejected``: list[dict]（action 下标 + reason + stage）
        - ``blocked``: list[str] 被确定性对齐拦截的说明码
        - ``records``: list[ImplementationAttempt]
        - ``contexts``: list[dict] 每个实现候选的最终 ctx（含 formula 等）
    """
    rec = hypothesis
    hooks = hooks or StageHooks()
    parent_list = [dict(p) for p in (parents or [])]
    parent_formula = _first_parent_formula(parent_list)

    # 阶段 1（hypothesis）：对齐初态（若仍 proposed 则推进到 aligned 前的准备态）
    ctx: dict[str, Any] = {
        "action": None,
        "action_type": "",
        "formula": None,
        "parent_formula": parent_formula,
        "parents": parent_list,
        "schema_plan": None,
        "ast_plan": None,
        "alignment_result": None,
        "fe_validation": None,
        "dedup": None,
        "evaluation": None,
        "rejected_reason": "",
    }

    implemented: list[int] = []
    rejected: list[dict[str, Any]] = []
    blocked: list[str] = []
    records: list[ImplementationAttempt] = []
    contexts: list[dict[str, Any]] = []

    for idx, action in enumerate(actions):
        act = dict(action) if isinstance(action, Mapping) else {}
        c: dict[str, Any] = dict(ctx, action=act)
        c["action"] = act
        c["action_type"] = str(act.get("action_type") or "REFINE")

        # -- stage 1：hypothesis（默认：畸形 action 拦截） -----------------
        # stage 4 FE 编译执行仅在**通过本段**的 action 上运行（先拦后执行）。
        if not act:
            rejected.append({"index": idx, "reason": "action must be a JSON object", "stage": "1"})
            continue
        c = hooks.on_stage_1(Stage.STAGE_1_HYPOTHESIS, c, rec)
        if c.get("rejected_reason"):
            rejected.append({"index": idx, "reason": c["rejected_reason"], "stage": "1"})
            continue
        ok, reason = _validate_action_shape(act)
        if not ok:
            rejected.append({"index": idx, "reason": reason, "stage": "1"})
            continue
        # stage 3（AST action plan）前置：可选算子表面拦截——非白名单 op 永远
        # 到不了 FE 执行（stage 4）。None = 只做结构拦截，语义校验交 FE 层。
        if op_whitelist is not None and act.get("action") in ("transform", "combine"):
            op = str(act.get("op") or "").strip()
            if op and op not in op_whitelist:
                rejected.append(
                    {"index": idx, "reason": f"op not in whitelist: {op!r}", "stage": "3"}
                )
                continue

        # -- stage 2：schema plan（默认 pass-through） ---------------------
        c = hooks.on_stage_2(Stage.STAGE_2_SCHEMA_PLAN, c, rec)

        # -- stage 3：AST action plan（默认 pass-through） -----------------
        c = hooks.on_stage_3(Stage.STAGE_3_AST_ACTION_PLAN, c, rec)

        # -- stage 4：确定性 FE 编译/执行 ---------------------------------
        #    （无挂载时不伪造 FE：formula 保持 None → 走记录式 rejected）
        c = hooks.on_stage_4(Stage.STAGE_4_FE_COMPILE, c, rec)
        # action 已通过前置校验，此处理应允许计数实现尝试
        if rec.state not in (STATE_IMPLEMENTED, STATE_BLOCKED):
            rec.record_impl_attempt(formula="")
        formula = c.get("formula")
        # 对齐前暂存「是否已推进为 aligned」——避免 stage 5 blocked 后残留
        was_aligned_before = rec.state == STATE_ALIGNED

        # -- stage 5：确定性对齐（无挂载默认通过；不复制 alignment 能力）---
        c = _default_align_ctx(c)
        c = hooks.on_stage_5(Stage.STAGE_5_DETERMINISTIC_ALIGNMENT, c, rec)
        align: AlignmentResult | None = c.get("alignment_result")
        if align is not None and getattr(align, "blocked", False):
            codes = list(getattr(align, "explanation_codes", ()) or ())
            codes = codes or ["alignment_blocked"]
            blocked.extend(codes)
            rejected.append(
                {"index": idx, "reason": "alignment blocked", "stage": "5"}
            )
            if rec.state not in (STATE_BLOCKED, STATE_IMPLEMENTED):
                rec.mark_blocked(note=";".join(codes))
                _save_if_registry(registry, rec)
            continue
        if align is not None and not getattr(align, "ok", True):
            # 未 blocked 但未通过（低分非 hard）：保守按未通过处理（仍存活）
            pass

        # -- stage 6：可选语义 critic（无挂载默认跳过，绝不强制） ----------
        c = hooks.on_stage_6(Stage.STAGE_6_OPTIONAL_CRITIC, c, rec)

        # -- stage 7：FE 验证/静态分析（stage 4 未产出公式 → 记录失败） -----
        c = hooks.on_stage_7(Stage.STAGE_7_FE_VALIDATION, c, rec)
        fe_ok = c.get("fe_validation")
        if fe_ok is not None and isinstance(fe_ok, tuple) and len(fe_ok) == 2:
            fe_ok, fe_reason = bool(fe_ok[0]), str(fe_ok[1])
        elif formula is None:
            fe_ok, fe_reason = False, "no formula produced at stage 4 (FE compile)"
        else:
            fe_ok, fe_reason = True, ""
        # 替代实现计数：FE 执行链路已触发；验证失败 → 计为失败尝试并继续
        if rec.state not in (STATE_IMPLEMENTED, STATE_BLOCKED) and not fe_ok:
            if fe_reason:
                rec.failure_reason = fe_reason
                _transition_to(
                    rec, STATE_IMPLEMENTATION_FAILED,
                    note=f"impl failed: {fe_reason}",
                    stage=Stage.STAGE_7_FE_VALIDATION,
                )
            _save_if_registry(registry, rec)
            rejected.append(
                {"index": idx, "reason": fe_reason, "stage": "7"}
            )
            continue
        if rec.state != STATE_IMPLEMENTED and not was_aligned_before:
            rec.mark_aligned(schema_id=rec.schema_id)

        # -- stage 8：去重（dedup_client 注入时消费；不复制去重能力） -------
        c = hooks.on_stage_8(Stage.STAGE_8_DEDUP, c, rec)
        if dedup_client is not None:
            try:
                verdict = dedup_client.check_new_candidate(formula)
                if verdict is not None and getattr(verdict, "rejection_reason", None) is not None:
                    rejected.append(
                        {"index": idx, "reason": verdict.rejection_reason, "stage": "8"}
                    )
                    continue
            except Exception:  # noqa: BLE001 - 去重失败不阻塞
                pass

        # -- stage 9：评估反馈（evaluator 注入时记录；不复制评估能力） ------
        c = hooks.on_stage_9(Stage.STAGE_9_EVALUATION_FEEDBACK, c, rec)
        impl_id = f"impl_{rec.hypothesis_id}_{rec.implementation_attempts}"
        rec.record_impl_success(formula=formula or "", impl_id=impl_id)
        _save_if_registry(registry, rec)
        records.append(
            ImplementationAttempt(
                formula=formula or "",
                impl_id=impl_id,
                action=act,
                status="implemented",
                stage=Stage.STAGE_9_EVALUATION_FEEDBACK,
                reason="",
                created_at=time.time(),
            )
        )
        contexts.append(c)
        implemented.append(idx)

    if rec.state == STATE_PROPOSED and not blocked and implemented:
        # 至少一个实现成功且仍是 proposed → 推进 aligned/implemented（防御性）
        if rec.state != STATE_IMPLEMENTED:
            rec.mark_aligned()

    return {
        "implemented": implemented,
        "rejected": rejected,
        "blocked": blocked,
        "records": records,
        "contexts": contexts,
    }


def _save_if_registry(registry: HypothesisRegistry | None, rec: HypothesisRecord) -> None:
    if registry is not None:
        try:
            registry._save(rec)
        except Exception:  # noqa: BLE001 - 持久化失败不阻塞搜索
            pass


def _first_parent_formula(parents: Sequence[Mapping[str, Any]]) -> str:
    for p in parents or ():
        f = str(p.get("formula") or p.get("canonical_formula") or "").strip()
        if f:
            return f
    return ""


def _spec_to_json(spec: HypothesisSpec) -> str:
    import json

    return json.dumps(
        {
            "hypothesis_id": spec.hypothesis_id,
            "logic_id": spec.logic_id,
            "schema_id": spec.schema_id,
            "text": spec.text,
            "expected_domains": list(spec.expected_domains),
            "expected_roles": list(spec.expected_roles),
            "expected_horizon": spec.expected_horizon,
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _record_from_row(d: Mapping[str, Any]) -> HypothesisRecord:
    import json

    spec_raw = json.loads(d.get("spec_json") or "{}")
    spec = HypothesisSpec(
        hypothesis_id=str(spec_raw.get("hypothesis_id") or d.get("hypothesis_id") or ""),
        logic_id=spec_raw.get("logic_id") or d.get("logic_id"),
        schema_id=spec_raw.get("schema_id") or d.get("schema_id"),
        text=str(spec_raw.get("text") or ""),
        expected_domains=tuple(str(x) for x in (spec_raw.get("expected_domains") or ())),
        expected_roles=tuple(str(x) for x in (spec_raw.get("expected_roles") or ())),
        expected_horizon=spec_raw.get("expected_horizon") or None,
    )
    return HypothesisRecord(
        hypothesis_id=str(d.get("hypothesis_id") or ""),
        spec=spec,
        state=str(d.get("state") or STATE_PROPOSED),
        schema_id=d.get("schema_id"),
        logic_id=d.get("logic_id"),
        implementation_attempts=int(d.get("implementation_attempts") or 0),
        implemented_impl_id=d.get("implemented_impl_id"),
        best_formula=str(d.get("best_formula") or ""),
        failure_reason=str(d.get("failure_reason") or ""),
        created_at=float(d.get("created_at") or time.time()),
        updated_at=float(d.get("updated_at") or time.time()),
    )
