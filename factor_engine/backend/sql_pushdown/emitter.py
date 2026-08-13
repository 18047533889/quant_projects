# -*- coding: utf-8 -*-
"""PlanNode → SQL 编译器（长表语义；支持 DuckDB / ClickHouse 方言）。

将逻辑计划树编译为 ``(ts, inst, _v)`` 长表 SQL，含 base CTE、子树 CSE 去重与
方言函数映射。对外入口为 ``compile_plan_to_sql`` / ``compile_plans_batch_to_sql``，
能力判断委托 ``plan_is_sql_capable``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Sequence

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode

FE_ROOT = Path(__file__).resolve().parents[2]


class CompileStatus(str, Enum):
    """R40 #216：编译结果三态——SUPPORTED / SEMANTICALLY_UNSUPPORTED /
    COMPILER_ERROR。

    旧实现 ``_compile_layer_impl`` 返回 ``_Layer | None``，None 同时表示
    「语义不支持」和「内部编译错误」。现在二者可区分：production 对
    COMPILER_ERROR hard fail（回归/内部 bug），对 SEMANTICALLY_UNSUPPORTED
    允许回退 pandas。
    """

    SUPPORTED = "supported"
    SEMANTICALLY_UNSUPPORTED = "semantically_unsupported"
    COMPILER_ERROR = "compiler_error"


class SqlCompileError(RuntimeError):
    """R40 #216：内部编译错误（区别于语义不支持）。"""


#: 最近一次编译的状态（compile_plan_to_sql / compile_plans_batch_to_sql 记录）。
_last_compile_status: CompileStatus | None = None
_compile_status_lock = threading.Lock()


def last_compile_status() -> CompileStatus | None:
    """最近一次 SQL 编译的状态（R40 #216）。"""
    with _compile_status_lock:
        return _last_compile_status


def _record_compile_status(status: CompileStatus) -> None:
    global _last_compile_status
    with _compile_status_lock:
        _last_compile_status = status


def _is_production_sql_mode() -> bool:
    try:
        from runtime.production_policy import is_production_mode

        return is_production_mode()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# R40 #214：emitter identity 从 build manifest 取（不用运行时 file inspection）。
# ---------------------------------------------------------------------------

EMITTER_IDENTITY_MANIFEST = FE_ROOT / "evidence" / "emitter_identity.json"


def _emitter_source_bytes() -> bytes:
    """emitter.py 当前源码字节（供构建期生成 manifest 用）。"""
    return Path(__file__).read_bytes()


def generate_emitter_identity_manifest(
    *, out_path: str | Path | None = None,
) -> dict[str, Any]:
    """构建期：把 emitter.py 的 source hash + build commit 固化进 manifest。

    运行期绝不读源码算 hash（那会受工作区脏改动影响且无法区分「真缺失」）。
    返回 manifest dict 并落盘。
    """
    from backend.evidence_provenance import compute_implementation_hash

    out = Path(out_path) if out_path is not None else EMITTER_IDENTITY_MANIFEST
    build_commit = ""
    try:
        import subprocess

        build_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(FE_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        build_commit = ""
    manifest = {
        "schema_version": 1,
        "emitter_source_hash": compute_implementation_hash(_emitter_source_bytes().decode("utf-8")),
        "build_commit_sha": build_commit,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class EmitterIdentity:
    """emitter identity（R40 #214）。``known=False`` 表示无法从 build manifest
    确定 emitter source hash——production SQL capability 必须 hard fail。"""

    known: bool
    source_hash: str = ""
    build_commit_sha: str = ""


def emitter_identity(manifest_path: str | Path | None = None) -> EmitterIdentity:
    """从 build manifest 读 emitter identity（不做运行时 file inspection）。

    manifest 缺失 / 损坏 / 字段缺失 → ``known=False``（与「真缺失」无法区分的
    情况一律视为未知——这是 #214 的核心：绝不把 unknown 缓存成可用身份）。
    ``FACTOR_ENGINE_EMITTER_IDENTITY_MANIFEST`` 可覆盖 manifest 路径（测试/部署）。
    """
    if manifest_path is None:
        override = __import__("os").environ.get("FACTOR_ENGINE_EMITTER_IDENTITY_MANIFEST", "")
        if override:
            manifest_path = override
    p = Path(manifest_path) if manifest_path is not None else EMITTER_IDENTITY_MANIFEST
    if not p.is_file():
        return EmitterIdentity(known=False)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return EmitterIdentity(known=False)
    if not isinstance(data, dict):
        return EmitterIdentity(known=False)
    source_hash = str(data.get("emitter_source_hash") or "")
    if not source_hash:
        return EmitterIdentity(known=False)
    return EmitterIdentity(
        known=True,
        source_hash=source_hash,
        build_commit_sha=str(data.get("build_commit_sha") or ""),
    )


def assert_emitter_identity_known(
    *, production: bool | None = None, manifest_path: str | Path | None = None,
) -> str:
    """R40 #214：production SQL capability 要求 emitter identity 已知。

    ``production`` 缺省按 run_mode 判定；production + unknown → raise
    ``SqlCompileError``（hard fail，绝不静默以 unknown 身份继续 SQL 下推）。
    返回已知的 emitter source hash。
    """
    if production is None:
        production = _is_production_sql_mode()
    ident = emitter_identity(manifest_path=manifest_path)
    if production and not ident.known:
        raise SqlCompileError(
            "emitter identity unknown（build manifest 缺失/损坏）：production SQL "
            "下推必须绑定已知 emitter source hash（R40 #214）。请运行 "
            "generate_emitter_identity_manifest() 生成 evidence/emitter_identity.json。"
        )
    return ident.source_hash


# ---------------------------------------------------------------------------
# R40 #65/#217：编译模板缓存（模板 = 计划形状；literal binding 分离）。
# ---------------------------------------------------------------------------


def _plan_shape_key(plan: PlanNode, dialect: SqlDialect) -> str:
    """计划「模板」key：结构 + 算子参数名/类型，**literal 值被类型化抽象**。

    两个计划仅 literal 值不同（如 ``ts_mean(w=5)`` 与 ``ts_mean(w=10)``）→
    同一 template key（结构形状相同）。这使缓存按「模板」而非「每个 literal
    组合」组织——不同 binding 不导致缓存条目数随 literal 值爆炸。
    """
    import json as _json

    def rec(n: PlanNode) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for k, v in sorted((n.attrs or {}).items()):
            if n.op == "literal" and k == "value":
                attrs[k] = f"<{type(v).__name__}>"  # 抽象 literal 值，保留类型
            else:
                try:
                    _json.dumps(v)
                    attrs[k] = v
                except (TypeError, ValueError):
                    attrs[k] = repr(v)
        return {"op": n.op, "a": attrs, "in": [rec(c) for c in n.inputs]}

    payload = _json.dumps(
        rec(plan), sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(
        f"{payload}|{getattr(dialect, 'value', dialect)}".encode("utf-8")
    ).hexdigest()[:20]


def _literal_binding_key(plan: PlanNode) -> str:
    """计划里全部 literal 值的有序摘要（binding 指纹）。"""
    vals: list[Any] = []

    def walk(n: PlanNode) -> None:
        if n.op == "literal" and (n.attrs or {}).get("value") is not None:
            vals.append(n.attrs["value"])
        for c in n.inputs:
            walk(c)

    walk(plan)
    payload = json.dumps(vals, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class SqlTemplate:
    """R40 #65/#217：已编译模板（按 plan 形状缓存）+ literal binding。

    - ``shape_key``   模板 key（结构，literal 值类型化抽象）；
    - ``binding_key`` 首次编译时使用的 literal 值指纹；
    - ``compiled``    该 binding 下的完整 ``CompiledSql``。
    """

    shape_key: str
    binding_key: str
    compiled: "CompiledSql"

    def bind(self, plan: PlanNode, **kw: Any) -> "CompiledSql | None":
        """把 literal binding 绑定回模板。

        binding 一致 → 直接复用缓存编译结果（零重编译）；binding 不一致 →
        诚实重编译（结构参数变化时 SQL 形状可能不同，绝不硬凑复用）。
        """
        from backend.sql_pushdown.emitter import compile_plan_to_sql

        if _literal_binding_key(plan) == self.binding_key:
            return self.compiled
        return compile_plan_to_sql(
            plan,
            dataset=kw.get("dataset"),
            table=kw.get("table"),
            time_column=kw["time_column"],
            instrument_column=kw["instrument_column"],
            filt=kw.get("filt"),
            dialect=kw.get("dialect", SqlDialect.DUCKDB),
        )


class _SqlTemplateCache:
    """R40 #65/#217：有界模板缓存（按 shape key；每个 shape 保留一个 binding）。

    与「per-literal 缓存」的区别：缓存条目数由**不同结构形状**决定，不随
    literal 值组合增长。超过 max_entries / max_bytes 按 LRU 逐出。
    """

    def __init__(self, *, max_entries: int = 64, max_bytes: int = 2 * 1024 * 1024) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._data: "OrderedDict[str, SqlTemplate]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, shape_key: str, binding_key: str) -> "CompiledSql | None":
        with self._lock:
            tpl = self._data.get(shape_key)
            if tpl is None or tpl.binding_key != binding_key:
                return None
            self._data.move_to_end(shape_key)
            return tpl.compiled

    def put(self, shape_key: str, binding_key: str, compiled: "CompiledSql") -> None:
        with self._lock:
            self._data[shape_key] = SqlTemplate(shape_key, binding_key, compiled)
            self._data.move_to_end(shape_key)
            size = sum(
                len(str(k)) + len(str(v.compiled.query)) + len(v.binding_key)
                for k, v in self._data.items()
            )
            while (len(self._data) > self._max_entries or size > self._max_bytes) \
                    and len(self._data) > 1:
                k, v = self._data.popitem(last=False)
                size -= len(str(k)) + len(str(v.compiled.query)) + len(v.binding_key)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._data),
                "max_entries": self._max_entries,
                "max_bytes": self._max_bytes,
                "shapes": list(self._data.keys()),
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


#: 进程级模板缓存（#65/#217）。
_SQL_TEMPLATE_CACHE: _SqlTemplateCache = _SqlTemplateCache()


def reset_sql_template_cache() -> None:
    """测试用：清空模板缓存。"""
    _SQL_TEMPLATE_CACHE.clear()


class SqlDialect(str, Enum):
    """SQL 下推目标方言：DuckDB 数据集或 ClickHouse 物理表。"""
    DUCKDB = "duckdb"
    CLICKHOUSE = "clickhouse"


# 可通过 SQL 下推的 canonical（与 sql_registry 同步）
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS as SQL_CAPABLE_OPS  # noqa: E402


class InstrumentFilterKind(str, Enum):
    """Instrument filter cardinality (R13 P0-69).

    ``None`` (ALL) and ``[]`` (EMPTY) must stay distinct: an explicit EMPTY
    universe means *zero rows* (``WHERE FALSE``), not "no restriction".
    """

    ALL = "all"      # no instrument restriction
    EMPTY = "empty"  # restriction to an empty set -> 0 rows
    LIST = "list"    # restriction to an explicit instrument list


@dataclass(frozen=True)
class SqlPushdownFilter:
    """下推至 base CTE 的过滤条件（与 DataAccessSource 对齐）。"""

    time_column: str | None = None
    start: Any = None
    end: Any = None
    instrument_column: str | None = None
    instruments: tuple[str, ...] = ()
    #: R13 P0-69: distinguishes ``instrument_filter=None`` (ALL) from ``[]``
    #: (EMPTY).  ``ALL`` emits no instrument clause; ``EMPTY`` emits ``WHERE
    #: FALSE`` (0 rows); ``LIST`` emits ``inst IN (...)``.
    instrument_filter_kind: InstrumentFilterKind = InstrumentFilterKind.ALL


@dataclass(frozen=True)
class CompiledSql:
    """单因子 SQL 编译结果：查询文本、引用列集与数据源定位信息。"""
    query: str
    read_datasets: tuple[str, ...]
    referenced_columns: frozenset[str]
    dialect: SqlDialect = SqlDialect.DUCKDB
    table: str | None = None  # ClickHouse 直读表名


def _resolve_canonical(op: str) -> str:
    """将算子别名解析为 canonical 名称。"""
    resolved = OperatorRegistry._aliases.get(op, op)
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    if op in SQL_IMPLEMENTED_CANONICALS and resolved not in SQL_IMPLEMENTED_CANONICALS:
        return op
    return resolved


def _window_int(node: PlanNode, default: int = 3) -> int:
    """从 attrs 或子 literal 输入解析窗口长度（与 PolarsLong 共用）。"""
    return _window_spec(node, default=default).size


def _window_spec(node: PlanNode, default: int = 3):
    """完整 WindowSpec。"""
    from backend.plan_params import PlanParamError, window_spec_from_plan_node

    spec = window_spec_from_plan_node(node, default=default)
    if spec.closed != "right":
        raise PlanParamError(f"closed={spec.closed!r} 暂未支持，仅 right")
    # #360：nan_policy 决定 NaN/NULL 在窗口聚合内的参与方式。SQL 层目前仅
    # 生成 RESPECT NULLS（propagate）语义；ignore 需要 IGNORE NULLS，尚未实现。
    if spec.nan_policy != "propagate":
        raise PlanParamError(
            f"window.nan_policy={spec.nan_policy!r} 需要 "
            f"IGNORE NULLS 语义，SQL emitter 尚未生成（仅 propagate/RESPECT NULLS）"
        )
    return spec


def _float_attr(node: PlanNode, *keys: str, default: float) -> float:
    """从 attrs 中按候选键读取有限浮点参数。

    #356：参数缺失（不在 attrs）→ 返回 ``default``；参数提供但非法
    （非有限 / 越界 / 非数值）→ raise ``PlanParamError``，不再静默回落。
    注意 ``lo``/``hi`` 在 clip 语境是任意实数界（非 [0,1] 概率），故只走
    ``parse_finite_float``，不按 unit-interval 校验。
    """
    from backend.plan_params import parse_finite_float, parse_unit_interval

    for key in keys:
        if key in (node.attrs or {}) and node.attrs[key] is not None:
            raw = node.attrs[key]
            if key in {"a", "p", "q", "min_pct", "max_pct", "lower", "upper"}:
                return parse_unit_interval(raw, label=key)
            need_pos = key in {"epsilon", "eps", "to", "ann_factor"}
            return parse_finite_float(raw, label=key, gt=0.0 if need_pos else None)
    return default


def _int_attr(node: PlanNode, *keys: str, default: int) -> int:
    """从 attrs 中按候选键读取整数参数（严格解析，无静默截断/钳制）。

    #357：整数直接用；非整数（float 5.9 / 字符串）→ raise ``PlanParamError``。
    需要的钳制（如 ``max(1, ...)``）由调用点显式完成——emitter 参数已
    canonicalize，非整数出现说明上游 bug，应 fail。
    """
    from backend.plan_params import parse_positive_int_literal

    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return parse_positive_int_literal(node.attrs[key], label=key)
    return default


def _literal_positional(node: PlanNode, index: int, *, default: float | None = None) -> float | None:
    """读取 positional literal 参数（``clip(x, lo, hi)`` 等）。"""
    pos = index + 1
    if pos >= len(node.inputs):
        return default
    child = node.inputs[pos]
    if child.op != "literal":
        return default
    val = child.attrs.get("value")
    if val is None:
        return default
    if val == "zero":
        return 0.0
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return default


def _clip_bounds(node: PlanNode, *, default_lo: float = -3.0, default_hi: float = 3.0) -> tuple[float, float]:
    """解析 clip 算子的上下界（attrs 与 positional literal 合并）。"""
    lo = _float_attr(node, "lo", "min", default=default_lo)
    hi = _float_attr(node, "max", "hi", default=default_hi)
    pos_lo = _literal_positional(node, 0)
    pos_hi = _literal_positional(node, 1)
    if pos_lo is not None:
        lo = pos_lo
    if pos_hi is not None:
        hi = pos_hi
    return lo, hi


def _truthy_sql(value_col: str, *, dialect: SqlDialect | None = None) -> str:
    """浮点条件真值：NULL/NaN/0 → false。"""
    from backend.logical_semantics import truthy_sql

    is_ch = dialect == SqlDialect.CLICKHOUSE if dialect is not None else False
    return truthy_sql(value_col, dialect_is_clickhouse=is_ch)


def _const_fill_value(node: PlanNode, *, default: float | None = None) -> float | None:
    """解析常量填充值（``fillna_const`` / ``nan_to_num`` / ``fillna(..., 0)``）。"""
    if "num" in node.attrs and node.attrs["num"] is not None:
        return float(node.attrs["num"])
    for key in ("value", "fill_value", "const", "c", "method"):
        if key in node.attrs and node.attrs[key] is not None:
            raw = node.attrs[key]
            if raw == "zero":
                return 0.0
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
    pos = _literal_positional(node, 0)
    if pos is not None:
        return pos
    if len(node.inputs) >= 2 and node.inputs[1].op == "literal":
        raw = node.inputs[1].attrs.get("value")
        if raw == "zero":
            return 0.0
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return default


#: SQL identifier validation pattern (alphanumeric, underscore, non-leading digit).
_SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_sql_identifier(name: str, *, context: str = "identifier") -> None:
    """Validate SQL identifier is injection-safe (fail-closed).

    Raises ValueError if identifier contains special chars that could enable
    injection (spaces, semicolons, quotes, braces, slashes, etc.).

    Args:
        name: Identifier to validate (dataset/table/column name).
        context: Human-readable context for error message.

    Raises:
        ValueError: If identifier is unsafe or exceeds length limit.
    """
    if not name:
        raise ValueError(f"SQL {context} cannot be empty")
    if len(name) > 128:
        raise ValueError(
            f"SQL {context} too long (max 128 chars): {name!r}"
        )
    if not _SAFE_IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"SQL {context} contains unsafe characters (only alphanumeric and "
            f"underscore allowed, no leading digit): {name!r}"
        )


def _quote_ident(name: str) -> str:
    """双引号转义 SQL 标识符。"""
    safe = name.replace('"', '""')
    return f'"{safe}"'


def _sql_literal(val: Any) -> str:
    """将 Python 值转为 SQL 字面量字符串。"""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, float):
        # #358：非有限浮点不能进 SQL 字面量。
        if not math.isfinite(val):
            raise ValueError("non-finite literal cannot be emitted to SQL")
        return str(val)
    if isinstance(val, int):
        return str(val)
    s = str(val).replace("'", "''")
    return f"'{s}'"


def _quantile_over(
    dialect: SqlDialect,
    col: str,
    p: float,
    partition: str,
) -> str:
    """窗口分位数：DuckDB ``quantile_cont`` / ClickHouse ``quantileExact``。"""
    if dialect == SqlDialect.CLICKHOUSE:
        return f"quantileExact({p})({col}) OVER ({partition})"
    return f"quantile_cont({col}, {p}) OVER ({partition})"


def _dialect_fn(dialect: SqlDialect, name: str) -> str:
    """将逻辑函数名映射为方言特定的 SQL 函数名。"""
    if dialect == SqlDialect.CLICKHOUSE:
        mapping = {
            "stddev": "stddevSamp",
            "stddev_pop": "stddevPop",
            "ln": "log",
            "greatest": "greatest",
            "least": "least",
            "nullif": "nullIf",
            "abs": "abs",
            "sign": "sign",
            "exp": "exp",
            "sqrt": "sqrt",
        }
        return mapping.get(name, name)
    mapping = {
        "stddev": "STDDEV_SAMP",
        "stddev_pop": "STDDEV_POP",
        "ln": "ln",
        "greatest": "GREATEST",
        "least": "LEAST",
        "nullif": "NULLIF",
        "abs": "abs",
        "sign": "sign",
        "exp": "exp",
        "sqrt": "sqrt",
    }
    return mapping.get(name, name)


@dataclass
class _Layer:
    """编译中间层：子查询 SQL 及窗口/partition 语义标记。"""
    sql: str
    has_inst_window: bool = False
    has_ts_partition: bool = False


@dataclass
class _SqlCompileMemo:
    """DAG-level CTE 去重：同一 structural_key 只编译一次。"""

    use_counts: dict[str, int]
    refs: dict[str, str] = field(default_factory=dict)
    bodies: list[tuple[str, str]] = field(default_factory=list)
    _counter: int = 0


_sql_memo_ctx: ContextVar[_SqlCompileMemo | None] = ContextVar("_sql_memo_ctx", default=None)


def _structural_use_counts(plan: PlanNode) -> dict[str, int]:
    """统计计划树中各 structural_key 的出现次数（用于 CTE 去重）。"""
    from planner.plan_hash import structural_key

    counts: dict[str, int] = {}

    def walk(n: PlanNode) -> None:
        key = structural_key(n)
        counts[key] = counts.get(key, 0) + 1
        for child in n.inputs:
            walk(child)

    walk(plan)
    return counts


def _inst_window(
    dialect: SqlDialect,
    w: int,
    agg: str,
    inner_sql: str,
    *,
    min_periods: int = 1,
) -> str:
    """按 instrument 分区的固定长度滚动窗口聚合 SQL（含 min_periods 门槛）。

    ±Inf is dropped to NULL before aggregation (pandas rolling machinery treats
    Inf as fully missing — excluded from BOTH the aggregate and the min_periods
    count), so a windowed aggregate over an Inf row matches the pandas reference
    and the polars long path.
    """
    isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
    isinf_fn = "isInfinite" if dialect == SqlDialect.CLICKHOUSE else "isinf"
    safe = f"CASE WHEN _v IS NOT NULL AND NOT {isnan_fn}(_v) AND NOT {isinf_fn}(_v) THEN _v END"
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    cnt = f"COUNT({safe}) OVER ({over})"
    rolled = f"{agg}({safe}) OVER ({over})"
    if min_periods <= 1:
        body = rolled
    else:
        body = f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {rolled} END"
    return f"SELECT ts, inst, {body} AS _v FROM ({inner_sql}) t"


def _ewm_adjust_false_sql(
    inner_sql: str,
    w: int,
    alpha: float,
    *,
    dialect: SqlDialect,
    value_col: str = "_v",
    min_periods: int | None = None,
) -> str:
    """``adjust=False`` EWM，精确匹配 pandas（DuckDB 可执行）。

    DuckDB 禁止嵌套窗口函数；改用有限窗口自连接加权和（
    ``alpha*SUM(x_i * decay^{t-i}) + (1-alpha)*decay^t * x0``），
    在 ``decay^W`` 可忽略的窗口宽度内收敛到 pandas 递归定义（1e-15）。
    ClickHouse 走原生 ``exponentialMovingAverage``。
    """
    if min_periods is None:
        min_periods = w
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"SELECT ts, inst, "
            f"exponentialMovingAverage({value_col}, {alpha}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner_sql}) t"
        )
    decay = 1.0 - alpha
    # 窗口宽度取 max(w, 32)，并对大 span 扩张到衰减到 ~1e-15 的宽度。
    _W = max(int(w), 32)
    while float(decay) ** _W > 1e-15 and _W < 1024:
        _W *= 2
    _W = max(_W, 256)  # 即便 span 很小，保持数值安全裕度
    # pandas adjust=False 以第一个非空值作为递归种子（领先 NaN 被跳过），
    # y_t = alpha*SUM_{i=k..t} decay^{t-i} x_i + decay^{t-k} x_k。
    # 加权和 ws 中 i=k 项已带完整权重 decay^{t-k}，故补 (1-alpha)*decay^{t-k} x_k。
    return (
        f"SELECT p.ts, p.inst, "
        f"CASE WHEN p.cnt < {min_periods} THEN NULL "
        f"WHEN p.pos = p.kpos THEN p.x0 "
        f"ELSE {alpha!r} * j.ws + {1.0 - alpha!r} * POW({decay}, p.pos - p.kpos) * p.x0 END AS _v "
        f"FROM ("
        f"SELECT ts, inst, pos, x0, kpos, "
        f"COUNT(_v) OVER (PARTITION BY inst ORDER BY ts "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cnt "
        f"FROM ("
        f"SELECT ts, inst, _v, pos, "
        f"FIRST_VALUE(_v IGNORE NULLS) OVER (PARTITION BY inst ORDER BY ts) AS x0, "
        f"MIN(CASE WHEN _v IS NOT NULL THEN pos END) OVER (PARTITION BY inst) AS kpos "
        f"FROM ("
        f"SELECT ts, inst, {value_col} AS _v, "
        f"ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - 1 AS pos "
        f"FROM ({inner_sql}) _e0"
        f") _e1"
        f") _e2"
        f") p "
        f"JOIN ("
        f"SELECT a.ts, a.inst, "
        f"SUM(b._v * POW({decay}, a.pos - b.pos)) AS ws "
        f"FROM ("
        f"SELECT ts, inst, {value_col} AS _v, "
        f"ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - 1 AS pos "
        f"FROM ({inner_sql}) _j0"
        f") a "
        f"JOIN ("
        f"SELECT ts, inst, {value_col} AS _v, "
        f"ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - 1 AS pos "
        f"FROM ({inner_sql}) _j1"
        f") b ON a.inst = b.inst AND b.pos BETWEEN a.pos - {_W - 1} AND a.pos "
        f"GROUP BY a.ts, a.inst"
        f") j ON p.ts = j.ts AND p.inst = j.inst"
    )


def _ema_span_over_inst(
    inner_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
    value_col: str = "_v",
    min_periods: int | None = None,
) -> str:
    """Span-based EWM（alpha=2/(w+1)），与 pandas ``ewm(span=w, adjust=False)`` 对齐。

    pandas 有两类 span EWM 参考：
    - MACD/ts_ema（signal.py / time_series.py）用 ``ewm(span=.., adjust=False)``，
      无 ``min_periods``，从第一个非空值即输出 → ``min_periods=1``。
    - 参数化指标（indicators_v2 ``_ema``）用 ``min_periods=span``，warmup 前为 NaN。
    """
    alpha = 2.0 / (float(max(w, 1)) + 1.0)
    if min_periods is None:
        min_periods = 1
    return _ewm_adjust_false_sql(
        inner_sql, w, alpha, dialect=dialect, value_col=value_col, min_periods=min_periods
    )


def _wilder_ewm_over_inst(
    inner_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
    value_col: str = "_v",
    min_periods: int | None = None,
) -> str:
    """Wilder 平滑（alpha=1/w），与 pandas ``ewm(alpha=1/w, adjust=False)`` 对齐。"""
    alpha = 1.0 / float(max(w, 1))
    return _ewm_adjust_false_sql(
        inner_sql, w, alpha, dialect=dialect, value_col=value_col, min_periods=min_periods
    )


def _rsi_wilder_sql(close_sql: str, w: int, *, dialect: SqlDialect) -> str:
    """Wilder RSI 指标 SQL（对齐 pandas ``RSI_WILDER``）。"""
    g = _dialect_fn(dialect, "greatest")
    abs_fn = _dialect_fn(dialect, "abs")
    delta = f"(_v - LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts))"
    if dialect == SqlDialect.CLICKHOUSE:
        gain_expr = f"if(isNull({delta}), NULL, {g}(0, {delta}))"
        loss_expr = f"if(isNull({delta}), NULL, {g}(0, -({delta})))"
    else:
        gain_expr = f"CASE WHEN {delta} IS NULL THEN NULL ELSE {g}(0, {delta}) END"
        loss_expr = f"CASE WHEN {delta} IS NULL THEN NULL ELSE {g}(0, -({delta})) END"
    gain_sql = f"SELECT ts, inst, {gain_expr} AS _v FROM ({close_sql}) c0"
    loss_sql = f"SELECT ts, inst, {loss_expr} AS _v FROM ({close_sql}) c1"
    avg_gain = _wilder_ewm_over_inst(gain_sql, w, dialect=dialect)
    avg_loss = _wilder_ewm_over_inst(loss_sql, w, dialect=dialect)
    if dialect == SqlDialect.CLICKHOUSE:
        rs = "g.gain / nullIf(l.loss, 0)"
        core = f"100 - (100 / (1 + {rs}))"
        return (
            f"SELECT c.ts, c.inst, "
            f"if(l.loss = 0 AND g.gain > 0, 100, "
            f"if(g.gain = 0 AND l.loss > 0, 0, "
            f"if(g.gain = 0 AND l.loss = 0, 50, {core}))) AS _v "
            f"FROM ({close_sql}) c "
            f"INNER JOIN (SELECT ts, inst, _v AS gain FROM ({avg_gain}) g0) g USING (ts, inst) "
            f"INNER JOIN (SELECT ts, inst, _v AS loss FROM ({avg_loss}) l0) l USING (ts, inst)"
        )
    return (
        f"SELECT c.ts, c.inst, "
        f"CASE "
        f"WHEN l.loss = 0 AND g.gain > 0 THEN 100 "
        f"WHEN g.gain = 0 AND l.loss > 0 THEN 0 "
        f"WHEN g.gain = 0 AND l.loss = 0 THEN 50 "
        f"ELSE 100 - (100 / (1 + g.gain / NULLIF(l.loss, 0))) END AS _v "
        f"FROM ({close_sql}) c "
        f"INNER JOIN (SELECT ts, inst, _v AS gain FROM ({avg_gain}) g0) g USING (ts, inst) "
        f"INNER JOIN (SELECT ts, inst, _v AS loss FROM ({avg_loss}) l0) l USING (ts, inst)"
    )


def _atr_wilder_sql(
    high_sql: str,
    low_sql: str,
    close_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
) -> str:
    """Wilder ATR 指标 SQL（对齐 pandas ``ATR_WILDER``）。"""
    abs_fn = _dialect_fn(dialect, "abs")
    g = _dialect_fn(dialect, "greatest")
    lag_close = f"LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
    if dialect == SqlDialect.CLICKHOUSE:
        tr_expr = (
            f"if(isNull({lag_close}), NULL, "
            f"{g}(h._v - l._v, {abs_fn}(h._v - {lag_close}), {abs_fn}(l._v - {lag_close})))"
        )
    else:
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {g}(h._v - l._v, {abs_fn}(h._v - {lag_close}), {abs_fn}(l._v - {lag_close})) END"
        )
    tr_sql = (
        f"SELECT h.ts, h.inst, {tr_expr} AS _v "
        f"FROM ({high_sql}) h "
        f"INNER JOIN ({low_sql}) l USING (ts, inst) "
        f"INNER JOIN ({close_sql}) c USING (ts, inst)"
    )
    return _wilder_ewm_over_inst(tr_sql, w, dialect=dialect)


def _ffill_over_inst(inner_sql: str, *, dialect: SqlDialect) -> str:
    """按 instrument 前向填充 NULL（``ffill`` 语义）。"""
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"SELECT ts, inst, "
            f"anyLast(if(isNotNull(_v), _v, NULL)) OVER ("
            f"PARTITION BY inst ORDER BY ts "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v "
            f"FROM ({inner_sql}) t"
        )
    return (
        f"SELECT ts, inst, "
        f"LAST_VALUE(_v IGNORE NULLS) OVER ("
        f"PARTITION BY inst ORDER BY ts "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v "
        f"FROM ({inner_sql}) t"
    )


def _average_rank_frac_correlated(
    *,
    row_value_col: str,
    partition_keys: list[str],
    numbered_sql: str,
    row_alias: str,
    dialect: SqlDialect,
    exclude_nan: bool = True,
    descending: bool = False,
) -> str:
    """pandas ``rank(method='average', pct=True)`` 相关子查询表达式。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    match = " AND ".join(f"p.{k} = {row_alias}.{k}" for k in partition_keys)
    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=exclude_nan)
    invalid_row = row_stat_invalid_sql(row_value_col, dialect=dialect, exclude_nan=exclude_nan)
    rank_cmp = ">=" if descending else "<="
    cnt_subq = (
        f"(SELECT COUNT(*) FILTER (WHERE {valid}) "
        f"FROM ({numbered_sql}) p WHERE {match})"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        valid_ch = valid.replace("p._v", "p._v")
        return (
            f"multiIf("
            f"{invalid_row}, NULL, "
            f"{cnt_subq} = 0, NULL, "
            f"("
            f"SELECT if(s.cnt = 0, NULL, (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt) "
            f"FROM ("
            f"SELECT "
            f"countIf({valid_ch}) AS cnt, "
            f"countIf({valid_ch} AND p._v {rank_cmp} {row_value_col}) AS cnt_le, "
            f"countIf({valid_ch} AND p._v = {row_value_col}) AS cnt_eq "
            f"FROM ({numbered_sql}) p WHERE {match}"
            f") s"
            f")"
            f")"
        )
    return (
        f"CASE WHEN {invalid_row} THEN NULL "
        f"WHEN {cnt_subq} = 0 THEN NULL "
        f"ELSE ("
        f"SELECT CASE WHEN s.cnt = 0 THEN NULL "
        f"ELSE (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt END "
        f"FROM ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v {rank_cmp} {row_value_col}) AS cnt_le, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v = {row_value_col}) AS cnt_eq "
        f"FROM ({numbered_sql}) p WHERE {match}"
        f") s"
        f") END"
    )


def _average_rank_01_correlated(
    *,
    row_value_col: str,
    partition_keys: list[str],
    numbered_sql: str,
    row_alias: str,
    dialect: SqlDialect,
    exclude_nan: bool = True,
) -> str:
    """截面 0-1 average rank，对齐 ``cs_rank_01``。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    match = " AND ".join(f"p.{k} = {row_alias}.{k}" for k in partition_keys)
    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=exclude_nan)
    invalid_row = row_stat_invalid_sql(row_value_col, dialect=dialect, exclude_nan=exclude_nan)
    cnt_subq = (
        f"(SELECT COUNT(*) FILTER (WHERE {valid}) "
        f"FROM ({numbered_sql}) p WHERE {match})"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        valid_ch = valid
        return (
            f"multiIf("
            f"{invalid_row}, NULL, "
            f"{cnt_subq} = 0, NULL, "
            f"{cnt_subq} = 1, 0.5, "
            f"("
            f"SELECT (s.cnt_le - (s.cnt_eq - 1) / 2.0 - 1) / nullIf(s.cnt - 1, 0) "
            f"FROM ("
            f"SELECT "
            f"countIf({valid_ch}) AS cnt, "
            f"countIf({valid_ch} AND p._v <= {row_value_col}) AS cnt_le, "
            f"countIf({valid_ch} AND p._v = {row_value_col}) AS cnt_eq "
            f"FROM ({numbered_sql}) p WHERE {match}"
            f") s"
            f")"
            f")"
        )
    return (
        f"CASE WHEN {invalid_row} THEN NULL "
        f"WHEN {cnt_subq} = 0 THEN NULL "
        f"WHEN {cnt_subq} = 1 THEN 0.5 "
        f"ELSE ("
        f"SELECT (s.cnt_le - (s.cnt_eq - 1) / 2.0 - 1.0) / NULLIF(s.cnt - 1, 0) "
        f"FROM ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v <= {row_value_col}) AS cnt_le, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v = {row_value_col}) AS cnt_eq "
        f"FROM ({numbered_sql}) p WHERE {match}"
        f") s"
        f") END"
    )


def _cs_average_rank_pct_sql(
    inner_sql: str,
    *,
    partition_keys: list[str],
    value_col: str = "_v",
    select_out: str = "ts, inst",
    dialect: SqlDialect,
) -> str:
    """截面/分组 average-rank 百分位 SQL。"""
    keys_csv = ", ".join(partition_keys)
    numbered = (
        f"SELECT {select_out}, {value_col} AS _v"
        + (f", {keys_csv}" if keys_csv else "")
        + f" FROM ({inner_sql}) t0"
    )
    frac = _average_rank_frac_correlated(
        row_value_col="b._v",
        partition_keys=partition_keys,
        numbered_sql=numbered,
        row_alias="b",
        dialect=dialect,
    )
    return f"SELECT {select_out}, {frac} AS _v FROM ({numbered}) b"


def _cs_average_rank_01_sql(
    inner_sql: str,
    *,
    partition_keys: list[str],
    value_col: str = "_v",
    select_out: str = "ts, inst",
    dialect: SqlDialect,
) -> str:
    """截面/分组 0-1 average rank SQL。"""
    keys_csv = ", ".join(partition_keys)
    numbered = (
        f"SELECT {select_out}, {value_col} AS _v"
        + (f", {keys_csv}" if keys_csv else "")
        + f" FROM ({inner_sql}) t0"
    )
    expr = _average_rank_01_correlated(
        row_value_col="b._v",
        partition_keys=partition_keys,
        numbered_sql=numbered,
        row_alias="b",
        dialect=dialect,
    )
    return f"SELECT {select_out}, {expr} AS _v FROM ({numbered}) b"


def _group_partition_wrap(
    inner_sql: str,
    grp_sql: str | None,
    *,
    value_alias: str = "_v",
) -> tuple[str, list[str]]:
    """分组 join 包装：返回 (wrapped_sql, partition_keys)。"""
    if grp_sql is not None:
        wrapped = (
            f"SELECT x.ts, x.inst, x._v AS {value_alias}, g._v AS _grp "
            f"FROM ({inner_sql}) x LEFT JOIN ({grp_sql}) g USING (ts, inst)"
        )
        return wrapped, ["ts", "_grp"]
    wrapped = f"SELECT ts, inst, _v AS {value_alias}, 1.0 AS _grp FROM ({inner_sql}) x"
    return wrapped, ["ts", "_grp"]


def _cs_rank_01_sql(inner_sql: str, *, partition: str, dialect: SqlDialect) -> str:
    """截面 0-1 排名 SQL 包装（average rank）。"""
    keys = [c.strip().split(".")[-1] for c in partition.replace("PARTITION BY", "").split(",") if c.strip()]
    return _cs_average_rank_01_sql(
        inner_sql,
        partition_keys=keys,
        dialect=dialect,
    )


def _cs_pct_rank_sql(inner_sql: str, *, partition: str, dialect: SqlDialect) -> str:
    """截面百分位 rank；NaN 保持 NULL（``rank_pct`` / ``cs_pct_rank`` 语义）。"""
    keys = [c.strip().split(".")[-1] for c in partition.replace("PARTITION BY", "").split(",") if c.strip()]
    return _cs_average_rank_pct_sql(
        inner_sql,
        partition_keys=keys,
        dialect=dialect,
    )


def _ts_pct_rank_sql(
    inner_sql: str,
    *,
    window: int,
    dialect: SqlDialect,
    min_periods: int = 1,
    exclude_nan: bool = True,
) -> str:
    """滚动百分位 rank，对齐 ``pandas rolling.rank(pct=True, method='average')``。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    w = max(int(window), 1)
    mp = max(int(min_periods), 1)
    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=exclude_nan)
    invalid_row = row_stat_invalid_sql("b._v", dialect=dialect, exclude_nan=exclude_nan)
    numbered = (
        f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn "
        f"FROM ({inner_sql}) t0"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        valid_ch = valid
        rank_expr = (
            f"multiIf("
            f"{invalid_row}, NULL, "
            f"("
            f"SELECT multiIf("
            f"s.cnt < {mp}, NULL, "
            f"s.cnt = 0, NULL, "
            f"s.cnt <= 1, 1.0, "
            f"(s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt"
            f") "
            f"FROM ("
            f"SELECT "
            f"countIf({valid_ch}) AS cnt, "
            f"countIf({valid_ch} AND p._v <= b._v) AS cnt_le, "
            f"countIf({valid_ch} AND p._v = b._v) AS cnt_eq "
            f"FROM ({numbered}) p "
            f"WHERE p.inst = b.inst AND p.rn BETWEEN b.rn - {w - 1} AND b.rn"
            f") s"
            f")"
            f")"
        )
    else:
        rank_expr = (
            f"CASE WHEN {invalid_row} THEN NULL ELSE ("
            f"SELECT CASE "
            f"WHEN s.cnt < {mp} THEN NULL "
            f"WHEN s.cnt = 0 THEN NULL "
            f"WHEN s.cnt <= 1 THEN 1.0 "
            f"ELSE (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt END "
            f"FROM ("
            f"SELECT "
            f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
            f"COUNT(*) FILTER (WHERE {valid} AND p._v <= b._v) AS cnt_le, "
            f"COUNT(*) FILTER (WHERE {valid} AND p._v = b._v) AS cnt_eq "
            f"FROM ({numbered}) p "
            f"WHERE p.inst = b.inst AND p.rn BETWEEN b.rn - {w - 1} AND b.rn"
            f") s"
            f") END"
        )
    return f"SELECT b.ts, b.inst, {rank_expr} AS _v FROM ({numbered}) b"


def _zscore_prev_window_expr(
    *,
    value_col: str,
    window: int,
    min_periods: int,
    dialect: SqlDialect,
) -> str:
    """``x.shift(1).rolling(w, min_periods=w)`` 型 z-score：当前值相对前一 w 根
    bar 的均值/样本 std（pandas ``rolling.std`` ddof=1）；std=0 → NULL（对齐
    ``s.replace(0, np.nan)``，区别于 ts_zscore 的 std=0→0）。"""
    std_fn = _dialect_fn(dialect, "stddev")
    over = (
        f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING"
    )
    cnt = f"COUNT({value_col}) OVER ({over})"
    m = f"AVG({value_col}) OVER ({over})"
    s = f"{std_fn}({value_col}) OVER ({over})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {cnt} < {min_periods} THEN NULL "
        f"WHEN {s} IS NULL OR {s} = 0 THEN NULL "
        f"ELSE ({value_col} - {m}) / {s} END"
    )


def _pct_rank_window_sql(
    inner_sql: str,
    *,
    window: int,
    dialect: SqlDialect,
) -> str:
    """``x.rolling(w, min_periods=w).apply`` 型百分位 rank（窗口含当前行）：
    ``(cnt_lt + 0.5 * cnt_eq) / (cnt_full - 1)``，对齐 ``_pct_rank``。"""
    w = max(int(window), 2)
    numbered = (
        f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn "
        f"FROM ({inner_sql}) t0"
    )
    rank_expr = (
        f"CASE WHEN b._v IS NULL THEN NULL "
        f"WHEN s.full_cnt < {w} THEN NULL "
        f"ELSE (s.lt + 0.5 * s.eq) / (s.full_cnt - 1) END"
    )
    return (
        f"SELECT b.ts, b.inst, {rank_expr} AS _v FROM ({numbered}) b "
        f"JOIN LATERAL ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE p._v IS NOT NULL AND p.rn BETWEEN b.rn - {w - 1} AND b.rn) AS full_cnt, "
        f"COUNT(*) FILTER (WHERE p._v IS NOT NULL AND p._v < b._v AND p.rn BETWEEN b.rn - {w - 1} AND b.rn - 1) AS lt, "
        f"COUNT(*) FILTER (WHERE p._v IS NOT NULL AND p._v = b._v AND p.rn BETWEEN b.rn - {w - 1} AND b.rn - 1) AS eq "
        f"FROM ({numbered}) p WHERE p.inst = b.inst"
        f") s"
    )


def _ts_expanding_rank_sql(
    inner_sql: str,
    *,
    dialect: SqlDialect,
) -> str:
    """``x.expanding(min_periods=1).rank(pct=True)``：从头到当前行的平均百分位
    rank（与 ``_ts_pct_rank_sql`` 同款相关子查询，frame 换成 ``rn <= b.rn``）。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=True)
    invalid_row = row_stat_invalid_sql("b._v", dialect=dialect, exclude_nan=True)
    numbered = (
        f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn "
        f"FROM ({inner_sql}) t0"
    )
    rank_expr = (
        f"CASE WHEN {invalid_row} THEN NULL ELSE ("
        f"SELECT CASE "
        f"WHEN s.cnt = 0 THEN NULL "
        f"WHEN s.cnt <= 1 THEN 1.0 "
        f"ELSE (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt END "
        f"FROM ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v <= b._v) AS cnt_le, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v = b._v) AS cnt_eq "
        f"FROM ({numbered}) p WHERE p.inst = b.inst AND p.rn <= b.rn"
        f") s"
        f") END"
    )
    return f"SELECT b.ts, b.inst, {rank_expr} AS _v FROM ({numbered}) b"


def _rolling_corr_pandas_compat_expr(
    *,
    corr_col: str,
    std_left_col: str,
    std_right_col: str,
    left_col: str,
    right_col: str,
    window_count_col: str,
    window: int,
    dialect: SqlDialect,
) -> str:
    """滚动相关；满窗零方差退化时对齐 pandas ``rolling.corr``（→ ``inf``）。"""
    w = max(int(window), 1)
    full = f"{window_count_col} >= {w}"
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"multiIf("
            f"isNull({left_col}) OR isNull({right_col}), NULL, "
            f"isFinite({corr_col}), {corr_col}, "
            f"{full} AND (isNull({std_left_col}) OR {std_left_col} = 0) AND {std_right_col} > 0, inf, "
            f"{full} AND (isNull({std_right_col}) OR {std_right_col} = 0) AND {std_left_col} > 0, inf, "
            f"NULL)"
        )
    return (
        f"CASE "
        f"WHEN {left_col} IS NULL OR {right_col} IS NULL THEN NULL "
        f"WHEN isfinite({corr_col}) THEN {corr_col} "
        f"WHEN {full} AND ({std_left_col} IS NULL OR {std_left_col} = 0) AND {std_right_col} > 0 THEN 'Infinity'::DOUBLE "
        f"WHEN {full} AND ({std_right_col} IS NULL OR {std_right_col} = 0) AND {std_left_col} > 0 THEN 'Infinity'::DOUBLE "
        f"ELSE NULL END"
    )


def _group_zscore_expr(
    *,
    value_col: str,
    partition: str,
    dialect: SqlDialect,
    canon: str = "group_zscore",
) -> str:
    """组内 zscore；零/缺失标准差时输出语义见 ``numeric_semantics``。"""
    from backend.numeric_semantics import sql_stddev_fn_key, zscore_zero_std_fill

    std_fn = _dialect_fn(dialect, sql_stddev_fn_key(canon))
    nf = _dialect_fn(dialect, "nullif")
    zero_fill = zscore_zero_std_fill(canon)
    zero_sql = "0" if zero_fill == 0.0 else ("NULL" if zero_fill is None else "NaN")
    avg = f"AVG({value_col}) OVER ({partition})"
    stdv = f"{std_fn}({value_col}) OVER ({partition})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {stdv} IS NULL OR {stdv} = 0 THEN {zero_sql} "
        f"ELSE ({value_col} - {avg}) / {nf}({stdv}, 0) END"
    )


def _group_decay_linear_expr(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """组内按排名线性衰减权重 × 原值（对齐 pandas group_decay_linear）。"""
    nf = _dialect_fn(dialect, "nullif")
    cnt = f"COUNT({value_col}) OVER ({partition})"
    denom = f"{nf}({cnt} * ({cnt} + 1.0) / 2.0, 0)"
    if dialect == SqlDialect.CLICKHOUSE:
        rank = f"toFloat64(RANK() OVER ({partition} ORDER BY {value_col}))"
    else:
        rank = f"RANK() OVER ({partition} ORDER BY {value_col} NULLS LAST)"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"ELSE {value_col} * ({rank} * 1.0 / {denom}) END"
    )


def _group_minmax_expr(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """组内 [0,1] min-max；零区间输出 0.5（对齐 pandas group_normalize）。"""
    nf = _dialect_fn(dialect, "nullif")
    lo = f"MIN({value_col}) OVER ({partition})"
    hi = f"MAX({value_col}) OVER ({partition})"
    span = f"{nf}({hi} - {lo}, 0)"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {span} IS NULL THEN 0.5 "
        f"ELSE ({value_col} - {lo}) / {span} END"
    )


def _cs_ols_components(
    dialect: SqlDialect,
    *,
    y_col: str,
    x_col: str,
    partition: str,
) -> tuple[str, str, str]:
    """截面 OLS y ~ x + const：pairwise-valid 样本，返回 (beta, alpha, n_valid)。"""
    if dialect == SqlDialect.DUCKDB:
        cov_fn = "covar_samp"
        var_fn = "var_samp"
    else:
        cov_fn = "covarSamp"
        var_fn = "varSamp"
    nf = _dialect_fn(dialect, "nullif")
    pair_mask = f"{y_col} IS NOT NULL AND {x_col} IS NOT NULL"
    pair_y = f"CASE WHEN {pair_mask} THEN {y_col} END"
    pair_x = f"CASE WHEN {pair_mask} THEN {x_col} END"
    mean_y = f"AVG({pair_y}) OVER ({partition})"
    mean_x = f"AVG({pair_x}) OVER ({partition})"
    cov_xy = f"{cov_fn}({pair_y}, {pair_x}) OVER ({partition})"
    var_x = f"{var_fn}({pair_x}) OVER ({partition})"
    beta = f"({cov_xy}) / {nf}({var_x}, 0)"
    alpha = f"({mean_y}) - ({beta}) * ({mean_x})"
    n_valid = f"COUNT(CASE WHEN {pair_mask} THEN 1 END) OVER ({partition})"
    return beta, alpha, n_valid


def _zscore_window_expr(
    *,
    value_col: str,
    partition: str,
    dialect: SqlDialect,
    std_fn_name: str = "stddev",
    min_periods: int = 1,
) -> str:
    """zscore / ts_zscore：std=0 → 0，NULL 保持缺失。"""
    std_fn = _dialect_fn(dialect, std_fn_name)
    avg = f"AVG({value_col}) OVER ({partition})"
    stdv = f"{std_fn}({value_col}) OVER ({partition})"
    core = (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {stdv} IS NULL THEN NULL "
        f"WHEN {stdv} = 0 THEN 0 "
        f"ELSE ({value_col} - {avg}) / {stdv} END"
    )
    if min_periods <= 1:
        return core
    cnt = f"COUNT({value_col}) OVER ({partition})"
    return f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {core} END"


def _normalize_window_expr(*, value_col: str, partition: str) -> str:
    """normalize / group_normalize：常数截面 → 0.5；单有效值 → NULL。"""
    lo = f"MIN({value_col}) OVER ({partition})"
    hi = f"MAX({value_col}) OVER ({partition})"
    span = f"({hi} - {lo})"
    cnt = f"COUNT({value_col}) OVER ({partition})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {cnt} <= 1 THEN NULL "
        f"WHEN {span} IS NULL OR {span} = 0 THEN 0.5 "
        f"ELSE ({value_col} - {lo}) / {span} END"
    )


def _scale_window_expr(*, value_col: str, partition: str, to_val: float, dialect: SqlDialect) -> str:
    """scale：sum(abs)=0 → 0，否则 x * to / sum(abs)。"""
    abs_fn = _dialect_fn(dialect, "abs")
    s = f"SUM({abs_fn}({value_col})) OVER ({partition})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {s} IS NULL OR {s} = 0 THEN 0 "
        f"ELSE {value_col} * ({to_val} / {s}) END"
    )


def _rolling_min_periods(window: int, *, default_mp: int | None = None) -> int:
    w = max(int(window), 1)
    return max(2, w // 2) if default_mp is None else max(int(default_mp), 1)


def _rolling_std_min_periods_sql(
    *,
    value_col: str,
    over: str,
    window: int,
    scale_expr: str,
    dialect: SqlDialect,
    min_periods: int | None = None,
    std_fn_name: str = "stddev",
) -> str:
    """带 min_periods 的 rolling std 表达式。"""
    mp = _rolling_min_periods(window, default_mp=min_periods)
    std_fn = _dialect_fn(dialect, std_fn_name)
    cnt = f"COUNT({value_col}) OVER ({over})"
    stdv = f"{std_fn}({value_col}) OVER ({over})"
    return (
        f"CASE WHEN {cnt} < {mp} THEN NULL ELSE {stdv} * {scale_expr} END"
    )


def _int_attr(node: PlanNode, *keys: str, input_index: int | None = None, default: int = 0) -> int:
    """从 attrs 或 positional literal 读取整数参数（严格解析，无静默截断）。

    #357：非整数（float 5.9 / 字符串）→ raise ``PlanParamError``；需要的钳制
    （如 ``max(1, ...)``）由调用点显式完成。
    """
    from backend.plan_params import parse_positive_int_literal

    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return parse_positive_int_literal(node.attrs[key], label=key)
    if input_index is not None:
        pos = _literal_positional(node, input_index, default=float(default))
        if pos is not None:
            if isinstance(pos, float) and pos != int(pos):
                from backend.plan_params import PlanParamError

                raise PlanParamError(
                    f"positional integer 参数必须为整数 literal，收到 {pos!r}"
                )
            return int(pos)
    return default


def _prior_window_zscore_sql(
    inner_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
    min_periods: int | None = None,
) -> str:
    """Prior-window z-score：先 shift(1) 再滚动 mean/std（``volume_zscore`` 等）。"""
    if min_periods is None:
        min_periods = w
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    std_fn = _dialect_fn(dialect, "stddev")
    return (
        f"SELECT ts, inst, "
        f"(x._v - AVG(x.prev) OVER ({over})) / "
        f"NULLIF({std_fn}(x.prev) OVER ({over}), 0) AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, "
        f"LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS prev "
        f"FROM ({inner_sql}) t"
        f") x"
    )


def _rolling_corr_lag_sql(
    inner_sql: str,
    w: int,
    lag: int,
    *,
    dialect: SqlDialect,
    min_periods: int | None = None,
) -> str:
    """Rolling correlation with a lag series (``volume_autocorr`` 等)。"""
    if min_periods is None:
        min_periods = w
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    pair_mask = "x._v IS NOT NULL AND x.pv IS NOT NULL"
    pair_l = f"CASE WHEN {pair_mask} THEN x._v END"
    pair_r = f"CASE WHEN {pair_mask} THEN x.pv END"
    cnt = f"COUNT(CASE WHEN {pair_mask} THEN 1 END) OVER ({over})"
    corr = f"CORR({pair_l}, {pair_r}) OVER ({over})"
    body = corr
    if min_periods > 1:
        body = f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {corr} END"
    return (
        f"SELECT ts, inst, {body} AS _v FROM ("
        f"SELECT ts, inst, _v, "
        f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS pv "
        f"FROM ({inner_sql}) t"
        f") x"
    )


def _linear_decay_over_inst(w: int, inner_sql: str, *, dialect: SqlDialect) -> str:
    """线性衰减加权均值：最近观测权重 ``w``，最早为 ``1``（对齐 ``ts_decay_linear``）。

    ±Inf rows are treated as missing (pandas ``_linear_weighted_1d`` drops them
    and renormalizes the surviving original weight slots).
    """
    isinf_fn = "isInfinite" if dialect == SqlDialect.CLICKHOUSE else "isinf"
    num_parts: list[str] = []
    den_parts: list[str] = []
    for lag in range(w):
        weight = w - lag
        if lag == 0:
            v = "_v"
        else:
            v = f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)"
        valid = f"{v} IS NOT NULL AND NOT {isinf_fn}({v})"
        num_parts.append(f"CASE WHEN {valid} THEN {weight}.0 * {v} ELSE 0 END")
        den_parts.append(f"CASE WHEN {valid} THEN {weight}.0 ELSE 0 END")
    num = " + ".join(num_parts)
    den = f"{_dialect_fn(dialect, 'nullif')}(" + " + ".join(den_parts) + ", 0)"
    return f"SELECT ts, inst, ({num}) / {den} AS _v FROM ({inner_sql}) t"


def _linear_weighted_mean_sql(inner_sql: str, w: int, *, dialect: SqlDialect) -> str:
    """WMA: 线性加权移动平均（权重递增: 1, 2, ..., w）。"""
    return _linear_decay_over_inst(w, inner_sql, dialect=dialect)


def _ts_rank_sql(inner_sql: str, w: int, *, dialect: SqlDialect) -> str:
    """滚动窗口内的 rank (0-1 scale)。"""
    return _ts_pct_rank_sql(inner_sql, window=w, dialect=dialect, min_periods=1)


def _scalar_from_plan(node: PlanNode, *, default: float | None = None) -> float | None:
    """从计划节点 attrs 或子 literal 提取标量浮点值。"""
    for key in ("value", "fill_value", "const", "c"):
        if key in node.attrs and node.attrs[key] is not None:
            return float(node.attrs[key])
    if len(node.inputs) >= 2 and node.inputs[1].op == "literal":
        val = node.inputs[1].attrs.get("value")
        if val is not None:
            return float(val)
    return default


def _compare_binary_sql(op: str, left_sql: str, right_sql: str, *, dialect: SqlDialect) -> str:
    """二元比较算子 SQL（NULL/NaN → NULL）。"""
    from backend.elementwise_semantics import compare_sql
    from backend.long_alignment import anchor_join_sql

    expr = compare_sql(op, "l._v", "r._v", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE)
    return f"SELECT l.ts, l.inst, {expr} AS _v {anchor_join_sql(left_sql, right_sql)}"


def _rolling_ols_partition(w: int, *, prefix: str = "l") -> str:
    """滚动 OLS 窗口的 PARTITION/ORDER BY 子句。"""
    return (
        f"PARTITION BY {prefix}.inst ORDER BY {prefix}.ts "
        f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    )


def _inst_cum_sum_sql(inner_sql: str) -> str:
    """pandas 对齐 cumsum：NULL 行输出 NULL，累计和跳过 NULL 输入。"""
    return _inst_cum_agg("SUM", inner_sql, sum_skip_null=True)


def _inst_cum_agg(
    agg: str,
    inner_sql: str,
    *,
    sum_skip_null: bool = False,
    null_guard: bool = True,
) -> str:
    """按 instrument 的累积窗口聚合 SQL；默认当前行 NULL → 输出 NULL。"""
    over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    agg_u = agg.upper()
    if sum_skip_null or agg_u == "SUM":
        rolled = f"SUM(CASE WHEN _v IS NULL THEN 0 ELSE _v END) OVER ({over})"
    elif agg_u == "COUNT":
        rolled = f"SUM(CASE WHEN _v IS NOT NULL THEN 1 ELSE 0 END) OVER ({over})"
        null_guard = False
    else:
        rolled = f"{agg}(_v) OVER ({over})"
    if null_guard:
        body = f"CASE WHEN _v IS NULL THEN NULL ELSE {rolled} END"
    else:
        body = rolled
    return (
        f"SELECT ts, inst, {body} AS _v "
        f"FROM ({inner_sql}) t"
    )


def _cs_broadcast_agg(agg: str, inner_sql: str, *, all_null_null: bool = False) -> str:
    """截面广播聚合：按 ts 分区将聚合值广播到各行。"""
    rolled = f"{agg}(_v) OVER (PARTITION BY ts)"
    if not all_null_null:
        return (
            f"SELECT ts, inst, {rolled} AS _v "
            f"FROM ({inner_sql}) t"
        )
    valid = "COUNT(_v) OVER (PARTITION BY ts)"
    body = f"CASE WHEN {valid} = 0 THEN NULL ELSE {rolled} END"
    return (
        f"SELECT ts, inst, {body} AS _v "
        f"FROM ({inner_sql}) t"
    )


def _cs_mad_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """截面 MAD：median(|x - median(x)|) 广播到各行（finite sample）。"""
    from backend.stat_valid import row_stat_invalid_sql

    med_fn = "median"
    invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
    cleaned = (
        f"SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS _v "
        f"FROM ({inner_sql}) t_clean"
    )
    return (
        f"SELECT ts, inst, "
        f"{med_fn}(abs(_v - med)) OVER (PARTITION BY ts) AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, {med_fn}(_v) OVER (PARTITION BY ts) AS med "
        f"FROM ({cleaned}) t0"
        f") t"
    )


def _cs_mad_zscore_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """MAD 稳健 zscore：(x - median) / MAD；MAD=0 → NULL（finite sample）。"""
    from backend.stat_valid import row_stat_invalid_sql

    med_fn = "median"
    invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
    cleaned = (
        f"SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS _v "
        f"FROM ({inner_sql}) t_clean"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        core = (
            f"if(isNull(mad) OR mad = 0 OR ({invalid}), NULL, (_v - med) / mad)"
        )
    else:
        core = (
            f"CASE WHEN {invalid} OR mad IS NULL OR mad = 0 THEN NULL "
            f"ELSE (_v - med) / mad END"
        )
    return (
        f"SELECT ts, inst, {core} AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, med, "
        f"{med_fn}(abs(_v - med)) OVER (PARTITION BY ts) AS mad "
        f"FROM ("
        f"SELECT ts, inst, _v, {med_fn}(_v) OVER (PARTITION BY ts) AS med "
        f"FROM ({cleaned}) t0"
        f") t1"
        f") t"
    )


def _cum_delta_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """累积差分：当前值减去窗口内首个非空值。"""
    over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    first_v = f"FIRST_VALUE(_v IGNORE NULLS) OVER ({over})"
    if dialect == SqlDialect.CLICKHOUSE:
        first_v = f"first_value(_v) IGNORE NULLS OVER ({over})"
        return (
            f"SELECT ts, inst, "
            f"if(isNull(_v) OR isNull({first_v}), NULL, _v - {first_v}) AS _v "
            f"FROM ({inner_sql}) t"
        )
    return (
        f"SELECT ts, inst, "
        f"CASE WHEN _v IS NULL OR {first_v} IS NULL THEN NULL "
        f"ELSE _v - {first_v} END AS _v "
        f"FROM ({inner_sql}) t"
    )


# #376：SQL 层 ts_ema / ts_ewm_std / ts_ewm_var / ts_ewm_cov / ts_ewm_corr 都是
# 有限窗近似（``ROWS BETWEEN w-1 PRECEDING`` 内的指数衰减加权和），并非
# canonical ``ts_ema`` 的精确递归 EWM。任何拿该 SQL 输出做 exact production
# parity 认证的代码点必须 fail（见 ``EWM_SQL_APPROX``），不得静默当作 exact。
_SQL_EWM_IS_APPROX = True
"""SQL EWM 是否为有限窗近似（#376）。True → 不可用于 exact parity。"""

EWM_SQL_APPROX = _SQL_EWM_IS_APPROX
"""导出的 EWM SQL 近似标记；外部 parity 认证代码应据此拒绝 exact 断言。"""


def _sql_ewm_exact_parity_guard() -> None:
    """#376：SQL 层 ``ts_ema``/``ts_ewm_*`` 是有限窗近似。

    任何拿该 SQL 近似给 canonical ``ts_ema`` 做 exact production parity 认证
    的代码点必须调用本守卫（强制 ``AssertionError``），而不是静默给出近似结果。
    """
    raise AssertionError(
        "SQL EWM 为有限窗近似（EWM_SQL_APPROX=True），不能用于 exact parity 认证"
    )


def _ewm_weighted_moment_sql(
    inner_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
    sqrt: bool,
) -> str:
    """指数衰减窗口矩（对齐 ``ewm(span, adjust=False)`` 的有限窗近似）。

    #376：有限窗近似（``_SQL_EWM_IS_APPROX=True``），勿用于 exact parity。
    """
    alpha = 2.0 / (float(w) + 1.0)
    decay = 1.0 - alpha
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    nf = _dialect_fn(dialect, "nullif")
    g = _dialect_fn(dialect, "greatest")
    sqrt_fn = _dialect_fn(dialect, "sqrt")
    mean = (
        f"SUM(_v * POW({decay}, rn)) OVER ({over}) / "
        f"{nf}(SUM(POW({decay}, rn)) OVER ({over}), 0)"
    )
    mean2 = (
        f"SUM(_v * _v * POW({decay}, rn)) OVER ({over}) / "
        f"{nf}(SUM(POW({decay}, rn)) OVER ({over}), 0)"
    )
    var = f"{g}(0, {mean2} - ({mean}) * ({mean}))"
    val = f"{sqrt_fn}({var})" if sqrt else var
    return (
        f"SELECT ts, inst, {val} AS _v FROM ("
        f"SELECT ts, inst, _v, "
        f"(COUNT(*) OVER ({over}) - 1 - "
        f"(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - "
        f"MIN(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts)) OVER ({over}))) AS rn "
        f"FROM ({inner_sql}) t0"
        f") t"
    )


def _ewm_cov_corr_sql(
    left_sql: str,
    right_sql: str,
    span: int,
    *,
    dialect: SqlDialect,
    corr: bool,
) -> str:
    """二元 EWM 协方差/相关系数（``adjust=False`` 有限窗近似）。"""
    w = span
    alpha = 2.0 / (float(w) + 1.0)
    decay = 1.0 - alpha
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    nf = _dialect_fn(dialect, "nullif")
    g = _dialect_fn(dialect, "greatest")
    sqrt_fn = _dialect_fn(dialect, "sqrt")
    joined = (
        f"SELECT l.ts, l.inst, l._v AS xv, r._v AS yv, "
        f"(COUNT(*) OVER ({over}) - 1 - "
        f"(ROW_NUMBER() OVER (PARTITION BY l.inst ORDER BY l.ts) - "
        f"MIN(ROW_NUMBER() OVER (PARTITION BY l.inst ORDER BY l.ts)) OVER ({over}))) AS rn "
        f"FROM ({left_sql}) l INNER JOIN ({right_sql}) r USING (ts, inst)"
    )
    wsum = f"SUM(POW({decay}, rn)) OVER ({over})"
    mx = f"SUM(xv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    my = f"SUM(yv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    mxy = f"SUM(xv * yv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    mxx = f"SUM(xv * xv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    myy = f"SUM(yv * yv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    cov = f"({mxy}) - ({mx}) * ({my})"
    if corr:
        varx = f"{g}(0, {mxx} - ({mx}) * ({mx}))"
        vary = f"{g}(0, {myy} - ({my}) * ({my}))"
        val = f"{cov} / {nf}({sqrt_fn}({varx}) * {sqrt_fn}({vary}), 0)"
    else:
        val = cov
    return f"SELECT ts, inst, {val} AS _v FROM ({joined}) t"


def _rolling_slope_over_inst(w: int, inner_sql: str, *, dialect: SqlDialect) -> str:
    """滚动 OLS 斜率（对齐 ``Slope(x, w)`` / ``rolling_time_slope``）。"""
    t_mean = (w - 1) / 2.0
    denom = sum((i - t_mean) ** 2 for i in range(w))
    num_parts: list[str] = []
    for lag in range(w):
        idx = w - 1 - lag
        weight = (idx - t_mean) / denom if denom else 0.0
        v = "_v" if lag == 0 else f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)"
        num_parts.append(f"CASE WHEN {v} IS NOT NULL THEN ({weight}) * {v} ELSE 0 END")
    num = " + ".join(num_parts)
    return f"SELECT ts, inst, ({num}) AS _v FROM ({inner_sql}) t"


def _ts_argext_sql(inner_sql: str, w: int, *, dialect: SqlDialect, pick: str) -> str:
    """滚动极值距当前 bar 的距离（0=当前；并列取最近）。"""
    over = "PARTITION BY inst ORDER BY ts"
    win = f"{over} ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    ext_fn = "MAX" if pick == "max" else "MIN"
    cases: list[str] = []
    for lag in range(w):
        val = "_v" if lag == 0 else f"LAG(_v, {lag}) OVER ({over})"
        cases.append(
            f"WHEN {val} IS NOT NULL AND w_ext IS NOT NULL AND ABS({val} - w_ext) < 1e-9 "
            f"THEN {float(lag)}"
        )
    case_expr = "CASE " + " ".join(cases) + " ELSE 0.0 END"
    return (
        f"SELECT ts, inst, "
        f"CASE WHEN w_ext IS NULL THEN 0.0 ELSE ({case_expr}) END AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, "
        f"{ext_fn}(_v) OVER ({win}) AS w_ext, "
        f"COUNT(_v) OVER ({win}) AS w_cnt "
        f"FROM ({inner_sql}) t0"
        f") t"
    )


def _duckdb_valid(value: str) -> str:
    return f"({value} IS NOT NULL AND NOT isnan({value}) AND NOT isinf({value}))"


def _raw_literal(node: PlanNode, input_index: int, default: Any = None) -> Any:
    if input_index >= len(node.inputs):
        return default
    child = node.inputs[input_index]
    if child.op != "literal":
        return default
    return child.attrs.get("value", default)


def _duckdb_join_layers(layers: list[_Layer], aliases: list[str]) -> str:
    sql = f"FROM ({layers[0].sql}) {aliases[0]}"
    for layer, alias in zip(layers[1:], aliases[1:], strict=True):
        sql += f" LEFT JOIN ({layer.sql}) {alias} USING (ts, inst)"
    return sql


def _duckdb_conditional_rolling_sql(
    value_sql: str | None,
    condition_sql: str,
    *,
    window: int,
    min_periods: int,
    op: str,
    ddof: int = 1,
) -> str:
    if value_sql is None:
        joined = f"FROM ({condition_sql}) c"
        over = (
            f"PARTITION BY c.inst ORDER BY c.ts "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        valid = _duckdb_valid("c._v")
        true = f"({valid} AND c._v <> 0)"
        count_valid = f"SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) OVER ({over})"
        value = f"SUM(CASE WHEN {true} THEN 1 ELSE 0 END) OVER ({over})"
        return (
            f"SELECT c.ts, c.inst, CASE WHEN {count_valid} < {min_periods} "
            f"THEN NULL ELSE CAST({value} AS DOUBLE) END AS _v {joined}"
        )

    joined = (
        f"FROM ({value_sql}) x LEFT JOIN ({condition_sql}) c USING (ts, inst)"
    )
    over = (
        f"PARTITION BY x.inst ORDER BY x.ts "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
    )
    selected = (
        f"({_duckdb_valid('x._v')} AND {_duckdb_valid('c._v')} AND c._v <> 0)"
    )
    count = f"SUM(CASE WHEN {selected} THEN 1 ELSE 0 END) OVER ({over})"
    if op == "sum":
        aggregate = f"SUM(CASE WHEN {selected} THEN x._v ELSE NULL END) OVER ({over})"
        required = min_periods
    elif op == "mean":
        aggregate = f"AVG(CASE WHEN {selected} THEN x._v ELSE NULL END) OVER ({over})"
        required = min_periods
    elif op == "std":
        fn = "STDDEV_POP" if ddof == 0 else "STDDEV_SAMP"
        aggregate = f"{fn}(CASE WHEN {selected} THEN x._v ELSE NULL END) OVER ({over})"
        required = max(min_periods, ddof + 1)
    else:  # pragma: no cover
        raise ValueError(op)
    return (
        f"SELECT x.ts, x.inst, CASE WHEN {count} < {required} "
        f"THEN NULL ELSE {aggregate} END AS _v {joined}"
    )


def _duckdb_multi_resid_sql(
    layers: list[_Layer],
    *,
    add_intercept: bool,
    min_obs: int,
) -> str:
    aliases = ["y"] + [f"x{i}" for i in range(1, len(layers))]
    joined = _duckdb_join_layers(layers, aliases)
    valid_parts = [_duckdb_valid("y._v")]
    valid_parts.extend(_duckdb_valid(f"{alias}._v") for alias in aliases[1:])
    ok = " AND ".join(valid_parts)
    raw_cols = ", ".join(
        ["y._v AS _y"] + [f"{alias}._v AS _x{i}" for i, alias in enumerate(aliases[1:], 1)]
    )
    base = (
        f"SELECT y.ts, y.inst, {raw_cols}, CASE WHEN {ok} THEN 1 ELSE 0 END AS _ok "
        f"{joined}"
    )
    n = "SUM(_ok) OVER (PARTITION BY ts)"
    if add_intercept:
        mean_y = "AVG(CASE WHEN _ok = 1 THEN _y END) OVER (PARTITION BY ts)"
        centered = [f"_x{i} - AVG(CASE WHEN _ok = 1 THEN _x{i} END) OVER (PARTITION BY ts)" for i in range(1, len(layers))]
        y_centered = f"_y - {mean_y}"
    else:
        mean_y = "0.0"
        centered = [f"_x{i}" for i in range(1, len(layers))]
        y_centered = "_y"
    stage = (
        f"SELECT *, {n} AS _n, {mean_y} AS _mean_y, "
        f"CASE WHEN _ok = 1 THEN {y_centered} END AS _yc, "
        + ", ".join(
            f"CASE WHEN _ok = 1 THEN {expr} END AS _v{i}"
            for i, expr in enumerate(centered, 1)
        )
        + f" FROM ({base}) b"
    )
    q_names: list[str] = []
    for i in range(1, len(layers)):
        residual = f"_v{i}"
        for q in q_names:
            coeff = (
                f"SUM(_v{i} * {q}) OVER (PARTITION BY ts) / "
                f"NULLIF(SUM({q} * {q}) OVER (PARTITION BY ts), 0)"
            )
            residual += f" - ({coeff}) * {q}"
        q = f"_q{i}"
        stage = f"SELECT *, ({residual}) AS {q} FROM ({stage}) qstage{i}"
        q_names.append(q)
    singular = " OR ".join(
        f"SUM({q} * {q}) OVER (PARTITION BY ts) IS NULL OR "
        f"SUM({q} * {q}) OVER (PARTITION BY ts) <= 1e-24"
        for q in q_names
    )
    fitted_terms = [
        (
            f"(SUM(_yc * {q}) OVER (PARTITION BY ts) / "
            f"NULLIF(SUM({q} * {q}) OVER (PARTITION BY ts), 0)) * {q}"
        )
        for q in q_names
    ]
    fitted = "_mean_y" + "".join(f" + {term}" for term in fitted_terms)
    return (
        f"SELECT ts, inst, CASE WHEN _ok = 0 OR _n < {min_obs} "
        f"OR ({singular}) THEN NULL ELSE _y - ({fitted}) END AS _v "
        f"FROM ({stage}) scored"
    )


def _duckdb_rolling_tstat_sql(
    y_sql: str,
    x_sql: str,
    *,
    window: int,
    min_periods: int,
    add_intercept: bool,
) -> str:
    over = (
        f"PARTITION BY y.inst ORDER BY y.ts "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
    )
    valid = f"({_duckdb_valid('y._v')} AND {_duckdb_valid('x._v')})"
    yv = f"CASE WHEN {valid} THEN y._v END"
    xv = f"CASE WHEN {valid} THEN x._v END"
    n = f"COUNT({yv}) OVER ({over})"
    sx = f"SUM({xv}) OVER ({over})"
    sy = f"SUM({yv}) OVER ({over})"
    sxx = f"SUM(({xv}) * ({xv})) OVER ({over})"
    sxy = f"SUM(({xv}) * ({yv})) OVER ({over})"
    syy = f"SUM(({yv}) * ({yv})) OVER ({over})"
    if add_intercept:
        xx = f"({sxx} - ({sx}) * ({sx}) / NULLIF({n}, 0))"
        xy = f"({sxy} - ({sx}) * ({sy}) / NULLIF({n}, 0))"
        yy = f"({syy} - ({sy}) * ({sy}) / NULLIF({n}, 0))"
        dof = f"({n} - 2)"
    else:
        xx, xy, yy = sxx, sxy, syy
        dof = f"({n} - 1)"
    beta = f"({xy}) / NULLIF(({xx}), 0)"
    sse = f"GREATEST(0.0, ({yy}) - ({beta}) * ({xy}))"
    se = f"SQRT(({sse}) / NULLIF({dof}, 0) / NULLIF(({xx}), 0))"
    return (
        f"SELECT y.ts, y.inst, CASE WHEN {n} < {min_periods} OR {dof} <= 0 "
        f"OR ({xx}) <= 0 OR ({se}) <= 0 THEN NULL ELSE ({beta}) / ({se}) END AS _v "
        f"FROM ({y_sql}) y LEFT JOIN ({x_sql}) x USING (ts, inst)"
    )


# 无法在 SQL 中精确复刻 pandas 语义、由混合后端回退到 polars（与 pandas 完全
# 一致）的算子——"不建议的不用强行加"原则。
#
# - EWMA/Wilder 平滑族：pandas ``ewm(adjust=False)`` 的 ignore_na=False 在
#   NaN 缺口时按绝对位置衰减 + 每次有效观测重新归一化，无法用有限窗口加权和/
#   运行积精确复刻（长序列 runprod 下溢，递归 CTE 无法并行）。
# - cdl_hammer / cdl_hanging_man：pandas 参考嵌入 prior-trend 上下文（audit
#   item 5），需要 AVG(LAG(close)) 嵌套窗口，DuckDB 禁止嵌套窗口函数。
# - ts_time_slope / ts_upside_deviation / ts_weighted_standardized_moment /
#   ts_abdi_ranaldo_spread / ts_value_at_argextreme：这些算子的 SQL 分支在
#   暖启动/NaN 缺口/窗口位置重索引上与 pandas 内核不一致（pandas 用窗口内
#   位置 OLS 与有限值重归一化），精确复刻成本高；polars 后端已与 pandas 完全
#   一致，回退到 polars。
_SQL_FALLBACK_CANONICALS: frozenset[str] = frozenset({
    "RSI_WILDER", "ATR_WILDER", "DMI_plus", "DMI_minus", "DX", "ADX",
    "MACD_line", "MACD_signal", "MACD_hist",
    "DEMA", "TEMA", "PPO", "PPO_signal", "PPO_hist",
    "PVO", "PVO_signal", "PVO_hist", "TSI", "TSI_signal",
    "KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition",
    "ADL", "ChaikinOscillator", "CMF", "ForceIndex",
    "cdl_hammer", "cdl_hanging_man",
    "ts_time_slope", "ts_upside_deviation", "ts_weighted_standardized_moment",
    "ts_abdi_ranaldo_spread", "ts_value_at_argextreme",
})


def _compile_layer_impl(node: PlanNode, *, dialect: SqlDialect) -> _Layer | None:
    """递归将 PlanNode 编译为 ``_Layer`` 子查询；不支持的算子返回 ``None``。"""
    op = _resolve_canonical(node.op)
    if op == "rolling_beta":
        op = "ts_beta"
    if op in _SQL_FALLBACK_CANONICALS:
        return None
    std = _dialect_fn(dialect, "stddev")
    ln = _dialect_fn(dialect, "ln")
    g = _dialect_fn(dialect, "greatest")
    l = _dialect_fn(dialect, "least")
    nf = _dialect_fn(dialect, "nullif")

    if op == "WMA":
        wma_node = PlanNode(op="ts_decay_linear", inputs=list(node.inputs), attrs=dict(node.attrs))
        return _compile_layer(wma_node, dialect=dialect)

    if op == "column":
        col = node.attrs.get("name") or node.attrs.get("column")
        if not col:
            return None
        c = _quote_ident(str(col))
        return _Layer(f"SELECT ts, inst, {c} AS _v FROM base")

    if op == "literal":
        val = node.attrs.get("value")
        lit = _sql_literal(val)
        return _Layer(f"SELECT ts, inst, {lit} AS _v FROM base")

    if op == "identity":
        inner = _compile_layer(node.inputs[0], dialect=dialect) if node.inputs else None
        if inner is None:
            return None
        return _Layer(
            inner.sql,
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"add", "subtract", "multiply", "divide", "maximum", "minimum"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        if op == "maximum":
            fn = _dialect_fn(dialect, "greatest")
            isnan = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
            nan = "nan" if dialect == SqlDialect.CLICKHOUSE else "'NaN'::DOUBLE"
            expr = (
                f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
                f"WHEN {isnan}(l._v) OR {isnan}(r._v) THEN {nan} "
                f"ELSE {fn}(l._v, r._v) END"
            )
        elif op == "minimum":
            fn = _dialect_fn(dialect, "least")
            isnan = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
            nan = "nan" if dialect == SqlDialect.CLICKHOUSE else "'NaN'::DOUBLE"
            expr = (
                f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
                f"WHEN {isnan}(l._v) OR {isnan}(r._v) THEN {nan} "
                f"ELSE {fn}(l._v, r._v) END"
            )
        else:
            sym = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]
            expr = f"(l._v {sym} r._v)"
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "protected_div":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.elementwise_semantics import protected_div_sql
        from backend.numeric_semantics import protected_div_default, protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        default = _float_attr(node, "default", default=protected_div_default())
        abs_fn = _dialect_fn(dialect, "abs")
        expr = protected_div_sql("l._v", "r._v", eps=eps, default=default, abs_fn=abs_fn)
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op in {
        "safe_div_null",
        "safe_div",
        "fin_ratio",
        "float_share_ratio",
        "free_float_share_ratio",
        "benchmark_relative_price",
        "holder_concentration",
    }:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.numeric_semantics import protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN {abs_fn}(r._v) <= {eps} THEN NULL "
            f"ELSE l._v / r._v END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "benchmark_excess_return":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            f"SELECT l.ts, l.inst, (l._v - r._v) AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "ashare_limit_distance":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.numeric_semantics import protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN {abs_fn}(r._v) <= {eps} THEN NULL "
            f"ELSE l._v / r._v - 1.0 END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "div_or_default":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.elementwise_semantics import div_or_default_sql
        from backend.numeric_semantics import protected_div_default, protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        default = _float_attr(node, "default", default=protected_div_default())
        abs_fn = _dialect_fn(dialect, "abs")
        expr = div_or_default_sql("l._v", "r._v", eps=eps, default=default, abs_fn=abs_fn)
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "protected_log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.elementwise_semantics import protected_log_sql

        eps = _float_attr(node, "epsilon", default=1e-12)
        expr = protected_log_sql("_v", eps=eps, ln_fn=ln)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "log_fill_invalid":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.elementwise_semantics import log_fill_invalid_sql

        eps = _float_attr(node, "epsilon", default=1e-12)
        expr = log_fill_invalid_sql("_v", eps=eps, ln_fn=ln)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "protected_sqrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "sqrt")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL ELSE {fn}({g}(_v, 0)) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "neg":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, (-_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "sqrt_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        sqrt_fn = _dialect_fn(dialect, "sqrt")
        return _Layer(
            f"SELECT ts, inst, {sqrt_fn}({abs_fn}(_v)) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"book_to_price", "earnings_yield"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v <= 0 THEN NULL ELSE 1.0 / _v END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "sign":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "sign")
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v < 0 THEN NULL "
            f"WHEN _v = 0 THEN -1.0/0.0 ELSE {ln}(_v) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "exp":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "exp")
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "sqrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "sqrt")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v < 0 THEN NULL ELSE {fn}(_v) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"clip", "cap"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lo, hi = _clip_bounds(node)
        clip = f"{g}({lo}, {l}({hi}, _v))"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE {clip} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_null":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.logical_semantics import is_null_sql

        return _Layer(
            f"SELECT ts, inst, {is_null_sql('_v')} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_not_null":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.logical_semantics import is_null_sql

        return _Layer(
            f"SELECT ts, inst, CASE WHEN {is_null_sql('_v')} = 1.0 THEN 0.0 ELSE 1.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_nan":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.logical_semantics import is_nan_sql

        expr = is_nan_sql("_v", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_finite":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        finite_fn = "isFinite" if dialect == SqlDialect.CLICKHOUSE else "isfinite"
        isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR {isnan_fn}(_v) THEN 0.0 "
            f"WHEN {finite_fn}(_v) THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_infinite":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.elementwise_semantics import is_infinite_sql

        expr = is_infinite_sql("_v", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if"}:
        if dialect != SqlDialect.DUCKDB:
            return None
        if op == "ts_count_if":
            condition = _compile_layer(node.inputs[0], dialect=dialect)
            if condition is None:
                return None
            window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
            min_periods = int(node.attrs.get("min_periods", _raw_literal(node, 2, 1)))
            sql = _duckdb_conditional_rolling_sql(
                None,
                condition.sql,
                window=window,
                min_periods=min_periods,
                op="count",
            )
        else:
            if len(node.inputs) < 2:
                return None
            value = _compile_layer(node.inputs[0], dialect=dialect)
            condition = _compile_layer(node.inputs[1], dialect=dialect)
            if value is None or condition is None:
                return None
            window = int(node.attrs.get("window", _raw_literal(node, 2, 3)))
            default_mp = 2 if op == "ts_std_if" else 1
            min_periods = int(
                node.attrs.get("min_periods", _raw_literal(node, 3, default_mp))
            )
            ddof = int(node.attrs.get("ddof", _raw_literal(node, 4, 1)))
            sql = _duckdb_conditional_rolling_sql(
                value.sql,
                condition.sql,
                window=window,
                min_periods=min_periods,
                op=op.removeprefix("ts_").removesuffix("_if"),
                ddof=ddof,
            )
        return _Layer(sql, has_inst_window=True)

    if op == "ts_last_if":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        value = _compile_layer(node.inputs[0], dialect=dialect)
        condition = _compile_layer(node.inputs[1], dialect=dialect)
        if value is None or condition is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 2, 3)))
        over = (
            f"PARTITION BY x.inst ORDER BY x.ts "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        selected = (
            f"({_duckdb_valid('x._v')} AND {_duckdb_valid('c._v')} AND c._v <> 0)"
        )
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"arg_max(x._v, x.ts) FILTER (WHERE {selected}) OVER ({over}) AS _v "
            f"FROM ({value.sql}) x LEFT JOIN ({condition.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_days_since":
        if dialect != SqlDialect.DUCKDB:
            return None
        condition = _compile_layer(node.inputs[0], dialect=dialect)
        if condition is None:
            return None
        raw_limit = node.attrs.get("max_lookback", _raw_literal(node, 1, None))
        limit = None if raw_limit is None else int(raw_limit)
        true = f"({_duckdb_valid('_v')} AND _v <> 0)"
        distance = "(_rn - _last_true)"
        limit_guard = "" if limit is None else f" OR {distance} >= {limit}"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _last_true IS NULL{limit_guard} "
            f"THEN NULL ELSE CAST({distance} AS DOUBLE) END AS _v FROM ("
            f"SELECT *, MAX(CASE WHEN {true} THEN _rn END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS _last_true FROM ("
            f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _rn "
            f"FROM ({condition.sql}) c0"
            f") c1"
            f") c2",
            has_inst_window=True,
        )

    if op == "ts_true_streak":
        if dialect != SqlDialect.DUCKDB:
            return None
        condition = _compile_layer(node.inputs[0], dialect=dialect)
        if condition is None:
            return None
        false = f"(NOT {_duckdb_valid('_v')} OR _v = 0)"
        return _Layer(
            f"SELECT ts, inst, CAST(_rn - COALESCE(_last_false, 0) AS DOUBLE) AS _v FROM ("
            f"SELECT *, MAX(CASE WHEN {false} THEN _rn END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS _last_false FROM ("
            f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _rn "
            f"FROM ({condition.sql}) s0"
            f") s1"
            f") s2",
            has_inst_window=True,
        )

    if op == "cs_bucket":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        buckets = int(node.attrs.get("buckets", _raw_literal(node, 1, 10)))
        ascending = bool(node.attrs.get("ascending", _raw_literal(node, 2, True)))
        direction = "ASC" if ascending else "DESC"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE "
            f"LEAST({buckets}.0, FLOOR(((_rank + (_ties - 1) / 2.0) / _n) * {buckets}) + 1.0) "
            f"END AS _v FROM ("
            f"SELECT ts, inst, _v, "
            f"RANK() OVER (PARTITION BY ts ORDER BY _v {direction} NULLS LAST) AS _rank, "
            f"COUNT(*) OVER (PARTITION BY ts, _v) AS _ties, "
            f"COUNT(_v) OVER (PARTITION BY ts) AS _n "
            f"FROM ({inner.sql}) b0"
            f") b1",
            has_ts_partition=True,
        )

    if op == "cs_multi_resid":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        layers: list[_Layer] = []
        for child in node.inputs:
            if child.op == "literal":
                return None
            layer = _compile_layer(child, dialect=dialect)
            if layer is None:
                return None
            layers.append(layer)
        add_intercept = bool(node.attrs.get("add_intercept", True))
        default_min = len(layers) + 1
        min_obs = int(node.attrs.get("min_obs") or default_min)
        return _Layer(
            _duckdb_multi_resid_sql(
                layers,
                add_intercept=add_intercept,
                min_obs=min_obs,
            ),
            has_ts_partition=True,
        )

    if op == "cs_wls_resid":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        weight = _compile_layer(node.inputs[2], dialect=dialect)
        if y is None or x is None or weight is None:
            return None
        add_intercept = bool(node.attrs.get("add_intercept", True))
        min_obs = int(node.attrs.get("min_obs", 5))
        valid = (
            f"({_duckdb_valid('y._v')} AND {_duckdb_valid('x._v')} "
            f"AND {_duckdb_valid('w._v')} AND w._v > 0)"
        )
        joined = (
            f"SELECT y.ts, y.inst, y._v AS _y, x._v AS _x, w._v AS _w, "
            f"CASE WHEN {valid} THEN 1 ELSE 0 END AS _ok "
            f"FROM ({y.sql}) y LEFT JOIN ({x.sql}) x USING (ts, inst) "
            f"LEFT JOIN ({weight.sql}) w USING (ts, inst)"
        )
        n_expr = "SUM(_ok) OVER (PARTITION BY ts)"
        sum_w_expr = "SUM(CASE WHEN _ok = 1 THEN _w END) OVER (PARTITION BY ts)"
        if add_intercept:
            mx_expr = f"SUM(CASE WHEN _ok = 1 THEN _w * _x END) OVER (PARTITION BY ts) / NULLIF({sum_w_expr}, 0)"
            my_expr = f"SUM(CASE WHEN _ok = 1 THEN _w * _y END) OVER (PARTITION BY ts) / NULLIF({sum_w_expr}, 0)"
        else:
            mx_expr = my_expr = "0.0"
        centered = (
            f"SELECT *, {n_expr} AS _n, {mx_expr} AS _mx, {my_expr} AS _my "
            f"FROM ({joined}) w0"
        )
        dx, dy = "(_x - _mx)", "(_y - _my)"
        cov = f"SUM(CASE WHEN _ok = 1 THEN _w * ({dx}) * ({dy}) END) OVER (PARTITION BY ts)"
        var = f"SUM(CASE WHEN _ok = 1 THEN _w * ({dx}) * ({dx}) END) OVER (PARTITION BY ts)"
        moments = (
            f"SELECT *, {cov} AS _cov, {var} AS _var FROM ({centered}) w1"
        )
        beta = "_cov / NULLIF(_var, 0)"
        fitted = f"_my + ({beta}) * ({dx})"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _ok = 0 OR _n < {min_obs} "
            f"OR _var <= 1e-24 THEN NULL ELSE _y - ({fitted}) END AS _v "
            f"FROM ({moments}) wr",
            has_ts_partition=True,
        )

    if op == "period_lag":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        value = _compile_layer(node.inputs[0], dialect=dialect)
        period = _compile_layer(node.inputs[1], dialect=dialect)
        if value is None or period is None:
            return None
        periods = int(node.attrs.get("periods", _raw_literal(node, 2, 1)))
        joined = (
            f"SELECT x.ts, x.inst, x._v AS xv, p._v AS pid "
            f"FROM ({value.sql}) x LEFT JOIN ({period.sql}) p USING (ts, inst)"
        )
        return _Layer(
            f"SELECT r.ts, r.inst, CASE WHEN r.pid IS NULL THEN NULL ELSE ("
            f"SELECT arg_max(h.xv, h.ts) FROM ({joined}) h "
            f"WHERE h.inst = r.inst AND h.ts <= r.ts AND h.pid = ("
            f"SELECT target.pid FROM ("
            f"SELECT q.pid, MIN(q.ts) AS first_seen FROM ({joined}) q "
            f"WHERE q.inst = r.inst AND q.ts <= r.ts AND q.pid IS NOT NULL "
            f"GROUP BY q.pid ORDER BY first_seen DESC LIMIT 1 OFFSET {periods}"
            f") target"
            f")) END AS _v FROM ({joined}) r",
            has_inst_window=True,
        )

    if op in {"period_change", "period_cagr", "yoy_by_period"}:
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        current = _compile_layer(node.inputs[0], dialect=dialect)
        periods = int(node.attrs.get("periods", _raw_literal(node, 2, 1 if op != "yoy_by_period" else 4)))
        previous = _compile_layer(
            PlanNode(op="period_lag", inputs=[node.inputs[0], node.inputs[1]], attrs={"periods": periods}),
            dialect=dialect,
        )
        if current is None or previous is None:
            return None
        joined = f"SELECT x.ts, x.inst, x._v AS xv, p._v AS pv FROM ({current.sql}) x LEFT JOIN ({previous.sql}) p USING (ts, inst)"
        valid = f"({_duckdb_valid('xv')} AND {_duckdb_valid('pv')})"
        if op == "period_change":
            mode = str(node.attrs.get("mode", _raw_literal(node, 3, "absolute"))).lower()
            expr = "xv - pv" if mode == "absolute" else "xv / NULLIF(pv, 0) - 1.0" if mode == "ratio" else "LN(xv / NULLIF(pv, 0))"
            if mode not in {"absolute", "ratio", "log"}:
                return None
            if mode == "log":
                valid += " AND xv / NULLIF(pv, 0) > 0"
        elif op == "period_cagr":
            ppy = int(node.attrs.get("periods_per_year", _raw_literal(node, 3, 4)))
            policy = str(node.attrs.get("sign_policy", _raw_literal(node, 4, "strict"))).lower()
            if policy == "strict":
                valid += " AND xv > 0 AND pv > 0"
                ratio = "xv / pv"
            elif policy == "absolute":
                valid += " AND ABS(pv) > 1e-12"
                ratio = "ABS(xv) / ABS(pv)"
            else:
                return None
            expr = f"POW({ratio}, {float(ppy) / float(periods)}) - 1.0"
        else:
            mode = str(node.attrs.get("denominator", _raw_literal(node, 3, "signed"))).lower()
            if mode not in {"signed", "absolute"}:
                return None
            valid += " AND ABS(pv) > 1e-12"
            denom = "pv" if mode == "signed" else "ABS(pv)"
            expr = f"(xv - pv) / {denom}"
        return _Layer(f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v FROM ({joined}) fp", has_inst_window=True)

    if op in {"period_average", "ttm_from_quarterly"}:
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        default_periods = 2 if op == "period_average" else 4
        count = int(node.attrs.get("periods", _raw_literal(node, 2, default_periods)))
        if count < 1:
            return None
        layers = [_compile_layer(node.inputs[0], dialect=dialect)]
        layers.extend(
            _compile_layer(PlanNode(op="period_lag", inputs=[node.inputs[0], node.inputs[1]], attrs={"periods": lag}), dialect=dialect)
            for lag in range(1, count)
        )
        if any(layer is None for layer in layers):
            return None
        aliases = [f"v{i}" for i in range(count)]
        sql = f"SELECT b.ts, b.inst, b._v AS {aliases[0]} FROM ({layers[0].sql}) b"
        for i in range(1, count):
            sql = f"SELECT j.*, p._v AS {aliases[i]} FROM ({sql}) j LEFT JOIN ({layers[i].sql}) p USING (ts, inst)"
        valid = " AND ".join(_duckdb_valid(alias) for alias in aliases)
        total = " + ".join(aliases)
        expr = f"({total}) / {float(count)}" if op == "period_average" else f"({total})"
        return _Layer(f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v FROM ({sql}) fs", has_inst_window=True)

    if op == "quarter_from_cumulative":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        current = _compile_layer(node.inputs[0], dialect=dialect)
        quarter = _compile_layer(node.inputs[2], dialect=dialect)
        previous = _compile_layer(PlanNode(op="period_lag", inputs=[node.inputs[0], node.inputs[1]], attrs={"periods": 1}), dialect=dialect)
        previous_q = _compile_layer(PlanNode(op="period_lag", inputs=[node.inputs[2], node.inputs[1]], attrs={"periods": 1}), dialect=dialect)
        if any(layer is None for layer in (current, quarter, previous, previous_q)):
            return None
        joined = (
            f"SELECT x.ts, x.inst, x._v AS xv, q._v AS qv, p._v AS pv, pq._v AS pqv "
            f"FROM ({current.sql}) x LEFT JOIN ({quarter.sql}) q USING (ts, inst) "
            f"LEFT JOIN ({previous.sql}) p USING (ts, inst) LEFT JOIN ({previous_q.sql}) pq USING (ts, inst)"
        )
        consecutive = "((pqv = 4 AND qv = 1) OR qv = pqv + 1)"
        expr = f"CASE WHEN qv = 1 THEN xv WHEN {consecutive} THEN xv - pv ELSE NULL END"
        return _Layer(f"SELECT ts, inst, {expr} AS _v FROM ({joined}) fq", has_inst_window=True)

    if op == "ttm_from_cumulative":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        quarterly = PlanNode(op="quarter_from_cumulative", inputs=list(node.inputs[:3]), attrs={})
        return _compile_layer(
            PlanNode(op="ttm_from_quarterly", inputs=[quarterly, node.inputs[1]], attrs={"periods": 4}),
            dialect=dialect,
        )

    if op == "ts_regression_tstat":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 2, 3)))
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 3, None))
        min_periods = window if raw_mp is None else int(raw_mp)
        add_intercept = bool(node.attrs.get("add_intercept", _raw_literal(node, 4, True)))
        return _Layer(
            _duckdb_rolling_tstat_sql(
                y.sql,
                x.sql,
                window=window,
                min_periods=min_periods,
                add_intercept=add_intercept,
            ),
            has_inst_window=True,
        )

    if op == "ts_trend_tstat":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 2, None))
        min_periods = window if raw_mp is None else int(raw_mp)
        time_sql = (
            f"SELECT ts, inst, _v, "
            f"CAST(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS DOUBLE) AS _time "
            f"FROM ({inner.sql}) tr0"
        )
        y_sql = f"SELECT ts, inst, _v FROM ({time_sql}) tr1"
        x_sql = f"SELECT ts, inst, _time AS _v FROM ({time_sql}) tr2"
        return _Layer(
            _duckdb_rolling_tstat_sql(
                y_sql,
                x_sql,
                window=window,
                min_periods=min_periods,
                add_intercept=True,
            ),
            has_inst_window=True,
        )

    if op == "ts_max_drawdown":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
        min_periods = int(node.attrs.get("min_periods", _raw_literal(node, 2, 2)))
        over = "PARTITION BY inst ORDER BY ts"
        lags = ["_v"] + [f"LAG(_v, {i}) OVER ({over})" for i in range(1, window)]
        values = f"list_value({', '.join(lags)})"
        valid_list = f"list_filter({values}, v -> v IS NOT NULL AND NOT isnan(v) AND NOT isinf(v))"
        drawdowns = ["0.0"]
        for i, value_i in enumerate(lags):
            earlier = f"list_max(list_value({', '.join(lags[i:])}))"
            drawdowns.append(
                f"CASE WHEN {value_i} IS NULL OR {earlier} IS NULL THEN NULL "
                f"ELSE {value_i} / {earlier} - 1.0 END"
            )
        return _Layer(
            f"SELECT ts, inst, CASE WHEN list_count({valid_list}) < {min_periods} "
            f"OR list_min({valid_list}) <= 0 THEN NULL "
            f"ELSE LEAST({', '.join(drawdowns)}) END AS _v FROM ({inner.sql}) md",
            has_inst_window=True,
        )

    if op == "ts_partial_corr":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        y = _compile_layer(node.inputs[1], dialect=dialect)
        z = _compile_layer(node.inputs[2], dialect=dialect)
        if x is None or y is None or z is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 3, 3)))
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 4, None))
        min_periods = window if raw_mp is None else int(raw_mp)
        over = (
            f"PARTITION BY x.inst ORDER BY x.ts "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        valid = (
            f"({_duckdb_valid('x._v')} AND {_duckdb_valid('y._v')} "
            f"AND {_duckdb_valid('z._v')})"
        )
        xv = f"CASE WHEN {valid} THEN x._v END"
        yv = f"CASE WHEN {valid} THEN y._v END"
        zv = f"CASE WHEN {valid} THEN z._v END"
        n = f"COUNT({xv}) OVER ({over})"
        rxy = f"corr({xv}, {yv}) OVER ({over})"
        rxz = f"corr({xv}, {zv}) OVER ({over})"
        ryz = f"corr({yv}, {zv}) OVER ({over})"
        denom = f"SQRT(GREATEST(0.0, 1 - ({rxz}) * ({rxz})) * GREATEST(0.0, 1 - ({ryz}) * ({ryz})))"
        return _Layer(
            f"SELECT x.ts, x.inst, CASE WHEN {n} < {min_periods} OR ({denom}) <= 0 "
            f"THEN NULL ELSE (({rxy}) - ({rxz}) * ({ryz})) / ({denom}) END AS _v "
            f"FROM ({x.sql}) x LEFT JOIN ({y.sql}) y USING (ts, inst) "
            f"LEFT JOIN ({z.sql}) z USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_nth_value":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
        nth = int(node.attrs.get("n", _raw_literal(node, 2, 1)))
        order = str(node.attrs.get("order", _raw_literal(node, 3, "largest"))).lower()
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 4, None))
        min_periods = nth if raw_mp is None else int(raw_mp)
        if order not in {"largest", "smallest"}:
            return None
        over = "PARTITION BY inst ORDER BY ts"
        lags = ["_v"] + [f"LAG(_v, {i}) OVER ({over})" for i in range(1, window)]
        values = f"list_filter(list_value({', '.join(lags)}), v -> v IS NOT NULL AND NOT isnan(v) AND NOT isinf(v))"
        sorted_values = (
            f"list_reverse_sort({values})"
            if order == "largest"
            else f"list_sort({values})"
        )
        required = max(min_periods, nth)
        return _Layer(
            f"SELECT ts, inst, CASE WHEN list_count({values}) < {required} THEN NULL "
            f"ELSE list_extract({sorted_values}, {nth}) END AS _v FROM ({inner.sql}) nth0",
            has_inst_window=True,
        )

    if op == "ts_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "AVG", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        )
        std_key = "stddev_pop" if spec.ddof == 0 else "stddev"
        expr = _rolling_std_min_periods_sql(
            value_col="_v",
            over=over,
            window=spec.size,
            scale_expr="1",
            dialect=dialect,
            min_periods=spec.min_periods,
            std_fn_name=std_key,
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "SUM", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "MAX", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "MIN", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    # ========== NEW WINDOW FUNCTION OPERATORS (50+) ==========
    # Simple SQL window function mappings for common operations

    if op == "ts_lag":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_lead":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lead = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, LEAD(_v, {lead}) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_first_value":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, FIRST_VALUE(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_last_value":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, LAST_VALUE(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_nth_value":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        n = _int_attr(node, "n", default=1)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, NTH_VALUE(_v, {n}) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_row_number":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_dense_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY _v ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, DENSE_RANK() OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_percent_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY _v ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, PERCENT_RANK() OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_cumsum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, SUM(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_cumprod":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        prod_fn = "PRODUCT" if dialect == SqlDialect.DUCKDB else "product"
        return _Layer(
            f"SELECT ts, inst, {prod_fn}(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_cummax":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, MAX(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_cummin":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, MIN(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_cumcount":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, COUNT(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, COUNT(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_avg":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "AVG", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_variance":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        var_fn = "VAR_POP" if spec.ddof == 0 else "VAR_SAMP"
        return _Layer(
            f"SELECT ts, inst, {var_fn}(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_stddev":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        std_fn = "STDDEV_POP" if spec.ddof == 0 else "STDDEV_SAMP"
        return _Layer(
            f"SELECT ts, inst, {std_fn}(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # Cross-sectional window functions
    if op == "cs_row_number":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, ROW_NUMBER() OVER (PARTITION BY ts ORDER BY _v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_dense_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, DENSE_RANK() OVER (PARTITION BY ts ORDER BY _v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_percent_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, PERCENT_RANK() OVER (PARTITION BY ts ORDER BY _v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_first_value":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, FIRST_VALUE(_v) OVER (PARTITION BY ts ORDER BY _v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_last_value":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, LAST_VALUE(_v) OVER (PARTITION BY ts ORDER BY _v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # Elementwise window aggregations (expanding windows)
    if op == "expanding_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, MIN(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "expanding_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, MAX(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "expanding_var":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        ddof = _int_attr(node, "ddof", default=1)
        var_fn = "VAR_POP" if ddof == 0 else "VAR_SAMP"
        return _Layer(
            f"SELECT ts, inst, {var_fn}(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "expanding_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, COUNT(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "expanding_product":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        prod_fn = "PRODUCT" if dialect == SqlDialect.DUCKDB else "product"
        return _Layer(
            f"SELECT ts, inst, {prod_fn}(_v) OVER (PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # Additional simple mappings
    if op == "ts_kurtosis":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, KURTOSIS(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_skewness":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, SKEWNESS(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_range":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, (MAX(_v) OVER ({over}) - MIN(_v) OVER ({over})) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_midpoint":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, (MAX(_v) OVER ({over}) + MIN(_v) OVER ({over})) / 2.0 AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # Additional aggregation variants
    if op == "ts_sum_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, SUM(ABS(_v)) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_mean_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, AVG(ABS(_v)) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_abs_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, MAX(ABS(_v)) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_positive_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, COUNT(CASE WHEN _v > 0 THEN 1 END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_negative_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, COUNT(CASE WHEN _v < 0 THEN 1 END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_zero_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, COUNT(CASE WHEN _v = 0 THEN 1 END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_positive_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, SUM(CASE WHEN _v > 0 THEN _v ELSE 0 END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_negative_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, SUM(CASE WHEN _v < 0 THEN _v ELSE 0 END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_positive_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, AVG(CASE WHEN _v > 0 THEN _v END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_negative_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, AVG(CASE WHEN _v < 0 THEN _v END) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # Cross-sectional aggregations (simple) — finite sample (R19-027).
    if op in {"cs_min", "cs_max", "cs_range"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.stat_valid import row_stat_invalid_sql

        invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
        cleaned = (
            f"SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS _v "
            f"FROM ({inner.sql}) t_clean"
        )
        if op == "cs_min":
            expr = "MIN(_v) OVER (PARTITION BY ts)"
        elif op == "cs_max":
            expr = "MAX(_v) OVER (PARTITION BY ts)"
        else:
            expr = (
                "(MAX(_v) OVER (PARTITION BY ts) - MIN(_v) OVER (PARTITION BY ts))"
            )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({cleaned}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_sum_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, SUM(ABS(_v)) OVER (PARTITION BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_mean_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, AVG(ABS(_v)) OVER (PARTITION BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_variance":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        ddof = _int_attr(node, "ddof", default=1)
        var_fn = "VAR_POP" if ddof == 0 else "VAR_SAMP"
        return _Layer(
            f"SELECT ts, inst, {var_fn}(_v) OVER (PARTITION BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_stddev":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        ddof = _int_attr(node, "ddof", default=1)
        std_fn = "STDDEV_POP" if ddof == 0 else "STDDEV_SAMP"
        return _Layer(
            f"SELECT ts, inst, {std_fn}(_v) OVER (PARTITION BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_skewness":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, SKEWNESS(_v) OVER (PARTITION BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_kurtosis":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, KURTOSIS(_v) OVER (PARTITION BY ts) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # ========== END NEW WINDOW OPERATORS ==========

    if op == "ts_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        )
        std_key = "stddev_pop" if spec.ddof == 0 else "stddev"
        expr = _zscore_window_expr(
            value_col="_v",
            partition=over,
            dialect=dialect,
            std_fn_name=std_key,
            min_periods=spec.min_periods,
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_sharpe":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        ann = _float_attr(node, "ann_factor", default=252.0)
        sqrt_af = ann**0.5
        mp = max(2, w // 3)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        cnt = f"COUNT(_v) OVER ({over})"
        std = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE "
            f"WHEN {cnt} < {mp} THEN NULL "
            f"WHEN {std}(_v) OVER ({over}) = 0 THEN NULL "
            f"ELSE AVG(_v) OVER ({over}) / {nf}({std}(_v) OVER ({over}), 0) "
            f"END * {sqrt_af} AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op in {"ts_delay", "delay"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_delta":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_pct":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v / {nf}(LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_rank_01_sql(inner.sql, partition="PARTITION BY ts", dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"rank_pct", "cs_pct_rank"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_pct_rank_sql(inner.sql, partition="PARTITION BY ts", dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"cs_quantile", "c_percentile"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_positional(node, 0)
        if pos_p is not None:
            p = pos_p
        qexpr = _quantile_over(dialect, "_v", p, "PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, {qexpr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"log_returns", "ts_log_return"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        d = _window_int(node, default=1)
        lag = f"LAG(_v, {d}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR {lag} IS NULL OR _v <= 0 OR {lag} <= 0 THEN NULL "
            f"ELSE {ln}(_v / {lag}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "volatility":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _rolling_min_periods(w)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        sqrt_fn = _dialect_fn(dialect, "sqrt")
        scale = f"{sqrt_fn}(252)"
        expr = _rolling_std_min_periods_sql(
            value_col="_v",
            over=over,
            window=w,
            scale_expr=scale,
            dialect=dialect,
            min_periods=mp,
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "vwap":
        if len(node.inputs) < 2:
            return None
        price = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if price is None or vol is None:
            return None
        from backend.pairwise_rolling import sql_pairwise_vwap_expr

        spec = _window_spec(node, default=20)
        over = (
            f"PARTITION BY p.inst ORDER BY p.ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        )
        vwap_expr = sql_pairwise_vwap_expr(
            price_col="p._v",
            volume_col="v._v",
            over=over,
            min_periods=spec.min_periods,
        )
        return _Layer(
            f"SELECT p.ts, p.inst, {vwap_expr} AS _v "
            f"FROM ({price.sql}) p LEFT JOIN ({vol.sql}) v USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"zscore", "standardize"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        expr = _zscore_window_expr(value_col="_v", partition="PARTITION BY ts", dialect=dialect)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        expr = _normalize_window_expr(value_col="_v", partition="PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_demean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - AVG(_v) OVER (PARTITION BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"AVG(_v) OVER (PARTITION BY ts) AS mean_val, "
            f"{std_fn}(_v) OVER (PARTITION BY ts) AS std_val "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN std_val IS NULL OR std_val = 0 THEN 0 "
            f"ELSE (_v - mean_val) / {nf}(std_val, 0) END AS _v "
            f"FROM stats",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"MIN(_v) OVER (PARTITION BY ts) AS min_val, "
            f"MAX(_v) OVER (PARTITION BY ts) AS max_val "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN max_val = min_val THEN 0.5 "
            f"ELSE (_v - min_val) / {nf}(max_val - min_val, 0) END AS _v "
            f"FROM stats",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_median":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        med_fn = "median"
        return _Layer(
            f"SELECT ts, inst, "
            f"{med_fn}(_v) OVER (PARTITION BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_iqr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if dialect == SqlDialect.CLICKHOUSE:
            q25 = "quantile(0.25)(_v)"
            q75 = "quantile(0.75)(_v)"
        else:
            q25 = "percentile_cont(0.25) WITHIN GROUP (ORDER BY _v)"
            q75 = "percentile_cont(0.75) WITHIN GROUP (ORDER BY _v)"
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"{q25} OVER (PARTITION BY ts) AS q25, "
            f"{q75} OVER (PARTITION BY ts) AS q75 "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, (q75 - q25) AS _v FROM stats",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_clip":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lower = node.attrs.get("lower")
        upper = node.attrs.get("upper")
        lo_fn = _dialect_fn(dialect, "least")
        hi_fn = _dialect_fn(dialect, "greatest")
        if lower is not None and upper is not None:
            expr = f"{lo_fn}({upper}, {hi_fn}({lower}, _v))"
        elif lower is not None:
            expr = f"{hi_fn}({lower}, _v)"
        elif upper is not None:
            expr = f"{lo_fn}({upper}, _v)"
        else:
            expr = "_v"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL ELSE {expr} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "unitize":
        # pandas：按行（截面）max(|x|) 归一化；max(|x|)=0 → NaN；clip(-1,1)。
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        m = f"MAX({abs_fn}(_v)) OVER (PARTITION BY ts)"
        lo = _dialect_fn(dialect, "least")
        hi = _dialect_fn(dialect, "greatest")
        expr = (
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN {m} IS NULL OR {m} = 0 THEN NULL "
            f"ELSE {hi}(-1.0, {lo}(1.0, _v / {m})) END"
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "index_weight":
        # Index constituent weight; normalize=True divides by the cross-section
        # total within each date (nansum semantics; SQL SUM ignores NULLs).
        # Guard matches the pandas reference: a zero/near-zero total yields NaN.
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        raw = node.attrs.get("normalize", True)
        if raw is None:
            raw = True
        if isinstance(raw, str):
            raw = raw.strip().lower() in {"1", "true", "yes", "on"}
        if not bool(raw):
            return _Layer(
                inner.sql,
                has_inst_window=inner.has_inst_window,
                has_ts_partition=inner.has_ts_partition,
            )
        denom = "SUM(_v) OVER (PARTITION BY ts)"
        expr = (
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN {denom} IS NULL OR ABS({denom}) <= 1e-12 THEN NULL "
            f"ELSE _v / {denom} END"
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_mad":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_mad_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_mad_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_mad_zscore_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    cs_aggregates = {
        "c_mean": "AVG", "cs_mean": "AVG",
        "c_sum": "SUM", "cs_sum": "SUM",
        "c_count": "COUNT", "cs_count": "COUNT",
        "c_std": _dialect_fn(dialect, "stddev"),
        "cs_std": _dialect_fn(dialect, "stddev"),
    }
    if op in cs_aggregates:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.stat_valid import row_stat_invalid_sql

        invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
        cleaned = (
            f"SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS _v "
            f"FROM ({inner.sql}) t_clean"
        )
        agg = cs_aggregates[op]
        return _Layer(
            _cs_broadcast_agg(agg, cleaned, all_null_null=agg != "COUNT"),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"cs_resid", "cs_regression"}:
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        x_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if y_layer is None or x_layer is None:
            return None
        part = "PARTITION BY y.ts"
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="y._v", x_col="x._v", partition=part
        )
        fit = f"({alpha}) + ({beta}) * x._v"
        from backend.plan_params import int_mode_from_plan_node

        mode = 0 if op == "cs_resid" else int_mode_from_plan_node(node, input_index=2, default=0)
        if mode == 1:
            core = beta
        elif mode == 2:
            core = fit
        else:
            core = f"y._v - ({fit})"
        expr = (
            f"CASE WHEN y._v IS NULL OR x._v IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE {core} END"
        )
        return _Layer(
            f"SELECT y.ts, y.inst, {expr} AS _v "
            f"FROM ({y_layer.sql}) y LEFT JOIN ({x_layer.sql}) x USING (ts, inst)",
            has_inst_window=y_layer.has_inst_window or x_layer.has_inst_window,
            has_ts_partition=True,
        )

    if op == "size_neutralize":
        # size_neutralize(x, market_cap) == cs_resid(x, ln(market_cap))
        # Legal caps only (R19-024): finite & >0 → ln; else NULL (no silent clip).
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        cap_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if y_layer is None or cap_layer is None:
            return None
        ln = _dialect_fn(dialect, "ln")
        ln_cap = (
            f"CASE WHEN c._v IS NOT NULL AND c._v > 0 THEN {ln}(c._v) ELSE NULL END"
        )
        part = "PARTITION BY y.ts"
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="y._v", x_col=ln_cap, partition=part
        )
        fit = f"({alpha}) + ({beta}) * ({ln_cap})"
        expr = (
            f"CASE WHEN y._v IS NULL OR ({ln_cap}) IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE y._v - ({fit}) END"
        )
        return _Layer(
            f"SELECT y.ts, y.inst, {expr} AS _v "
            f"FROM ({y_layer.sql}) y LEFT JOIN ({cap_layer.sql}) c USING (ts, inst)",
            has_inst_window=y_layer.has_inst_window or cap_layer.has_inst_window,
            has_ts_partition=True,
        )

    if op == "industry_size_neutralize":
        # FWL (R19-023): demean y and log(size) within industry, then residual.
        # Legal caps only (R19-024): finite & >0 → ln; else NULL (no silent clip).
        if len(node.inputs) < 3:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        ind_layer = _compile_layer(node.inputs[1], dialect=dialect)
        cap_layer = _compile_layer(node.inputs[2], dialect=dialect)
        if y_layer is None or ind_layer is None or cap_layer is None:
            return None
        ln = _dialect_fn(dialect, "ln")
        ln_cap = (
            f"CASE WHEN c._v IS NOT NULL AND c._v > 0 THEN {ln}(c._v) ELSE NULL END"
        )
        demeaned_sql = (
            f"SELECT x.ts AS ts, x.inst AS inst, "
            f"(x._v - AVG(x._v) OVER (PARTITION BY x.ts, g._v)) AS dm, "
            f"(({ln_cap}) - AVG({ln_cap}) OVER (PARTITION BY x.ts, g._v)) AS ln_dm "
            f"FROM ({y_layer.sql}) x "
            f"LEFT JOIN ({ind_layer.sql}) g USING (ts, inst) "
            f"LEFT JOIN ({cap_layer.sql}) c USING (ts, inst)"
        )
        part = "PARTITION BY d.ts"
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="d.dm", x_col="d.ln_dm", partition=part
        )
        fit = f"({alpha}) + ({beta}) * d.ln_dm"
        expr = (
            f"CASE WHEN d.dm IS NULL OR d.ln_dm IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE d.dm - ({fit}) END"
        )
        return _Layer(
            f"SELECT d.ts, d.inst, {expr} AS _v FROM ({demeaned_sql}) d",
            has_inst_window=(
                y_layer.has_inst_window
                or ind_layer.has_inst_window
                or cap_layer.has_inst_window
            ),
            has_ts_partition=True,
        )

    if op == "scale":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        to_val = _float_attr(node, "to", default=1.0)
        if "to" not in (node.attrs or {}):
            for idx in range(1, len(node.inputs)):
                child = node.inputs[idx]
                if child.op == "literal":
                    val = child.attrs.get("value")
                    if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                        to_val = float(val)
                        break
        expr = _scale_window_expr(
            value_col="_v", partition="PARTITION BY ts", to_val=to_val, dialect=dialect
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_neutralize":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            return _Layer(
                f"SELECT x.ts, x.inst, (x._v - AVG(x._v) OVER (PARTITION BY x.ts, g._v)) AS _v "
                f"FROM ({inner.sql}) x "
                f"LEFT JOIN ({grp.sql}) g USING (ts, inst)",
                has_inst_window=inner.has_inst_window or grp.has_inst_window,
                has_ts_partition=True,
            )
        return _Layer(
            f"SELECT ts, inst, (_v - AVG(_v) OVER (PARTITION BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_ex_self_mean":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH agg AS ("
            f"SELECT x.ts, g._v AS grp, "
            f"SUM(x._v) AS grp_sum, "
            f"COUNT(x._v) AS grp_cnt "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst) "
            f"GROUP BY x.ts, g._v"
            f") "
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN x._v IS NULL THEN NULL "
            f"ELSE (a.grp_sum - x._v) / {nf}(a.grp_cnt - 1, 0) END AS _v "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst) "
            f"LEFT JOIN agg a ON x.ts = a.ts AND g._v = a.grp",
            has_inst_window=val.has_inst_window or grp.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_ex_self_std":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            f"WITH grp_stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"{std_fn}(x._v) OVER (PARTITION BY x.ts, g._v) AS grp_std "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, grp_std AS _v FROM grp_stats",
            has_inst_window=val.has_inst_window or grp.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_ex_self_quantile":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        q = _float_attr(node, "q", default=0.5)
        if dialect == SqlDialect.CLICKHOUSE:
            qfn = f"quantile({q})(x._v)"
        else:
            qfn = f"percentile_cont({q}) WITHIN GROUP (ORDER BY x._v)"
        return _Layer(
            f"WITH grp_stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, "
            f"{qfn} OVER (PARTITION BY x.ts, g._v) AS grp_q "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, grp_q AS _v FROM grp_stats",
            has_inst_window=val.has_inst_window or grp.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_impute_median":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        med_fn = "median"
        return _Layer(
            f"WITH grp_med AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"{med_fn}(x._v) OVER (PARTITION BY x.ts, g._v) AS med "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, COALESCE(val, med) AS _v FROM grp_med",
            has_inst_window=val.has_inst_window or grp.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_corr":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.pair_window_spec import PairWindowSpec
        from backend.stat_valid import stat_valid_sql

        pspec = PairWindowSpec.from_plan_node(node, default_min_periods=2)
        w = pspec.size
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        corr_fn = "corr" if dialect == SqlDialect.DUCKDB else "corrStable"
        # Finite-pair mask (R19-027): ±Inf excluded like pandas rolling.corr after
        # masking non-finite inputs. Current-row may still be NULL (R19-030).
        l_ok = stat_valid_sql("l._v", dialect=dialect, exclude_nan=True)
        r_ok = stat_valid_sql("r._v", dialect=dialect, exclude_nan=True)
        pair = f"({l_ok} AND {r_ok})"
        pair_l = f"CASE WHEN {pair} THEN l._v END"
        pair_r = f"CASE WHEN {pair} THEN r._v END"
        cnt = f"COUNT(CASE WHEN {pair} THEN 1 END) OVER ({over})"
        body = f"{corr_fn}({pair_l}, {pair_r}) OVER ({over})"
        expr = (
            body
            if pspec.min_periods <= 2
            else f"CASE WHEN {cnt} < {pspec.min_periods} THEN NULL ELSE {body} END"
        )
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "abs_return_volume_corr":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        corr_fn = "corr" if dialect == SqlDialect.DUCKDB else "corrStable"
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"ELSE {corr_fn}({abs_fn}(l._v), r._v) OVER ({over}) END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_autocorr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag_pos = _literal_positional(node, 1, default=1.0)
        lag = max(int(lag_pos or 1), 1)
        w = max(lag + 2, _window_int(node))
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        corr_expr = _rolling_corr_pandas_compat_expr(
            corr_col="_corr",
            std_left_col="_std_l",
            std_right_col="_std_r",
            left_col="curr_v",
            right_col="lagged_v",
            window_count_col="_win_cnt",
            window=w,
            dialect=dialect,
        )
        return _Layer(
            f"SELECT ts, inst, {corr_expr} AS _v "
            f"FROM ("
            f"SELECT ts, inst, curr_v, lagged_v, "
            f"COUNT(*) OVER ({over}) AS _win_cnt, "
            f"{'corr' if dialect == SqlDialect.DUCKDB else 'corrStable'}(curr_v, lagged_v) OVER ({over}) AS _corr, "
            f"{'stddev_samp' if dialect == SqlDialect.DUCKDB else 'stddevSamp'}(curr_v) OVER ({over}) AS _std_l, "
            f"{'stddev_samp' if dialect == SqlDialect.DUCKDB else 'stddevSamp'}(lagged_v) OVER ({over}) AS _std_r "
            f"FROM ("
            f"SELECT ts, inst, _v AS curr_v, "
            f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS lagged_v "
            f"FROM ({inner.sql}) inner0"
            f") aligned"
            f") scored",
            has_inst_window=True,
        )

    if op == "ts_covariance":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.pair_window_spec import PairWindowSpec
        from backend.stat_valid import stat_valid_sql

        pspec = PairWindowSpec.from_plan_node(node, default_min_periods=2)
        w = pspec.size
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        cov_fn = "covar_samp" if dialect == SqlDialect.DUCKDB else "covarSamp"
        l_ok = stat_valid_sql("l._v", dialect=dialect, exclude_nan=True)
        r_ok = stat_valid_sql("r._v", dialect=dialect, exclude_nan=True)
        pair = f"({l_ok} AND {r_ok})"
        pair_l = f"CASE WHEN {pair} THEN l._v END"
        pair_r = f"CASE WHEN {pair} THEN r._v END"
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"{cov_fn}({pair_l}, {pair_r}) OVER ({over}) AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_beta":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.pair_window_spec import PairWindowSpec
        from backend.pairwise_rolling import sql_pairwise_beta_expr

        pspec = PairWindowSpec.from_plan_node(node, default_min_periods=5)
        w = pspec.size
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        beta_expr = sql_pairwise_beta_expr(
            left_col="l._v",
            right_col="r._v",
            over=over,
            dialect_is_duckdb=dialect == SqlDialect.DUCKDB,
            min_periods=pspec.min_periods,
        )
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL ELSE ({beta_expr}) END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "cs_trim_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        pct = _float_attr(node, "proportiontocut", default=0.1)
        lower_pct = pct
        upper_pct = 1.0 - pct
        if dialect == SqlDialect.CLICKHOUSE:
            lower_q = f"quantile({lower_pct})(_v)"
            upper_q = f"quantile({upper_pct})(_v)"
        else:
            lower_q = f"percentile_cont({lower_pct}) WITHIN GROUP (ORDER BY _v)"
            upper_q = f"percentile_cont({upper_pct}) WITHIN GROUP (ORDER BY _v)"
        return _Layer(
            f"WITH bounds AS ("
            f"SELECT ts, inst, _v, "
            f"{lower_q} OVER (PARTITION BY ts) AS lower_b, "
            f"{upper_q} OVER (PARTITION BY ts) AS upper_b "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"AVG(CASE WHEN _v >= lower_b AND _v <= upper_b THEN _v ELSE NULL END) "
            f"OVER (PARTITION BY ts) AS _v "
            f"FROM bounds",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_robust_scale":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        med_fn = "median"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"{med_fn}(_v) OVER (PARTITION BY ts) AS med, "
            f"{med_fn}(ABS(_v - {med_fn}(_v) OVER (PARTITION BY ts))) "
            f"OVER (PARTITION BY ts) AS mad "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN mad IS NULL OR mad = 0 THEN 0 "
            f"ELSE (_v - med) / {nf}(mad, 0) END AS _v "
            f"FROM stats",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_rank_gaussian":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"WITH ranks AS ("
            f"SELECT ts, inst, _v, "
            f"ROW_NUMBER() OVER (PARTITION BY ts ORDER BY _v NULLS LAST) AS rn, "
            f"COUNT(*) OVER (PARTITION BY ts) AS n "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"ELSE (rn - 0.5) / n END AS _v "
            f"FROM ranks",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "RSI_WILDER":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(_window_int(node, default=14), 2)
        return _Layer(
            _rsi_wilder_sql(inner.sql, w, dialect=dialect),
            has_inst_window=True,
        )

    if op == "ATR_WILDER":
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or close is None:
            return None
        w = max(_window_int(node, default=14), 2)
        return _Layer(
            _atr_wilder_sql(high.sql, low.sql, close.sql, w, dialect=dialect),
            has_inst_window=True,
        )

    if op == "NATR":
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or close is None:
            return None
        w = max(_window_int(node, default=14), 2)
        atr_sql = _atr_wilder_sql(high.sql, low.sql, close.sql, w, dialect=dialect)
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT a.ts, a.inst, "
            f"CASE WHEN c._v IS NULL OR {abs_fn}(c._v) <= 0 THEN NULL "
            f"ELSE 100.0 * a._v / {abs_fn}(c._v) END AS _v "
            f"FROM ({atr_sql}) a LEFT JOIN ({close.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "MACD_line":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fast = int(_literal_positional(node, 0, default=12) or 12)
        slow = int(_literal_positional(node, 1, default=26) or 26)
        fast_sql = _ema_span_over_inst(inner.sql, fast, dialect=dialect)
        slow_sql = _ema_span_over_inst(inner.sql, slow, dialect=dialect)
        return _Layer(
            f"SELECT f.ts, f.inst, (f._v - s._v) AS _v "
            f"FROM ({fast_sql}) f LEFT JOIN ({slow_sql}) s USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "MACD_signal":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fast = int(_literal_positional(node, 0, default=12) or 12)
        slow = int(_literal_positional(node, 1, default=26) or 26)
        signal = int(_literal_positional(node, 2, default=9) or 9)
        macd_sql = (
            f"SELECT ts, inst, (f._v - s._v) AS _v "
            f"FROM ({_ema_span_over_inst(inner.sql, fast, dialect=dialect)}) f "
            f"LEFT JOIN ({_ema_span_over_inst(inner.sql, slow, dialect=dialect)}) s USING (ts, inst)"
        )
        return _Layer(
            _ema_span_over_inst(macd_sql, signal, dialect=dialect),
            has_inst_window=True,
        )

    if op == "MACD_hist":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fast = int(_literal_positional(node, 0, default=12) or 12)
        slow = int(_literal_positional(node, 1, default=26) or 26)
        signal = int(_literal_positional(node, 2, default=9) or 9)
        macd_sql = (
            f"SELECT ts, inst, (f._v - s._v) AS _v "
            f"FROM ({_ema_span_over_inst(inner.sql, fast, dialect=dialect)}) f "
            f"LEFT JOIN ({_ema_span_over_inst(inner.sql, slow, dialect=dialect)}) s USING (ts, inst)"
        )
        sig_sql = _ema_span_over_inst(macd_sql, signal, dialect=dialect)
        return _Layer(
            f"SELECT m.ts, m.inst, (m._v - g._v) AS _v "
            f"FROM ({macd_sql}) m LEFT JOIN ({sig_sql}) g USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"DEMA", "TEMA"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=int(_literal_positional(node, 0, default=20) or 20))
        w = max(int(w), 2)
        pos_w = _literal_positional(node, 0)
        if pos_w is not None:
            w = max(int(pos_w), 2)
        e1 = _ema_span_over_inst(inner.sql, w, dialect=dialect, min_periods=w)
        e2 = _ema_span_over_inst(e1, w, dialect=dialect, min_periods=w)
        if op == "DEMA":
            return _Layer(
                f"SELECT ts, inst, (2.0 * e1._v - e2._v) AS _v "
                f"FROM ({e1}) e1 JOIN ({e2}) e2 USING (ts, inst)",
                has_inst_window=True,
            )
        e3 = _ema_span_over_inst(e2, w, dialect=dialect, min_periods=w)
        return _Layer(
            f"SELECT ts, inst, (3.0 * e1._v - 3.0 * e2._v + e3._v) AS _v "
            f"FROM ({e1}) e1 JOIN ({e2}) e2 USING (ts, inst) "
            f"JOIN ({e3}) e3 USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"PPO", "PPO_signal", "PPO_hist", "PVO", "PVO_signal", "PVO_hist"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fast = int(_literal_positional(node, 0, default=12) or 12)
        slow = int(_literal_positional(node, 1, default=26) or 26)
        fast = max(int(fast), 2)
        slow = max(int(slow), 2)
        ef = _ema_span_over_inst(inner.sql, fast, dialect=dialect, min_periods=fast)
        es = _ema_span_over_inst(inner.sql, slow, dialect=dialect, min_periods=slow)
        osc = (
            f"SELECT f.ts, f.inst, (100.0 * (f._v - s._v)) / NULLIF(s._v, 0) AS _v "
            f"FROM ({ef}) f JOIN ({es}) s USING (ts, inst)"
        )
        if op in {"PPO", "PVO"}:
            return _Layer(osc, has_inst_window=True)
        signal = int(_literal_positional(node, 2, default=9) or 9)
        signal = max(int(signal), 2)
        sig = _ema_span_over_inst(osc, signal, dialect=dialect, min_periods=signal)
        if op in {"PPO_signal", "PVO_signal"}:
            return _Layer(sig, has_inst_window=True)
        return _Layer(
            f"SELECT o.ts, o.inst, (o._v - g._v) AS _v "
            f"FROM ({osc}) o LEFT JOIN ({sig}) g USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"TSI", "TSI_signal"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        long_w = max(int(_literal_positional(node, 0, default=25) or 25), 2)
        short_w = max(int(_literal_positional(node, 1, default=13) or 13), 2)
        m = (
            f"SELECT ts, inst, "
            f"(_v - LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts)) AS _v "
            f"FROM ({inner.sql}) t"
        )
        num = _ema_span_over_inst(
            _ema_span_over_inst(m, long_w, dialect=dialect, min_periods=long_w), short_w, dialect=dialect, min_periods=short_w
        )
        abs_m = f"SELECT ts, inst, ABS(_v) AS _v FROM ({m}) t"
        den = _ema_span_over_inst(
            _ema_span_over_inst(abs_m, long_w, dialect=dialect, min_periods=long_w), short_w, dialect=dialect, min_periods=short_w
        )
        tsi = (
            f"SELECT n.ts, n.inst, (100.0 * n._v) / NULLIF(d._v, 0) AS _v "
            f"FROM ({num}) n JOIN ({den}) d USING (ts, inst)"
        )
        if op == "TSI":
            return _Layer(tsi, has_inst_window=True)
        signal = max(int(_literal_positional(node, 2, default=9) or 9), 2)
        return _Layer(_ema_span_over_inst(tsi, signal, dialect=dialect, min_periods=signal), has_inst_window=True)

    if op == "CMO":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=14) or 14), 2)
        d = (
            f"SELECT ts, inst, "
            f"(_v - LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts)) AS _v "
            f"FROM ({inner.sql}) t"
        )
        up = _inst_window(dialect, w, "SUM", f"SELECT ts, inst, GREATEST(_v, 0) AS _v FROM ({d}) t", min_periods=w)
        dn = _inst_window(dialect, w, "SUM", f"SELECT ts, inst, GREATEST(-_v, 0) AS _v FROM ({d}) t", min_periods=w)
        return _Layer(
            f"SELECT u.ts, u.inst, (100.0 * (u._v - n._v)) / NULLIF(u._v + n._v, 0) AS _v "
            f"FROM ({up}) u JOIN ({dn}) n USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"VortexPlus", "VortexMinus"}:
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or close is None:
            return None
        w = max(int(_literal_positional(node, 2, default=14) or 14), 2)
        g = _dialect_fn(dialect, "greatest")
        abs_fn = _dialect_fn(dialect, "abs")
        lag_close = f"LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {g}(h._v - l._v, {abs_fn}(h._v - {lag_close}), {abs_fn}(l._v - {lag_close})) END"
        )
        tr_sql = (
            f"SELECT h.ts, h.inst, {tr_expr} AS _v "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({close.sql}) c USING (ts, inst)"
        )
        tr_sum = _inst_window(dialect, w, "SUM", tr_sql, min_periods=w)
        if op == "VortexPlus":
            vm = (
                f"SELECT h.ts, h.inst, {abs_fn}(h._v - "
                f"LAG(l._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)) AS _v "
                f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst)"
            )
        else:
            vm = (
                f"SELECT l.ts, l.inst, {abs_fn}(l._v - "
                f"LAG(h._v, 1) OVER (PARTITION BY l.inst ORDER BY l.ts)) AS _v "
                f"FROM ({low.sql}) l INNER JOIN ({high.sql}) h USING (ts, inst)"
            )
        vm_sum = _inst_window(dialect, w, "SUM", vm, min_periods=w)
        return _Layer(
            f"SELECT v.ts, v.inst, v._v / NULLIF(t._v, 0) AS _v "
            f"FROM ({vm_sum}) v JOIN ({tr_sum}) t USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"donchian_upper", "donchian_lower"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 1)
        # pandas: x.shift(1).rolling(w).max/min() —— 先 shift 再滚动，排除当前 bar。
        prev = f"SELECT ts, inst, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({inner.sql}) t"
        agg = "MAX" if op == "donchian_upper" else "MIN"
        return _Layer(_inst_window(dialect, w, agg, prev, min_periods=w), has_inst_window=True)

    if op == "donchian_mid":
        if len(node.inputs) < 2:
            return None
        h = _compile_layer(node.inputs[0], dialect=dialect)
        l = _compile_layer(node.inputs[1], dialect=dialect)
        if h is None or l is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        hp = f"SELECT ts, inst, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({h.sql}) t"
        lp = f"SELECT ts, inst, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({l.sql}) t"
        up = _inst_window(dialect, w, "MAX", hp, min_periods=w)
        lo = _inst_window(dialect, w, "MIN", lp, min_periods=w)
        return _Layer(
            f"SELECT u.ts, u.inst, ((u._v + o._v) / 2.0) AS _v "
            f"FROM ({up}) u JOIN ({lo}) o USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "donchian_position":
        if len(node.inputs) < 3:
            return None
        c = _compile_layer(node.inputs[0], dialect=dialect)
        h = _compile_layer(node.inputs[1], dialect=dialect)
        l = _compile_layer(node.inputs[2], dialect=dialect)
        if c is None or h is None or l is None:
            return None
        w = max(int(_literal_positional(node, 2, default=20) or 20), 1)
        hp = f"SELECT ts, inst, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({h.sql}) t"
        lp = f"SELECT ts, inst, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({l.sql}) t"
        up = _inst_window(dialect, w, "MAX", hp, min_periods=w)
        lo = _inst_window(dialect, w, "MIN", lp, min_periods=w)
        return _Layer(
            f"SELECT x.ts, x.inst, (x._v - o._v) / NULLIF(u._v - o._v, 0) AS _v "
            f"FROM ({c.sql}) x JOIN ({up}) u USING (ts, inst) JOIN ({lo}) o USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"bollinger_pct_b", "bollinger_width"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 2)
        k = float(_literal_positional(node, 1, default=2.0) or 2.0)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        cnt = f"COUNT(_v) OVER ({over})"
        mean = f"AVG(_v) OVER ({over})"
        std = _dialect_fn(dialect, "stddev")
        stdv = f"{std}(_v) OVER ({over})"
        guard = f"CASE WHEN {cnt} < {w} THEN NULL ELSE"
        if op == "bollinger_pct_b":
            # pct_b = (x - (mean - k*std)) / ((mean + k*std) - (mean - k*std))
            expr = f"{guard} ((_v - ({mean} - {k} * {stdv})) / NULLIF((2.0 * {k} * {stdv}), 0)) END"
        else:
            expr = f"{guard} ((2.0 * {k} * {stdv}) / NULLIF(ABS({mean}), 0)) END"
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op in {"KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition"}:
        if len(node.inputs) < 1:
            return None
        close_layer = _compile_layer(node.inputs[0] if op == "KeltnerMid" else node.inputs[2], dialect=dialect)
        if close_layer is None:
            return None
        if op == "KeltnerMid":
            ema_w = max(int(_literal_positional(node, 0, default=20) or 20), 2)
            return _Layer(_ema_span_over_inst(close_layer.sql, ema_w, dialect=dialect, min_periods=ema_w), has_inst_window=True)
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        if high is None or low is None:
            return None
        ema_w = max(int(_literal_positional(node, 2, default=20) or 20), 2)
        atr_w = max(int(_literal_positional(node, 3, default=14) or 14), 2)
        mult = float(_literal_positional(node, 4, default=2.0) or 2.0)
        mid = _ema_span_over_inst(close_layer.sql, ema_w, dialect=dialect, min_periods=ema_w)
        atr = _atr_wilder_sql(high.sql, low.sql, close_layer.sql, atr_w, dialect=dialect)
        if op == "KeltnerUpper":
            return _Layer(
                f"SELECT m.ts, m.inst, (m._v + {mult} * a._v) AS _v "
                f"FROM ({mid}) m JOIN ({atr}) a USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "KeltnerLower":
            return _Layer(
                f"SELECT m.ts, m.inst, (m._v - {mult} * a._v) AS _v "
                f"FROM ({mid}) m JOIN ({atr}) a USING (ts, inst)",
                has_inst_window=True,
            )
        upper = f"SELECT ts, inst, (m._v + {mult} * a._v) AS _v FROM ({mid}) m JOIN ({atr}) a USING (ts, inst)"
        lower = f"SELECT ts, inst, (m._v - {mult} * a._v) AS _v FROM ({mid}) m JOIN ({atr}) a USING (ts, inst)"
        return _Layer(
            f"SELECT c.ts, c.inst, "
            f"(c._v - l._v) / NULLIF(u._v - l._v, 0) AS _v "
            f"FROM ({close_layer.sql}) c "
            f"JOIN ({upper}) u USING (ts, inst) JOIN ({lower}) l USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"DMI_plus", "DMI_minus", "DX", "ADX"}:
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or close is None:
            return None
        w = max(int(_literal_positional(node, 2, default=14) or 14), 2)
        g = _dialect_fn(dialect, "greatest")
        abs_fn = _dialect_fn(dialect, "abs")
        lag_close = f"LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {g}(h._v - l._v, {abs_fn}(h._v - {lag_close}), {abs_fn}(l._v - {lag_close})) END"
        )
        tr_sql = (
            f"SELECT h.ts, h.inst, {tr_expr} AS _v "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({close.sql}) c USING (ts, inst)"
        )
        atr = _wilder_ewm_over_inst(tr_sql, w, dialect=dialect, min_periods=w)
        base = (
            f"SELECT h.ts, h.inst, "
            f"(h._v - LAG(h._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)) AS up, "
            f"(LAG(l._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts) - l._v) AS dn, "
            f"a._v AS atr "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"JOIN ({atr}) a USING (ts, inst)"
        )
        plus = (
            f"SELECT ts, inst, "
            f"CASE WHEN up IS NULL OR dn IS NULL THEN NULL "
            f"WHEN up > dn AND up > 0 THEN up ELSE 0.0 END AS _v FROM ({base}) t"
        )
        minus = (
            f"SELECT ts, inst, "
            f"CASE WHEN up IS NULL OR dn IS NULL THEN NULL "
            f"WHEN dn > up AND dn > 0 THEN dn ELSE 0.0 END AS _v FROM ({base}) t"
        )
        plus_di = _wilder_ewm_over_inst(plus, w, dialect=dialect, min_periods=w)
        minus_di = _wilder_ewm_over_inst(minus, w, dialect=dialect, min_periods=w)
        di = (
            f"SELECT p.ts, p.inst, (100.0 * p._v) / NULLIF(b.atr, 0) AS plus, "
            f"(100.0 * m._v) / NULLIF(b.atr, 0) AS minus "
            f"FROM ({plus_di}) p JOIN ({minus_di}) m USING (ts, inst) "
            f"JOIN ({base}) b USING (ts, inst)"
        )
        if op == "DMI_plus":
            return _Layer(
                f"SELECT ts, inst, plus AS _v FROM ({di}) t",
                has_inst_window=True,
            )
        if op == "DMI_minus":
            return _Layer(
                f"SELECT ts, inst, minus AS _v FROM ({di}) t",
                has_inst_window=True,
            )
        if op == "DX":
            return _Layer(
                f"SELECT ts, inst, (100.0 * ABS(plus - minus)) / NULLIF(plus + minus, 0) AS _v "
                f"FROM ({di}) t",
                has_inst_window=True,
            )
        dx = (
            f"SELECT ts, inst, (100.0 * ABS(plus - minus)) / NULLIF(plus + minus, 0) AS _v "
            f"FROM ({di}) t"
        )
        return _Layer(_wilder_ewm_over_inst(dx, w, dialect=dialect, min_periods=1), has_inst_window=True)

    if op == "UltimateOscillator":
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or close is None:
            return None
        s = max(int(_literal_positional(node, 2, default=7) or 7), 2)
        m = max(int(_literal_positional(node, 3, default=14) or 14), 2)
        l = max(int(_literal_positional(node, 4, default=28) or 28), 2)
        ws = float(_literal_positional(node, 5, default=4.0) or 4.0)
        wm = float(_literal_positional(node, 6, default=2.0) or 2.0)
        wl = float(_literal_positional(node, 7, default=1.0) or 1.0)
        g = _dialect_fn(dialect, "greatest")
        l_fn = _dialect_fn(dialect, "least")
        lag_close = f"LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        bp_expr = f"(c._v - {l_fn}(l._v, {lag_close}))"
        tr_expr = f"({g}(h._v, {lag_close}) - {l_fn}(l._v, {lag_close}))"
        raw = (
            f"SELECT h.ts, h.inst, {bp_expr} AS bp, {tr_expr} AS tr "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({close.sql}) c USING (ts, inst)"
        )
        bps = f"SELECT ts, inst, bp AS _v FROM ({raw}) t"
        trs = f"SELECT ts, inst, tr AS _v FROM ({raw}) t"

        def _avg(w_):
            bsum = _inst_window(dialect, w_, "SUM", bps, min_periods=w_)
            tsum = _inst_window(dialect, w_, "SUM", trs, min_periods=w_)
            return f"SELECT p.ts, p.inst, p._v / NULLIF(t._v, 0) AS _v FROM ({bsum}) p JOIN ({tsum}) t USING (ts, inst)"

        avs, avm, avl = _avg(s), _avg(m), _avg(l)
        den = ws + wm + wl
        return _Layer(
            f"SELECT a.ts, a.inst, "
            f"(100.0 * ({ws} * a._v + {wm} * b._v + {wl} * c._v)) / {den} AS _v "
            f"FROM ({avs}) a JOIN ({avm}) b USING (ts, inst) JOIN ({avl}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ADL":
        if len(node.inputs) < 4:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        volume = _compile_layer(node.inputs[3], dialect=dialect)
        if high is None or low is None or close is None or volume is None:
            return None
        w = max(int(_literal_positional(node, 3, default=20) or 20), 1)
        flow = (
            f"SELECT h.ts, h.inst, "
            f"(((c._v - l._v) - (h._v - c._v)) / NULLIF(h._v - l._v, 0)) * v._v AS _v "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({close.sql}) c USING (ts, inst) "
            f"INNER JOIN ({volume.sql}) v USING (ts, inst)"
        )
        return _Layer(_inst_window(dialect, w, "SUM", flow, min_periods=w), has_inst_window=True)

    if op == "ChaikinOscillator":
        if len(node.inputs) < 4:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        volume = _compile_layer(node.inputs[3], dialect=dialect)
        if high is None or low is None or close is None or volume is None:
            return None
        fast = max(int(_literal_positional(node, 3, default=3) or 3), 2)
        slow = max(int(_literal_positional(node, 4, default=10) or 10), 2)
        adl_w = max(int(_literal_positional(node, 5, default=20) or 20), 1)
        adl_node = PlanNode(op="ADL", inputs=list(node.inputs[:4]), attrs={"window": adl_w})
        adl_layer = _compile_layer(adl_node, dialect=dialect)
        if adl_layer is None:
            return None
        ef = _ema_span_over_inst(adl_layer.sql, fast, dialect=dialect, min_periods=fast)
        es = _ema_span_over_inst(adl_layer.sql, slow, dialect=dialect, min_periods=slow)
        return _Layer(
            f"SELECT f.ts, f.inst, (f._v - s._v) AS _v "
            f"FROM ({ef}) f JOIN ({es}) s USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ForceIndex":
        if len(node.inputs) < 2:
            return None
        close = _compile_layer(node.inputs[0], dialect=dialect)
        volume = _compile_layer(node.inputs[1], dialect=dialect)
        if close is None or volume is None:
            return None
        w = max(int(_literal_positional(node, 1, default=13) or 13), 2)
        raw = (
            f"SELECT c.ts, c.inst, "
            f"(c._v - LAG(c._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts)) * v._v AS _v "
            f"FROM ({close.sql}) c INNER JOIN ({volume.sql}) v USING (ts, inst)"
        )
        return _Layer(_ema_span_over_inst(raw, w, dialect=dialect, min_periods=w), has_inst_window=True)

    if op == "EaseOfMovement":
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        volume = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or volume is None:
            return None
        w = max(int(_literal_positional(node, 2, default=14) or 14), 2)
        scale = float(_literal_positional(node, 3, default=1.0) or 1.0)
        raw = (
            f"SELECT h.ts, h.inst, "
            f"(((h._v + l._v) / 2.0) - "
            f"LAG((h._v + l._v) / 2.0, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)) "
            f"* ((h._v - l._v) / NULLIF(v._v / {scale}, 0)) AS _v "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({volume.sql}) v USING (ts, inst)"
        )
        return _Layer(_inst_window(dialect, w, "AVG", raw, min_periods=w), has_inst_window=True)

    if op == "CMF":
        if len(node.inputs) < 4:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        volume = _compile_layer(node.inputs[3], dialect=dialect)
        if high is None or low is None or close is None or volume is None:
            return None
        w = max(int(_literal_positional(node, 3, default=20) or 20), 1)
        mfv = (
            f"SELECT h.ts, h.inst, "
            f"((((c._v - l._v) - (h._v - c._v)) / NULLIF(h._v - l._v, 0)) * v._v) AS _v "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({close.sql}) c USING (ts, inst) "
            f"INNER JOIN ({volume.sql}) v USING (ts, inst)"
        )
        vol = f"SELECT ts, inst, _v FROM ({volume.sql}) t"
        mfv_sum = _inst_window(dialect, w, "SUM", mfv, min_periods=w)
        vol_sum = _inst_window(dialect, w, "SUM", vol, min_periods=w)
        return _Layer(
            f"SELECT m.ts, m.inst, m._v / NULLIF(v._v, 0) AS _v "
            f"FROM ({mfv_sum}) m JOIN ({vol_sum}) v USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "MFI":
        if len(node.inputs) < 4:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        volume = _compile_layer(node.inputs[3], dialect=dialect)
        if high is None or low is None or close is None or volume is None:
            return None
        w = max(int(_literal_positional(node, 3, default=14) or 14), 1)
        tp = (
            f"SELECT h.ts, h.inst, ((h._v + l._v + c._v) / 3.0) * v._v AS raw, "
            f"(((h._v + l._v + c._v) / 3.0) - "
            f"LAG((h._v + l._v + c._v) / 3.0, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)) AS delta "
            f"FROM ({high.sql}) h INNER JOIN ({low.sql}) l USING (ts, inst) "
            f"INNER JOIN ({close.sql}) c USING (ts, inst) "
            f"INNER JOIN ({volume.sql}) v USING (ts, inst)"
        )
        pos = f"SELECT ts, inst, CASE WHEN delta > 0 THEN raw ELSE 0.0 END AS _v FROM ({tp}) t"
        neg = f"SELECT ts, inst, CASE WHEN delta < 0 THEN raw ELSE 0.0 END AS _v FROM ({tp}) t"
        pos_sum = _inst_window(dialect, w, "SUM", pos, min_periods=w)
        neg_sum = _inst_window(dialect, w, "SUM", neg, min_periods=w)
        return _Layer(
            f"SELECT p.ts, p.inst, "
            f"CASE "
            f"WHEN n._v = 0 AND p._v > 0 THEN 100.0 "
            f"WHEN p._v = 0 AND n._v > 0 THEN 0.0 "
            f"WHEN p._v = 0 AND n._v = 0 THEN 50.0 "
            f"ELSE 100.0 - (100.0 / (1.0 + p._v / NULLIF(n._v, 0))) END AS _v "
            f"FROM ({pos_sum}) p JOIN ({neg_sum}) n USING (ts, inst)",
            has_inst_window=True,
        )

    # ------------------------------------------------------------------
    # Candle geometry / OHLC per-row + return decomposition ops.
    # ------------------------------------------------------------------
    _abs_fn = _dialect_fn(dialect, "abs")
    _g = _dialect_fn(dialect, "greatest")
    _l_fn = _dialect_fn(dialect, "least")

    def _row_join_candle(layers: dict[str, _Layer]) -> tuple[str, str]:
        """构建 OHLC 行级 JOIN：返回 (first_alias, joins_sql)。"""
        joins = []
        first_alias = None
        for name, layer in layers.items():
            if first_alias is None:
                first_alias = name
            else:
                joins.append(f"LEFT JOIN ({layer.sql}) {name} USING (ts, inst)")
        return first_alias, " ".join(joins)

    if op in {"candle_body", "candle_abs_body"}:
        if len(node.inputs) < 2:
            return None
        o = _compile_layer(node.inputs[0], dialect=dialect)
        c = _compile_layer(node.inputs[1], dialect=dialect)
        if o is None or c is None:
            return None
        expr = "(c._v - o._v)" if op == "candle_body" else f"{_abs_fn}(c._v - o._v)"
        return _Layer(
            f"SELECT o.ts, o.inst, {expr} AS _v "
            f"FROM ({o.sql}) o LEFT JOIN ({c.sql}) c USING (ts, inst)",
            has_inst_window=o.has_inst_window or c.has_inst_window,
            has_ts_partition=o.has_ts_partition or c.has_ts_partition,
        )

    if op == "candle_range":
        if len(node.inputs) < 2:
            return None
        h = _compile_layer(node.inputs[0], dialect=dialect)
        l = _compile_layer(node.inputs[1], dialect=dialect)
        if h is None or l is None:
            return None
        return _Layer(
            f"SELECT h.ts, h.inst, (h._v - l._v) AS _v "
            f"FROM ({h.sql}) h LEFT JOIN ({l.sql}) l USING (ts, inst)",
            has_inst_window=h.has_inst_window or l.has_inst_window,
            has_ts_partition=h.has_ts_partition or l.has_ts_partition,
        )

    if op in {"candle_body_ratio", "candle_upper_shadow_ratio", "candle_lower_shadow_ratio",
              "candle_upper_shadow", "candle_lower_shadow", "candle_close_location",
              "candle_body_position", "candle_close_strength",
              "candle_rejection_upper", "candle_rejection_lower", "candle_direction"}:
        # 4-input (open, high, low, close) / 3-input subsets.
        req = {"open", "high", "low", "close"}
        if op in {"candle_close_location", "candle_close_strength"}:
            req = {"high", "low", "close"}
        elif op in {"candle_upper_shadow", "candle_lower_shadow"}:
            req = {"open", "high", "close"} if op == "candle_upper_shadow" else {"open", "low", "close"}
        elif op == "candle_direction":
            req = {"open", "close"}
        layers: dict[str, _Layer] = {}
        ordered = ("open", "high", "low", "close")
        req_names = [name for name in ordered if name in req]
        if len(node.inputs) < len(req_names):
            return None
        for name, layer_node in zip(req_names, node.inputs):
            layer = _compile_layer(layer_node, dialect=dialect)
            if layer is None:
                return None
            layers[name] = layer
        first_alias, join_sql = _row_join_candle(layers)
        refs = {name: f"{name}._v" for name in layers}
        o = refs.get("open", "NULL")
        h = refs.get("high", "NULL")
        l = refs.get("low", "NULL")
        c = refs.get("close", "NULL")
        def _sv(num, den):
            return f"CASE WHEN {num} IS NULL OR ({den}) IS NULL OR ({den}) = 0 THEN NULL ELSE ({num}) / ({den}) END"
        max_expr = f"{_g}({o}, {c})"
        min_expr = f"{_l_fn}({o}, {c})"
        if op == "candle_body_ratio":
            expr = _sv(f"{_abs_fn}({c} - {o})", f"{h} - {l}")
        elif op == "candle_upper_shadow_ratio":
            expr = _sv(f"{h} - {max_expr}", f"{h} - {l}")
        elif op == "candle_lower_shadow_ratio":
            expr = _sv(f"{min_expr} - {l}", f"{h} - {l}")
        elif op == "candle_upper_shadow":
            expr = f"{h} - {max_expr}"
        elif op == "candle_lower_shadow":
            expr = f"{min_expr} - {l}"
        elif op == "candle_close_location":
            expr = _sv(f"{c} - {l}", f"{h} - {l}")
        elif op == "candle_body_position":
            expr = _sv(f"(({o} + {c}) / 2.0) - {l}", f"{h} - {l}")
        elif op == "candle_close_strength":
            expr = f"2.0 * ({_sv(f'{c} - {l}', f'{h} - {l}')}) - 1.0"
        elif op == "candle_rejection_upper":
            expr = _sv(f"{h} - {max_expr}", f"{h} - {l}")
        elif op == "candle_rejection_lower":
            expr = _sv(f"{min_expr} - {l}", f"{h} - {l}")
        else:  # candle_direction
            expr = f"CASE WHEN {c} IS NULL OR {o} IS NULL THEN NULL "
            expr += f"WHEN {c} > {o} THEN 1.0 WHEN {c} < {o} THEN -1.0 ELSE 0.0 END"
        has_win = any(layer.has_inst_window for layer in layers.values())
        has_ts = any(layer.has_ts_partition for layer in layers.values())
        return _Layer(
            f"SELECT {first_alias}.ts, {first_alias}.inst, {expr} AS _v "
            f"FROM ({layers[first_alias].sql}) {first_alias} {join_sql}",
            has_inst_window=has_win,
            has_ts_partition=has_ts,
        )

    if op in {"candle_body_zscore", "candle_range_zscore",
              "candle_upper_shadow_zscore", "candle_lower_shadow_zscore",
              "candle_body_percentile", "candle_range_percentile"}:
        col_names = {
            "candle_body_zscore": ("open", "close"),
            "candle_range_zscore": ("high", "low"),
            "candle_upper_shadow_zscore": ("open", "high", "close"),
            "candle_lower_shadow_zscore": ("open", "low", "close"),
            "candle_body_percentile": ("open", "close"),
            "candle_range_percentile": ("high", "low"),
        }[op]
        if len(node.inputs) < len(col_names):
            return None
        w = _window_int(node)
        layers: dict[str, _Layer] = {}
        for name, input_node in zip(col_names, node.inputs):
            layer = _compile_layer(input_node, dialect=dialect)
            if layer is None:
                return None
            layers[name] = layer
        first_alias, join_sql = _row_join_candle(layers)
        o = f"{'open._v' if 'open' in layers else 'NULL'}"
        h = f"{'high._v' if 'high' in layers else 'NULL'}"
        l = f"{'low._v' if 'low' in layers else 'NULL'}"
        c = f"{'close._v' if 'close' in layers else 'NULL'}"
        if op in {"candle_body_zscore", "candle_body_percentile"}:
            mid = f"{_abs_fn}({c} - {o})"
        elif op in {"candle_range_zscore", "candle_range_percentile"}:
            mid = f"({h} - {l})"
        elif op == "candle_upper_shadow_zscore":
            mid = f"{h} - {_g}({o}, {c})"
        else:  # candle_lower_shadow_zscore
            mid = f"{_l_fn}({o}, {c}) - {l}"
        intermediate = _Layer(
            f"SELECT {first_alias}.ts, {first_alias}.inst, {mid} AS _v "
            f"FROM ({layers[first_alias].sql}) {first_alias} {join_sql}",
            has_inst_window=any(layer.has_inst_window for layer in layers.values()),
            has_ts_partition=any(layer.has_ts_partition for layer in layers.values()),
        )
        if op.endswith("_zscore"):
            body = _zscore_prev_window_expr(
                value_col="_v", window=w, min_periods=w, dialect=dialect,
            )
            return _Layer(
                f"SELECT ts, inst, {body} AS _v FROM ({intermediate.sql}) t",
                has_inst_window=True,
            )
        return _Layer(
            _pct_rank_window_sql(intermediate.sql, window=w, dialect=dialect),
            has_inst_window=True,
        )

    if op in {"candle_gap", "candle_gap_pct"}:
        if len(node.inputs) < 2:
            return None
        o = _compile_layer(node.inputs[0], dialect=dialect)
        c = _compile_layer(node.inputs[1], dialect=dialect)
        if o is None or c is None:
            return None
        lag_close = "LAG(c._v, 1) OVER (PARTITION BY o.inst ORDER BY o.ts)"
        if op == "candle_gap":
            expr = f"(o._v - {lag_close})"
        else:
            expr = f"CASE WHEN {lag_close} IS NULL THEN NULL WHEN {lag_close} = 0 THEN NULL ELSE o._v / {lag_close} - 1.0 END"
        return _Layer(
            f"SELECT o.ts, o.inst, {expr} AS _v "
            f"FROM ({o.sql}) o LEFT JOIN ({c.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "candle_range_atr":
        if len(node.inputs) < 4:
            return None
        h = _compile_layer(node.inputs[1], dialect=dialect)
        l = _compile_layer(node.inputs[2], dialect=dialect)
        c = _compile_layer(node.inputs[3], dialect=dialect)
        if h is None or l is None or c is None:
            return None
        w = max(int(_literal_positional(node, 4, default=14) or 14), 2)
        lag_close = "LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {_g}(h._v - l._v, {_abs_fn}(h._v - {lag_close}), {_abs_fn}(l._v - {lag_close})) END"
        )
        tr_sql = (
            f"SELECT h.ts, h.inst, {tr_expr} AS _v "
            f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
            f"INNER JOIN ({c.sql}) c USING (ts, inst)"
        )
        atr = _inst_window(dialect, w, "AVG", tr_sql, min_periods=w)
        rng_sql = (
            f"SELECT h.ts, h.inst, (h._v - l._v) AS _v "
            f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst)"
        )
        return _Layer(
            f"SELECT r.ts, r.inst, r._v / NULLIF(a._v, 0) AS _v "
            f"FROM ({rng_sql}) r JOIN ({atr}) a USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "candle_gap_atr":
        if len(node.inputs) < 4:
            return None
        o = _compile_layer(node.inputs[0], dialect=dialect)
        h = _compile_layer(node.inputs[1], dialect=dialect)
        l = _compile_layer(node.inputs[2], dialect=dialect)
        c = _compile_layer(node.inputs[3], dialect=dialect)
        if o is None or h is None or l is None or c is None:
            return None
        w = max(int(_literal_positional(node, 4, default=14) or 14), 2)
        lag_close = "LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {_g}(h._v - l._v, {_abs_fn}(h._v - {lag_close}), {_abs_fn}(l._v - {lag_close})) END"
        )
        tr_sql = (
            f"SELECT h.ts, h.inst, {tr_expr} AS _v "
            f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
            f"INNER JOIN ({c.sql}) c USING (ts, inst)"
        )
        atr = _inst_window(dialect, w, "AVG", tr_sql, min_periods=w)
        gap_sql = (
            f"SELECT o.ts, o.inst, (o._v - {lag_close}) AS _v "
            f"FROM ({o.sql}) o LEFT JOIN ({c.sql}) c USING (ts, inst)"
        )
        return _Layer(
            f"SELECT g.ts, g.inst, g._v / NULLIF(a._v, 0) AS _v "
            f"FROM ({gap_sql}) g JOIN ({atr}) a USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "candle_overlap_ratio":
        if len(node.inputs) < 2:
            return None
        h = _compile_layer(node.inputs[0], dialect=dialect)
        l = _compile_layer(node.inputs[1], dialect=dialect)
        if h is None or l is None:
            return None
        prev_h = "LAG(h._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        prev_l = "LAG(l._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        overlap = f"{_g}(0.0, ({_l_fn}(h._v, {prev_h}) - {_g}(l._v, {prev_l})))"
        union = f"({_g}(h._v, {prev_h}) - {_l_fn}(l._v, {prev_l}))"
        expr = f"CASE WHEN {prev_h} IS NULL THEN NULL WHEN {union} = 0 THEN NULL ELSE ({overlap}) / ({union}) END"
        return _Layer(
            f"SELECT h.ts, h.inst, {expr} AS _v "
            f"FROM ({h.sql}) h LEFT JOIN ({l.sql}) l USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "candle_inside_ratio":
        if len(node.inputs) < 2:
            return None
        h = _compile_layer(node.inputs[0], dialect=dialect)
        l = _compile_layer(node.inputs[1], dialect=dialect)
        if h is None or l is None:
            return None
        prev_h = "LAG(h._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        prev_l = "LAG(l._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        prev_rng = f"({prev_h} - {prev_l})"
        cur_rng = "(h._v - l._v)"
        ratio = f"CASE WHEN {prev_rng} = 0 THEN NULL ELSE {cur_rng} / {prev_rng} END"
        inside = f"(h._v <= {prev_h} AND l._v >= {prev_l})"
        outside = f"(h._v >= {prev_h} AND l._v <= {prev_l})"
        # pandas：inside→current/prev；outside&~inside→1+prev/current；既非 inside
        # 也非 outside 的 shifted/gap bar → NaN（P1-12），不能落入 >1 的 else。
        expr = (
            f"CASE WHEN {prev_h} IS NULL THEN NULL "
            f"WHEN {prev_rng} = 0 THEN NULL "
            f"WHEN {inside} THEN {ratio} "
            f"WHEN {outside} THEN 1.0 + {prev_rng} / NULLIF({cur_rng}, 0) "
            f"ELSE NULL END"
        )
        return _Layer(
            f"SELECT h.ts, h.inst, {expr} AS _v "
            f"FROM ({h.sql}) h LEFT JOIN ({l.sql}) l USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {
        "cdl_doji",
        "cdl_hammer",
        "cdl_inverted_hammer",
        "cdl_shooting_star",
        "cdl_marubozu",
        "cdl_spinning_top",
        "cdl_engulfing",
        "cdl_inside_bar",
        "cdl_outside_bar",
        "cdl_dragonfly_doji",
        "cdl_gravestone_doji",
        "cdl_hanging_man",
        "cdl_harami",
        "cdl_harami_cross",
        "cdl_piercing",
        "cdl_dark_cloud_cover",
        "cdl_morning_star",
        "cdl_evening_star",
        "cdl_three_white_soldiers",
        "cdl_three_black_crows",
        "cdl_tweezer_top",
        "cdl_tweezer_bottom",
    }:
        # 全部 4-input (open, high, low, close)。元素级 OHLC 几何 + 至多 LAG(2)。
        if len(node.inputs) < 4:
            return None
        o_l, h_l, l_l, c_l = (
            _compile_layer(node.inputs[0], dialect=dialect),
            _compile_layer(node.inputs[1], dialect=dialect),
            _compile_layer(node.inputs[2], dialect=dialect),
            _compile_layer(node.inputs[3], dialect=dialect),
        )
        if any(x is None for x in (o_l, h_l, l_l, c_l)):
            return None
        _eps = 1e-12
        o, h, l, c = "o._v", "h._v", "l._v", "c._v"
        lag1 = "LAG({}, 1) OVER (PARTITION BY o.inst ORDER BY o.ts)"
        lag2 = "LAG({}, 2) OVER (PARTITION BY o.inst ORDER BY o.ts)"
        o1, c1 = lag1.format(o), lag1.format(c)
        h1, l1 = lag1.format(h), lag1.format(l)
        o2, c2 = lag2.format(o), lag2.format(c)
        h2, l2 = lag2.format(h), lag2.format(l)
        body = f"{_abs_fn}({c} - {o})"
        rng = f"({h} - {l})"
        upper = f"({h} - {_g}({o}, {c}))"
        lower = f"({_l_fn}({o}, {c}) - {l})"
        sign_expr = (
            f"CASE WHEN {c} > {o} THEN 1.0 WHEN {c} < {o} THEN -1.0 ELSE 0.0 END"
        )
        sign0_expr = f"CASE WHEN {c} >= {o} THEN 1.0 ELSE -1.0 END"

        def _flag(cond: str) -> str:
            return f"CASE WHEN {cond} THEN 1.0 ELSE 0.0 END"

        def _sflag(cond: str) -> str:
            return f"-{_flag(cond)}"

        def _sign_flag(cond: str, signed: str) -> str:
            return f"CASE WHEN {cond} THEN {signed} ELSE 0.0 END"

        if op == "cdl_doji":
            expr = _flag(f"{body} <= 0.10 * {rng}")
        elif op == "cdl_hammer":
            expr = _flag(
                f"{body} <= 0.35 * {rng} AND {lower} >= 2.0 * {body} "
                f"AND {upper} <= 0.35 * {_g}({body}, {_eps})"
            )
        elif op == "cdl_inverted_hammer":
            expr = _flag(
                f"{body} <= 0.35 * {rng} AND {upper} >= 2.0 * {body} "
                f"AND {lower} <= 0.35 * {_g}({body}, {_eps})"
            )
        elif op == "cdl_shooting_star":
            expr = _sflag(
                f"{body} <= 0.35 * {rng} AND {upper} >= 2.0 * {body} "
                f"AND {lower} <= 0.35 * {_g}({body}, {_eps})"
            )
        elif op == "cdl_marubozu":
            expr = _sign_flag(
                f"{body} >= 0.90 * {rng} AND {upper} <= 0.05 * {rng} "
                f"AND {lower} <= 0.05 * {rng}",
                sign_expr,
            )
        elif op == "cdl_spinning_top":
            expr = _sign_flag(
                f"{body} <= 0.35 * {rng} AND {upper} >= {body} AND {lower} >= {body}",
                sign0_expr,
            )
        elif op == "cdl_engulfing":
            bull = f"({c} > {o}) AND ({c1} < {o1}) AND ({o} <= {c1}) AND ({c} >= {o1})"
            bear = f"({c} < {o}) AND ({c1} > {o1}) AND ({o} >= {c1}) AND ({c} <= {o1})"
            expr = f"({_flag(bull)}) - ({_flag(bear)})"
        elif op == "cdl_inside_bar":
            expr = _flag(f"({h} < {h1}) AND ({l} > {l1})")
        elif op == "cdl_outside_bar":
            expr = _sign_flag(f"({h} > {h1}) AND ({l} < {l1})", sign_expr)
        elif op == "cdl_dragonfly_doji":
            expr = _flag(
                f"{body} <= 0.10 * {rng} AND {lower} >= 0.60 * {rng} "
                f"AND {upper} <= 0.10 * {rng}"
            )
        elif op == "cdl_gravestone_doji":
            expr = _flag(
                f"{body} <= 0.10 * {rng} AND {upper} >= 0.60 * {rng} "
                f"AND {lower} <= 0.10 * {rng}"
            )
        elif op == "cdl_hanging_man":
            expr = _sflag(
                f"{body} <= 0.35 * {rng} AND {lower} >= 2.0 * {body} "
                f"AND {upper} <= 0.35 * {_g}({body}, {_eps})"
            )
        elif op == "cdl_harami":
            prev_hi = f"{_g}({o1}, {c1})"
            prev_lo = f"{_l_fn}({o1}, {c1})"
            cur_hi = f"{_g}({o}, {c})"
            cur_lo = f"{_l_fn}({o}, {c})"
            inside = f"({cur_hi} < {prev_hi}) AND ({cur_lo} > {prev_lo})"
            bullish = f"{inside} AND ({c1} < {o1}) AND ({c} > {o})"
            bearish = f"{inside} AND ({c1} > {o1}) AND ({c} < {o})"
            expr = f"({_flag(bullish)}) - ({_flag(bearish)})"
        elif op == "cdl_harami_cross":
            prev_hi = f"{_g}({o1}, {c1})"
            prev_lo = f"{_l_fn}({o1}, {c1})"
            cur_hi = f"{_g}({o}, {c})"
            cur_lo = f"{_l_fn}({o}, {c})"
            inside = f"({cur_hi} < {prev_hi}) AND ({cur_lo} > {prev_lo})"
            bullish = f"{inside} AND ({c1} < {o1}) AND ({c} > {o})"
            bearish = f"{inside} AND ({c1} > {o1}) AND ({c} < {o})"
            base = f"({_flag(bullish)}) - ({_flag(bearish)})"
            doji = f"{body} <= 0.10 * {rng}"
            expr = f"({base}) * ({_flag(doji)})"
        elif op == "cdl_piercing":
            midpoint = f"(({o1} + {c1}) / 2.0)"
            expr = _flag(
                f"({c1} < {o1}) AND ({c} > {o}) AND ({o} <= {c1}) "
                f"AND ({c} > {midpoint}) AND ({c} < {o1})"
            )
        elif op == "cdl_dark_cloud_cover":
            midpoint = f"(({o1} + {c1}) / 2.0)"
            expr = _sflag(
                f"({c1} > {o1}) AND ({c} < {o}) AND ({o} >= {c1}) "
                f"AND ({c} < {midpoint}) AND ({c} > {o1})"
            )
        elif op == "cdl_morning_star":
            body2 = f"{_abs_fn}({c2} - {o2})"
            body1 = f"{_abs_fn}({c1} - {o1})"
            range2 = f"({h2} - {l2})"
            range1 = f"({h1} - {l1})"
            midpoint2 = f"(({o2} + {c2}) / 2.0)"
            expr = _flag(
                f"({c2} < {o2}) AND ({body2} >= 0.50 * {nf}({range2}, 0.0)) "
                f"AND ({body1} <= 0.35 * {nf}({range1}, 0.0)) "
                f"AND ({c} > {o}) AND ({c} > {midpoint2})"
            )
        elif op == "cdl_evening_star":
            body2 = f"{_abs_fn}({c2} - {o2})"
            body1 = f"{_abs_fn}({c1} - {o1})"
            range2 = f"({h2} - {l2})"
            range1 = f"({h1} - {l1})"
            midpoint2 = f"(({o2} + {c2}) / 2.0)"
            expr = _sflag(
                f"({c2} > {o2}) AND ({body2} >= 0.50 * {nf}({range2}, 0.0)) "
                f"AND ({body1} <= 0.35 * {nf}({range1}, 0.0)) "
                f"AND ({c} < {o}) AND ({c} < {midpoint2})"
            )
        elif op == "cdl_three_white_soldiers":
            expr = _flag(
                f"({c} > {o}) AND ({c1} > {o1}) AND ({c2} > {o2}) "
                f"AND ({c} > {c1}) AND ({c1} > {c2}) "
                f"AND ({o} >= {o1}) AND ({o} <= {c1}) "
                f"AND ({o1} >= {o2}) AND ({o1} <= {c2})"
            )
        elif op == "cdl_three_black_crows":
            expr = _sflag(
                f"({c} < {o}) AND ({c1} < {o1}) AND ({c2} < {o2}) "
                f"AND ({c} < {c1}) AND ({c1} < {c2}) "
                f"AND ({o} <= {o1}) AND ({o} >= {c1}) "
                f"AND ({o1} <= {o2}) AND ({o1} >= {c2})"
            )
        elif op == "cdl_tweezer_top":
            # audit item 6: A-share tick 容差（|p|<10→0.01, <100→0.05, else 0.1）×2，
            # 非旧的 1e-4*price 相对带。
            tick_h = (
                f"CASE WHEN {_abs_fn}({h}) < 10.0 THEN 0.01 "
                f"WHEN {_abs_fn}({h}) < 100.0 THEN 0.05 ELSE 0.1 END"
            )
            same_high = f"{_abs_fn}({h} - {h1}) <= {tick_h} * 2.0"
            reversal = f"({c1} > {o1}) AND ({c} < {o})"
            expr = _sflag(f"{same_high} AND {reversal}")
        else:  # cdl_tweezer_bottom
            tick_l = (
                f"CASE WHEN {_abs_fn}({l}) < 10.0 THEN 0.01 "
                f"WHEN {_abs_fn}({l}) < 100.0 THEN 0.05 ELSE 0.1 END"
            )
            same_low = f"{_abs_fn}({l} - {l1}) <= {tick_l} * 2.0"
            reversal = f"({c1} < {o1}) AND ({c} > {o})"
            expr = _flag(f"{same_low} AND {reversal}")

        full_expr = (
            f"CASE WHEN {o} IS NULL OR {h} IS NULL OR {l} IS NULL OR {c} IS NULL "
            f"THEN NULL ELSE {expr} END"
        )
        has_win = any(x.has_inst_window for x in (o_l, h_l, l_l, c_l))
        has_ts = any(x.has_ts_partition for x in (o_l, h_l, l_l, c_l))
        return _Layer(
            f"SELECT o.ts, o.inst, {full_expr} AS _v "
            f"FROM ({o_l.sql}) o "
            f"LEFT JOIN ({h_l.sql}) h USING (ts, inst) "
            f"LEFT JOIN ({l_l.sql}) l USING (ts, inst) "
            f"LEFT JOIN ({c_l.sql}) c USING (ts, inst)",
            has_inst_window=True,
            has_ts_partition=has_ts,
        )

    if op in {
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "ichimoku_cloud_width",
        "ichimoku_cloud_position",
    }:
        # Ichimoku 族 = (rolling_max(high, w) + rolling_min(low, w)) / 2 的组合，
        # 均为因果（min_periods=w），无 chart-forward 位移。
        if len(node.inputs) < 2:
            return None
        h_l = _compile_layer(node.inputs[0], dialect=dialect)
        l_l = _compile_layer(node.inputs[1], dialect=dialect)
        if h_l is None or l_l is None:
            return None

        def _midpoint(w: int) -> _Layer:
            hmax = _Layer(
                _inst_window(dialect, w, "MAX", h_l.sql, min_periods=w),
                has_inst_window=True,
            )
            lmin = _Layer(
                _inst_window(dialect, w, "MIN", l_l.sql, min_periods=w),
                has_inst_window=True,
            )
            return _Layer(
                f"SELECT a.ts, a.inst, ((a._v + b._v) / 2.0) AS _v "
                f"FROM ({hmax.sql}) a LEFT JOIN ({lmin.sql}) b USING (ts, inst)",
                has_inst_window=True,
            )

        if op == "ichimoku_tenkan":
            w = int(_literal_positional(node, 1, default=9) or 9)
            return _midpoint(w)
        if op == "ichimoku_kijun":
            w = int(_literal_positional(node, 1, default=26) or 26)
            return _midpoint(w)
        if op == "ichimoku_senkou_b":
            w = int(_literal_positional(node, 1, default=52) or 52)
            return _midpoint(w)
        if op == "ichimoku_senkou_a":
            w1 = int(_literal_positional(node, 1, default=9) or 9)
            w2 = int(_literal_positional(node, 2, default=26) or 26)
            t = _midpoint(w1)
            k = _midpoint(w2)
            return _Layer(
                f"SELECT a.ts, a.inst, ((a._v + b._v) / 2.0) AS _v "
                f"FROM ({t.sql}) a LEFT JOIN ({k.sql}) b USING (ts, inst)",
                has_inst_window=True,
            )
        # cloud_width / cloud_position
        if op == "ichimoku_cloud_width":
            w1 = int(_literal_positional(node, 1, default=9) or 9)
            w2 = int(_literal_positional(node, 2, default=26) or 26)
            w3 = int(_literal_positional(node, 3, default=52) or 52)
        else:  # cloud_position: params [high, low, close, tenkan, kijun, senkou_b]
            if len(node.inputs) < 3:
                return None
            w1 = int(_literal_positional(node, 2, default=9) or 9)
            w2 = int(_literal_positional(node, 3, default=26) or 26)
            w3 = int(_literal_positional(node, 4, default=52) or 52)
        t = _midpoint(w1)
        k = _midpoint(w2)
        b = _midpoint(w3)
        sa = _Layer(
            f"SELECT a.ts, a.inst, ((a._v + b._v) / 2.0) AS _v "
            f"FROM ({t.sql}) a LEFT JOIN ({k.sql}) b USING (ts, inst)",
            has_inst_window=True,
        )
        if op == "ichimoku_cloud_width":
            return _Layer(
                f"SELECT a.ts, a.inst, {_abs_fn}(a._v - b._v) AS _v "
                f"FROM ({sa.sql}) a LEFT JOIN ({b.sql}) b USING (ts, inst)",
                has_inst_window=True,
            )
        c_l = _compile_layer(node.inputs[2], dialect=dialect)
        if c_l is None:
            return None
        lo_expr = f"{_l_fn}(a._v, b._v)"
        hi_expr = f"{_g}(a._v, b._v)"
        expr = (
            f"CASE WHEN a._v IS NULL OR b._v IS NULL OR c._v IS NULL THEN NULL "
            f"WHEN ({hi_expr} - {lo_expr}) = 0 THEN NULL "
            f"ELSE (c._v - {lo_expr}) / ({hi_expr} - {lo_expr}) END"
        )
        return _Layer(
            f"SELECT a.ts, a.inst, {expr} AS _v "
            f"FROM ({sa.sql}) a LEFT JOIN ({b.sql}) b USING (ts, inst) "
            f"LEFT JOIN ({c_l.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "efficiency_ratio":
        # Kaufman efficiency ratio = |close - close.shift(w)| / rolling_sum(|diff|, w)
        if len(node.inputs) < 1:
            return None
        c_l = _compile_layer(node.inputs[0], dialect=dialect)
        if c_l is None:
            return None
        w = int(_literal_positional(node, 0, default=20) or 20)
        diff_sql = f"SELECT ts, inst, {_abs_fn}(_v - LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts)) AS _v FROM ({c_l.sql}) _e"
        path_l = _Layer(_inst_window(dialect, w, "SUM", diff_sql, min_periods=w), has_inst_window=True)
        change_l = _Layer(
            f"SELECT ts, inst, {_abs_fn}(_v - LAG(_v, {w}) OVER (PARTITION BY inst ORDER BY ts)) AS _v FROM ({c_l.sql}) _c",
            has_inst_window=True,
        )
        expr = (
            f"CASE WHEN a._v IS NULL OR b._v IS NULL OR b._v = 0 THEN NULL "
            f"ELSE a._v / b._v END"
        )
        return _Layer(
            f"SELECT a.ts, a.inst, {expr} AS _v "
            f"FROM ({change_l.sql}) a LEFT JOIN ({path_l.sql}) b USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "choppiness_index":
        # 100 * log10( (ΣTR / (max(h)-min(l))) clip(lower=EPS) ) / log10(w)
        if len(node.inputs) < 3:
            return None
        h_l = _compile_layer(node.inputs[0], dialect=dialect)
        l_l = _compile_layer(node.inputs[1], dialect=dialect)
        c_l = _compile_layer(node.inputs[2], dialect=dialect)
        if h_l is None or l_l is None or c_l is None:
            return None
        w = int(_literal_positional(node, 2, default=20) or 20)
        _eps = 1e-12
        tr_node = PlanNode(op="true_range", inputs=[node.inputs[0], node.inputs[1], node.inputs[2]], attrs={})
        tr_l = _compile_layer(tr_node, dialect=dialect)
        if tr_l is None:
            return None
        num_l = _Layer(_inst_window(dialect, w, "SUM", tr_l.sql, min_periods=w), has_inst_window=True)
        hmax_l = _Layer(_inst_window(dialect, w, "MAX", h_l.sql, min_periods=w), has_inst_window=True)
        lmin_l = _Layer(_inst_window(dialect, w, "MIN", l_l.sql, min_periods=w), has_inst_window=True)
        expr = (
            f"CASE WHEN a._v IS NULL OR b._v IS NULL OR c._v IS NULL THEN NULL "
            f"WHEN b._v - c._v <= 0 THEN NULL "
            f"ELSE 100.0 * LOG10({_g}(a._v / (b._v - c._v), {_eps})) / LOG10({w}) END"
        )
        return _Layer(
            f"SELECT a.ts, a.inst, {expr} AS _v "
            f"FROM ({num_l.sql}) a "
            f"LEFT JOIN ({hmax_l.sql}) b USING (ts, inst) "
            f"LEFT JOIN ({lmin_l.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "coskewness_to_market":
        # 协偏度 = E[(r-μ_r)(m-μ_m)²] / (σ_r · var(m))，窗口内 ≥5 对有效观测。
        # 展开交叉项为独立窗口聚合（避免嵌套窗口函数）：
        # numer = E[rm²] - 2·μ_m·E[rm] - μ_r·E[m²] + 2·μ_r·μ_m²
        if len(node.inputs) < 2:
            return None
        r_l = _compile_layer(node.inputs[0], dialect=dialect)
        m_l = _compile_layer(node.inputs[1], dialect=dialect)
        if r_l is None or m_l is None:
            return None
        w = int(_literal_positional(node, 1, default=60) or 60)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        sub = (
            f"SELECT r.ts, r.inst, "
            f"CASE WHEN r._v IS NULL OR m._v IS NULL THEN NULL ELSE r._v END AS r, "
            f"CASE WHEN r._v IS NULL OR m._v IS NULL THEN NULL ELSE m._v END AS m "
            f"FROM ({r_l.sql}) r LEFT JOIN ({m_l.sql}) m USING (ts, inst)"
        )
        agg = (
            f"SELECT t.ts, t.inst, "
            f"COUNT(r) OVER ({over}) AS cnt, "
            f"AVG(r) OVER ({over}) AS mr, "
            f"AVG(m) OVER ({over}) AS mm, "
            f"AVG(r*m) OVER ({over}) AS mrm, "
            f"AVG(r*m*m) OVER ({over}) AS mrm2, "
            f"AVG(m*m) OVER ({over}) AS mm2, "
            f"STDDEV(r) OVER ({over}) AS sr, "
            f"VAR(m) OVER ({over}) AS vm "
            f"FROM ({sub}) t"
        )
        expr = (
            f"CASE WHEN a.cnt < 5 THEN NULL "
            f"WHEN a.sr IS NULL OR a.vm IS NULL OR a.sr * a.vm <= 0 THEN NULL "
            f"ELSE (a.mrm2 - 2.0 * a.mm * a.mrm - a.mr * a.mm2 + 2.0 * a.mr * a.mm * a.mm) / (a.sr * a.vm) END"
        )
        return _Layer(
            f"SELECT a.ts, a.inst, {expr} AS _v FROM ({agg}) a",
            has_inst_window=True,
        )

    if op in {
        "ts_valid_count",
        "ts_coverage_ratio",
        "ts_abs_concentration",
        "ts_abs_entropy",
        "ts_downside_deviation",
        "ts_upside_deviation",
        "ts_impulse_return",
        "ts_impulse_strength",
        "ts_impulse_volume",
    }:
        # 窗口内聚合统计（ts_valid_count / coverage / concentration / entropy /
        # downside·upside deviation / impulse 族）。仅需单一输入列 + literal 参数。
        if len(node.inputs) < 1:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        if x_l is None:
            return None
        w = int(_literal_positional(node, 0, default=20) or 20)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        cnt = f"COUNT(_v) OVER ({over})"

        if op == "ts_valid_count":
            mp = int(_literal_positional(node, 1, default=1) or 1)
            expr = f"CASE WHEN {cnt} < {mp} THEN NULL ELSE {cnt} END"
        elif op == "ts_coverage_ratio":
            mp = int(_literal_positional(node, 1, default=1) or 1)
            expr = f"CASE WHEN {cnt} < {mp} THEN NULL ELSE {cnt} / {w} END"
        elif op == "ts_abs_concentration":
            mp = int(_literal_positional(node, 1, default=1) or 1)
            s_abs = f"SUM({_abs_fn}(_v)) OVER ({over})"
            s_sq = f"SUM(POW({_abs_fn}(_v), 2)) OVER ({over})"
            expr = (
                f"CASE WHEN {cnt} < {mp} THEN NULL "
                f"WHEN {s_abs} IS NULL OR {s_abs} <= 0 THEN NULL "
                f"ELSE {s_sq} / ({s_abs} * {s_abs}) END"
            )
        elif op == "ts_abs_entropy":
            mp = int(_literal_positional(node, 2, default=1) or 1)
            norm = bool(_literal_positional(node, 1, default=1.0) or 1.0)
            a = f"{_abs_fn}(_v)"
            s_abs = f"SUM({a}) OVER ({over})"
            s_ln = (
                f"SUM(CASE WHEN {a} = 0 THEN 0.0 "
                f"ELSE ({a} / {s_abs}) * LN({a} / {s_abs} + 1e-300) END) OVER ({over})"
            )
            entropy = f"(-1.0 * {s_ln})"
            if norm:
                entropy = f"(CASE WHEN {cnt} > 1 THEN {entropy} / LN({cnt}) ELSE {entropy} END)"
            expr = (
                f"CASE WHEN {cnt} < {mp} THEN NULL "
                f"WHEN {s_abs} IS NULL OR {s_abs} <= 0 THEN NULL "
                f"ELSE {entropy} END"
            )
        elif op in {"ts_downside_deviation", "ts_upside_deviation"}:
            tgt = _literal_positional(node, 1, default=0.0) or 0.0
            mp = int(_literal_positional(node, 2, default=2) or 2)
            wf = f"LEAST(_v - {tgt}, 0.0)" if op == "ts_downside_deviation" else f"GREATEST(_v - {tgt}, 0.0)"
            avg = f"AVG(POW({wf}, 2)) OVER ({over})"
            expr = f"CASE WHEN {cnt} < {mp} THEN NULL WHEN {avg} IS NULL THEN NULL ELSE SQRT({avg}) END"
        elif op == "ts_impulse_return":
            prev = f"LAG(_v, {w}) OVER (PARTITION BY inst ORDER BY ts)"
            expr = f"CASE WHEN {prev} IS NULL THEN NULL ELSE _v / {prev} - 1.0 END"
        elif op == "ts_impulse_strength":
            vol_w = int(_literal_positional(node, 1, default=20) or 20)
            prev_w = f"LAG(_v, {w}) OVER (PARTITION BY inst ORDER BY ts)"
            impulse = f"CASE WHEN {prev_w} IS NULL THEN NULL ELSE _v / {prev_w} - 1.0 END"
            # R16-062: the realized-vol BASELINE must be STRICTLY PRIOR — a
            # current-row ``ret`` (``_v / LAG(_v,1)``) mixed the current value
            # into the normalizer.  Use only prior 1-step returns and a frame
            # that ENDS at 1 PRECEDING.
            ret = (
                f"CASE WHEN LAG(_v, 2) OVER (PARTITION BY inst ORDER BY ts) IS NULL "
                f"THEN NULL ELSE "
                f"LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) / "
                f"LAG(_v, 2) OVER (PARTITION BY inst ORDER BY ts) - 1.0 END"
            )
            rv_over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {vol_w} PRECEDING AND 1 PRECEDING"
            rv = f"STDDEV({ret}) OVER ({rv_over})"
            rv_cnt = f"COUNT({ret}) OVER ({rv_over})"
            expr = (
                f"CASE WHEN {impulse} IS NULL THEN NULL "
                f"WHEN {rv_cnt} < {vol_w} THEN NULL "
                f"WHEN {rv} IS NULL OR {rv} * SQRT({w}) = 0 THEN NULL "
                f"ELSE {impulse} / ({rv} * SQRT({w})) END"
            )
        else:  # ts_impulse_volume
            b = int(_literal_positional(node, 1, default=20) or 20)
            recent = f"AVG(_v) OVER ({over})"
            recent_cnt = f"COUNT(_v) OVER ({over})"
            lagged = f"SELECT ts, inst, LAG(_v, {w}) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({x_l.sql}) _t"
            base_l = _Layer(_inst_window(dialect, b, "AVG", lagged, min_periods=b), has_inst_window=True)
            expr = (
                f"CASE WHEN {recent_cnt} < {w} THEN NULL "
                f"WHEN {recent} IS NULL OR c._v IS NULL OR c._v = 0 THEN NULL "
                f"ELSE {recent} / c._v - 1.0 END"
            )
            return _Layer(
                f"SELECT a.ts, a.inst, {expr} AS _v "
                f"FROM ({x_l.sql}) a LEFT JOIN ({base_l.sql}) c USING (ts, inst)",
                has_inst_window=True,
            )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({x_l.sql}) t",
            has_inst_window=True,
        )

    # Alpha-language SQL subset (2026-08): pandas-reference parity on clean
    # daily panels.  Only the window-function-natural ops are lowered; the
    # run/hysteresis/quantile/cs-locality family stays pandas_numpy-only
    # (fail-closed, same as the 2026-08 final pack).
    if op in {
        "event_frequency",
        "ts_semivariance_balance",
        "ts_realized_quarticity",
        "ts_vol_of_vol",
        "ts_vol_acceleration",
        "ts_vol_term_structure",
    }:
        if len(node.inputs) < 1:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        if x_l is None:
            return None

        if op == "event_frequency":
            w = int(_literal_positional(node, 0, default=20) or 20)
            mp = int(_literal_positional(node, 1, default=1) or 1)
            over_w = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(_v) OVER ({over_w})"
            truth = _truthy_sql("_v", dialect=dialect)
            hits = f"SUM(CASE WHEN {truth} THEN 1 ELSE 0 END) OVER ({over_w})"
            expr = f"CASE WHEN {cnt} < {mp} THEN NULL WHEN {cnt} = 0 THEN NULL ELSE {hits} / {cnt} END"
        elif op == "ts_semivariance_balance":
            w = int(_literal_positional(node, 0, default=20) or 20)
            mp = max(2, int(_literal_positional(node, 1, default=2) or 2))
            over_w = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(_v) OVER ({over_w})"
            pos = f"SUM(CASE WHEN _v > 0 THEN POW(_v, 2) ELSE 0 END) OVER ({over_w})"
            neg = f"SUM(CASE WHEN _v < 0 THEN POW(_v, 2) ELSE 0 END) OVER ({over_w})"
            expr = (
                f"CASE WHEN {cnt} < {mp} THEN NULL "
                f"WHEN ({pos} + {neg}) < 1e-12 THEN NULL "
                f"ELSE ({pos} - {neg}) / ({pos} + {neg}) END"
            )
        elif op == "ts_realized_quarticity":
            w = int(_literal_positional(node, 0, default=20) or 20)
            mp = max(3, int(_literal_positional(node, 1, default=3) or 3))
            over_w = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(_v) OVER ({over_w})"
            rv2 = f"SUM(POW(_v, 2)) OVER ({over_w})"
            rv4 = f"SUM(POW(_v, 4)) OVER ({over_w})"
            expr = (
                f"CASE WHEN {cnt} < {mp} THEN NULL "
                f"WHEN {rv2} < 1e-12 THEN NULL "
                f"ELSE {cnt} * {rv4} / (3 * POW({rv2}, 2) + 1e-12) END"
            )
        elif op == "ts_vol_of_vol":
            wi = int(_literal_positional(node, 0, default=5) or 5)
            wo = int(_literal_positional(node, 1, default=40) or 40)
            sd = _dialect_fn(dialect, "stddev_pop")
            inner = _inst_window(dialect, wi, sd, x_l.sql, min_periods=2)
            logv = f"SELECT ts, inst, LN(_v + 1e-12) AS _v FROM ({inner}) _lv"
            outer = _inst_window(dialect, wo, sd, logv, min_periods=2)
            return _Layer(outer, has_inst_window=True)
        elif op == "ts_vol_acceleration":
            wi = int(_literal_positional(node, 0, default=5) or 5)
            la = max(1, int(_literal_positional(node, 1, default=5) or 5))
            sd = _dialect_fn(dialect, "stddev_pop")
            inner = _inst_window(dialect, wi, sd, x_l.sql, min_periods=2)
            lagged = (
                f"SELECT ts, inst, _v, LAG(_v, {la}) OVER (PARTITION BY inst ORDER BY ts) AS _p "
                f"FROM ({inner}) _la"
            )
            expr = (
                f"CASE WHEN _v IS NULL OR _p IS NULL THEN NULL "
                f"ELSE LN((_v + 1e-12) / (_p + 1e-12)) END"
            )
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({lagged}) _f",
                has_inst_window=True,
            )
        else:  # ts_vol_term_structure
            ws = int(_literal_positional(node, 0, default=5) or 5)
            wl = int(_literal_positional(node, 1, default=40) or 40)
            sd = _dialect_fn(dialect, "stddev_pop")
            over_s = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {ws - 1} PRECEDING AND CURRENT ROW"
            over_l = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {wl - 1} PRECEDING AND CURRENT ROW"
            both = (
                f"SELECT ts, inst, {sd}(_v) OVER ({over_s}) AS _s, {sd}(_v) OVER ({over_l}) AS _l, "
                f"COUNT(_v) OVER ({over_s}) AS _cs FROM ({x_l.sql}) _b"
            )
            expr = (
                f"CASE WHEN _cs < 2 THEN NULL WHEN _s IS NULL OR _l IS NULL THEN NULL "
                f"ELSE LN((_s + 1e-12) / (_l + 1e-12)) END"
            )
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({both}) _f",
                has_inst_window=True,
            )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({x_l.sql}) t",
            has_inst_window=True,
        )

    if op in {
        "ts_argmax_age",
        "ts_argmin_age",
        "ts_argmax_index_from_oldest",
        "ts_argmin_index_from_oldest",
        "ts_staleness",
    }:
        # 极值年龄 / 极值位置 / 最近有限值 bar 数。
        # DuckDB 禁止嵌套窗口函数，用三级子查询：先求窗口极值 + 计数，
        # 再标记命中行号，最后在窗口内取命中行号的极值。
        if len(node.inputs) < 1:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        if x_l is None:
            return None
        w = int(_literal_positional(node, 0, default=20) or 20)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        rn_sql = f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn FROM ({x_l.sql}) _s"
        if op == "ts_staleness":
            mp = 0
            sub1 = f"SELECT ts, inst, _v, rn, COUNT(_v) OVER ({over}) AS cnt FROM ({rn_sql}) _1"
            sub2 = f"SELECT ts, inst, rn, cnt, CASE WHEN _v IS NOT NULL THEN rn END AS hit_rn FROM ({sub1}) _2"
            gate = "WHEN cnt = 0 THEN NULL"
        else:
            mp = int(_literal_positional(node, 1, default=1) or 1)
            agg = "MAX" if op in {"ts_argmax_age", "ts_argmax_index_from_oldest"} else "MIN"
            sub1 = (
                f"SELECT ts, inst, _v, rn, {agg}(_v) OVER ({over}) AS wmax, "
                f"COUNT(_v) OVER ({over}) AS cnt FROM ({rn_sql}) _1"
            )
            sub2 = f"SELECT ts, inst, rn, wmax, cnt, CASE WHEN _v = wmax THEN rn END AS hit_rn FROM ({sub1}) _2"
            gate = f"WHEN cnt < {mp} THEN NULL"
        sub3 = f"SELECT ts, inst, rn, cnt, MAX(hit_rn) OVER ({over}) AS last_hit FROM ({sub2}) _3"
        if op == "ts_staleness":
            expr = f"CASE {gate} ELSE rn - last_hit END"
        elif op in {"ts_argmax_age", "ts_argmin_age"}:
            expr = f"CASE {gate} WHEN last_hit IS NULL THEN NULL ELSE rn - last_hit END"
        else:  # index_from_oldest
            start_rn = f"GREATEST(1, rn - {w} + 1)"
            expr = f"CASE {gate} WHEN last_hit IS NULL THEN NULL ELSE last_hit - {start_rn} END"
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({sub3}) _4",
            has_inst_window=True,
        )

    if op in {"ts_days_since_high", "ts_days_since_low"}:
        # x.shift(1).rolling(w, min_periods=w)：在「前 w 个 bar」里找极值，
        # 返回距最近一个 prior bar 的 bar 数。
        # R16-061: the canonical Pandas path resolves TIES to the MOST RECENT
        # hit (``np.argmax``/``argmin`` on the reversed window).  The old SQL
        # ``MIN(hit_rn)`` returned the FIRST hit — a tie-rich window diverged
        # from Pandas.  ``MAX(hit_rn)`` = most-recent hit.
        if len(node.inputs) < 1:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        if x_l is None:
            return None
        w = int(_literal_positional(node, 0, default=20) or 20)
        over_prev = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING"
        agg = "MAX" if op == "ts_days_since_high" else "MIN"
        rn_sql = f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn FROM ({x_l.sql}) _s"
        sub0 = f"SELECT ts, inst, rn, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS sh FROM ({rn_sql}) _0"
        sub1 = f"SELECT ts, inst, rn, sh, {agg}(sh) OVER ({over_prev}) AS pxt, COUNT(sh) OVER ({over_prev}) AS pcnt FROM ({sub0}) _1"
        sub2 = f"SELECT ts, inst, rn, pxt, pcnt, CASE WHEN sh = pxt THEN rn END AS hit_rn FROM ({sub1}) _2"
        sub3 = f"SELECT ts, inst, rn, pcnt, MAX(hit_rn) OVER ({over_prev}) AS last_hit FROM ({sub2}) _3"
        expr = f"CASE WHEN pcnt < {w} THEN NULL ELSE (rn - 1) - last_hit END"
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({sub3}) _4",
            has_inst_window=True,
        )

    if op in {
        "cs_valid_count",
        "cs_coverage_ratio",
        "cs_fill_mean",
        "cs_fill_median",
        "cs_impute_mean",
        "cs_impute_median",
        "cs_residual_percentile",
    }:
        # 横截面（同一 ts 内所有 inst）聚合。partition by ts。
        if len(node.inputs) < 1:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        if x_l is None:
            return None
        p = "PARTITION BY ts"
        if op == "cs_valid_count":
            expr = f"COUNT(_v) OVER ({p})"
        elif op == "cs_coverage_ratio":
            expr = f"COUNT(_v) OVER ({p}) / COUNT(*) OVER ({p})"
        elif op in {"cs_fill_mean", "cs_impute_mean"}:
            expr = f"CASE WHEN _v IS NULL THEN AVG(_v) OVER ({p}) ELSE _v END"
        elif op in {"cs_fill_median", "cs_impute_median"}:
            expr = f"CASE WHEN _v IS NULL THEN MEDIAN(_v) OVER ({p}) ELSE _v END"
        else:  # cs_residual_percentile
            rn = f"ROW_NUMBER() OVER ({p} ORDER BY _v)"
            cnt = f"COUNT(_v) OVER ({p})"
            expr = (
                f"CASE WHEN {cnt} < 2 THEN NULL "
                f"WHEN _v IS NULL THEN NULL "
                f"ELSE ({rn} - 1) / ({cnt} - 1) END"
            )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({x_l.sql}) t",
            has_inst_window=x_l.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"cs_weighted_mean", "cs_weighted_demean", "cs_weighted_zscore"}:
        # 横截面加权统计：仅对 x 有限、weight 有限且 >0 的位置聚合并输出；
        # 权重和 <= EPS 时整行 NaN。
        if len(node.inputs) < 2:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        w_l = _compile_layer(node.inputs[1], dialect=dialect)
        if x_l is None or w_l is None:
            return None
        _eps = 1e-12
        p = "PARTITION BY ts"
        sub = (
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN x._v IS NULL OR w._v IS NULL OR w._v <= 0 THEN NULL ELSE x._v END AS x, "
            f"CASE WHEN x._v IS NULL OR w._v IS NULL OR w._v <= 0 THEN NULL ELSE w._v END AS w "
            f"FROM ({x_l.sql}) x LEFT JOIN ({w_l.sql}) w USING (ts, inst)"
        )
        sub2 = (
            f"SELECT ts, inst, x, w, "
            f"SUM(x*w) OVER ({p}) AS swx, SUM(w) OVER ({p}) AS sw FROM ({sub}) _1"
        )
        sub3 = (
            f"SELECT ts, inst, x, w, swx, sw, "
            f"CASE WHEN sw IS NULL OR sw <= {_eps} THEN NULL ELSE swx / sw END AS m, "
            f"x - CASE WHEN sw IS NULL OR sw <= {_eps} THEN NULL ELSE swx / sw END AS dx "
            f"FROM ({sub2}) _2"
        )
        if op == "cs_weighted_mean":
            expr = "CASE WHEN x IS NULL THEN NULL ELSE m END"
        elif op == "cs_weighted_demean":
            expr = "CASE WHEN x IS NULL THEN NULL ELSE dx END"
        else:
            sub4 = (
                f"SELECT ts, inst, x, dx, sw, "
                f"SUM(dx*dx*w) OVER ({p}) AS sdx2 FROM ({sub3}) _3"
            )
            expr = (
                f"CASE WHEN x IS NULL THEN NULL "
                f"WHEN sw IS NULL OR sw <= {_eps} THEN NULL "
                f"WHEN sdx2 / sw <= {_eps} THEN NULL "
                f"ELSE dx / SQRT(sdx2 / sw) END"
            )
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({sub4}) _4",
                has_inst_window=x_l.has_inst_window or w_l.has_inst_window,
                has_ts_partition=True,
            )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({sub3}) _3",
            has_inst_window=x_l.has_inst_window or w_l.has_inst_window,
            has_ts_partition=True,
        )

    if op in {
        "ts_prev_high",
        "ts_prev_low",
        "ts_distance_to_high",
        "ts_distance_to_low",
        "ts_breakout_high",
        "ts_breakdown_low",
        "ts_channel_position",
        "ts_new_high",
        "ts_new_low",
    }:
        # Prior-window 极值族：x.shift(1).rolling(w, min_periods=w)（排除当前 bar）。
        if len(node.inputs) < 1:
            return None
        x_l = _compile_layer(node.inputs[0], dialect=dialect)
        if x_l is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 1)
        lagged = f"SELECT ts, inst, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v FROM ({x_l.sql}) _t"
        hi = _inst_window(dialect, w, "MAX", lagged, min_periods=w)
        lo = _inst_window(dialect, w, "MIN", lagged, min_periods=w)
        if op == "ts_prev_high":
            return _Layer(hi, has_inst_window=True)
        if op == "ts_prev_low":
            return _Layer(lo, has_inst_window=True)
        if op == "ts_new_high":
            return _Layer(
                f"SELECT x.ts, x.inst, "
                f"CASE WHEN x._v > h._v THEN 1.0 ELSE 0.0 END AS _v "
                f"FROM ({x_l.sql}) x LEFT JOIN ({hi}) h USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "ts_new_low":
            return _Layer(
                f"SELECT x.ts, x.inst, "
                f"CASE WHEN x._v < l._v THEN 1.0 ELSE 0.0 END AS _v "
                f"FROM ({x_l.sql}) x LEFT JOIN ({lo}) l USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "ts_channel_position":
            return _Layer(
                f"SELECT x.ts, x.inst, "
                f"CASE WHEN h._v IS NULL OR l._v IS NULL OR (h._v - l._v) = 0 THEN NULL "
                f"ELSE (x._v - l._v) / (h._v - l._v) END AS _v "
                f"FROM ({x_l.sql}) x LEFT JOIN ({hi}) h USING (ts, inst) LEFT JOIN ({lo}) l USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "ts_breakout_high":
            return _Layer(
                f"SELECT x.ts, x.inst, "
                f"CASE WHEN h._v IS NULL THEN NULL WHEN h._v = 0 THEN NULL "
                f"ELSE {_g}(x._v / h._v - 1.0, 0.0) END AS _v "
                f"FROM ({x_l.sql}) x LEFT JOIN ({hi}) h USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "ts_breakdown_low":
            return _Layer(
                f"SELECT x.ts, x.inst, "
                f"CASE WHEN l._v IS NULL THEN NULL WHEN x._v = 0 THEN NULL "
                f"ELSE {_g}(l._v / x._v - 1.0, 0.0) END AS _v "
                f"FROM ({x_l.sql}) x LEFT JOIN ({lo}) l USING (ts, inst)",
                has_inst_window=True,
            )
        # ts_distance_to_high / ts_distance_to_low
        if op == "ts_distance_to_high":
            return _Layer(
                f"SELECT x.ts, x.inst, "
                f"CASE WHEN h._v IS NULL THEN NULL WHEN h._v = 0 THEN NULL "
                f"ELSE x._v / h._v - 1.0 END AS _v "
                f"FROM ({x_l.sql}) x LEFT JOIN ({hi}) h USING (ts, inst)",
                has_inst_window=True,
            )
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN l._v IS NULL THEN NULL WHEN l._v = 0 THEN NULL "
            f"ELSE x._v / l._v - 1.0 END AS _v "
            f"FROM ({x_l.sql}) x LEFT JOIN ({lo}) l USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"open_close_return", "overnight_return", "open_to_vwap_return", "vwap_to_close_return"}:
        if len(node.inputs) != 2:
            return None
        a = _compile_layer(node.inputs[0], dialect=dialect)
        b = _compile_layer(node.inputs[1], dialect=dialect)
        if a is None or b is None:
            return None
        num, den = "a._v", "b._v"
        if op in {"open_close_return", "open_to_vwap_return", "vwap_to_close_return"}:
            num, den = "b._v", "a._v"
        return _Layer(
            f"SELECT a.ts, a.inst, "
            f"CASE WHEN {num} IS NULL OR {den} IS NULL OR {den} = 0 THEN NULL "
            f"ELSE {num} / {den} - 1.0 END AS _v "
            f"FROM ({a.sql}) a LEFT JOIN ({b.sql}) b USING (ts, inst)",
            has_inst_window=a.has_inst_window or b.has_inst_window,
            has_ts_partition=a.has_ts_partition or b.has_ts_partition,
        )

    if op in {"limit_up_close", "limit_down_close"}:
        if len(node.inputs) < 2:
            return None
        close = _compile_layer(node.inputs[0], dialect=dialect)
        limit = _compile_layer(node.inputs[1], dialect=dialect)
        if close is None or limit is None:
            return None
        tol = float(_literal_positional(node, 2, default=0.005) or 0.005)
        if op == "limit_up_close":
            cond = f"close._v >= limit._v - {tol}"
        else:
            cond = f"close._v <= limit._v + {tol}"
        return _Layer(
            f"SELECT close.ts, close.inst, "
            f"CASE WHEN close._v IS NULL OR limit._v IS NULL THEN NULL "
            f"WHEN {cond} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({close.sql}) close LEFT JOIN ({limit.sql}) limit USING (ts, inst)",
            has_inst_window=close.has_inst_window or limit.has_inst_window,
            has_ts_partition=close.has_ts_partition or limit.has_ts_partition,
        )

    if op == "true_range":
        if len(node.inputs) < 3:
            return None
        h = _compile_layer(node.inputs[0], dialect=dialect)
        l = _compile_layer(node.inputs[1], dialect=dialect)
        c = _compile_layer(node.inputs[2], dialect=dialect)
        if h is None or l is None or c is None:
            return None
        lag_close = "LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {_g}(h._v - l._v, {_abs_fn}(h._v - {lag_close}), {_abs_fn}(l._v - {lag_close})) END"
        )
        return _Layer(
            f"SELECT h.ts, h.inst, {tr_expr} AS _v "
            f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
            f"INNER JOIN ({c.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"parkinson_vol", "garman_klass_vol", "rogers_satchell_vol", "yang_zhang_vol",
              "overnight_volatility", "intraday_volatility", "range_volatility", "ulcer_index",
              "high_low_spread_proxy"}:
        if len(node.inputs) < 1:
            return None
        # 只收集列输入（literal 参数跳过），按各算子实际签名赋予角色。
        col_inputs = [n for n in node.inputs if n.op != "literal"]
        if op == "ulcer_index":
            role_order = ("close",)
        elif op in {"parkinson_vol", "high_low_spread_proxy"}:
            role_order = ("high", "low")
        elif op in {"overnight_volatility", "intraday_volatility"}:
            role_order = ("open", "close")
        elif op == "range_volatility":
            role_order = ("high", "low", "close")
        else:  # garman_klass / rogers_satchell / yang_zhang
            role_order = ("open", "high", "low", "close")
        if len(col_inputs) < len(role_order):
            return None
        layer_map = {}
        for name, col_node in zip(role_order, col_inputs[: len(role_order)]):
            layer_map[name] = _compile_layer(col_node, dialect=dialect)
        if any(layer is None for layer in layer_map.values()):
            return None
        o, h, l, c = layer_map.get("open"), layer_map.get("high"), layer_map.get("low"), layer_map.get("close")
        ocol, hcol, lcol, ccol = "o._v", "h._v", "l._v", "c._v"
        def _log_div(num, den):
            return f"CASE WHEN {num} IS NULL OR {den} IS NULL OR {den} = 0 THEN NULL ELSE ln({num} / {den}) END"
        if op == "parkinson_vol":
            if h is None or l is None:
                return None
            w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
            rs = f"POWER({_log_div(hcol, lcol)}, 2)"
            inner = f"SELECT h.ts, h.inst, {rs} AS _v FROM ({h.sql}) h LEFT JOIN ({l.sql}) l USING (ts, inst)"
            mean = _inst_window(dialect, w, "AVG", inner, min_periods=w)
            var = f"SELECT ts, inst, _v / {2.772588722239781!r} AS _v FROM ({mean}) t"
            expr = f"POWER(GREATEST(_v, 0) * 252.0, 0.5)"
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({var}) t", has_inst_window=True
            )
        if op == "garman_klass_vol":
            if o is None or h is None or l is None or c is None:
                return None
            w = max(int(_literal_positional(node, 3, default=20) or 20), 2)
            hl = _log_div(hcol, lcol)
            co = _log_div(ccol, ocol)
            daily = f"(0.5 * POWER({hl}, 2) - {0.3862943611198906!r} * POWER({co}, 2))"
            inner = (
                f"SELECT h.ts, h.inst, {daily} AS _v "
                f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
                f"INNER JOIN ({c.sql}) c USING (ts, inst) INNER JOIN ({o.sql}) o USING (ts, inst)"
            )
            mean = _inst_window(dialect, w, "AVG", inner, min_periods=w)
            return _Layer(
                f"SELECT ts, inst, POWER(GREATEST(_v, 0) * 252.0, 0.5) AS _v FROM ({mean}) t",
                has_inst_window=True,
            )
        if op == "rogers_satchell_vol":
            if o is None or h is None or l is None or c is None:
                return None
            w = max(int(_literal_positional(node, 3, default=20) or 20), 2)
            ho = _log_div(hcol, ocol)
            hc = _log_div(hcol, ccol)
            lo = _log_div(lcol, ocol)
            lc = _log_div(lcol, ccol)
            daily = f"(({ho}) * ({hc}) + ({lo}) * ({lc}))"
            inner = (
                f"SELECT h.ts, h.inst, {daily} AS _v "
                f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
                f"INNER JOIN ({c.sql}) c USING (ts, inst) INNER JOIN ({o.sql}) o USING (ts, inst)"
            )
            mean = _inst_window(dialect, w, "AVG", inner, min_periods=w)
            return _Layer(
                f"SELECT ts, inst, POWER(GREATEST(_v, 0) * 252.0, 0.5) AS _v FROM ({mean}) t",
                has_inst_window=True,
            )
        if op == "yang_zhang_vol":
            if o is None or h is None or l is None or c is None:
                return None
            w = max(int(_literal_positional(node, 3, default=20) or 20), 3)
            prev_close = "LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
            overnight = _log_div(ocol, prev_close)
            intraday = _log_div(ccol, ocol)
            ho = _log_div(hcol, ocol)
            hc = _log_div(hcol, ccol)
            lo = _log_div(lcol, ocol)
            lc = _log_div(lcol, ccol)
            rs = f"(({ho}) * ({hc}) + ({lo}) * ({lc}))"
            k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
            raw = (
                f"SELECT h.ts, h.inst, "
                f"{overnight} AS ov, {intraday} AS intr, {rs} AS rs_v "
                f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
                f"INNER JOIN ({c.sql}) c USING (ts, inst) INNER JOIN ({o.sql}) o USING (ts, inst)"
            )
            over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(ov) OVER ({over})"
            ov_var = f"VAR_SAMP(ov) OVER ({over})"
            in_var = f"VAR_SAMP(intr) OVER ({over})"
            rs_mean = f"AVG(rs_v) OVER ({over})"
            var_expr = (
                f"CASE WHEN {cnt} < {w} THEN NULL "
                f"ELSE {ov_var} + {k!r} * {in_var} + {1.0 - k!r} * {rs_mean} END"
            )
            return _Layer(
                f"SELECT ts, inst, POWER(GREATEST({var_expr}, 0) * 252.0, 0.5) AS _v FROM ({raw}) t",
                has_inst_window=True,
            )
        if op == "overnight_volatility":
            if o is None or c is None:
                return None
            w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
            prev_close = "LAG(c._v, 1) OVER (PARTITION BY o.inst ORDER BY o.ts)"
            inner = (
                f"SELECT o.ts, o.inst, {_log_div(ocol, prev_close)} AS _v "
                f"FROM ({o.sql}) o LEFT JOIN ({c.sql}) c USING (ts, inst)"
            )
            over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(_v) OVER ({over})"
            std = _dialect_fn(dialect, "stddev")
            expr = f"CASE WHEN {cnt} < {w} THEN NULL ELSE {std}(_v) OVER ({over}) * POWER(252.0, 0.5) END"
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({inner}) t", has_inst_window=True
            )
        if op == "intraday_volatility":
            if o is None or c is None:
                return None
            w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
            inner = (
                f"SELECT o.ts, o.inst, {_log_div(ccol, ocol)} AS _v "
                f"FROM ({o.sql}) o LEFT JOIN ({c.sql}) c USING (ts, inst)"
            )
            over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(_v) OVER ({over})"
            std = _dialect_fn(dialect, "stddev")
            expr = f"CASE WHEN {cnt} < {w} THEN NULL ELSE {std}(_v) OVER ({over}) * POWER(252.0, 0.5) END"
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({inner}) t", has_inst_window=True
            )
        if op == "range_volatility":
            if h is None or l is None or c is None:
                return None
            w = max(int(_literal_positional(node, 2, default=20) or 20), 2)
            x = f"CASE WHEN {ccol} IS NULL OR {ccol} = 0 THEN NULL ELSE ({hcol} - {lcol}) / ABS({ccol}) END"
            inner = (
                f"SELECT h.ts, h.inst, {x} AS _v "
                f"FROM ({h.sql}) h INNER JOIN ({l.sql}) l USING (ts, inst) "
                f"INNER JOIN ({c.sql}) c USING (ts, inst)"
            )
            over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            cnt = f"COUNT(_v) OVER ({over})"
            std = _dialect_fn(dialect, "stddev")
            expr = f"CASE WHEN {cnt} < {w} THEN NULL ELSE {std}(_v) OVER ({over}) * POWER(252.0, 0.5) END"
            return _Layer(
                f"SELECT ts, inst, {expr} AS _v FROM ({inner}) t", has_inst_window=True
            )
        if op == "ulcer_index":
            if c is None:
                return None
            w = max(int(_literal_positional(node, 0, default=20) or 20), 2)
            rolling_high = _inst_window(dialect, w, "MAX", c.sql, min_periods=w)
            dd_sql = (
                f"SELECT ts, inst, POWER(100.0 * (_v / NULLIF(rh._v, 0) - 1.0), 2) AS _v "
                f"FROM ({c.sql}) x JOIN ({rolling_high}) rh USING (ts, inst)"
            )
            mean = _inst_window(dialect, w, "AVG", dd_sql, min_periods=w)
            return _Layer(
                f"SELECT ts, inst, POWER(GREATEST(_v, 0), 0.5) AS _v FROM ({mean}) t",
                has_inst_window=True,
            )
        if op == "high_low_spread_proxy":
            if h is None or l is None:
                return None
            w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
            inner = (
                f"SELECT h.ts, h.inst, {_log_div(hcol, lcol)} AS _v "
                f"FROM ({h.sql}) h LEFT JOIN ({l.sql}) l USING (ts, inst)"
            )
            return _Layer(_inst_window(dialect, w, "AVG", inner, min_periods=w), has_inst_window=True)

    # ------------------------------------------------------------------
    # Volume / turnover rolling ops.
    # ------------------------------------------------------------------
    if op in {"average_volume", "ts_average_volume", "average_turnover"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 1)
        return _Layer(_inst_window(dialect, w, "AVG", inner.sql, min_periods=w), has_inst_window=True)

    if op == "volume_to_range":
        # volume_to_range(volume, high, low, window)
        if len(node.inputs) < 3:
            return None
        vol = _compile_layer(node.inputs[0], dialect=dialect)
        h = _compile_layer(node.inputs[1], dialect=dialect)
        l = _compile_layer(node.inputs[2], dialect=dialect)
        if vol is None or h is None or l is None:
            return None
        w = max(int(_literal_positional(node, 3, default=20) or 20), 1)
        raw = (
            f"SELECT v.ts, v.inst, v._v / NULLIF(h._v - l._v, 0) AS _v "
            f"FROM ({vol.sql}) v LEFT JOIN ({h.sql}) h USING (ts, inst) "
            f"LEFT JOIN ({l.sql}) l USING (ts, inst)"
        )
        return _Layer(_inst_window(dialect, w, "AVG", raw, min_periods=w), has_inst_window=True)

    if op in {"abnormal_volume", "abnormal_turnover", "relative_volume"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 1)
        base = (
            f"SELECT ts, inst, _v, "
            f"AVG(LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts)) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS prev_avg "
            f"FROM ({inner.sql}) t"
        )
        if op == "relative_volume":
            expr = f"CASE WHEN prev_avg IS NULL THEN NULL WHEN prev_avg = 0 THEN NULL ELSE _v / prev_avg END"
        else:
            expr = f"CASE WHEN prev_avg IS NULL THEN NULL WHEN prev_avg = 0 THEN NULL ELSE _v / prev_avg - 1.0 END"
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({base}) t", has_inst_window=True
        )

    if op in {"volume_zscore", "turnover_zscore", "volume_shock", "turnover_shock"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 2)
        return _Layer(
            _prior_window_zscore_sql(inner.sql, w, dialect=dialect, min_periods=w),
            has_inst_window=True,
        )

    if op in {"volume_momentum", "turnover_momentum"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 1)
        lag = f"LAG(_v, {w}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {lag} IS NULL THEN NULL WHEN {lag} = 0 THEN NULL "
            f"ELSE _v / {lag} - 1.0 END AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op in {"volume_volatility", "turnover_volatility"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 2)
        pct = (
            f"SELECT ts, inst, "
            f"(_v / NULLIF(LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0) AS _v "
            f"FROM ({inner.sql}) t"
        )
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        cnt = f"COUNT(_v) OVER ({over})"
        std = _dialect_fn(dialect, "stddev")
        expr = f"CASE WHEN {cnt} < {w} THEN NULL ELSE {std}(_v) OVER ({over}) END"
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({pct}) t", has_inst_window=True
        )

    if op in {"volume_autocorr", "turnover_autocorr"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 3)
        lag = max(int(_literal_positional(node, 1, default=1) or 1), 1)
        return _Layer(
            _rolling_corr_lag_sql(inner.sql, w, lag, dialect=dialect, min_periods=w),
            has_inst_window=True,
        )

    if op in {"volume_acceleration", "turnover_acceleration"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        short_w = max(int(_literal_positional(node, 0, default=5) or 5), 1)
        long_w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        short_mean = _inst_window(dialect, short_w, "AVG", inner.sql, min_periods=short_w)
        long_mean = _inst_window(dialect, long_w, "AVG", inner.sql, min_periods=long_w)
        return _Layer(
            f"SELECT s.ts, s.inst, s._v / NULLIF(l._v, 0) - 1.0 AS _v "
            f"FROM ({short_mean}) s JOIN ({long_mean}) l USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"up_volume_ratio", "down_volume_ratio", "signed_volume_imbalance", "up_down_volume_ratio"}:
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or vol is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        base = (
            f"SELECT r.ts, r.inst, r._v AS ret, v._v AS vol "
            f"FROM ({ret.sql}) r LEFT JOIN ({vol.sql}) v USING (ts, inst)"
        )
        up = _inst_window(
            dialect, w, "SUM",
            f"SELECT ts, inst, CASE WHEN ret > 0 THEN vol ELSE 0.0 END AS _v FROM ({base}) t",
            min_periods=w,
        )
        dn = _inst_window(
            dialect, w, "SUM",
            f"SELECT ts, inst, CASE WHEN ret < 0 THEN vol ELSE 0.0 END AS _v FROM ({base}) t",
            min_periods=w,
        )
        tot = _inst_window(
            dialect, w, "SUM",
            f"SELECT ts, inst, ABS(vol) AS _v FROM ({base}) t",
            min_periods=w,
        )
        if op == "up_volume_ratio":
            return _Layer(
                f"SELECT u.ts, u.inst, u._v / NULLIF(t._v, 0) AS _v "
                f"FROM ({up}) u JOIN ({tot}) t USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "down_volume_ratio":
            return _Layer(
                f"SELECT d.ts, d.inst, d._v / NULLIF(t._v, 0) AS _v "
                f"FROM ({dn}) d JOIN ({tot}) t USING (ts, inst)",
                has_inst_window=True,
            )
        if op == "signed_volume_imbalance":
            return _Layer(
                f"SELECT u.ts, u.inst, (u._v / NULLIF(t._v, 0) - d._v / NULLIF(t2._v, 0)) AS _v "
                f"FROM ({up}) u JOIN ({tot}) t USING (ts, inst) "
                f"JOIN ({dn}) d USING (ts, inst) JOIN ({tot}) t2 USING (ts, inst)",
                has_inst_window=True,
            )
        return _Layer(
            f"SELECT u.ts, u.inst, (u._v / NULLIF(d._v, 0)) AS _v "
            f"FROM ({up}) u JOIN ({dn}) d USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"volume_weighted_return", "volume_weighted_momentum"}:
        if len(node.inputs) < 2:
            return None
        a = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if a is None or vol is None:
            return None
        w = max(int(_literal_positional(node, 2, default=20) or 20), 1)
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        if op == "volume_weighted_momentum":
            val = (
                f"SELECT ts, inst, "
                f"(_v / NULLIF(LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0) AS _v "
                f"FROM ({a.sql}) t"
            )
        else:
            val = a.sql
        pair = f"SELECT x.ts, x.inst, x._v * v._v AS pv, ABS(v._v) AS av FROM ({val}) x LEFT JOIN ({vol.sql}) v USING (ts, inst)"
        num = _inst_window(dialect, w, "SUM", f"SELECT ts, inst, pv AS _v FROM ({pair}) t", min_periods=w)
        den = _inst_window(dialect, w, "SUM", f"SELECT ts, inst, av AS _v FROM ({pair}) t", min_periods=w)
        return _Layer(
            f"SELECT n.ts, n.inst, n._v / NULLIF(d._v, 0) AS _v "
            f"FROM ({num}) n JOIN ({den}) d USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"rolling_vwap", "vwap_deviation"}:
        if len(node.inputs) < 2:
            return None
        price = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if price is None or vol is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        pair = f"SELECT x.ts, x.inst, x._v * v._v AS pv, v._v AS av FROM ({price.sql}) x LEFT JOIN ({vol.sql}) v USING (ts, inst)"
        num = _inst_window(dialect, w, "SUM", f"SELECT ts, inst, pv AS _v FROM ({pair}) t", min_periods=w)
        den = _inst_window(dialect, w, "SUM", f"SELECT ts, inst, av AS _v FROM ({pair}) t", min_periods=w)
        vwap = f"SELECT n.ts, n.inst, n._v / NULLIF(d._v, 0) AS _v FROM ({num}) n JOIN ({den}) d USING (ts, inst)"
        if op == "rolling_vwap":
            return _Layer(vwap, has_inst_window=True)
        return _Layer(
            f"SELECT v.ts, v.inst, p._v / NULLIF(v._v, 0) - 1.0 AS _v "
            f"FROM ({price.sql}) p JOIN ({vwap}) v USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"rolling_obv", "rolling_pvt"}:
        if len(node.inputs) < 2:
            return None
        close = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if close is None or vol is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        if op == "rolling_obv":
            flow = (
                f"SELECT c.ts, c.inst, "
                f"CASE WHEN c._v IS NULL THEN NULL "
                f"WHEN c._v > LAG(c._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts) THEN v._v "
                f"WHEN c._v < LAG(c._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts) THEN -v._v "
                f"ELSE 0.0 END AS _v "
                f"FROM ({close.sql}) c LEFT JOIN ({vol.sql}) v USING (ts, inst)"
            )
        else:
            flow = (
                f"SELECT c.ts, c.inst, "
                f"(c._v / NULLIF(LAG(c._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts), 0) - 1.0) * v._v AS _v "
                f"FROM ({close.sql}) c LEFT JOIN ({vol.sql}) v USING (ts, inst)"
            )
        return _Layer(_inst_window(dialect, w, "SUM", flow, min_periods=w), has_inst_window=True)

    if op in {"signed_volume", "signed_dollar_volume"}:
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        if op == "signed_volume":
            vol = _compile_layer(node.inputs[1], dialect=dialect)
            if vol is None:
                return None
            return _Layer(
                f"SELECT r.ts, r.inst, (CASE WHEN r._v > 0 THEN 1.0 WHEN r._v < 0 THEN -1.0 ELSE 0.0 END) * v._v AS _v "
                f"FROM ({ret.sql}) r LEFT JOIN ({vol.sql}) v USING (ts, inst)",
                has_inst_window=ret.has_inst_window or vol.has_inst_window,
                has_ts_partition=ret.has_ts_partition or vol.has_ts_partition,
            )
        close = _compile_layer(node.inputs[1], dialect=dialect)
        vol = _compile_layer(node.inputs[2], dialect=dialect)
        if close is None or vol is None:
            return None
        return _Layer(
            f"SELECT r.ts, r.inst, (CASE WHEN r._v > 0 THEN 1.0 WHEN r._v < 0 THEN -1.0 ELSE 0.0 END) * c._v * v._v AS _v "
            f"FROM ({ret.sql}) r LEFT JOIN ({close.sql}) c USING (ts, inst) LEFT JOIN ({vol.sql}) v USING (ts, inst)",
            has_inst_window=ret.has_inst_window or close.has_inst_window or vol.has_inst_window,
            has_ts_partition=ret.has_ts_partition or close.has_ts_partition or vol.has_ts_partition,
        )

    if op in {"dollar_volume", "signed_dollar_volume"}:
        # dollar_volume = close * volume（无窗口）。
        if len(node.inputs) < 2:
            return None
        c = _compile_layer(node.inputs[0], dialect=dialect)
        v = _compile_layer(node.inputs[1], dialect=dialect)
        if c is None or v is None:
            return None
        return _Layer(
            f"SELECT c.ts, c.inst, (c._v * v._v) AS _v "
            f"FROM ({c.sql}) c LEFT JOIN ({v.sql}) v USING (ts, inst)",
            has_inst_window=c.has_inst_window or v.has_inst_window,
            has_ts_partition=c.has_ts_partition or v.has_ts_partition,
        )

    if op == "dollar_volume_zscore":
        if len(node.inputs) < 2:
            return None
        c = _compile_layer(node.inputs[0], dialect=dialect)
        v = _compile_layer(node.inputs[1], dialect=dialect)
        if c is None or v is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
        dv = f"SELECT c.ts, c.inst, (c._v * v._v) AS _v FROM ({c.sql}) c LEFT JOIN ({v.sql}) v USING (ts, inst)"
        return _Layer(
            _prior_window_zscore_sql(dv, w, dialect=dialect, min_periods=w),
            has_inst_window=True,
        )

    if op == "adv":
        if len(node.inputs) < 2:
            return None
        c = _compile_layer(node.inputs[0], dialect=dialect)
        v = _compile_layer(node.inputs[1], dialect=dialect)
        if c is None or v is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        dv = f"SELECT c.ts, c.inst, (ABS(c._v) * v._v) AS _v FROM ({c.sql}) c LEFT JOIN ({v.sql}) v USING (ts, inst)"
        return _Layer(_inst_window(dialect, w, "AVG", dv, min_periods=w), has_inst_window=True)

    if op == "amihud_illiquidity":
        if len(node.inputs) < 3:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        c = _compile_layer(node.inputs[1], dialect=dialect)
        v = _compile_layer(node.inputs[2], dialect=dialect)
        if ret is None or c is None or v is None:
            return None
        w = max(int(_literal_positional(node, 2, default=20) or 20), 1)
        inner = (
            f"SELECT r.ts, r.inst, "
            f"ABS(r._v) / NULLIF(ABS(c._v) * v._v, 0) AS _v "
            f"FROM ({ret.sql}) r LEFT JOIN ({c.sql}) c USING (ts, inst) LEFT JOIN ({v.sql}) v USING (ts, inst)"
        )
        return _Layer(_inst_window(dialect, w, "AVG", inner, min_periods=w), has_inst_window=True)

    if op == "price_impact":
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        dv = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or dv is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        inner = (
            f"SELECT r.ts, r.inst, ABS(r._v) / NULLIF(d._v, 0) AS _v "
            f"FROM ({ret.sql}) r LEFT JOIN ({dv.sql}) d USING (ts, inst)"
        )
        return _Layer(_inst_window(dialect, w, "AVG", inner, min_periods=w), has_inst_window=True)

    if op == "return_per_turnover":
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        to = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or to is None:
            return None
        return _Layer(
            f"SELECT r.ts, r.inst, r._v / NULLIF(t._v, 0) AS _v "
            f"FROM ({ret.sql}) r LEFT JOIN ({to.sql}) t USING (ts, inst)",
            has_inst_window=ret.has_inst_window or to.has_inst_window,
            has_ts_partition=ret.has_ts_partition or to.has_ts_partition,
        )

    if op == "return_volume_corr":
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or vol is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
        pair = f"SELECT r.ts, r.inst, r._v AS lv, v._v AS rv FROM ({ret.sql}) r LEFT JOIN ({vol.sql}) v USING (ts, inst)"
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        mask = "lv IS NOT NULL AND rv IS NOT NULL"
        cnt = f"COUNT(CASE WHEN {mask} THEN 1 END) OVER ({over})"
        corr = f"CORR(CASE WHEN {mask} THEN lv END, CASE WHEN {mask} THEN rv END) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN {cnt} < {w} THEN NULL ELSE {corr} END AS _v FROM ({pair}) t",
            has_inst_window=True,
        )

    if op in {"return_volume_beta", "return_turnover_beta"}:
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or vol is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 3)
        vc = (
            f"SELECT ts, inst, "
            f"(_v / NULLIF(LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0) AS _v "
            f"FROM ({vol.sql}) t"
        )
        pair = f"SELECT r.ts, r.inst, r._v AS lv, v._v AS rv FROM ({ret.sql}) r LEFT JOIN ({vc}) v USING (ts, inst)"
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        mask = "lv IS NOT NULL AND rv IS NOT NULL"
        cnt = f"COUNT(CASE WHEN {mask} THEN 1 END) OVER ({over})"
        cov = f"COVAR_SAMP(CASE WHEN {mask} THEN lv END, CASE WHEN {mask} THEN rv END) OVER ({over})"
        var = f"VAR_SAMP(CASE WHEN {mask} THEN rv END) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN {cnt} < {w} THEN NULL ELSE {cov} / NULLIF({var}, 0) END AS _v FROM ({pair}) t",
            has_inst_window=True,
        )

    if op in {"price_volume_divergence", "price_turnover_divergence"}:
        if len(node.inputs) < 2:
            return None
        price = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if price is None or vol is None:
            return None
        pw = max(int(_literal_positional(node, 1, default=20) or 20), 1)
        vw = max(int(_literal_positional(node, 2, default=20) or 20), 1)
        pr = f"(_v / NULLIF(LAG(_v, {pw}) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0)"
        vr = f"(_v / NULLIF(LAG(_v, {vw}) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0)"
        return _Layer(
            f"SELECT p.ts, p.inst, ({pr} - {vr}) AS _v "
            f"FROM ({price.sql}) p LEFT JOIN ({vol.sql}) v USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "turnover_adjusted_volatility":
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        to = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or to is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        vol_expr = f"{_dialect_fn(dialect, 'stddev')}(ret._v) OVER ({over})"
        act_expr = f"AVG(t._v) OVER ({over})"
        cnt_v = f"COUNT(ret._v) OVER ({over})"
        cnt_a = f"COUNT(t._v) OVER ({over})"
        return _Layer(
            f"SELECT ret.ts, ret.inst, "
            f"CASE WHEN {cnt_v} < {w} OR {cnt_a} < {w} THEN NULL "
            f"ELSE {vol_expr} / NULLIF({act_expr}, 0) END AS _v "
            f"FROM ({ret.sql}) ret LEFT JOIN ({to.sql}) t USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"zero_return_ratio"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 1)
        eps = float(_literal_positional(node, 1, default=1e-12) or 1e-12)
        flag = f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL WHEN ABS(_v) <= {eps} THEN 1.0 ELSE 0.0 END AS _v FROM ({inner.sql}) t"
        return _Layer(_inst_window(dialect, w, "AVG", flag, min_periods=w), has_inst_window=True)

    if op == "roll_spread_proxy":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(int(_literal_positional(node, 0, default=20) or 20), 3)
        pair = f"SELECT ts, inst, _v, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS pv FROM ({inner.sql}) t"
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        mask = "_v IS NOT NULL AND pv IS NOT NULL"
        cnt = f"COUNT(CASE WHEN {mask} THEN 1 END) OVER ({over})"
        cov = f"COVAR_SAMP(CASE WHEN {mask} THEN _v END, CASE WHEN {mask} THEN pv END) OVER ({over})"
        expr = f"2.0 * POWER(GREATEST(-({cov}), 0), 0.5)"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN {cnt} < {w} THEN NULL ELSE {expr} END AS _v FROM ({pair}) t",
            has_inst_window=True,
        )

    if op == "corwin_schultz_spread":
        if len(node.inputs) < 2:
            return None
        h = _compile_layer(node.inputs[0], dialect=dialect)
        l = _compile_layer(node.inputs[1], dialect=dialect)
        if h is None or l is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
        ph = f"LAG(h._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        pl = f"LAG(l._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
        loghl = f"CASE WHEN h._v IS NULL OR l._v IS NULL OR l._v = 0 THEN NULL ELSE ln(h._v / l._v) END"
        loghl_p = f"CASE WHEN {ph} IS NULL OR {pl} IS NULL OR {pl} = 0 THEN NULL ELSE ln({ph} / {pl}) END"
        beta = f"(({loghl}) * ({loghl}) + ({loghl_p}) * ({loghl_p}))"
        h2 = f"GREATEST(h._v, {ph})"
        l2 = f"LEAST(l._v, {pl})"
        gamma = f"POWER(CASE WHEN {h2} IS NULL OR {l2} = 0 THEN NULL ELSE ln({h2} / {l2}) END, 2)"
        den = 3.0 - 2.0 * math.sqrt(2.0)
        alpha = (
            f"(POWER(2.0 * {beta}, 0.5) - POWER({beta}, 0.5)) / {den!r} - POWER({gamma} / {den!r}, 0.5)"
        )
        alpha_c = f"GREATEST({alpha}, 0.0)"
        daily = f"(2.0 * (EXP({alpha_c}) - 1.0) / (1.0 + EXP({alpha_c})))"
        raw = f"SELECT h.ts, h.inst, {daily} AS _v FROM ({h.sql}) h LEFT JOIN ({l.sql}) l USING (ts, inst)"
        return _Layer(_inst_window(dialect, w, "AVG", raw, min_periods=w), has_inst_window=True)

    if op == "bounded_nvi":
        if len(node.inputs) < 2:
            return None
        c = _compile_layer(node.inputs[0], dialect=dialect)
        v = _compile_layer(node.inputs[1], dialect=dialect)
        if c is None or v is None:
            return None
        w = max(int(_literal_positional(node, 1, default=20) or 20), 2)
        r = f"(c._v / NULLIF(LAG(c._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts), 0) - 1.0)"
        flag = (
            f"SELECT c.ts, c.inst, "
            f"CASE WHEN v._v < LAG(v._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts) "
            f"THEN GREATEST({r}, -0.999999) ELSE 0.0 END AS _v "
            f"FROM ({c.sql}) c LEFT JOIN ({v.sql}) v USING (ts, inst)"
        )
        logsum = f"SELECT ts, inst, LN(1.0 + _v) AS _v FROM ({flag}) t"
        agg = _inst_window(dialect, w, "SUM", logsum, min_periods=w)
        return _Layer(
            f"SELECT ts, inst, EXP(_v) - 1.0 AS _v FROM ({agg}) t", has_inst_window=True
        )

    if op == "bounded_pvi":
        if len(node.inputs) < 2:
            return None
        c = _compile_layer(node.inputs[0], dialect=dialect)
        v = _compile_layer(node.inputs[1], dialect=dialect)
        if c is None or v is None:
            return None
        w = max(int(_literal_positional(node, 2, default=20) or 20), 2)
        r = f"(c._v / NULLIF(LAG(c._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts), 0) - 1.0)"
        flag = (
            f"SELECT c.ts, c.inst, "
            f"CASE WHEN v._v > LAG(v._v, 1) OVER (PARTITION BY c.inst ORDER BY c.ts) "
            f"THEN GREATEST({r}, -0.999999) ELSE 0.0 END AS _v "
            f"FROM ({c.sql}) c LEFT JOIN ({v.sql}) v USING (ts, inst)"
        )
        logsum = f"SELECT ts, inst, LN(1.0 + _v) AS _v FROM ({flag}) t"
        agg = _inst_window(dialect, w, "SUM", logsum, min_periods=w)
        return _Layer(
            f"SELECT ts, inst, EXP(_v) - 1.0 AS _v FROM ({agg}) t", has_inst_window=True
        )

    if op == "where" or op == "if_else":
        if len(node.inputs) != 3:
            return None
        cond = _compile_layer(node.inputs[0], dialect=dialect)
        a = _compile_layer(node.inputs[1], dialect=dialect)
        b = _compile_layer(node.inputs[2], dialect=dialect)
        if cond is None or a is None or b is None:
            return None
        # Audit #19: NULL/NaN condition stays missing — `where` is a value
        # selector and must not silently take the else branch (pandas reference
        # ``ScalarBroadcastWhere`` propagates unknown conditions).
        isnan_fn = _dialect_fn(dialect, "isnan")
        return _Layer(
            f"SELECT c.ts, c.inst, "
            f"CASE WHEN c._v IS NULL OR {isnan_fn}(c._v) THEN NULL "
            f"WHEN {_truthy_sql('c._v')} THEN a._v ELSE b._v END AS _v "
            f"FROM ({cond.sql}) c "
            f"INNER JOIN ({a.sql}) a USING (ts, inst) "
            f"INNER JOIN ({b.sql}) b USING (ts, inst)",
            has_inst_window=cond.has_inst_window or a.has_inst_window or b.has_inst_window,
            has_ts_partition=cond.has_ts_partition or a.has_ts_partition or b.has_ts_partition,
        )

    if op in {
        "group_rank",
        "group_mean",
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
        "group_zscore",
        "group_normalize",
        "group_std",
    }:
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        grp_layer = None
        if len(node.inputs) >= 2:
            grp_layer = _compile_layer(node.inputs[1], dialect=dialect)
            if grp_layer is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x LEFT JOIN ({grp_layer.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        if op == "group_rank":
            wrapped, keys = _group_partition_wrap(
                inner.sql,
                grp_layer.sql if grp_layer is not None else None,
            )
            return _Layer(
                _cs_average_rank_pct_sql(wrapped, partition_keys=keys, dialect=dialect),
                has_inst_window=inner.has_inst_window,
                has_ts_partition=True,
            )
        if op == "group_mean":
            # NaN/NULL members stay NaN/NULL (audit #19): the window aggregate
            # is only assigned to finite members, mirroring the pandas
            # reference and the group_sum/group_std guards below.
            expr = f"CASE WHEN x._v IS NULL THEN NULL ELSE AVG(x._v) OVER ({part}) END"
        elif op == "group_sum":
            expr = f"CASE WHEN x._v IS NULL THEN NULL ELSE SUM(x._v) OVER ({part}) END"
        elif op == "group_min":
            expr = f"CASE WHEN x._v IS NULL THEN NULL ELSE MIN(x._v) OVER ({part}) END"
        elif op == "group_max":
            expr = f"CASE WHEN x._v IS NULL THEN NULL ELSE MAX(x._v) OVER ({part}) END"
        elif op == "group_count":
            expr = (
                f"CASE WHEN x._v IS NULL THEN NULL "
                f"ELSE CAST(COUNT(x._v) OVER ({part}) AS DOUBLE) END"
            )
        elif op == "group_std":
            std_fn = _dialect_fn(dialect, "stddev")
            cnt = f"COUNT(x._v) OVER ({part})"
            std_val = f"{std_fn}(x._v) OVER ({part})"
            expr = (
                f"CASE WHEN x._v IS NULL THEN NULL "
                f"WHEN {cnt} < 2 THEN 0.0 "
                f"WHEN {std_val} IS NULL THEN 0.0 "
                f"ELSE {std_val} END"
            )
        elif op == "group_zscore":
            expr = _group_zscore_expr(value_col="x._v", partition=part, dialect=dialect)
        elif op == "group_normalize":
            expr = _group_minmax_expr(value_col="x._v", partition=part, dialect=dialect)
        else:
            expr = "NULL"
        return _Layer(
            f"SELECT x.ts, x.inst, {expr} AS _v {join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_percentile":
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_positional(node, 1)
        if pos_p is not None:
            p = pos_p
        if len(node.inputs) < 2:
            return None
        grp_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if grp_layer is None:
            return None
        side = str(node.attrs.get("side", "top")).lower()
        if side not in {"top", "bottom"}:
            return None
        wrapped, keys = _group_partition_wrap(
            inner.sql,
            grp_layer.sql if grp_layer is not None else None,
            value_alias="_oval",
        )
        keys_csv = ", ".join(keys)
        numbered = (
            f"SELECT ts, inst, _oval, {keys_csv}, _oval AS _v "
            f"FROM ({wrapped}) t0"
        )
        rank_frac = _average_rank_frac_correlated(
            row_value_col="b._v",
            partition_keys=keys,
            numbered_sql=numbered,
            row_alias="b",
            dialect=dialect,
            descending=side == "top",
        )
        expr = (
            f"CASE WHEN b._oval IS NULL THEN NULL "
            f"WHEN ({rank_frac}) <= {p} THEN 1.0 ELSE 0.0 END"
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({numbered}) b",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"group_skewness", "group_kurtosis"}:
        # 组内截面偏度/峰度：population std（ddof=0），赋值给组内全部成员
        # （含 NaN 成员，对齐 _group_shape 的 result.iloc[mask]=value）。
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if grp is None:
            return None
        part = "PARTITION BY x.ts, g._v"
        join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
        std_pop = _dialect_fn(dialect, "stddev_pop")
        need = 3 if op == "group_skewness" else 4
        order = 3 if op == "group_skewness" else 4
        base = (
            f"SELECT x.ts, x.inst, x._v, g._v AS _grp, "
            f"AVG(x._v) OVER ({part}) AS _m, "
            f"COUNT(x._v) OVER ({part}) AS _cnt, "
            f"{std_pop}(x._v) OVER ({part}) AS _std "
            f"{join}"
        )
        part2 = "PARTITION BY ts, _grp"
        expr = (
            f"CASE WHEN _cnt < {need} THEN NULL "
            f"WHEN _std IS NULL OR _std < 1e-12 THEN NULL "
            f"ELSE AVG(POWER(_v - _m, {order})) OVER ({part2}) / POWER(_std, {order}) END"
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({base}) m",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_quantile_spread":
        # 组内 (Q_high - Q_low)，线性插值分位数（np.quantile），赋值给全部成员。
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if grp is None:
            return None
        ql = _float_attr(node, "q_low", default=0.25)
        qh = _float_attr(node, "q_high", default=0.75)
        if not 0.0 < ql < qh < 1.0:
            return None
        part = "PARTITION BY x.ts, g._v"
        join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
        std_pop = _dialect_fn(dialect, "stddev_pop")
        base = (
            f"SELECT x.ts, x.inst, x._v, "
            f"COUNT(x._v) OVER ({part}) AS _cnt, "
            f"{std_pop}(x._v) OVER ({part}) AS _std, "
            f"quantile_cont(x._v, {ql}) OVER ({part}) AS _qlo, "
            f"quantile_cont(x._v, {qh}) OVER ({part}) AS _qhi "
            f"{join}"
        )
        expr = (
            f"CASE WHEN _cnt < 3 THEN NULL "
            f"WHEN _std IS NULL OR _std < 1e-12 THEN NULL "
            f"ELSE _qhi - _qlo END"
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({base}) m",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_ex_self_mean":
        # leave-one-out peer mean：仅有限成员有值；(total - x)/(count - 1)。
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if grp is None:
            return None
        part = "PARTITION BY x.ts, g._v"
        join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
        total = f"SUM(x._v) OVER ({part})"
        cnt = f"COUNT(x._v) OVER ({part})"
        expr = (
            f"CASE WHEN x._v IS NULL THEN NULL "
            f"WHEN {cnt} <= 1 THEN NULL "
            f"ELSE ({total} - x._v) / ({cnt} - 1) END"
        )
        return _Layer(
            f"SELECT x.ts, x.inst, {expr} AS _v {join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_decay_linear":
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        expr = _group_decay_linear_expr(value_col="x._v", partition=part, dialect=dialect)
        return _Layer(
            f"SELECT x.ts, x.inst, {expr} AS _v {join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_median":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        med = "median" if dialect == SqlDialect.DUCKDB else "median"
        return _Layer(
            _inst_window(dialect, spec.size, med, inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_var":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        if spec.ddof == 0:
            var_fn = "VAR_POP" if dialect == SqlDialect.DUCKDB else "varPop"
        else:
            var_fn = "VAR_SAMP" if dialect == SqlDialect.DUCKDB else "varSamp"
        return _Layer(
            _inst_window(dialect, spec.size, var_fn, inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "winsorize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        # pandas 参考用 lower/upper（非对称），group_winsorize 才用 a。SQL 分支
        # 原来读 ``a``（对称 5%/95%），与 pandas lower/upper 语义不一致。
        lo = _float_attr(node, "lower", "lo", default=0.05)
        hi = _float_attr(node, "upper", "hi", default=0.95)
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        q_lo = _quantile_over(dialect, "_v", lo, "PARTITION BY ts")
        q_hi = _quantile_over(dialect, "_v", hi, "PARTITION BY ts")
        clip = f"{g}({q_lo}, {l}({q_hi}, _v))"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE {clip} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_winsorize":
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        a = _float_attr(node, "a", "p", default=0.05)
        lo = a
        hi = 1.0 - a
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        q_lo = _quantile_over(dialect, "x._v", lo, part)
        q_hi = _quantile_over(dialect, "x._v", hi, part)
        clip = f"{g}({q_lo}, {l}({q_hi}, x._v))"
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN x._v IS NULL THEN NULL ELSE {clip} END AS _v "
            f"{join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_mad":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        w = spec.size
        mp = spec.min_periods if node.attrs.get("min_periods") is not None else w
        scale = _float_attr(node, "scale", default=1.0)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.DUCKDB:
            vals = (
                f"list(_v) FILTER (WHERE _v IS NOT NULL AND isfinite(_v)) "
                f"OVER ({over})"
            )
            inf_count = (
                f"COUNT(*) FILTER (WHERE _v IS NOT NULL AND isinf(_v)) "
                f"OVER ({over})"
            )
            return _Layer(
                f"SELECT ts, inst, "
                f"CASE WHEN _inf_count > 0 OR length(_vals) < {mp} THEN NULL "
                f"ELSE {scale} * list_median(list_transform("
                f"_vals, x -> abs(x - list_median(_vals)))) END AS _v "
                f"FROM (SELECT ts, inst, {vals} AS _vals, "
                f"{inf_count} AS _inf_count FROM ({inner.sql}) t0) t",
                has_inst_window=True,
            )
        vals = f"groupArrayIf(_v, isFinite(_v)) OVER ({over})"
        inf_count = f"countIf(isInfinite(_v)) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, "
            f"if(_inf_count > 0 OR length(_vals) < {mp}, NULL, {scale} * "
            f"arrayReduce('medianExact', arrayMap("
            f"x -> abs(x - arrayReduce('medianExact', _vals)), _vals))) AS _v "
            f"FROM (SELECT ts, inst, {vals} AS _vals, "
            f"{inf_count} AS _inf_count FROM ({inner.sql}) t0) t",
            has_inst_window=True,
        )

    if op in {"ts_ema", "ema", "ewm_mean"}:
        # #376：这是有限窗近似（EWM_SQL_APPROX=True），不是 canonical ts_ema
        # 的精确递归 EWM——勿用其做 exact production parity 认证。
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        alpha = 2.0 / (float(w) + 1.0)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.CLICKHOUSE:
            return _Layer(
                f"SELECT ts, inst, "
                f"exponentialMovingAverage(_v, {alpha}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
                f"FROM ({inner.sql}) t",
                has_inst_window=True,
            )
        decay = 1.0 - alpha
        return _Layer(
            f"SELECT ts, inst, "
            f"SUM(_v * POW({decay}, rn)) OVER ({over}) / "
            f"NULLIF(SUM(POW({decay}, rn)) OVER ({over}), 0) AS _v "
            f"FROM ("
            f"SELECT ts, inst, _v, "
            f"(COUNT(*) OVER ({over}) - 1 - "
            f"(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - "
            f"MIN(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts)) OVER ({over}))) AS rn "
            f"FROM ({inner.sql}) t0"
            f") t",
            has_inst_window=True,
        )

    if op == "ts_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _ts_pct_rank_sql(
                inner.sql,
                window=spec.size,
                dialect=dialect,
                min_periods=spec.min_periods,
            ),
            has_inst_window=True,
        )

    if op == "ffill":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _ffill_over_inst(inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "bfill":
        return None

    if op in {"fillna_const", "nan_to_num", "fillna"}:
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if op == "nan_to_num":
            const = _const_fill_value(node, default=0.0)
        elif op == "fillna":
            const = _const_fill_value(node, default=None)
            if const is None:
                return None
        else:
            const = _const_fill_value(node)
            if const is None:
                const = _scalar_from_plan(node, default=0.0)
        lit = _sql_literal(const)
        coalesce = "coalesce" if dialect == SqlDialect.CLICKHOUSE else "COALESCE"
        if op == "nan_to_num":
            isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
            isinf_fn = "isInfinite" if dialect == SqlDialect.CLICKHOUSE else "isinf"
            val_expr = (
                f"CASE WHEN _v IS NULL OR {isnan_fn}(_v) OR {isinf_fn}(_v) THEN {lit} ELSE _v END"
            )
        else:
            val_expr = f"{coalesce}(_v, {lit})"
        return _Layer(
            f"SELECT ts, inst, {val_expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "row_sum_skipna":
        if not node.inputs:
            return None
        min_count = int(node.attrs.get("min_count", 1))
        if min_count < 1:
            return None
        layers = [_compile_layer(inp, dialect=dialect) for inp in node.inputs]
        if any(layer is None for layer in layers):
            return None
        layers = [layer for layer in layers if layer is not None]
        aliases = [f"t{i}" for i in range(len(layers))]
        join = f"FROM ({layers[0].sql}) {aliases[0]}"
        for alias, layer in zip(aliases[1:], layers[1:], strict=True):
            join += f" INNER JOIN ({layer.sql}) {alias} USING (ts, inst)"
        finite = [f"{alias}._v IS NOT NULL AND NOT isnan({alias}._v) AND NOT isinf({alias}._v)" for alias in aliases]
        count = " + ".join(f"CASE WHEN {expr} THEN 1 ELSE 0 END" for expr in finite)
        total = " + ".join(f"CASE WHEN {expr} THEN {alias}._v ELSE 0.0 END" for alias, expr in zip(aliases, finite, strict=True))
        return _Layer(
            f"SELECT {aliases[0]}.ts, {aliases[0]}.inst, CASE WHEN ({count}) >= {min_count} THEN ({total}) ELSE NULL END AS _v {join}",
            has_inst_window=any(layer.has_inst_window for layer in layers),
            has_ts_partition=any(layer.has_ts_partition for layer in layers),
        )

    if op == "coalesce":
        if len(node.inputs) < 2:
            return None
        layers: list[_Layer] = []
        for inp in node.inputs:
            layer = _compile_layer(inp, dialect=dialect)
            if layer is None:
                return None
            layers.append(layer)
        aliases = [f"t{i}" for i in range(len(layers))]
        coalesce_fn = "coalesce" if dialect == SqlDialect.CLICKHOUSE else "COALESCE"
        cols = ", ".join(f"{alias}._v" for alias in aliases)
        join = f"FROM ({layers[0].sql}) {aliases[0]}"
        for alias, layer in zip(aliases[1:], layers[1:], strict=True):
            join += f" INNER JOIN ({layer.sql}) {alias} USING (ts, inst)"
        return _Layer(
            f"SELECT {aliases[0]}.ts, {aliases[0]}.inst, "
            f"{coalesce_fn}({cols}) AS _v {join}",
            has_inst_window=any(layer.has_inst_window for layer in layers),
            has_ts_partition=any(layer.has_ts_partition for layer in layers),
        )

    if op in {"ts_decay_linear", "decay_linear"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _linear_decay_over_inst(w, inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "power":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.inf_sanitize import apply_inf_policy_sql

        pow_fn = "pow" if dialect == SqlDialect.CLICKHOUSE else "POW"
        raw = (
            f"CASE "
            f"WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN l._v < 0 AND r._v <> FLOOR(r._v) THEN NULL "
            f"WHEN l._v = 0 AND r._v < 0 THEN NULL "
            f"ELSE {pow_fn}(l._v, r._v) END"
        )
        expr = apply_inf_policy_sql(
            raw, "power", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE
        )
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op in {"gt", "lt", "eq", "ge", "le", "ne"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            _compare_binary_sql(op, left.sql, right.sql, dialect=dialect),
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "and_":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN {_truthy_sql('l._v')} AND {_truthy_sql('r._v')} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "or_":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN {_truthy_sql('l._v')} OR {_truthy_sql('r._v')} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "not_":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN 1.0 "
            f"WHEN {_truthy_sql('_v')} THEN 0.0 ELSE 1.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ts_cov":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.pair_window_spec import PairWindowSpec
        from backend.stat_valid import stat_valid_sql

        pspec = PairWindowSpec.from_plan_node(node, default_min_periods=2)
        w = pspec.size
        over = _rolling_ols_partition(w, prefix="l")
        cov_fn = "covar_samp" if dialect == SqlDialect.DUCKDB else "covarSamp"
        l_ok = stat_valid_sql("l._v", dialect=dialect, exclude_nan=True)
        r_ok = stat_valid_sql("r._v", dialect=dialect, exclude_nan=True)
        pair = f"({l_ok} AND {r_ok})"
        pair_l = f"CASE WHEN {pair} THEN l._v END"
        pair_r = f"CASE WHEN {pair} THEN r._v END"
        cnt = f"COUNT(CASE WHEN {pair} THEN 1 END) OVER ({over})"
        body = f"{cov_fn}({pair_l}, {pair_r}) OVER ({over})"
        expr = (
            body
            if pspec.min_periods <= 2
            else f"CASE WHEN {cnt} < {pspec.min_periods} THEN NULL ELSE {body} END"
        )
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_quantile":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        p = _float_attr(node, "q", "p", default=0.5)
        pos_p = _literal_positional(node, 1)
        if pos_p is not None:
            p = pos_p
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        qexpr = _quantile_over(dialect, "_v", p, over)
        return _Layer(
            f"SELECT ts, inst, {qexpr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_product":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        w = spec.size
        # pandas 参考 ``np.exp(log.rolling(w, min_periods=1).sum())`` 用
        # min_periods=1：窗口内部分 NaN 时仍输出有效值的乘积，而非整窗全有效。
        mp = spec.min_periods if node.attrs.get("min_periods") is not None else 1
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.DUCKDB:
            valid = "_v IS NOT NULL AND isfinite(_v)"
            count_valid = f"COUNT(*) FILTER (WHERE {valid}) OVER ({over})"
            count_zero = f"COUNT(*) FILTER (WHERE {valid} AND _v = 0) OVER ({over})"
            count_neg = f"COUNT(*) FILTER (WHERE {valid} AND _v < 0) OVER ({over})"
            count_inf = (
                f"COUNT(*) FILTER (WHERE _v IS NOT NULL AND isinf(_v)) OVER ({over})"
            )
        else:
            valid = "isFinite(_v)"
            count_valid = f"countIf({valid}) OVER ({over})"
            count_zero = f"countIf({valid} AND _v = 0) OVER ({over})"
            count_neg = f"countIf({valid} AND _v < 0) OVER ({over})"
            count_inf = f"countIf(isInfinite(_v)) OVER ({over})"
        log_sum = (
            f"SUM(CASE WHEN {valid} AND _v != 0 THEN {ln}(ABS(_v)) ELSE NULL END) "
            f"OVER ({over})"
        )
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {count_inf} > 0 OR {count_valid} < {mp} THEN NULL "
            f"WHEN {count_zero} > 0 THEN 0.0 "
            f"WHEN {log_sum} > 709.782712893384 THEN NULL "
            f"ELSE CASE WHEN MOD({count_neg}, 2) = 1 THEN -1.0 ELSE 1.0 END "
            f"* EXP({log_sum}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_skew":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        skew_fn = "skewness" if dialect == SqlDialect.DUCKDB else "skewSamp"
        return _Layer(
            f"SELECT ts, inst, {skew_fn}(_v) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_regression_slope":
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        x_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if y_layer is None or x_layer is None:
            return None
        w = _window_int(node)
        over = _rolling_ols_partition(w, prefix="y")
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="y._v", x_col="x._v", partition=over
        )
        fit = f"({alpha}) + ({beta}) * x._v"
        raw_ret = node.attrs.get("retval", node.attrs.get("mode", "slope"))
        retval = str(raw_ret).lower() if raw_ret is not None else "slope"
        if retval in {"1", "intercept", "alpha"}:
            core = alpha
        elif retval in {"2", "fit", "predict", "prediction"}:
            core = fit
        elif retval in {"0", "resid", "residual", "residuals"}:
            core = f"y._v - ({fit})"
        else:
            core = beta
        expr = (
            f"CASE WHEN y._v IS NULL OR x._v IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE {core} END"
        )
        return _Layer(
            f"SELECT y.ts, y.inst, {expr} AS _v "
            f"FROM ({y_layer.sql}) y LEFT JOIN ({x_layer.sql}) x USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "cum_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_sum_sql(inner.sql),
            has_inst_window=True,
        )

    if op == "cum_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("MAX", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("MIN", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_prod":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        prod_fn = "product"
        return _Layer(
            _inst_cum_agg(prod_fn, inner.sql),
            has_inst_window=True,
        )

    if op == "cum_delta":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cum_delta_sql(inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "expanding_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        run = f"SUM(CASE WHEN _v IS NULL THEN 0 ELSE _v END) OVER ({over})"
        cnt = f"SUM(CASE WHEN _v IS NOT NULL THEN 1 ELSE 0 END) OVER ({over})"
        body = f"CASE WHEN _v IS NULL THEN NULL WHEN {cnt} <= 0 THEN NULL ELSE {run} / {cnt} END"
        return _Layer(
            f"SELECT ts, inst, {body} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "expanding_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("SUM", inner.sql, sum_skip_null=True),
            has_inst_window=True,
        )

    if op == "expanding_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _ts_expanding_rank_sql(inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op in {"cum_std", "expanding_std"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            _inst_cum_agg(std_fn, inner.sql),
            has_inst_window=True,
        )

    if op == "count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("COUNT", inner.sql),
            has_inst_window=True,
        )

    if op == "floor":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = "floor"
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ceil":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = "ceil"
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "inverse":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v = 0 THEN NULL ELSE (1.0 / _v) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "log_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        ln_fn = _dialect_fn(dialect, "ln")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN {abs_fn}(_v) = 0 THEN NULL "
            f"ELSE {ln_fn}({abs_fn}(_v)) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "signed_log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        ln_fn = _dialect_fn(dialect, "ln")
        sign_fn = _dialect_fn(dialect, "sign")
        eps = 1e-10
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"ELSE {sign_fn}(_v) * {ln_fn}({abs_fn}(_v) + {eps}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "signed_sqrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        sqrt_fn = _dialect_fn(dialect, "sqrt")
        sign_fn = _dialect_fn(dialect, "sign")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"ELSE {sign_fn}(_v) * {sqrt_fn}({abs_fn}(_v)) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "tanh":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        # Both supported dialects expose tanh. NULL remains NULL and IEEE
        # infinities saturate to +/-1 in DuckDB, matching NumPy/Polars.
        return _Layer(
            f"SELECT ts, inst, tanh(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "cbrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            "SELECT ts, inst, sign(_v) * power(abs(_v), 1.0 / 3.0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "truncate":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        decimals = int(node.attrs.get("decimals", node.attrs.get("k", node.attrs.get("d", 0))))
        scale = 10.0 ** decimals
        return _Layer(
            f"SELECT ts, inst, trunc(_v * {scale!r}) / {scale!r} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    # ------------------------------------------------------------------
    # Elementwise math: single-input unary SQL (NULL preserves NULL, NaN
    # follows IEEE semantics of each dialect like the numpy reference).
    # ------------------------------------------------------------------
    _UNARY_SQL_FN = {
        "sin": "sin(_v)",
        "cos": "cos(_v)",
        "tan": "tan(_v)",
        "asin": "asin(_v)",
        "acos": "acos(_v)",
        "atan": "atan(_v)",
        "sinh": "sinh(_v)",
        "cosh": "cosh(_v)",
        "log2": "log2(_v)",
        "log10": "log10(_v)",
        "csc": "(1.0 / sin(_v))",
        "sec": "(1.0 / cos(_v))",
        "cot": "(1.0 / tan(_v))",
    }
    if op in _UNARY_SQL_FN:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, {_UNARY_SQL_FN[op]} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "atan2":
        if len(node.inputs) != 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None
        return _Layer(
            f"SELECT y.ts, y.inst, atan2(y._v, x._v) AS _v "
            f"FROM ({y.sql}) y LEFT JOIN ({x.sql}) x USING (ts, inst)",
            has_inst_window=y.has_inst_window or x.has_inst_window,
            has_ts_partition=y.has_ts_partition or x.has_ts_partition,
        )

    if op in {"square", "cube"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        pow_n = 2 if op == "square" else 3
        # Pandas reference replaces +/-inf with NaN after the power.
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN ABS(_v) >= 1e154 THEN NULL "
            f"ELSE POWER(_v, {pow_n}) END AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "sigmoid":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, (1.0 / (1.0 + exp(-_v))) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "exp_neg":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL ELSE exp(-_v) END AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "saturate":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        return _Layer(
            f"SELECT ts, inst, {l}({g}(_v, 0.0), 1.0) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "round":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        decimals = _literal_positional(node, 0)
        if decimals is None:
            decimals = float(node.attrs.get("decimals", 0))
        return _Layer(
            f"SELECT ts, inst, round(_v, {int(decimals)}) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "fix":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        # 向 0 取整 == trunc，与 numpy fix 一致。
        return _Layer(
            f"SELECT ts, inst, trunc(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "signed_power":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        c = _literal_positional(node, 0)
        if c is None:
            c = float(node.attrs.get("c", 2))
        abs_fn = _dialect_fn(dialect, "abs")
        sign_fn = _dialect_fn(dialect, "sign")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"ELSE {sign_fn}(_v) * POWER({abs_fn}(_v), {float(c)}) END AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "lerp":
        if len(node.inputs) < 2:
            return None
        a = _compile_layer(node.inputs[0], dialect=dialect)
        b = _compile_layer(node.inputs[1], dialect=dialect)
        if a is None or b is None:
            return None
        fraction = _literal_positional(node, 1)
        if fraction is None:
            fraction = float(node.attrs.get("fraction", 0.5))
        return _Layer(
            f"SELECT a.ts, a.inst, (a._v + {float(fraction)} * (b._v - a._v)) AS _v "
            f"FROM ({a.sql}) a LEFT JOIN ({b.sql}) b USING (ts, inst)",
            has_inst_window=a.has_inst_window or b.has_inst_window,
            has_ts_partition=a.has_ts_partition or b.has_ts_partition,
        )

    if op in {"ewm_std", "ts_ewm_std"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_weighted_moment_sql(inner.sql, span, dialect=dialect, sqrt=True),
            has_inst_window=True,
        )

    if op in {"ewm_var", "ts_ewm_var"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_weighted_moment_sql(inner.sql, span, dialect=dialect, sqrt=False),
            has_inst_window=True,
        )

    if op in {"Slope", "ts_time_slope"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _rolling_slope_over_inst(w, inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "ts_argmax":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _ts_argext_sql(inner.sql, w, dialect=dialect, pick="max"),
            has_inst_window=True,
        )

    if op == "ts_argmin":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _ts_argext_sql(inner.sql, w, dialect=dialect, pick="min"),
            has_inst_window=True,
        )

    if op in {"ewm_cov", "ewm_corr", "ts_ewm_cov", "ts_ewm_corr"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_cov_corr_sql(
                left.sql,
                right.sql,
                span,
                dialect=dialect,
                corr=(op in {"ewm_corr", "ts_ewm_corr"}),
            ),
            has_inst_window=True,
        )

    # 2026-08 geometry/math expansion — DuckDB SQL pushdown subset.
    # Intraday volatility shape: minute-return panels, trailing-window reduction
    # with the same >=5-finite / RV>eps guards as the pandas kernels.
    if op in {"intraday_volatility_concentration", "intraday_volatility_entropy"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node, default=240)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        # Each output row uses ONE window RV (sum of squared returns at that row)
        # applied uniformly to every element of the window.  HHI = SUM(r^4)/RV^2;
        # entropy decomposes as -(S1 - RV*ln(RV)) / (RV*ln(n)) with
        # S1 = SUM(r^2 ln r^2) and 0*ln0 -> 0 masked per element.
        expr = (
            f"r4 / (rv * rv)"
            if op == "intraday_volatility_concentration"
            else f"-(S1 - rv * LN(rv)) / (rv * LN(CAST(n AS DOUBLE)))"
        )
        return _Layer(
            f"SELECT ts, inst, CASE WHEN n < 5 OR rv <= 1e-12 THEN NULL "
            f"ELSE {expr} END AS _v FROM ("
            f"SELECT ts, inst, rv, n, SUM(r4raw) OVER ({win}) AS r4, "
            f"SUM(s1raw) OVER ({win}) AS S1 FROM ("
            f"SELECT ts, inst, "
            f"_v*_v*_v*_v AS r4raw, "
            f"CASE WHEN _v = 0 THEN 0 ELSE _v*_v*LN(_v*_v) END AS s1raw, "
            f"SUM(_v*_v) OVER ({win}) AS rv, "
            f"COUNT(_v) OVER ({win}) AS n "
            f"FROM ({inner.sql}) t) t0) t1",
            has_inst_window=True,
        )

    if op == "intraday_realized_semivariance_balance":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node, default=240)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN n < 5 THEN NULL "
            f"ELSE (up - dn) / (up + dn + 1e-12) END AS _v FROM ("
            f"SELECT ts, inst, up, dn, COUNT(_v) OVER ({win}) AS n FROM ("
            f"SELECT ts, inst, _v, "
            f"SUM(CASE WHEN _v > 0 THEN _v*_v ELSE 0 END) OVER ({win}) AS up, "
            f"SUM(CASE WHEN _v < 0 THEN _v*_v ELSE 0 END) OVER ({win}) AS dn "
            f"FROM ({inner.sql}) t) t1) t2",
            has_inst_window=True,
        )

    # Crossing quality: z = x - y, population std over the window, LAG z.
    if op in {"ts_crossing_speed", "ts_crossing_acceleration"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        spec = _window_spec(node, default=20)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        part = "PARTITION BY inst ORDER BY ts"
        zsql = (
            f"SELECT l.ts, l.inst, l._v - r._v AS z "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)"
        )
        if op == "ts_crossing_speed":
            return _Layer(
                f"SELECT ts, inst, "
                f"CASE WHEN z IS NULL OR zprev IS NULL OR sd IS NULL THEN 0 "
                f"WHEN zprev <= 0 AND z > 0 THEN ABS(z - zprev) / (sd + 1e-12) "
                f"WHEN zprev >= 0 AND z < 0 THEN -ABS(z - zprev) / (sd + 1e-12) "
                f"ELSE 0 END AS _v FROM ("
                f"SELECT ts, inst, z, LAG(z) OVER ({part}) AS zprev, "
                f"STDDEV_POP(z) OVER ({win}) AS sd FROM ({zsql}) t) t2",
                has_inst_window=True,
            )
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN z IS NULL OR zprev IS NULL OR dz IS NULL OR dzprev IS NULL OR sd IS NULL THEN 0 "
            f"WHEN (zprev <= 0 AND z > 0) OR (zprev >= 0 AND z < 0) THEN (dz - dzprev) / (sd + 1e-12) "
            f"ELSE 0 END AS _v FROM ("
            f"SELECT ts, inst, z, zprev, dz, LAG(dz) OVER ({part}) AS dzprev, sd FROM ("
            f"SELECT ts, inst, z, LAG(z) OVER ({part}) AS zprev, "
            f"z - LAG(z) OVER ({part}) AS dz, STDDEV_POP(z) OVER ({win}) AS sd "
            f"FROM ({zsql}) t) t2) t3",
            has_inst_window=True,
        )

    # 2026-08-08 Gemini-recommended primitives — DuckDB SQL pushdown subset.
    # group_topk_mean: exact top-k of peers by score (exclude_self supported)
    # via correlated top-(k+1) selection, matching the pandas kernel.
    if op == "group_topk_mean":
        if len(node.inputs) < 3:
            return None
        tl = _compile_layer(node.inputs[0], dialect=dialect)
        sl = _compile_layer(node.inputs[1], dialect=dialect)
        gl = _compile_layer(node.inputs[2], dialect=dialect)
        if tl is None or sl is None or gl is None:
            return None
        try:
            k = int(node.attrs.get("k", 3) or 3)
        except (TypeError, ValueError):
            return None
        exclude_self = bool(node.attrs.get("exclude_self", True))
        if k < 1:
            return None
        keep = k + (1 if exclude_self else 0)
        excl = "AND t2.inst != x.inst" if exclude_self else ""
        topk = (
            f"SELECT zz._v AS tv FROM ({tl.sql}) zz "
            f"JOIN ({sl.sql}) ss USING (ts, inst) "
            f"JOIN ({gl.sql}) gg USING (ts, inst) "
            f"WHERE gg._v = g._v AND zz.ts = x.ts {excl} "
            f"ORDER BY ss._v DESC LIMIT {keep}"
        )
        return _Layer(
            f"SELECT x.ts AS ts, x.inst AS inst, "
            f"CASE WHEN (SELECT COUNT(*) FROM ({topk}) c) < {k} THEN NULL "
            f"ELSE (SELECT AVG(tv) FROM ({topk}) q) END AS _v "
            f"FROM ({tl.sql}) x "
            f"JOIN ({sl.sql}) s USING (ts, inst) "
            f"JOIN ({gl.sql}) g USING (ts, inst)",
            has_inst_window=True,
        )

    # cs_weighted_percentile_rank: weighted empirical CDF with mid-rank ties.
    if op == "cs_weighted_percentile_rank":
        if len(node.inputs) < 2:
            return None
        xl = _compile_layer(node.inputs[0], dialect=dialect)
        wl = _compile_layer(node.inputs[1], dialect=dialect)
        if xl is None or wl is None:
            return None
        xv = f"SELECT x.ts, x.inst, x._v AS xv FROM ({xl.sql}) x"
        xw = f"SELECT x.ts, x.inst, x._v AS xv, w._v AS wv FROM ({xl.sql}) x JOIN ({wl.sql}) w USING (ts, inst)"
        # pandas：total 只计 x、weight 均有限的行；任一有限负权重 → 整行失败
        # （P1-02）；total<=EPS → NaN。
        valid_w = "wv IS NOT NULL AND xv IS NOT NULL"
        return _Layer(
            f"SELECT a.ts AS ts, a.inst AS inst, "
            f"CASE WHEN wt.neg > 0 OR wt.total_w <= 1e-12 THEN NULL "
            f"ELSE (pl.less_w + 0.5 * pl.eq_w) / wt.total_w END AS _v "
            f"FROM ({xv}) a "
            f"JOIN (SELECT ts, SUM(CASE WHEN {valid_w} THEN wv ELSE 0 END) AS total_w, "
            f"MAX(CASE WHEN wv IS NOT NULL AND wv < 0 THEN 1 ELSE 0 END) AS neg "
            f"FROM ({xw}) w2 GROUP BY ts) wt USING (ts) "
            f"JOIN (SELECT a2.ts, a2.inst, "
            f"SUM(CASE WHEN b2.xv < a2.xv THEN b2.wv ELSE 0 END) AS less_w, "
            f"SUM(CASE WHEN b2.xv = a2.xv THEN b2.wv ELSE 0 END) AS eq_w "
            f"FROM ({xv}) a2 JOIN ({xw}) b2 ON b2.ts = a2.ts "
            f"GROUP BY a2.ts, a2.inst) pl USING (ts, inst)",
            has_inst_window=True,
        )

    # ts_cov_if: conditional sample covariance (ddof=1) over selected pairs.
    if op == "ts_cov_if":
        if len(node.inputs) < 3:
            return None
        xl = _compile_layer(node.inputs[0], dialect=dialect)
        yl = _compile_layer(node.inputs[1], dialect=dialect)
        cl = _compile_layer(node.inputs[2], dialect=dialect)
        if xl is None or yl is None or cl is None:
            return None
        spec = _window_spec(node, default=20)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        try:
            mp = int(node.attrs.get("min_periods", 2) or 2)
        except (TypeError, ValueError):
            mp = 2
        mp = max(mp, 2)
        # pandas：窗口覆盖所有原始行（回看 w 个位置），窗口**内部**掩码
        # cond≠0 / x,y finite。不能先过滤再开窗——那会把窗口回看得更远。
        valid = "cond <> 0 AND x IS NOT NULL AND y IS NOT NULL"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN cnt < {mp} THEN NULL "
            f"ELSE (SUM(CASE WHEN {valid} THEN x * y END) OVER ({win}) "
            f"- SUM(CASE WHEN {valid} THEN x END) OVER ({win}) "
            f"* SUM(CASE WHEN {valid} THEN y END) OVER ({win}) / cnt) "
            f"/ (cnt - 1.0) END AS _v "
            f"FROM (SELECT ts, inst, x, y, "
            f"SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) OVER ({win}) AS cnt "
            f"FROM (SELECT x.ts, x.inst, x._v AS x, y._v AS y, c._v AS cond "
            f"FROM ({xl.sql}) x JOIN ({yl.sql}) y USING (ts, inst) "
            f"JOIN ({cl.sql}) c USING (ts, inst)) j) s",
            has_inst_window=True,
        )

    # ts_value_at_argextreme: argmax/argmin gather over a row frame.
    if op == "ts_value_at_argextreme":
        if len(node.inputs) < 2:
            return None
        vl = _compile_layer(node.inputs[0], dialect=dialect)
        sl = _compile_layer(node.inputs[1], dialect=dialect)
        if vl is None or sl is None:
            return None
        spec = _window_spec(node, default=20)
        include_current = bool(node.attrs.get("include_current", False))
        end = "CURRENT ROW" if include_current else "1 PRECEDING"
        mode = str(node.attrs.get("mode", "max") or "max")
        fn = "arg_max" if mode == "max" else "arg_min"
        win = f"PARTITION BY v.inst ORDER BY v.ts ROWS BETWEEN {spec.size - 1} PRECEDING AND {end}"
        return _Layer(
            f"SELECT v.ts AS ts, v.inst AS inst, {fn}(v._v, s._v) OVER ({win}) AS _v "
            f"FROM ({vl.sql}) v LEFT JOIN ({sl.sql}) s USING (ts, inst)",
            has_inst_window=True,
        )

    # ts_weighted_standardized_moment: weighted standardized central moment
    # (order 3/4) via nested window reductions (mu -> var -> moment).
    if op == "ts_weighted_standardized_moment":
        if len(node.inputs) < 2:
            return None
        xl = _compile_layer(node.inputs[0], dialect=dialect)
        wl = _compile_layer(node.inputs[1], dialect=dialect)
        if xl is None or wl is None:
            return None
        spec = _window_spec(node, default=20)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        try:
            p = int(node.attrs.get("order", 3) or 3)
        except (TypeError, ValueError):
            return None
        if p not in (3, 4):
            return None
        # P1-007: match the pandas kernel exactly — a window containing a NEGATIVE
        # weight fail-closes (never silently dropped), the effective sample size
        # (sw^2/sw2) must meet the order's minimum (3 for skew, 4 for kurtosis).
        min_samp = 3 if p == 3 else 4
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN neg_w > 0 OR n < {min_samp} OR sw <= 0 OR var <= 0 "
            f"OR (sw * sw) / NULLIF(sw2, 0) < {min_samp} THEN NULL "
            f"ELSE (SUM(w * POWER(x - mu, {p})) OVER ({win}) / sw) "
            f"/ (POWER(var, {p} / 2.0) + 1e-12) END AS _v "
            f"FROM (SELECT ts, inst, x, w, n, neg_w, sw, sw2, mu, "
            f"SUM(w * (x - mu) * (x - mu)) OVER ({win}) / sw AS var "
            f"FROM (SELECT ts, inst, x, w, "
            f"COUNT(*) OVER ({win}) AS n, "
            f"COUNT(CASE WHEN w < 0 THEN 1 END) OVER ({win}) AS neg_w, "
            f"SUM(w) OVER ({win}) AS sw, "
            f"SUM(w * w) OVER ({win}) AS sw2, "
            f"SUM(w * x) OVER ({win}) / NULLIF(SUM(w) OVER ({win}), 0) AS mu "
            f"FROM (SELECT x.ts, x.inst, x._v AS x, w._v AS w "
            f"FROM ({xl.sql}) x JOIN ({wl.sql}) w USING (ts, inst)) j "
            f"WHERE x IS NOT NULL AND w IS NOT NULL) t0) t1",
            has_inst_window=True,
        )

    # ts_abdi_ranaldo_spread: PIT-safe Abdi-Ranaldo effective spread.  The
    # estimator needs (c_s, eta_s, eta_{s+1}); output[T] uses only completed
    # pairs s <= T-1, so eta is lead()ed one row and the rolling mean frame
    # EXCLUDES the current row (ROWS ... AND 1 PRECEDING).  eta_{T} (the
    # current day's mid-range) is known at day-T close, matching the kernel.
    if op == "ts_abdi_ranaldo_spread":
        if len(node.inputs) < 3:
            return None
        cl = _compile_layer(node.inputs[0], dialect=dialect)
        hl = _compile_layer(node.inputs[1], dialect=dialect)
        ll = _compile_layer(node.inputs[2], dialect=dialect)
        if cl is None or hl is None or ll is None:
            return None
        spec = _window_spec(node, default=20)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND 1 PRECEDING"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN cnt < 2 THEN NULL "
            f"ELSE SQRT(GREATEST(4.0 * SUM(term) OVER ({win}) / cnt, 0.0)) END AS _v "
            f"FROM (SELECT ts, inst, term, COUNT(*) OVER ({win}) AS cnt "
            f"FROM (SELECT ts, inst, (c - eta) * (c - eta_next) AS term "
            f"FROM (SELECT ts, inst, c, eta, "
            f"LEAD(eta) OVER (PARTITION BY inst ORDER BY ts) AS eta_next "
            f"FROM (SELECT c.ts, c.inst, LN(c._v) AS c, "
            f"(LN(h._v) + LN(l._v)) / 2.0 AS eta "
            f"FROM ({cl.sql}) c JOIN ({hl.sql}) h USING (ts, inst) "
            f"JOIN ({ll.sql}) l USING (ts, inst) "
            f"WHERE c._v > 0 AND h._v > 0 AND l._v > 0) m) "
            f"WHERE eta_next IS NOT NULL) t) w",
            has_inst_window=True,
        )

    # Round-7 gap closure: windowed sign-ratio / moment / group reducers that are
    # genuinely SQL-expressible.  NaN is cleaned to NULL first (the emitter's
    # convention), then windowed/group aggregates mirror the pandas kernels
    # exactly (min_periods gate for ratios, nanmean central moment, group count
    # over finite members, weighted mean over w>0 & finite members).
    if op in {"ts_positive_ratio", "ts_negative_ratio", "ts_zero_ratio"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node, default=20)
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        try:
            thr = float(_raw_literal(node, 2, _float_attr(node, "threshold", "tolerance", default=0.0)) or 0.0)
        except (TypeError, ValueError):
            return None
        mp = max(1, int(_raw_literal(node, 3, _int_attr(node, "min_periods", default=1)) or 1))
        from backend.stat_valid import row_stat_invalid_sql

        invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
        if op == "ts_positive_ratio":
            cmp_sql = f"_v > {thr!r}"
        elif op == "ts_negative_ratio":
            cmp_sql = f"_v < {thr!r}"
        else:
            cmp_sql = f"ABS(_v) <= {thr!r}"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN cnt < {mp} THEN NULL "
            f"ELSE SUM(CASE WHEN {cmp_sql} THEN 1.0 ELSE 0.0 END) OVER ({win}) / cnt END AS _v "
            f"FROM (SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS _v, "
            f"COUNT(_v) OVER ({win}) AS cnt "
            f"FROM ({inner.sql}) t) s",
            has_inst_window=True,
        )

    if op == "ts_moment":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        d = _window_int(node, default=20)
        try:
            k = int(_raw_literal(node, 2, _int_attr(node, "k", default=3)) or 3)
        except (TypeError, ValueError):
            return None
        # ``E[(x-μ)^k]`` with μ = window mean broadcast over the whole window.
        # A plain ``SUM(POWER(_v - mu, k)) OVER (win)`` is WRONG: each window row
        # would use its own μ.  Use the binomial expansion in raw power sums
        # (algebraically identical to ``nanmean((w-μ)^k)``) for the common
        # orders; other k falls back to the polars/pandas path.
        if k not in (2, 3, 4):
            return None
        win = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {d - 1} PRECEDING AND CURRENT ROW"
        from backend.stat_valid import row_stat_invalid_sql

        invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
        power_terms = ", ".join(
            f"SUM(POWER(v, {j})) OVER ({win}) AS s{j}" for j in range(1, k + 1)
        )
        cnt = f"COUNT(v) OVER ({win})"
        nrows = f"COUNT(*) OVER ({win})"
        if k == 2:
            moment = f"(s2 - (s1 * s1) / cnt) / cnt"
        elif k == 3:
            moment = f"(s3 - 3 * s2 * s1 / cnt + 2 * POWER(s1, 3) / POWER(cnt, 2)) / cnt"
        else:
            moment = (
                f"(s4 - 4 * s3 * s1 / cnt + 6 * s2 * POWER(s1, 2) / POWER(cnt, 2) "
                f"- 3 * POWER(s1, 4) / POWER(cnt, 3)) / cnt"
            )
        # Cold start mirrors the pandas ``_rolling_apply`` (NaN for the first
        # d-1 rows regardless of NaN content): gate on raw row count, not the
        # cleaned count, so NaN rows still occupy window slots.
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN nrows < {d} OR cnt = 0 THEN NULL ELSE {moment} END AS _v "
            f"FROM (SELECT ts, inst, {nrows} AS nrows, {cnt} AS cnt, {power_terms} "
            f"FROM (SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS v "
            f"FROM ({inner.sql}) t) c) s",
            has_inst_window=True,
        )

    if op == "group_valid_count":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        from backend.stat_valid import row_stat_invalid_sql

        x_inv = row_stat_invalid_sql("x._v", dialect=dialect, exclude_nan=True)
        g_inv = row_stat_invalid_sql("g._v", dialect=dialect, exclude_nan=True)
        part = "PARTITION BY x.ts, g._v"
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN {g_inv} THEN NULL "
            f"ELSE COUNT(CASE WHEN NOT ({x_inv}) THEN 1 END) OVER ({part}) END AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_weighted_mean":
        if len(node.inputs) < 3:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        wgt = _compile_layer(node.inputs[2], dialect=dialect)
        if inner is None or grp is None or wgt is None:
            return None
        from backend.stat_valid import row_stat_invalid_sql

        x_inv = row_stat_invalid_sql("x._v", dialect=dialect, exclude_nan=True)
        w_inv = row_stat_invalid_sql("w._v", dialect=dialect, exclude_nan=True)
        g_inv = row_stat_invalid_sql("g._v", dialect=dialect, exclude_nan=True)
        part = "PARTITION BY x.ts, g._v"
        valid = f"(NOT ({x_inv}) AND NOT ({w_inv}) AND w._v > 0 AND NOT ({g_inv}))"
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN {valid} THEN "
            f"CASE WHEN SUM(CASE WHEN {valid} THEN w._v ELSE 0 END) OVER ({part}) <= 1e-12 THEN NULL "
            f"ELSE SUM(CASE WHEN {valid} THEN x._v * w._v ELSE 0 END) OVER ({part}) "
            f"/ SUM(CASE WHEN {valid} THEN w._v ELSE 0 END) OVER ({part}) END "
            f"ELSE NULL END AS _v "
            f"FROM ({inner.sql}) x JOIN ({wgt.sql}) w USING (ts, inst) "
            f"JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # ========================================================================
    # Batch implementation of missing simple operators
    # ========================================================================

    if op == "ts_lead":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        periods = max(int(periods), 1)
        return _Layer(
            f"SELECT ts, inst, LEAD(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "cs_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_rank_01_sql(inner.sql, partition="PARTITION BY ts", dialect=dialect),
            has_ts_partition=True,
        )

    if op == "cs_normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        expr = _normalize_window_expr(value_col="_v", partition="PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        expr = _zscore_window_expr(value_col="_v", partition="PARTITION BY ts", dialect=dialect)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_median":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        med_fn = "median"
        return _Layer(
            f"SELECT ts, inst, {med_fn}(_v) OVER (PARTITION BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_iqr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        q25 = _quantile_over(dialect, "_v", 0.25, "PARTITION BY ts")
        q75 = _quantile_over(dialect, "_v", 0.75, "PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, ({q75} - {q25}) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_clip":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lower_q = _float_attr(node, "lower", default=0.05)
        upper_q = _float_attr(node, "upper", default=0.95)
        lo_expr = _quantile_over(dialect, "_v", lower_q, "PARTITION BY ts")
        hi_expr = _quantile_over(dialect, "_v", upper_q, "PARTITION BY ts")
        clipped = f"{g}({lo_expr}, {l}({hi_expr}, _v))"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE {clipped} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "winsorize_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lower_q = _float_attr(node, "lower", default=0.05)
        upper_q = _float_attr(node, "upper", default=0.95)
        lo_expr = _quantile_over(dialect, "_v", lower_q, "PARTITION BY ts")
        hi_expr = _quantile_over(dialect, "_v", upper_q, "PARTITION BY ts")
        clipped = f"{g}({lo_expr}, {l}({hi_expr}, _v))"
        return _Layer(
            f"SELECT ts, inst, AVG({clipped}) OVER (PARTITION BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # Financial operators - simple transformations
    if op == "fin_lag":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        return _Layer(
            f"SELECT ts, inst, LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_diff":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        return _Layer(
            f"SELECT ts, inst, (_v - LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_pct_change":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        lag_v = f"LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {lag_v} IS NULL OR {lag_v} = 0 THEN NULL "
            f"ELSE (_v - {lag_v}) / {lag_v} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_growth":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=4)
        if periods is None:
            periods = 4
        lag_v = f"LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {lag_v} IS NULL OR {lag_v} = 0 THEN NULL "
            f"ELSE (_v - {lag_v}) / ABS({lag_v}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_yoy":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag_v = f"LAG(_v, 4) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {lag_v} IS NULL OR {lag_v} = 0 THEN NULL "
            f"ELSE (_v - {lag_v}) / ABS({lag_v}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_qoq":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag_v = f"LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {lag_v} IS NULL OR {lag_v} = 0 THEN NULL "
            f"ELSE (_v - {lag_v}) / ABS({lag_v}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        w = max(int(w), 2)
        std_fn = _dialect_fn(dialect, "stddev")
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, {std_fn}(_v) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_cv":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        w = max(int(w), 2)
        std_fn = _dialect_fn(dialect, "stddev")
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        mean_expr = f"AVG(_v) OVER ({over})"
        std_expr = f"{std_fn}(_v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {mean_expr} IS NULL OR ABS({mean_expr}) <= 1e-12 THEN NULL "
            f"ELSE {std_expr} / ABS({mean_expr}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_range":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        w = max(int(w), 2)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, (MAX(_v) OVER ({over}) - MIN(_v) OVER ({over})) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_lag":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        periods = max(int(periods), 1)
        return _Layer(
            f"SELECT ts, inst, LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_cumprod":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if dialect == SqlDialect.DUCKDB:
            over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            return _Layer(
                f"SELECT ts, inst, EXP(SUM(LN(CASE WHEN _v > 0 THEN _v ELSE NULL END)) OVER ({over})) AS _v "
                f"FROM ({inner.sql}) t",
                has_inst_window=True,
            )
        return None

    if op == "ts_cumsum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(_inst_cum_sum_sql(inner.sql), has_inst_window=True)

    if op == "cs_trim_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lower_q = _float_attr(node, "lower", default=0.1)
        upper_q = _float_attr(node, "upper", default=0.9)
        lo_expr = _quantile_over(dialect, "_v", lower_q, "PARTITION BY ts")
        hi_expr = _quantile_over(dialect, "_v", upper_q, "PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, AVG(CASE WHEN _v >= {lo_expr} AND _v <= {hi_expr} THEN _v END) "
            f"OVER (PARTITION BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # More simple operators
    if op == "HMA":
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        if w < 2:
            return None
        import math
        w2 = max(int(w / 2), 1)
        ws = max(int(math.sqrt(w)), 1)
        # WMA for half period
        wma_half = _linear_weighted_mean_sql(inner.sql, w2, dialect=dialect)
        # WMA for full period
        wma_full = _linear_weighted_mean_sql(inner.sql, w, dialect=dialect)
        # 2*WMA(n/2) - WMA(n)
        diff_sql = (
            f"SELECT h.ts, h.inst, (2.0 * h._v - f._v) AS _v "
            f"FROM ({wma_half}) h JOIN ({wma_full}) f USING (ts, inst)"
        )
        # Final WMA of sqrt(n)
        return _Layer(_linear_weighted_mean_sql(diff_sql, ws, dialect=dialect), has_inst_window=True)

    if op == "KAMA":
        # Kaufman's Adaptive Moving Average
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=10)
        fast_sc = 2.0 / (2.0 + 1.0)  # fast smoothing constant
        slow_sc = 2.0 / (30.0 + 1.0)  # slow smoothing constant
        if dialect != SqlDialect.DUCKDB:
            return None
        # Need recursive computation for KAMA - complex, skip for now
        return None

    if op == "ALMA":
        # Arnaud Legoux Moving Average - needs Gaussian weights, complex
        return None

    if op == "WMA":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        return _Layer(_linear_weighted_mean_sql(inner.sql, w, dialect=dialect), has_inst_window=True)

    if op == "ts_sma":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "AVG", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "rank_corr":
        # Spearman rank correlation
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        w = _window_int(node, default=20)
        # Rank each series then compute correlation
        left_ranked = _ts_rank_sql(left.sql, w, dialect=dialect)
        right_ranked = _ts_rank_sql(right.sql, w, dialect=dialect)
        over = f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        corr_fn = "corr" if dialect == SqlDialect.DUCKDB else "corrStable"
        return _Layer(
            f"SELECT l.ts, l.inst, {corr_fn}(l._v, r._v) OVER ({over}) AS _v "
            f"FROM ({left_ranked}) l LEFT JOIN ({right_ranked}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "cs_neutralize":
        # Same as cs_demean
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, (_v - AVG(_v) OVER (PARTITION BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_mean":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        return _Layer(
            f"SELECT x.ts, x.inst, AVG(x._v) OVER (PARTITION BY x.ts, g._v) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_std":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            f"SELECT x.ts, x.inst, {std_fn}(x._v) OVER (PARTITION BY x.ts, g._v) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_rank":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        wrapped, partition_keys = _group_partition_wrap(inner.sql, grp.sql)
        return _Layer(
            _cs_average_rank_01_sql(wrapped, partition_keys=partition_keys, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "fin_log_change":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        lag_v = f"LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR {lag_v} IS NULL OR _v <= 0 OR {lag_v} <= 0 THEN NULL "
            f"ELSE {ln}(_v / {lag_v}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_returns":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        lag_v = f"LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {lag_v} IS NULL OR {lag_v} = 0 THEN NULL "
            f"ELSE (_v - {lag_v}) / {lag_v} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "flex_max":
        # Flexible maximum - treat as ts_max with flexible window
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "MAX", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "flex_min":
        # Flexible minimum - treat as ts_min with flexible window
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "MIN", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ffill_limit":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        limit = _int_attr(node, "limit", default=5)
        if limit is None:
            limit = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        # Complex - needs to track consecutive nulls
        return None

    if op == "digital_count":
        # Count distinct values in a rolling window
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, COUNT(DISTINCT _v) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_ratio":
        # Current value divided by lagged value
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = _int_attr(node, "periods", default=1)
        if periods is None:
            periods = 1
        lag_v = f"LAG(_v, {periods}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, _v / {nf}({lag_v}, 0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # Additional CTE-based operators
    if op == "ts_weighted_mean":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        wgt = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or wgt is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY v.inst ORDER BY v.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH weighted AS ("
            f"SELECT v.ts, v.inst, v._v AS val, w._v AS wt, "
            f"SUM(v._v * w._v) OVER ({over}) AS wsum, "
            f"SUM(w._v) OVER ({over}) AS wtotal "
            f"FROM ({val.sql}) v LEFT JOIN ({wgt.sql}) w USING (ts, inst)"
            f") "
            f"SELECT ts, inst, wsum / {nf}(wtotal, 0) AS _v FROM weighted",
            has_inst_window=True,
        )

    if op == "ts_weighted_std":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        wgt = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or wgt is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY v.inst ORDER BY v.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT v.ts, v.inst, v._v AS val, w._v AS wt, "
            f"SUM(v._v * w._v) OVER ({over}) / {nf}(SUM(w._v) OVER ({over}), 0) AS wmean, "
            f"SUM(w._v) OVER ({over}) AS wtotal "
            f"FROM ({val.sql}) v LEFT JOIN ({wgt.sql}) w USING (ts, inst)"
            f"), "
            f"variance AS ("
            f"SELECT ts, inst, "
            f"SQRT(SUM(wt * (val - wmean) * (val - wmean)) OVER "
            f"(PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) / "
            f"{nf}(wtotal, 0)) AS wstd "
            f"FROM stats"
            f") "
            f"SELECT ts, inst, wstd AS _v FROM variance",
            has_inst_window=True,
        )

    if op == "cs_weighted_mean":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        wgt = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or wgt is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH weighted AS ("
            f"SELECT v.ts, v.inst, v._v AS val, w._v AS wt, "
            f"SUM(v._v * w._v) OVER (PARTITION BY v.ts) AS wsum, "
            f"SUM(w._v) OVER (PARTITION BY v.ts) AS wtotal "
            f"FROM ({val.sql}) v LEFT JOIN ({wgt.sql}) w USING (ts, inst)"
            f") "
            f"SELECT ts, inst, wsum / {nf}(wtotal, 0) AS _v FROM weighted",
            has_inst_window=(val.has_inst_window or wgt.has_inst_window),
            has_ts_partition=True,
        )

    if op == "cs_weighted_std":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        wgt = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or wgt is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT v.ts, v.inst, v._v AS val, w._v AS wt, "
            f"SUM(v._v * w._v) OVER (PARTITION BY v.ts) / {nf}(SUM(w._v) OVER (PARTITION BY v.ts), 0) AS wmean, "
            f"SUM(w._v) OVER (PARTITION BY v.ts) AS wtotal "
            f"FROM ({val.sql}) v LEFT JOIN ({wgt.sql}) w USING (ts, inst)"
            f"), "
            f"variance AS ("
            f"SELECT ts, inst, "
            f"SQRT(SUM(wt * (val - wmean) * (val - wmean)) OVER (PARTITION BY ts) / "
            f"{nf}(wtotal, 0)) AS wstd "
            f"FROM stats"
            f") "
            f"SELECT ts, inst, wstd AS _v FROM variance",
            has_inst_window=(val.has_inst_window or wgt.has_inst_window),
            has_ts_partition=True,
        )

    if op == "ts_skewness":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"AVG(_v) OVER ({over}) AS mean, "
            f"STDDEV(_v) OVER ({over}) AS std, "
            f"COUNT(_v) OVER ({over}) AS n "
            f"FROM ({inner.sql}) t0"
            f"), "
            f"moments AS ("
            f"SELECT ts, inst, "
            f"SUM(POW((_v - mean) / {nf}(std, 0), 3)) OVER ({over}) / {nf}(n, 0) AS skew "
            f"FROM stats"
            f") "
            f"SELECT ts, inst, skew AS _v FROM moments",
            has_inst_window=True,
        )

    if op == "ts_kurtosis":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"AVG(_v) OVER ({over}) AS mean, "
            f"STDDEV(_v) OVER ({over}) AS std, "
            f"COUNT(_v) OVER ({over}) AS n "
            f"FROM ({inner.sql}) t0"
            f"), "
            f"moments AS ("
            f"SELECT ts, inst, "
            f"SUM(POW((_v - mean) / {nf}(std, 0), 4)) OVER ({over}) / {nf}(n, 0) - 3 AS kurt "
            f"FROM stats"
            f") "
            f"SELECT ts, inst, kurt AS _v FROM moments",
            has_inst_window=True,
        )

    if op == "cs_skewness":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"AVG(_v) OVER (PARTITION BY ts) AS mean, "
            f"STDDEV(_v) OVER (PARTITION BY ts) AS std, "
            f"COUNT(_v) OVER (PARTITION BY ts) AS n "
            f"FROM ({inner.sql}) t0"
            f"), "
            f"moments AS ("
            f"SELECT ts, inst, "
            f"SUM(POW((_v - mean) / {nf}(std, 0), 3)) OVER (PARTITION BY ts) / {nf}(n, 0) AS skew "
            f"FROM stats"
            f") "
            f"SELECT ts, inst, skew AS _v FROM moments",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_kurtosis":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"AVG(_v) OVER (PARTITION BY ts) AS mean, "
            f"STDDEV(_v) OVER (PARTITION BY ts) AS std, "
            f"COUNT(_v) OVER (PARTITION BY ts) AS n "
            f"FROM ({inner.sql}) t0"
            f"), "
            f"moments AS ("
            f"SELECT ts, inst, "
            f"SUM(POW((_v - mean) / {nf}(std, 0), 4)) OVER (PARTITION BY ts) / {nf}(n, 0) - 3 AS kurt "
            f"FROM stats"
            f") "
            f"SELECT ts, inst, kurt AS _v FROM moments",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_zscore":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"AVG(x._v) OVER (PARTITION BY x.ts, g._v) AS mean, "
            f"{std_fn}(x._v) OVER (PARTITION BY x.ts, g._v) AS std "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN val IS NULL THEN NULL "
            f"WHEN std IS NULL OR std = 0 THEN 0 "
            f"ELSE (val - mean) / {nf}(std, 0) END AS _v "
            f"FROM stats",
            has_inst_window=(val.has_inst_window or grp.has_inst_window),
            has_ts_partition=True,
        )

    if op == "group_normalize":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"MIN(x._v) OVER (PARTITION BY x.ts, g._v) AS min_val, "
            f"MAX(x._v) OVER (PARTITION BY x.ts, g._v) AS max_val "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN val IS NULL THEN NULL "
            f"WHEN max_val = min_val THEN 0.5 "
            f"ELSE (val - min_val) / {nf}(max_val - min_val, 0) END AS _v "
            f"FROM stats",
            has_inst_window=(val.has_inst_window or grp.has_inst_window),
            has_ts_partition=True,
        )

    if op == "ts_alpha":
        if len(node.inputs) < 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY y.inst ORDER BY y.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        cov_fn = "covar_samp" if dialect == SqlDialect.DUCKDB else "covarSamp"
        var_fn = "var_samp" if dialect == SqlDialect.DUCKDB else "varSamp"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT y.ts, y.inst, "
            f"AVG(y._v) OVER ({over}) AS mean_y, "
            f"AVG(x._v) OVER ({over}) AS mean_x, "
            f"{cov_fn}(y._v, x._v) OVER ({over}) AS cov_xy, "
            f"{var_fn}(x._v) OVER ({over}) AS var_x "
            f"FROM ({y.sql}) y LEFT JOIN ({x.sql}) x USING (ts, inst)"
            f") "
            f"SELECT ts, inst, "
            f"mean_y - (cov_xy / {nf}(var_x, 0)) * mean_x AS _v "
            f"FROM stats",
            has_inst_window=True,
        )

    if op == "ts_information_ratio":
        if len(node.inputs) < 2:
            return None
        ret = _compile_layer(node.inputs[0], dialect=dialect)
        bench = _compile_layer(node.inputs[1], dialect=dialect)
        if ret is None or bench is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY r.inst ORDER BY r.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        std_fn = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH excess AS ("
            f"SELECT r.ts, r.inst, (r._v - b._v) AS ex_ret "
            f"FROM ({ret.sql}) r LEFT JOIN ({bench.sql}) b USING (ts, inst)"
            f"), "
            f"stats AS ("
            f"SELECT ts, inst, "
            f"AVG(ex_ret) OVER ({over}) AS mean_ex, "
            f"{std_fn}(ex_ret) OVER ({over}) AS std_ex "
            f"FROM excess"
            f") "
            f"SELECT ts, inst, mean_ex / {nf}(std_ex, 0) AS _v FROM stats",
            has_inst_window=True,
        )

    if op == "ts_sharpe":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        std_fn = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, "
            f"AVG(_v) OVER ({over}) AS mean_ret, "
            f"{std_fn}(_v) OVER ({over}) AS std_ret "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, mean_ret / {nf}(std_ret, 0) AS _v FROM stats",
            has_inst_window=True,
        )

    if op == "ts_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        std_fn = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"AVG(_v) OVER ({over}) AS mean, "
            f"{std_fn}(_v) OVER ({over}) AS std "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN std IS NULL OR std = 0 THEN 0 "
            f"ELSE (_v - mean) / {nf}(std, 0) END AS _v "
            f"FROM stats",
            has_inst_window=True,
        )

    if op == "ts_normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH stats AS ("
            f"SELECT ts, inst, _v, "
            f"MIN(_v) OVER ({over}) AS min_val, "
            f"MAX(_v) OVER ({over}) AS max_val "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN max_val = min_val THEN 0.5 "
            f"ELSE (_v - min_val) / {nf}(max_val - min_val, 0) END AS _v "
            f"FROM stats",
            has_inst_window=True,
        )

    if op == "group_rank":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        return _Layer(
            f"WITH ranked AS ("
            f"SELECT x.ts, x.inst, x._v AS val, "
            f"ROW_NUMBER() OVER (PARTITION BY x.ts, g._v ORDER BY x._v ASC) AS rn, "
            f"COUNT(*) OVER (PARTITION BY x.ts, g._v) AS n "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN val IS NULL THEN NULL "
            f"ELSE (rn - 1.0) / NULLIF(n - 1, 0) END AS _v "
            f"FROM ranked",
            has_inst_window=(val.has_inst_window or grp.has_inst_window),
            has_ts_partition=True,
        )

    if op == "group_median":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        med_fn = "median"
        return _Layer(
            f"WITH stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"{med_fn}(x._v) OVER (PARTITION BY x.ts, g._v) AS med "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, med AS _v FROM stats",
            has_inst_window=(val.has_inst_window or grp.has_inst_window),
            has_ts_partition=True,
        )

    if op == "group_min":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        return _Layer(
            f"WITH stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"MIN(x._v) OVER (PARTITION BY x.ts, g._v) AS min_val "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, min_val AS _v FROM stats",
            has_inst_window=(val.has_inst_window or grp.has_inst_window),
            has_ts_partition=True,
        )

    if op == "group_max":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        return _Layer(
            f"WITH stats AS ("
            f"SELECT x.ts, g._v AS grp, x.inst, x._v AS val, "
            f"MAX(x._v) OVER (PARTITION BY x.ts, g._v) AS max_val "
            f"FROM ({val.sql}) x "
            f"LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, max_val AS _v FROM stats",
            has_inst_window=(val.has_inst_window or grp.has_inst_window),
            has_ts_partition=True,
        )

    if op == "cs_winsorize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lower_q = _float_attr(node, "lower", default=0.05)
        upper_q = _float_attr(node, "upper", default=0.95)
        lo_fn = _dialect_fn(dialect, "greatest")
        hi_fn = _dialect_fn(dialect, "least")
        if dialect == SqlDialect.CLICKHOUSE:
            lower_expr = f"quantile({lower_q})(_v)"
            upper_expr = f"quantile({upper_q})(_v)"
        else:
            lower_expr = f"percentile_cont({lower_q}) WITHIN GROUP (ORDER BY _v)"
            upper_expr = f"percentile_cont({upper_q}) WITHIN GROUP (ORDER BY _v)"
        return _Layer(
            f"WITH bounds AS ("
            f"SELECT ts, inst, _v, "
            f"{lower_expr} OVER (PARTITION BY ts) AS lower_b, "
            f"{upper_expr} OVER (PARTITION BY ts) AS upper_b "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"{hi_fn}(upper_b, {lo_fn}(lower_b, _v)) AS _v "
            f"FROM bounds",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_winsorize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        lower_q = _float_attr(node, "lower", default=0.05)
        upper_q = _float_attr(node, "upper", default=0.95)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        lo_fn = _dialect_fn(dialect, "greatest")
        hi_fn = _dialect_fn(dialect, "least")
        if dialect == SqlDialect.CLICKHOUSE:
            lower_expr = f"quantile({lower_q})(_v)"
            upper_expr = f"quantile({upper_q})(_v)"
        else:
            lower_expr = f"percentile_cont({lower_q}) WITHIN GROUP (ORDER BY _v)"
            upper_expr = f"percentile_cont({upper_q}) WITHIN GROUP (ORDER BY _v)"
        return _Layer(
            f"WITH bounds AS ("
            f"SELECT ts, inst, _v, "
            f"{lower_expr} OVER ({over}) AS lower_b, "
            f"{upper_expr} OVER ({over}) AS upper_b "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"{hi_fn}(upper_b, {lo_fn}(lower_b, _v)) AS _v "
            f"FROM bounds",
            has_inst_window=True,
        )

    # ========================================================================
    # Second batch: More time series, cross-sectional, and financial operators
    # ========================================================================

    if op == "ts_skew":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, SKEWNESS(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_kurt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        # pandas rolling.kurt: any ±Inf in the window → NaN (does not skip Inf).
        inf_cnt = (
            f"COUNT(*) FILTER (WHERE _v IS NOT NULL AND isinf(_v)) OVER ({over})"
        )
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {inf_cnt} > 0 THEN NULL "
            f"ELSE KURTOSIS(_v) OVER ({over}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_trimmed_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        lower_q = _float_attr(node, "lower", default=0.1)
        upper_q = _float_attr(node, "upper", default=0.9)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        lo_q = f"PERCENTILE_CONT({lower_q}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        hi_q = f"PERCENTILE_CONT({upper_q}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, AVG(CASE WHEN _v >= {lo_q} AND _v <= {hi_q} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_topk_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        k = _int_attr(node, "k", default=5)
        if k is None:
            k = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        threshold_pct = max(0.0, 1.0 - float(k) / float(w))
        threshold = f"PERCENTILE_CONT({threshold_pct}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, AVG(CASE WHEN _v >= {threshold} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_topk_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        k = _int_attr(node, "k", default=5)
        if k is None:
            k = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        threshold_pct = max(0.0, 1.0 - float(k) / float(w))
        threshold = f"PERCENTILE_CONT({threshold_pct}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, SUM(CASE WHEN _v >= {threshold} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_topk_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        k = _int_attr(node, "k", default=5)
        if k is None:
            k = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        threshold_pct = max(0.0, 1.0 - float(k) / float(w))
        threshold = f"PERCENTILE_CONT({threshold_pct}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, {std_fn}(CASE WHEN _v >= {threshold} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_bottomk_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        k = _int_attr(node, "k", default=5)
        if k is None:
            k = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        threshold_pct = min(1.0, float(k) / float(w))
        threshold = f"PERCENTILE_CONT({threshold_pct}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, AVG(CASE WHEN _v <= {threshold} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_bottomk_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        k = _int_attr(node, "k", default=5)
        if k is None:
            k = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        threshold_pct = min(1.0, float(k) / float(w))
        threshold = f"PERCENTILE_CONT({threshold_pct}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, SUM(CASE WHEN _v <= {threshold} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_bottomk_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        k = _int_attr(node, "k", default=5)
        if k is None:
            k = 5
        if dialect != SqlDialect.DUCKDB:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        threshold_pct = min(1.0, float(k) / float(w))
        threshold = f"PERCENTILE_CONT({threshold_pct}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, {std_fn}(CASE WHEN _v <= {threshold} THEN _v END) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # More financial operators
    if op == "fin_ttm":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, SUM(_v) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_mean_abs_deviation":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        mean_expr = f"AVG(_v) OVER ({over})"
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT ts, inst, AVG({abs_fn}(_v - {mean_expr})) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_median_abs_deviation":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        med = f"MEDIAN(_v) OVER ({over})"
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT ts, inst, MEDIAN({abs_fn}(_v - {med})) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_positive_streak":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if dialect != SqlDialect.DUCKDB:
            return None
        false_cond = f"(_v IS NULL OR _v <= 0)"
        return _Layer(
            f"SELECT ts, inst, CAST(_rn - COALESCE(_last_false, 0) AS DOUBLE) AS _v FROM ("
            f"SELECT *, MAX(CASE WHEN {false_cond} THEN _rn END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS _last_false FROM ("
            f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _rn "
            f"FROM ({inner.sql}) s0"
            f") s1"
            f") s2",
            has_inst_window=True,
        )

    if op == "fin_negative_streak":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if dialect != SqlDialect.DUCKDB:
            return None
        false_cond = f"(_v IS NULL OR _v >= 0)"
        return _Layer(
            f"SELECT ts, inst, CAST(_rn - COALESCE(_last_false, 0) AS DOUBLE) AS _v FROM ("
            f"SELECT *, MAX(CASE WHEN {false_cond} THEN _rn END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS _last_false FROM ("
            f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _rn "
            f"FROM ({inner.sql}) s0"
            f") s1"
            f") s2",
            has_inst_window=True,
        )

    if op == "fin_sign_change_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        sign_fn = _dialect_fn(dialect, "sign")
        prev_sign = f"LAG({sign_fn}(_v), 1) OVER (PARTITION BY inst ORDER BY ts)"
        changed = f"CASE WHEN {sign_fn}(_v) <> {prev_sign} AND {prev_sign} IS NOT NULL THEN 1 ELSE 0 END"
        return _Layer(
            f"SELECT ts, inst, SUM({changed}) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_monotonicity":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _int_attr(node, "window", default=4)
        if w is None:
            w = 4
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        sign_fn = _dialect_fn(dialect, "sign")
        delta = f"_v - LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, AVG({sign_fn}({delta})) OVER ({over}) AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "fin_turnover":
        if len(node.inputs) < 2:
            return None
        numerator = _compile_layer(node.inputs[0], dialect=dialect)
        denominator = _compile_layer(node.inputs[1], dialect=dialect)
        if numerator is None or denominator is None:
            return None
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"SELECT n.ts, n.inst, n._v / {nf}(d._v, 0) AS _v "
            f"FROM ({numerator.sql}) n LEFT JOIN ({denominator.sql}) d USING (ts, inst)",
            has_inst_window=numerator.has_inst_window or denominator.has_inst_window,
        )

    # More group operators
    if op == "group_max":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        return _Layer(
            f"SELECT x.ts, x.inst, MAX(x._v) OVER (PARTITION BY x.ts, g._v) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_min":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        return _Layer(
            f"SELECT x.ts, x.inst, MIN(x._v) OVER (PARTITION BY x.ts, g._v) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_count":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        return _Layer(
            f"SELECT x.ts, x.inst, CAST(COUNT(x._v) OVER (PARTITION BY x.ts, g._v) AS DOUBLE) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_sum":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        return _Layer(
            f"SELECT x.ts, x.inst, SUM(x._v) OVER (PARTITION BY x.ts, g._v) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_median":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        if dialect != SqlDialect.DUCKDB:
            return None
        return _Layer(
            f"SELECT x.ts, x.inst, MEDIAN(x._v) OVER (PARTITION BY x.ts, g._v) AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_quantile":
        if len(node.inputs) < 2:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if inner is None or grp is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        if dialect != SqlDialect.DUCKDB:
            return None
        qexpr = f"PERCENTILE_CONT({p}) WITHIN GROUP (ORDER BY x._v) OVER (PARTITION BY x.ts, g._v)"
        return _Layer(
            f"SELECT x.ts, x.inst, {qexpr} AS _v "
            f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # Cross-sectional operators
    if op == "cs_robust_scale":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if dialect != SqlDialect.DUCKDB:
            return None
        nf = _dialect_fn(dialect, "nullif")
        med = "MEDIAN(_v) OVER (PARTITION BY ts)"
        q25 = "PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY _v) OVER (PARTITION BY ts)"
        q75 = "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY _v) OVER (PARTITION BY ts)"
        iqr = f"({q75} - {q25})"
        return _Layer(
            f"SELECT ts, inst, (_v - {med}) / {nf}({iqr}, 0) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # Simple utility operators
    if op == "acos_bounded":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        return _Layer(
            f"SELECT ts, inst, ACOS({g}(-1.0, {l}(1.0, _v))) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "asin_bounded":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        return _Layer(
            f"SELECT ts, inst, ASIN({g}(-1.0, {l}(1.0, _v))) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "cos_phase":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        phase = _float_attr(node, "phase", default=0.0)
        return _Layer(
            f"SELECT ts, inst, COS(_v + {phase}) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "sin_phase":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        phase = _float_attr(node, "phase", default=0.0)
        return _Layer(
            f"SELECT ts, inst, SIN(_v + {phase}) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    # ------------------------------------------------------------------
    # Advanced Analytics: CAGR, Percentiles, Correlation, Conditional Aggregation
    # ------------------------------------------------------------------

    # ts_cagr: Time-series compound annual growth rate
    if op == "ts_cagr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        if window < 2:
            return None
        periods_per_year = _float_attr(node, "periods_per_year", default=252.0)
        annualization_factor = periods_per_year / (window - 1)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        lag_expr = f"LAG(_v, {window - 1}) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v > 0 AND ({lag_expr}) > 0 "
            f"THEN POWER(_v / ({lag_expr}), {annualization_factor}) - 1.0 "
            f"ELSE NULL END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # fin_cagr: Financial CAGR (compound annual growth rate)
    if op == "fin_cagr":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        periods = int(node.attrs.get("periods", _raw_literal(node, 2, 4)))
        periods_per_year = _float_attr(node, "periods_per_year", default=4.0)
        if periods < 1:
            return None
        annualization_factor = periods_per_year / periods
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {periods} PRECEDING AND CURRENT ROW"
        lag_expr = f"LAG(_v, {periods}) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v > 0 AND ({lag_expr}) > 0 "
            f"THEN POWER(_v / ({lag_expr}), {annualization_factor}) - 1.0 "
            f"ELSE NULL END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # ts_expected_shortfall: Expected Shortfall (CVaR)
    if op == "ts_expected_shortfall":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        alpha = _float_attr(node, "alpha", default=0.05)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        var_expr = f"PERCENTILE_CONT({alpha}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"WITH var_calc AS ("
            f"SELECT ts, inst, _v, {var_expr} AS var_threshold FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"AVG(CASE WHEN _v <= var_threshold THEN _v END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
            f") AS _v "
            f"FROM var_calc",
            has_inst_window=True,
        )

    # ts_expected_shortfall_asymmetry: Ratio of upper to lower expected shortfall
    if op == "ts_expected_shortfall_asymmetry":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        alpha = _float_attr(node, "alpha", default=0.05)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        lower_var = f"PERCENTILE_CONT({alpha}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        upper_var = f"PERCENTILE_CONT({1.0 - alpha}) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"WITH var_calc AS ("
            f"SELECT ts, inst, _v, {lower_var} AS lower_t, {upper_var} AS upper_t "
            f"FROM ({inner.sql}) t0"
            f"), es_calc AS ("
            f"SELECT ts, inst, "
            f"AVG(CASE WHEN _v <= lower_t THEN _v END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
            f") AS lower_es, "
            f"AVG(CASE WHEN _v >= upper_t THEN _v END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
            f") AS upper_es "
            f"FROM var_calc"
            f") "
            f"SELECT ts, inst, upper_es / NULLIF(ABS(lower_es), 0) AS _v FROM es_calc",
            has_inst_window=True,
        )

    # fin_percentile_history: Historical percentile rank of current value
    if op == "fin_percentile_history":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 2, 20)))
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, "
            f"PERCENT_RANK() OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # fin_percentile_vs_prior_history: Current percentile vs previous period percentile
    if op == "fin_percentile_vs_prior_history":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 2, 20)))
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        pct_rank = f"PERCENT_RANK() OVER ({over})"
        return _Layer(
            f"WITH pct AS ("
            f"SELECT ts, inst, _v, {pct_rank} AS pr FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, pr - LAG(pr, 1) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM pct",
            has_inst_window=True,
        )

    # fin_seasonal_percentile: Percentile within same season (e.g., same quarter)
    if op == "fin_seasonal_percentile":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        season_window = int(node.attrs.get("season_window", _raw_literal(node, 2, 4)))
        if dialect != SqlDialect.DUCKDB:
            return None
        # Use modulo arithmetic to identify season
        return _Layer(
            f"WITH seasonal AS ("
            f"SELECT ts, inst, _v, "
            f"ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn "
            f"FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"PERCENT_RANK() OVER (PARTITION BY inst, (rn % {season_window}) ORDER BY _v) AS _v "
            f"FROM seasonal",
            has_inst_window=True,
        )

    # rank_corr: Spearman rank correlation between two series
    if op == "rank_corr":
        if len(node.inputs) < 2:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        y = _compile_layer(node.inputs[1], dialect=dialect)
        if x is None or y is None:
            return None
        window = _window_int(node)
        if dialect != SqlDialect.DUCKDB:
            return None
        nf = _dialect_fn(dialect, "nullif")
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        # Rank both series, then compute Pearson correlation on ranks
        return _Layer(
            f"WITH joined AS ("
            f"SELECT x.ts, x.inst, x._v AS xv, y._v AS yv "
            f"FROM ({x.sql}) x LEFT JOIN ({y.sql}) y USING (ts, inst)"
            f"), ranked AS ("
            f"SELECT ts, inst, "
            f"RANK() OVER ({over} ORDER BY xv) AS rx, "
            f"RANK() OVER ({over} ORDER BY yv) AS ry, "
            f"COUNT(*) OVER ({over}) AS n "
            f"FROM joined"
            f"), moments AS ("
            f"SELECT ts, inst, rx, ry, n, "
            f"AVG(rx) OVER ({over}) AS mx, "
            f"AVG(ry) OVER ({over}) AS my "
            f"FROM ranked"
            f"), corr_calc AS ("
            f"SELECT ts, inst, "
            f"SUM((rx - mx) * (ry - my)) OVER ({over}) AS cov, "
            f"SQRT(SUM(POWER(rx - mx, 2)) OVER ({over})) AS sx, "
            f"SQRT(SUM(POWER(ry - my, 2)) OVER ({over})) AS sy "
            f"FROM moments"
            f") "
            f"SELECT ts, inst, cov / {nf}(sx * sy, 0) AS _v FROM corr_calc",
            has_inst_window=True,
        )

    # ts_corr_if: Conditional correlation (only when condition is met)
    if op == "ts_corr_if":
        if len(node.inputs) < 3:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        y = _compile_layer(node.inputs[1], dialect=dialect)
        cond = _compile_layer(node.inputs[2], dialect=dialect)
        if x is None or y is None or cond is None:
            return None
        window = _window_int(node)
        nf = _dialect_fn(dialect, "nullif")
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"WITH joined AS ("
            f"SELECT x.ts, x.inst, x._v AS xv, y._v AS yv, "
            f"CASE WHEN c._v != 0 AND c._v IS NOT NULL THEN 1 ELSE 0 END AS ok "
            f"FROM ({x.sql}) x "
            f"LEFT JOIN ({y.sql}) y USING (ts, inst) "
            f"LEFT JOIN ({cond.sql}) c USING (ts, inst)"
            f"), moments AS ("
            f"SELECT ts, inst, xv, yv, ok, "
            f"SUM(ok) OVER ({over}) AS n, "
            f"SUM(CASE WHEN ok = 1 THEN xv END) OVER ({over}) / {nf}(SUM(ok) OVER ({over}), 0) AS mx, "
            f"SUM(CASE WHEN ok = 1 THEN yv END) OVER ({over}) / {nf}(SUM(ok) OVER ({over}), 0) AS my "
            f"FROM joined"
            f"), corr_calc AS ("
            f"SELECT ts, inst, n, "
            f"SUM(CASE WHEN ok = 1 THEN (xv - mx) * (yv - my) END) OVER ({over}) AS cov, "
            f"SQRT(SUM(CASE WHEN ok = 1 THEN POWER(xv - mx, 2) END) OVER ({over})) AS sx, "
            f"SQRT(SUM(CASE WHEN ok = 1 THEN POWER(yv - my, 2) END) OVER ({over})) AS sy "
            f"FROM moments"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN n >= 2 THEN cov / {nf}(sx * sy, 0) ELSE NULL END AS _v "
            f"FROM corr_calc",
            has_inst_window=True,
        )

    # ts_autocorr_decay_half_life: Half-life of autocorrelation decay
    if op == "ts_autocorr_decay_half_life":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        max_lag = int(node.attrs.get("max_lag", _raw_literal(node, 2, 10)))
        if dialect != SqlDialect.DUCKDB:
            return None
        # Simplified: compute autocorr at lag 1, estimate half-life
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH lagged AS ("
            f"SELECT ts, inst, _v, LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts) AS lag1 "
            f"FROM ({inner.sql}) t0"
            f"), moments AS ("
            f"SELECT ts, inst, _v, lag1, "
            f"AVG(_v) OVER ({over}) AS mx, "
            f"AVG(lag1) OVER ({over}) AS my "
            f"FROM lagged"
            f"), corr_calc AS ("
            f"SELECT ts, inst, "
            f"SUM((_v - mx) * (lag1 - my)) OVER ({over}) AS cov, "
            f"SQRT(SUM(POWER(_v - mx, 2)) OVER ({over})) AS sx, "
            f"SQRT(SUM(POWER(lag1 - my, 2)) OVER ({over})) AS sy "
            f"FROM moments"
            f") "
            f"SELECT ts, inst, "
            f"CASE WHEN cov / {nf}(sx * sy, 0) > 0 "
            f"THEN -LN(2.0) / LN(cov / {nf}(sx * sy, 0)) "
            f"ELSE NULL END AS _v "
            f"FROM corr_calc",
            has_inst_window=True,
        )

    # ts_distance_corr: Distance correlation (simplified approximation)
    if op == "ts_distance_corr":
        if len(node.inputs) < 2:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        y = _compile_layer(node.inputs[1], dialect=dialect)
        if x is None or y is None:
            return None
        window = _window_int(node)
        # Simplified: use absolute differences as proxy for distance
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH joined AS ("
            f"SELECT x.ts, x.inst, x._v AS xv, y._v AS yv "
            f"FROM ({x.sql}) x LEFT JOIN ({y.sql}) y USING (ts, inst)"
            f"), distances AS ("
            f"SELECT ts, inst, "
            f"ABS(xv - AVG(xv) OVER ({over})) AS dx, "
            f"ABS(yv - AVG(yv) OVER ({over})) AS dy "
            f"FROM joined"
            f"), moments AS ("
            f"SELECT ts, inst, dx, dy, "
            f"AVG(dx) OVER ({over}) AS mdx, "
            f"AVG(dy) OVER ({over}) AS mdy "
            f"FROM distances"
            f"), corr_calc AS ("
            f"SELECT ts, inst, "
            f"SUM((dx - mdx) * (dy - mdy)) OVER ({over}) AS cov, "
            f"SQRT(SUM(POWER(dx - mdx, 2)) OVER ({over})) AS sx, "
            f"SQRT(SUM(POWER(dy - mdy, 2)) OVER ({over})) AS sy "
            f"FROM moments"
            f") "
            f"SELECT ts, inst, cov / {nf}(sx * sy, 0) AS _v FROM corr_calc",
            has_inst_window=True,
        )

    # ts_quantile_range: Difference between upper and lower quantiles
    if op == "ts_quantile_range":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        lower = _float_attr(node, "lower", default=0.25)
        upper = _float_attr(node, "upper", default=0.75)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, "
            f"PERCENTILE_CONT({upper}) WITHIN GROUP (ORDER BY _v) OVER ({over}) - "
            f"PERCENTILE_CONT({lower}) WITHIN GROUP (ORDER BY _v) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # ts_expectile: Expectile (asymmetric squared loss quantile)
    if op == "ts_expectile":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        tau = _float_attr(node, "tau", default=0.5)
        if dialect != SqlDialect.DUCKDB:
            return None
        # Simplified: use weighted average with asymmetric weights
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        median_expr = f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY _v) OVER ({over})"
        return _Layer(
            f"WITH med AS ("
            f"SELECT ts, inst, _v, {median_expr} AS m FROM ({inner.sql}) t0"
            f") "
            f"SELECT ts, inst, "
            f"SUM(CASE WHEN _v >= m THEN {tau} * _v ELSE {1.0 - tau} * _v END) OVER ({over}) / "
            f"SUM(CASE WHEN _v >= m THEN {tau} ELSE {1.0 - tau} END) OVER ({over}) AS _v "
            f"FROM med",
            has_inst_window=True,
        )

    # cs_quantile_resid: Residual from cross-sectional quantile regression
    if op == "cs_quantile_resid":
        if len(node.inputs) < 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None
        tau = _float_attr(node, "tau", default=0.5)
        if dialect != SqlDialect.DUCKDB:
            return None
        # Simplified: compute quantile regression using percentile
        return _Layer(
            f"WITH joined AS ("
            f"SELECT y.ts, y.inst, y._v AS yv, x._v AS xv "
            f"FROM ({y.sql}) y LEFT JOIN ({x.sql}) x USING (ts, inst)"
            f"), fit AS ("
            f"SELECT ts, inst, yv, xv, "
            f"PERCENTILE_CONT({tau}) WITHIN GROUP (ORDER BY yv) OVER (PARTITION BY ts) AS fitted "
            f"FROM joined"
            f") "
            f"SELECT ts, inst, yv - fitted AS _v FROM fit",
            has_ts_partition=True,
        )

    # ts_lower_partial_moment: Lower partial moment (downside risk)
    if op == "ts_lower_partial_moment":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        threshold = _float_attr(node, "threshold", default=0.0)
        order = _float_attr(node, "order", default=2.0)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"SELECT ts, inst, "
            f"AVG(CASE WHEN _v < {threshold} THEN POWER({threshold} - _v, {order}) ELSE 0 END) "
            f"OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # ts_upper_partial_moment: Upper partial moment (upside potential)
    if op == "ts_upper_partial_moment":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        threshold = _float_attr(node, "threshold", default=0.0)
        order = _float_attr(node, "order", default=2.0)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, "
            f"AVG(CASE WHEN _v > {threshold} THEN POWER(_v - {threshold}, {order}) ELSE 0 END) "
            f"OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # ts_tail_ratio: Ratio of upper to lower tail quantiles
    if op == "ts_tail_ratio":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        alpha = _float_attr(node, "alpha", default=0.05)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"SELECT ts, inst, "
            f"PERCENTILE_CONT({1.0 - alpha}) WITHIN GROUP (ORDER BY _v) OVER ({over}) / "
            f"{nf}(ABS(PERCENTILE_CONT({alpha}) WITHIN GROUP (ORDER BY _v) OVER ({over})), 0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # ts_upside_deviation: Semi-deviation above threshold
    if op == "ts_upside_deviation":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = _window_int(node)
        threshold = _float_attr(node, "threshold", default=0.0)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        return _Layer(
            f"SELECT ts, inst, "
            f"SQRT(AVG(CASE WHEN _v > {threshold} THEN POWER(_v - {threshold}, 2) ELSE 0 END) "
            f"OVER ({over})) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    # intra_realized_correlation: Realized correlation from intraday data
    if op == "intra_realized_correlation":
        if len(node.inputs) < 2:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        y = _compile_layer(node.inputs[1], dialect=dialect)
        if x is None or y is None:
            return None
        # Simplified: cross-sectional correlation at each timestamp
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH joined AS ("
            f"SELECT x.ts, x.inst, x._v AS xv, y._v AS yv "
            f"FROM ({x.sql}) x LEFT JOIN ({y.sql}) y USING (ts, inst)"
            f"), moments AS ("
            f"SELECT ts, inst, xv, yv, "
            f"AVG(xv) OVER (PARTITION BY ts) AS mx, "
            f"AVG(yv) OVER (PARTITION BY ts) AS my "
            f"FROM joined"
            f"), corr_calc AS ("
            f"SELECT ts, inst, "
            f"SUM((xv - mx) * (yv - my)) OVER (PARTITION BY ts) AS cov, "
            f"SQRT(SUM(POWER(xv - mx, 2)) OVER (PARTITION BY ts)) AS sx, "
            f"SQRT(SUM(POWER(yv - my, 2)) OVER (PARTITION BY ts)) AS sy "
            f"FROM moments"
            f") "
            f"SELECT ts, inst, cov / {nf}(sx * sy, 0) AS _v FROM corr_calc",
            has_ts_partition=True,
        )

    # ts_quantile_beta_spread: Spread between upper and lower quantile betas
    if op == "ts_quantile_beta_spread":
        if len(node.inputs) < 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None
        window = _window_int(node)
        lower_q = _float_attr(node, "lower_q", default=0.25)
        upper_q = _float_attr(node, "upper_q", default=0.75)
        if dialect != SqlDialect.DUCKDB:
            return None
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        # Simplified: compute slope at different quantiles
        return _Layer(
            f"WITH joined AS ("
            f"SELECT y.ts, y.inst, y._v AS yv, x._v AS xv "
            f"FROM ({y.sql}) y LEFT JOIN ({x.sql}) x USING (ts, inst)"
            f"), quantiles AS ("
            f"SELECT ts, inst, yv, xv, "
            f"PERCENTILE_CONT({lower_q}) WITHIN GROUP (ORDER BY xv) OVER ({over}) AS x_lower, "
            f"PERCENTILE_CONT({upper_q}) WITHIN GROUP (ORDER BY xv) OVER ({over}) AS x_upper, "
            f"PERCENTILE_CONT({lower_q}) WITHIN GROUP (ORDER BY yv) OVER ({over}) AS y_lower, "
            f"PERCENTILE_CONT({upper_q}) WITHIN GROUP (ORDER BY yv) OVER ({over}) AS y_upper "
            f"FROM joined"
            f") "
            f"SELECT ts, inst, "
            f"(y_upper - y_lower) / NULLIF(x_upper - x_lower, 0) AS _v "
            f"FROM quantiles",
            has_inst_window=True,
        )

    # group_ex_self_quantile: Group quantile excluding self
    if op == "group_ex_self_quantile":
        if len(node.inputs) < 2:
            return None
        val = _compile_layer(node.inputs[0], dialect=dialect)
        grp = _compile_layer(node.inputs[1], dialect=dialect)
        if val is None or grp is None:
            return None
        q = _float_attr(node, "q", default=0.5)
        if dialect != SqlDialect.DUCKDB:
            return None
        # Approximate by computing group quantile (excluding self is complex in SQL)
        return _Layer(
            f"WITH grp_data AS ("
            f"SELECT x.ts, x.inst, x._v AS val, g._v AS grp "
            f"FROM ({val.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
            f") "
            f"SELECT ts, inst, "
            f"PERCENTILE_CONT({q}) WITHIN GROUP (ORDER BY val) OVER (PARTITION BY ts, grp) AS _v "
            f"FROM grp_data",
            has_ts_partition=True,
        )

    # fiscal_autocorr: Autocorrelation for fiscal data
    if op == "fiscal_autocorr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = int(node.attrs.get("lag", _raw_literal(node, 2, 1)))
        window = int(node.attrs.get("window", _raw_literal(node, 3, 20)))
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"WITH lagged AS ("
            f"SELECT ts, inst, _v, LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS lag_v "
            f"FROM ({inner.sql}) t0"
            f"), moments AS ("
            f"SELECT ts, inst, _v, lag_v, "
            f"AVG(_v) OVER ({over}) AS mx, "
            f"AVG(lag_v) OVER ({over}) AS my "
            f"FROM lagged"
            f"), corr_calc AS ("
            f"SELECT ts, inst, "
            f"SUM((_v - mx) * (lag_v - my)) OVER ({over}) AS cov, "
            f"SQRT(SUM(POWER(_v - mx, 2)) OVER ({over})) AS sx, "
            f"SQRT(SUM(POWER(lag_v - my, 2)) OVER ({over})) AS sy "
            f"FROM moments"
            f") "
            f"SELECT ts, inst, cov / {nf}(sx * sy, 0) AS _v FROM corr_calc",
            has_inst_window=True,
        )

    return None


def _collect_columns(node: PlanNode, out: set[str]) -> None:
    """递归收集计划树引用的列名到 ``out`` 集合。"""
    if node.op == "column":
        name = node.attrs.get("name") or node.attrs.get("column")
        if name:
            out.add(str(name))
    for child in node.inputs:
        _collect_columns(child, out)


def plan_is_sql_capable(plan: PlanNode) -> bool:
    """判断计划树是否全部由 SQL backend 支持（委托 ``sql_registry.is_sql_capable``）。"""
    from backend.sql_pushdown.sql_registry import is_sql_capable

    return is_sql_capable(plan)


def _build_filter_clause(
    filt: SqlPushdownFilter | None,
    *,
    dialect: SqlDialect,
) -> str:
    """根据 ``SqlPushdownFilter`` 生成 base CTE 的 WHERE 子句。

    R13 P0-69：显式 EMPTY 仪器过滤器（``instrument_filter=[]``）与 ``None``
    （ALL）不再合并——EMPTY 直接生成 ``WHERE FALSE``（0 行）。
    """
    if filt is None:
        return ""
    if filt.instrument_filter_kind == InstrumentFilterKind.EMPTY:
        # Explicit empty universe: zero rows regardless of time bounds.
        return " WHERE FALSE"
    parts: list[str] = []
    if filt.time_column and filt.start is not None:
        tc = _quote_ident(filt.time_column)
        parts.append(f"{tc} >= {_sql_literal(filt.start)}")
    if filt.time_column and filt.end is not None:
        tc = _quote_ident(filt.time_column)
        parts.append(f"{tc} <= {_sql_literal(filt.end)}")
    # Legacy callers pass a non-empty ``instruments`` without setting the kind;
    # a non-empty tuple is an implicit LIST.  ``ALL`` carries ``()`` and ``EMPTY``
    # returned above, so the two never collide.
    if filt.instrument_column and filt.instruments:
        ic = _quote_ident(filt.instrument_column)
        insts = ", ".join(_sql_literal(x) for x in filt.instruments)
        parts.append(f"{ic} IN ({insts})")
    if not parts:
        return ""
    return " WHERE " + " AND ".join(parts)


def _duckdb_dataset_ref(dataset: str) -> str:
    """DuckDB sql() 要求 ``{{dataset}}`` 占位符。

    Validates dataset name before interpolation to prevent injection.
    """
    _validate_sql_identifier(dataset, context="dataset")
    return f"{{{{{dataset}}}}}"


def _build_base_cte(
    *,
    source_from: str,
    time_column: str,
    instrument_column: str,
    columns: Sequence[str],
    filt: SqlPushdownFilter | None,
    dialect: SqlDialect,
) -> str:
    """构造 base CTE：标准化 ``ts``/``inst`` 轴列并应用过滤。

    注意：source_from 在调用前已由 compile_plan_to_sql 验证
    （通过 _duckdb_dataset_ref 或直接表名验证）。
    """
    col_list = ", ".join(_quote_ident(c) for c in sorted(columns))
    where = _build_filter_clause(filt, dialect=dialect)
    return (
        f"base AS (SELECT {_quote_ident(time_column)} AS ts, "
        f"{_quote_ident(instrument_column)} AS inst, {col_list} "
        f"FROM {source_from}{where})"
    )


def _compile_layer(node: PlanNode, *, dialect: SqlDialect) -> _Layer | None:
    """带 optional CTE memo 的编译入口（递归经此函数以共享子树）。"""
    memo = _sql_memo_ctx.get()
    if memo is not None:
        from planner.plan_hash import structural_key

        key = structural_key(node)
        if key in memo.refs:
            return _Layer(f"SELECT ts, inst, _v FROM {memo.refs[key]}")
    layer = _compile_layer_impl(node, dialect=dialect)
    if layer is None or memo is None:
        return layer
    from planner.plan_hash import structural_key

    key = structural_key(node)
    canon = _resolve_canonical(node.op)
    if canon in {"column", "literal"} or memo.use_counts.get(key, 0) < 2:
        return layer
    if key not in memo.refs:
        memo._counter += 1
        name = f"s{memo._counter}"
        memo.refs[key] = name
        memo.bodies.append((name, layer.sql))
    return _Layer(f"SELECT ts, inst, _v FROM {memo.refs[key]}")


def _template_cache_key(
    plan: PlanNode,
    dialect: SqlDialect,
    *,
    dataset: str | None,
    table: str | None,
    time_column: str,
    instrument_column: str,
    filt: SqlPushdownFilter | None,
) -> str:
    """模板缓存 key = 计划形状 + 编译上下文（dataset/table/轴/过滤）。

    形状（literal 值类型化抽象）决定「模板」；上下文决定最终 SQL 的 base CTE
    与源引用。二者任一变化 → 不同缓存条目。
    """
    shape = _plan_shape_key(plan, dialect)
    filt_sig = "none"
    if filt is not None:
        filt_sig = "|".join([
            str(getattr(filt, "instrument_filter_kind", "")),
            str(filt.time_column), str(filt.instrument_column),
            str(filt.start), str(filt.end),
            ",".join(str(x) for x in (filt.instruments or ())),
        ])
    ctx = f"{dataset}|{table}|{time_column}|{instrument_column}|{filt_sig}"
    ctx_hash = hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:16]
    return f"{shape}:{ctx_hash}"


def compile_plan_to_sql(
    plan: PlanNode,
    *,
    dataset: str | None = None,
    table: str | None = None,
    time_column: str,
    instrument_column: str,
    filt: SqlPushdownFilter | None = None,
    dialect: SqlDialect = SqlDialect.DUCKDB,
) -> CompiledSql | None:
    """将单因子逻辑计划编译为可执行 SQL。

    DuckDB 使用 registry 数据集名（``{{dataset}}`` 占位符），ClickHouse 使用物理表名。
    计划不可 SQL 化或编译失败时返回 ``None``。

    R40：
        #214  production 下 emitter identity 未知 → hard fail；
        #216  编译三态（SUPPORTED / SEMANTICALLY_UNSUPPORTED / COMPILER_ERROR）
              记录到 ``last_compile_status()``；
        #65/217  按模板（计划形状）+ literal binding 的有界缓存复用。
    """
    # #214：production SQL capability 要求 emitter identity 已知。
    assert_emitter_identity_known()
    if not plan_is_sql_capable(plan):
        _record_compile_status(CompileStatus.SEMANTICALLY_UNSUPPORTED)
        return None

    tkey = _template_cache_key(
        plan, dialect, dataset=dataset, table=table,
        time_column=time_column, instrument_column=instrument_column, filt=filt,
    )
    bkey = _literal_binding_key(plan)
    cached = _SQL_TEMPLATE_CACHE.get(tkey, bkey)
    if cached is not None:
        _record_compile_status(CompileStatus.SUPPORTED)
        return cached

    # Validate dataset/table names BEFORE compilation (fail-closed, always raise).
    if dialect == SqlDialect.DUCKDB and dataset:
        _validate_sql_identifier(dataset, context="dataset")
    if dialect == SqlDialect.CLICKHOUSE and table:
        _validate_sql_identifier(table, context="table")

    try:
        use_counts = _structural_use_counts(plan)
        memo = _SqlCompileMemo(use_counts=use_counts) if any(c > 1 for c in use_counts.values()) else None
        token = _sql_memo_ctx.set(memo)
        try:
            layer = _compile_layer(plan, dialect=dialect)
        finally:
            _sql_memo_ctx.reset(token)
    except Exception as exc:
        # #216：内部编译错误（COMPILER_ERROR）与「语义不支持」严格区分。
        _record_compile_status(CompileStatus.COMPILER_ERROR)
        if _is_production_sql_mode():
            raise SqlCompileError(
                f"SQL 编译器内部错误（R40 #216 COMPILER_ERROR）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return None
    if layer is None:
        _record_compile_status(CompileStatus.SEMANTICALLY_UNSUPPORTED)
        return None

    cols: set[str] = set()
    _collect_columns(plan, cols)
    if not cols:
        _record_compile_status(CompileStatus.SEMANTICALLY_UNSUPPORTED)
        return None

    source_from = table if dialect == SqlDialect.CLICKHOUSE else (dataset or table)
    if source_from and dialect == SqlDialect.DUCKDB and dataset:
        source_from = _duckdb_dataset_ref(dataset)
    if not source_from:
        _record_compile_status(CompileStatus.SEMANTICALLY_UNSUPPORTED)
        return None

    push_filter = filt or SqlPushdownFilter(
        time_column=time_column,
        instrument_column=instrument_column,
    )
    base = _build_base_cte(
        source_from=source_from,
        time_column=time_column,
        instrument_column=instrument_column,
        columns=sorted(cols),
        filt=push_filter,
        dialect=dialect,
    )
    cte_parts = [base]
    if memo and memo.bodies:
        for name, sql in memo.bodies:
            cte_parts.append(f"{name} AS (SELECT ts, inst, _v FROM ({sql}) t)")
    with_body = ", ".join(cte_parts)
    query = (
        f"WITH {with_body} "
        f"SELECT ts, inst, _v AS value FROM ({layer.sql}) result "
        f"ORDER BY ts, inst"
    )
    read_datasets = (dataset,) if dataset and dialect == SqlDialect.DUCKDB else tuple()
    compiled = CompiledSql(
        query=query,
        read_datasets=read_datasets,
        referenced_columns=frozenset(cols),
        dialect=dialect,
        table=table if dialect == SqlDialect.CLICKHOUSE else None,
    )
    _SQL_TEMPLATE_CACHE.put(tkey, bkey, compiled)
    _record_compile_status(CompileStatus.SUPPORTED)
    return compiled


def compile_plan_to_sql_template(
    plan: PlanNode,
    *,
    dataset: str | None = None,
    table: str | None = None,
    time_column: str,
    instrument_column: str,
    filt: SqlPushdownFilter | None = None,
    dialect: SqlDialect = SqlDialect.DUCKDB,
) -> SqlTemplate | None:
    """R40 #65/#217：返回 :class:`SqlTemplate`（模板 + 首个 literal binding）。

    调用方随后用 ``SqlTemplate.bind(plan, **ctx)`` 把不同 literal binding 绑定
    回模板——binding 一致时零重编译；不一致时诚实重编译。
    """
    compiled = compile_plan_to_sql(
        plan, dataset=dataset, table=table, time_column=time_column,
        instrument_column=instrument_column, filt=filt, dialect=dialect,
    )
    if compiled is None:
        return None
    tkey = _template_cache_key(
        plan, dialect, dataset=dataset, table=table,
        time_column=time_column, instrument_column=instrument_column, filt=filt,
    )
    return SqlTemplate(shape_key=tkey, binding_key=_literal_binding_key(plan), compiled=compiled)


@dataclass(frozen=True)
class BatchCompiledSql:
    """多子树单条 SQL（WITH CSE 批执行）。"""

    query: str
    read_datasets: tuple[str, ...]
    referenced_columns: frozenset[str]
    column_aliases: tuple[tuple[str, str], ...]  # (sid, sql_alias)
    dialect: SqlDialect = SqlDialect.DUCKDB
    table: str | None = None


def compile_plans_batch_to_sql(
    plans: dict[str, PlanNode],
    *,
    dataset: str | None = None,
    table: str | None = None,
    time_column: str,
    instrument_column: str,
    filt: SqlPushdownFilter | None = None,
    dialect: SqlDialect = SqlDialect.DUCKDB,
) -> BatchCompiledSql | None:
    """将多个 SQL 可编译子树合并为一条 WITH 查询（共享 base CTE）。

    返回 ``BatchCompiledSql``，含各 ``sid`` 对应的 SQL 列别名；任一子树
    不可编译时返回 ``None``。
    """
    if not plans:
        return None
    if len(plans) == 1:
        sid, plan = next(iter(plans.items()))
        single = compile_plan_to_sql(
            plan,
            dataset=dataset,
            table=table,
            time_column=time_column,
            instrument_column=instrument_column,
            filt=filt,
            dialect=dialect,
        )
        if single is None:
            return None
        return BatchCompiledSql(
            query=single.query,
            read_datasets=single.read_datasets,
            referenced_columns=single.referenced_columns,
            column_aliases=((sid, "value"),),
            dialect=single.dialect,
            table=single.table,
        )

    cols: set[str] = set()
    layers: list[tuple[str, str, _Layer]] = []
    for sid, plan in plans.items():
        if not plan_is_sql_capable(plan):
            return None
        layer = _compile_layer(plan, dialect=dialect)
        if layer is None:
            return None
        _collect_columns(plan, cols)
        alias = f"v_{len(layers)}"
        layers.append((sid, alias, layer))

    if not cols:
        return None

    # Validate dataset/table names before FROM clause interpolation (fail-closed).
    if dialect == SqlDialect.DUCKDB and dataset:
        _validate_sql_identifier(dataset, context="dataset")
    if dialect == SqlDialect.CLICKHOUSE and table:
        _validate_sql_identifier(table, context="table")

    source_from = table if dialect == SqlDialect.CLICKHOUSE else (dataset or table)
    if source_from and dialect == SqlDialect.DUCKDB and dataset:
        source_from = _duckdb_dataset_ref(dataset)
    if not source_from:
        return None

    push_filter = filt or SqlPushdownFilter(
        time_column=time_column,
        instrument_column=instrument_column,
    )
    base = _build_base_cte(
        source_from=source_from,
        time_column=time_column,
        instrument_column=instrument_column,
        columns=sorted(cols),
        filt=push_filter,
        dialect=dialect,
    )

    sub_ctes: list[str] = []
    alias_map: list[tuple[str, str]] = []
    for sid, col_alias, layer in layers:
        sub_name = f"sub_{col_alias}"
        sub_ctes.append(
            f"{sub_name} AS (SELECT ts, inst, _v AS {col_alias} FROM ({layer.sql}) t)"
        )
        alias_map.append((sid, col_alias))

    first = f"sub_v_0"
    join_from = first
    select_cols = [f"{first}.ts", f"{first}.inst"]
    for _sid, col_alias in alias_map:
        sub_name = f"sub_{col_alias}"
        select_cols.append(f"{sub_name}.{col_alias}")
        if sub_name != first:
            join_from += f" INNER JOIN {sub_name} USING (ts, inst)"

    with_body = ", ".join([base] + sub_ctes)
    query = (
        f"WITH {with_body} "
        f"SELECT {', '.join(select_cols)} FROM {join_from} "
        f"ORDER BY {first}.ts, {first}.inst"
    )
    read_datasets = (dataset,) if dataset and dialect == SqlDialect.DUCKDB else tuple()
    return BatchCompiledSql(
        query=query,
        read_datasets=read_datasets,
        referenced_columns=frozenset(cols),
        column_aliases=tuple(alias_map),
        dialect=dialect,
        table=table if dialect == SqlDialect.CLICKHOUSE else None,
    )
