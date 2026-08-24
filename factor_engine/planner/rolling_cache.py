"""Rolling 子表达式缓存摘要：配合 CSE shared_nodes 可观测与调度。

R20-010..013: ``ROLLING_OPS`` 不再是手写清单。它从 registry 的 canonical alias、
``OperatorMetadata``（``param_names`` / ``window_semantics``）、``ParamSpec``
（``history_semantics`` / ``history_formula``）动态派生；窗口解析优先使用声明
的 history param 名，而不是凭名字猜测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from factor_engine.planner.logical_plan import PlanNode

# 静态 bootstrap 基集：registry 尚未加载时（bootstrap / compiler 工具链）作为
# 快速路径与兜底；registry 加载后 ``refresh_rolling_ops()`` 会用派生集合替换
# ``ROLLING_OPS``。派生集合必须覆盖本基集（一致性由测试断言）。
_BASE_ROLLING_OPS: frozenset[str] = frozenset(
    {
        "ts_mean",
        "ts_std",
        "ts_std_dev",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_delta",
        "ts_rank",
        "ts_corr",
        "ts_correlation",
        "ts_cov",
        "ts_covariance",
        "ts_decay_linear",
        "ts_delay",
    }
)

# 快速路径（R20-013）：cache hit 判定使用的内存化派生集合。初始为基集，
# ``refresh_rolling_ops()`` 后为 registry 派生集合（两者一致性由测试断言）。
ROLLING_OPS: frozenset[str] = frozenset(_BASE_ROLLING_OPS)

_rolling_ops_refreshed = False
_last_derived_count = -1

# 兜底 history-param 名字优先级表：仅当 OperatorMetadata / ParamSpec 都没有
# 声明 history 语义时才按名字猜测（R20-011 —— 声明优先，名字猜测是最后手段）。
_HISTORY_PARAM_PRIORITY: tuple[str, ...] = (
    "window",
    "d",
    "period",
    "n",
    "lag",
    "periods",
    "span",
    "outer_window",
    "inner_window",
    "lookback",
    "ema_window",
    "atr_window",
    "er_window",
    "short_window",
    "long_window",
)


@dataclass(frozen=True)
class RollingCacheEntry:
    """单条 rolling 共享子树的摘要条目。

    字段：
        structural_id: CSE 结构键 / sid
        op: rolling 算子名
        window: 窗口长度（无法解析时为 ``None``）
        input_column: 主输入列名（无法解析时为 ``None``）
    """

    structural_id: str
    op: str
    window: int | None
    input_column: str | None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。

        返回：
            含 structural_id、op、window、input_column 的字典
        """
        return {
            "structural_id": self.structural_id,
            "op": self.op,
            "window": self.window,
            "input_column": self.input_column,
        }


_COLUMN_SOURCE_ATTRS = ("source_table", "field_id", "source_field", "field_registry_hash")


def _column_full_ref(node: PlanNode) -> str:
    """完整列引用：name + source identity（Review-8 #G）。

    两列可能共享 display name（sourceA.Close 与 sourceB.Close）却是完全不同的
    数据。列节点携带 analyzer 写出的 field_id / source_table / source_field 时，
    identity 必须参与 CSE / 摘要键 —— 否则 sourceA 的 rolling 结果会被复用给
    sourceB（错误缓存命中）。synthetic 列无 source identity 时退回 name。
    """
    name = str(node.attrs.get("name") or node.attrs.get("column") or "")
    parts = [str(node.attrs.get(k) or "") for k in _COLUMN_SOURCE_ATTRS]
    if any(parts):
        return f"{name}@{'|'.join(parts)}"
    return name


def _first_col_ref(node: PlanNode) -> str | None:
    """深度优先查找子树中首个列引用（含 source identity）。"""
    if node.op in {"col", "column"}:
        return _column_full_ref(node)
    for child in node.inputs:
        found = _first_col_ref(child)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# R20-010..013：rolling 算子集合与窗口参数的 registry 派生
# ---------------------------------------------------------------------------
def _resolve_canonical(op: str) -> str:
    """将 DSL 名 / 别名解析为 canonical 名；registry 未加载时原样返回。"""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(str(op))
    except Exception:  # pragma: no cover - bootstrap/compiler tooling
        return str(op)


def _metadata_for(canonical: str) -> Any:
    """按 canonical 获取 ``OperatorMetadata``（注册算子）或 catalog 字典。"""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, backend="pandas_numpy")
        if op is not None:
            meta = getattr(op, "metadata", None)
            if meta is not None:
                return meta
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is not None:
            return catalog
    except Exception:  # pragma: no cover - bootstrap/compiler tooling
        pass
    return None


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    """从 ``OperatorMetadata`` 或 catalog 字典按名取值（兼容两种元数据形态）。"""
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _history_param_for(canonical: str) -> str | None:
    """返回算子声明/推断的 history（window/lag）参数名，无则 ``None``。

    R20-011: 解析顺序 —— (1) ``ParamSpec.history_semantics`` /
    ``history_formula`` 声明的参数； (2) ``metadata.window_semantics`` 声明后
    在 ``param_names`` 中按优先级表寻找窗口参数； (3) 兜底按名字优先级表猜测。
    第 (1) 项是机器可读的权威声明，绝不被名字猜测覆盖。
    """
    meta = _metadata_for(canonical)
    if meta is None:
        return None
    specs = _meta_get(meta, "param_specs", None) or {}
    declared: list[str] = []
    for name, spec in specs.items():
        if getattr(spec, "history_semantics", None) or getattr(spec, "history_formula", None):
            declared.append(name)
    if declared:
        return sorted(declared)[0]
    pnames = list(_meta_get(meta, "param_names", None) or [])
    if _meta_get(meta, "window_semantics", None):
        for p in pnames:
            if p in _HISTORY_PARAM_PRIORITY:
                return p
        return None
    for p in pnames:
        if p in _HISTORY_PARAM_PRIORITY:
            return p
    return None


def _canonical_is_rolling(canonical: str) -> bool:
    """按 ``OperatorMetadata`` / ``ParamSpec`` / history 语义判定 rolling 算子。"""
    if canonical in _BASE_ROLLING_OPS:
        return True
    meta = _metadata_for(canonical)
    if meta is None:
        return False
    return _history_param_for(canonical) is not None


def _derive_rolling_ops() -> frozenset[str]:
    """从 registry 扫描全部 canonical，筛出 rolling 算子（惰性计算）。

    返回：
        rolling canonical 的 frozenset；registry 未加载 / 扫描失败时返回空集
        （调用方保留 bootstrap 基集）。
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        canonicals = OperatorRegistry.list_canonical()
    except Exception:  # pragma: no cover - bootstrap/compiler tooling
        return frozenset()
    out = {c for c in canonicals if _canonical_is_rolling(c)}
    return frozenset(out)


def refresh_rolling_ops() -> frozenset[str]:
    """重新派生 ``ROLLING_OPS``（registry 已加载后调用）。

    派生集合为空时（registry 尚未加载 / 全部失败）保留 bootstrap 基集，避免
    把快速路径清空。返回更新后的 ``ROLLING_OPS``。

    返回：
        更新后的 ``ROLLING_OPS`` frozenset
    """
    global ROLLING_OPS, _rolling_ops_refreshed, _last_derived_count
    derived = _derive_rolling_ops()
    if derived:
        ROLLING_OPS = frozenset(derived)
    _rolling_ops_refreshed = True
    _last_derived_count = _registry_size()
    return ROLLING_OPS


def _registry_size() -> int:
    """O(1) registry 规模信号（``_operators`` + ``_catalog`` 的 dict 长度）。

    避免在热路径（``is_rolling_operator`` 逐节点调用）里反复
    ``list_canonical()``——那是 set 并集 + 排序，1300 项约 170µs/次；这里仅两个
    O(1) ``len()``。
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return len(OperatorRegistry._operators) + len(OperatorRegistry._catalog)
    except Exception:  # pragma: no cover - bootstrap/compiler tooling
        return 0


def _ensure_rolling_ops_refreshed() -> None:
    """进程内首次使用时派生一次；registry 随后增长时重新派生。

    registry 可能在首次（空）派生之后才由 ``load_all`` 加载 —— canonical 计数
    一旦显著增长就重派，避免快速路径停留在 bootstrap 基集而漏掉新增 rolling 算子。
    """
    if not _rolling_ops_refreshed:
        refresh_rolling_ops()
        return
    n = _registry_size()
    if n != _last_derived_count and n > len(_BASE_ROLLING_OPS):
        refresh_rolling_ops()


def is_rolling_operator(op: str) -> bool:
    """判定一个算子（canonical 或别名）是否为 rolling / history 算子。

    R20-010: 别名先解析到 canonical，再查派生 ``ROLLING_OPS``；bootstrap 基集
    作为 registry 尚未加载时的兜底。这是 ``ROLLING_OPS`` 之外唯一的权威判定
    入口，语义键 / 缓存摘要全部走这里。
    """
    _ensure_rolling_ops_refreshed()
    resolved = _resolve_canonical(op)
    if resolved in ROLLING_OPS:
        return True
    if resolved in _BASE_ROLLING_OPS or str(op) in _BASE_ROLLING_OPS:
        return True
    return False


# ---------------------------------------------------------------------------
# 窗口解析
# ---------------------------------------------------------------------------
def _window_from_attrs(op: str, attrs: dict[str, Any]) -> int | None:
    """从算子 attrs 解析整数窗口参数。

    R20-011: 优先使用 registry 声明的 history param 名，而不是固定的
    ``("window", "d", "period", "n")`` 顺序；声明缺失时才回落名字猜测。
    """
    declared = _history_param_for(op)
    if declared is not None:
        if declared in attrs and attrs[declared] is not None:
            try:
                return int(attrs[declared])
            except (TypeError, ValueError):
                pass
    for key in ("window", "d", "period", "n"):
        if key in attrs and attrs[key] is not None:
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                continue
    if op in {"ts_delay", "delay", "ts_delta"}:
        for key in ("periods", "lag"):
            if key in attrs and attrs[key] is not None:
                try:
                    return int(attrs[key])
                except (TypeError, ValueError):
                    continue
    return None


def _window_from_literals(node: PlanNode) -> int | None:
    """从 positional literal child inputs 解析 window/lag 参数。

    ``ts_mean(close, 20)`` 的 20 是 ``op=="literal"`` 的 child input 而非 attrs
    （见 ``planner/lowerings/_helpers.py`` 的 ``ts_mean``/``ts_delay`` 等），因此
    ``_window_from_attrs`` 会漏掉窗口值。本函数遍历 ``node.inputs[1:]``，跳过
    column / materialized_series / plan_ref 等非 literal 子节点，取第一个
    ``op=="literal"`` 子节点作为窗口值；对 ``ts_delay``/``delay``/``ts_delta``
    该 literal 即 lag（同一解析逻辑）。

    参数：
        node: 候选 rolling 计划节点

    返回：
        整数窗口/lag；非 rolling 算子或无法解析时返回 ``None``
    """
    if not is_rolling_operator(node.op):
        return None
    for child in node.inputs[1:]:
        if getattr(child, "op", None) != "literal":
            continue
        value = (child.attrs or {}).get("value")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def rolling_entry_from_node(structural_id: str, node: PlanNode) -> RollingCacheEntry | None:
    """从计划节点构造 rolling 缓存摘要条目。

    参数：
        structural_id: 共享子树的结构 sid
        node: 候选 rolling 算子节点

    返回：
        ``RollingCacheEntry``；非 rolling 算子时返回 ``None``
    """
    if not is_rolling_operator(node.op):
        return None
    return RollingCacheEntry(
        structural_id=structural_id,
        op=node.op,
        window=_window_from_attrs(node.op, node.attrs),
        input_column=_first_col_ref(node),
    )


def summarize_rolling_cache(shared_nodes: dict[str, PlanNode]) -> dict[str, Any]:
    """从 CSE ``shared_nodes`` 提取 rolling 共享项摘要。

    参数：
        shared_nodes: 结构 CSE 产出的共享子树字典

    返回：
        含 ``rolling_shared_count`` 与 ``entries`` 列表的摘要字典
    """
    entries: list[RollingCacheEntry] = []
    for sid, node in shared_nodes.items():
        entry = rolling_entry_from_node(sid, node)
        if entry is not None:
            entries.append(entry)
    return {
        "rolling_shared_count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
