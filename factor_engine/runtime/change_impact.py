# -*- coding: utf-8 -*-
"""R31-P1-038 (CHANGE_IMPACT_RECOMPUTE_PASS)：source-change → affected 区间传播。

从「factor watermark」升级到 Change Impact DAG（R31 §55）：输入一个变化集合
``(dataset, field, instrument, time interval, source_version)``，沿 dependency
DAG 传播出受影响的 operator intervals 与 root partitions——只重算真正受影响的数据。

    - elementwise（add/log/…）：change at T → output affected [T, T]。
    - rolling（ts_mean/ts_std/…）：change at T → affected [T, T+window-1]
      （按 ``forward_impact``）。
    - stateful（EMA/recursive）：affected [T, ∞) 直到 checkpoint/replay end。
    - cs/group：change at T 影响当天整截面 → 至少 [T, T]，跨截面算子的
      forward_impact 由 contract 提供。

``affected_root_window(root, field, changed_start, changed_end)`` 返回根因子输出上
必须重算的最小连续区间（None 终点 = 直到 checkpoint 的重放）。

R39 fixes:
  - P0-045：shared DAG node 的 synthetic node id 恒定（``id(object) -> assigned``）。
  - P0-046：column 匹配用完整 Source Identity（dataset/field/market/grain/
    price_basis/revision_identity），不再是 ``attrs["name"] == field``。
  - P0-047：forward window 按真实 session 日历偏移（``CalendarSnapshot``），
    不再用 ``pandas.offsets.BDay`` 近似（CNY/US 节假日错误）。
  - P0-048：``_own_impact`` 不吞 contract 异常后按 operator-name 猜窗口；
    contract 失败 = UNBOUNDED（全量重算）。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AffectedWindow:
    """一个算子/因子根受 source change 影响的输出区间。"""

    node_id: str
    op: str
    start: str
    end: str | None  # None = 无限延伸到 checkpoint/replay end
    unbounded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "start": self.start,
            "end": self.end,
            "unbounded": self.unbounded,
        }


# ---------------------------------------------------------------------------
# R39-P0-046: full source identity for column matching
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnIdentity:
    """Changed column's full source identity.

    Matching a source change to a plan's ``column`` nodes by ``attrs["name"]``
    alone is unsafe when multiple datasets each carry ``close``/``ret``/
    ``industry`` — a revision to dataset B's ``close`` would dirty dataset A's
    ``close``.  This identity keys the match on (dataset, field, market, grain,
    price_basis, revision_identity).

    Empty fields in the CHANGED identity are wildcards (unconstrained); a
    non-empty changed field must EQUAL the node's field exactly.  The column
    node's own identity is extracted from its attrs / semantic_attrs (or the
    embedded ``SourceRefSpec`` for source-ref-encoded names).
    """

    dataset: str = ""
    field: str = ""
    market: str = ""
    grain: str = ""
    price_basis: str = ""
    revision_identity: str = ""

    _FIELDS = (
        "dataset", "field", "market", "grain", "price_basis", "revision_identity",
    )

    def matches(self, other: "ColumnIdentity") -> bool:
        """True when every NON-EMPTY field of ``self`` equals ``other``'s field."""
        for name in self._FIELDS:
            wanted = getattr(self, name)
            if wanted and wanted != getattr(other, name):
                return False
        return True


_SOURCE_REF_PREFIX = "__fe_source_ref_v1__"


def _column_identity(node: Any) -> "ColumnIdentity | None":
    """Extract a plan ``column`` node's source identity.

    SourceRef-encoded column names (``__fe_source_ref_v1__...``) decode to a
    ``SourceRefSpec`` carrying the full v2 identity (table/dataset/market/
    source_version).  Otherwise read the analyzer's typed leaf attrs /
    semantic_attrs (``field`` / ``source_table`` / ``price_basis`` / ``grain`` /
    ``source_vintage`` …).  Returns ``None`` when the node carries no field.
    """
    attrs = dict(getattr(node, "attrs", None) or {})
    sem = dict(getattr(node, "semantic_attrs", None) or {})
    name = str(
        attrs.get("name") or attrs.get("field") or attrs.get("source_field") or ""
    )
    dataset = str(
        attrs.get("source_table") or attrs.get("dataset") or attrs.get("table") or ""
    )
    market = str(attrs.get("market") or sem.get("market") or "")
    price_basis = str(attrs.get("price_basis") or sem.get("price_basis") or "")
    grain_raw = sem.get("grain") or ()
    if not isinstance(grain_raw, (tuple, list)):
        grain_raw = (grain_raw,)
    grain = ",".join(str(g) for g in grain_raw)
    revision = str(
        attrs.get("revision_identity")
        or attrs.get("source_version")
        or attrs.get("revision")
        or sem.get("source_vintage")
        or ""
    )
    if name.startswith(_SOURCE_REF_PREFIX):
        try:
            from api.source_ref import decode_source_ref

            spec = decode_source_ref(name)
        except Exception:
            spec = None
        if spec is not None:
            name = str(spec.field or name)
            if spec.dataset:
                dataset = str(spec.dataset)
            elif spec.table:
                dataset = str(spec.table)
            if spec.market:
                market = str(spec.market)
            if spec.source_version:
                revision = str(spec.source_version)
    if not name:
        return None
    return ColumnIdentity(
        dataset=dataset,
        field=name,
        market=market,
        grain=grain,
        price_basis=price_basis,
        revision_identity=revision,
    )


def _coerce_column_identity(column_identity: Any, *, field: str) -> ColumnIdentity:
    if column_identity is None:
        # Legacy name-only matching: an empty identity constrains only ``field``.
        return ColumnIdentity(field=field)
    if isinstance(column_identity, ColumnIdentity):
        return column_identity
    if isinstance(column_identity, dict):
        coerced = {
            k: str(v) for k, v in column_identity.items() if v is not None
        }
        known = set(ColumnIdentity._FIELDS)
        unknown = set(coerced) - known
        if unknown:
            raise TypeError(
                f"column_identity dict has unknown field(s): {sorted(unknown)!r} "
                f"(allowed: {sorted(known)!r})"
            )
        return ColumnIdentity(**coerced)
    raise TypeError(
        f"column_identity must be ColumnIdentity or dict, got "
        f"{type(column_identity).__name__}"
    )


# ---------------------------------------------------------------------------
# R39-P0-047: frozen session-day calendar for forward-window offsets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalendarSnapshot:
    """Frozen session-day calendar view for forward-window offsets.

    Wraps a :class:`storage.trading_calendar.TradingCalendar`; ``offset(base,
    n)`` advances by REAL trading sessions (CNY / US holidays and weekends are
    skipped).  Out-of-coverage targets clamp to the first/last known session
    rather than fabricating a pandas business-day guess — consistent with the
    AutoShard calendar semantics.
    """

    calendar: Any

    def offset(self, base: str | pd.Timestamp, n: int) -> pd.Timestamp:
        days = getattr(self.calendar, "days", None)
        if not days:
            raise ValueError(
                "CalendarSnapshot requires a non-empty TradingCalendar"
            )
        base_ts = pd.Timestamp(base).normalize()
        # first known session on/after base (a holiday/weekend base -> next
        # real session, i.e. the first bar whose value reflects the change).
        pos = bisect.bisect_left(days, base_ts)
        if pos == len(days):
            return days[-1]
        target = pos + int(n)
        if target < 0:
            return days[0]
        if target >= len(days):
            return days[-1]
        return days[target]


def _coerce_calendar(calendar: Any) -> "CalendarSnapshot | None":
    if calendar is None:
        return None
    if isinstance(calendar, CalendarSnapshot):
        return calendar
    return CalendarSnapshot(calendar)


def _forward_offset(
    anchor: Any,
    impact: int,
    *,
    calendar: "CalendarSnapshot | None",
) -> pd.Timestamp:
    """Advance ``anchor`` by ``impact`` session days.

    Uses the frozen calendar snapshot when provided (real trading sessions);
    falls back to ``pandas.offsets.BDay`` only for research / no-calendar
    environments (backward compatible).
    """
    if impact <= 0:
        return _parse_date(anchor)
    if calendar is not None:
        return calendar.offset(anchor, impact)
    return _parse_date(anchor) + pd.offsets.BDay(impact)


# ---------------------------------------------------------------------------
# R39-P0-048: contract-only forward impact (no name-guess fallback)
# ---------------------------------------------------------------------------
def _production_mode() -> bool:
    try:
        from runtime.production_policy import is_production_mode

        return bool(is_production_mode())
    except Exception:
        return False


def _own_impact(op: str, attrs: dict[str, Any]) -> int | None:
    """算子自身的 forward impact（多少个 session 之后仍受影响）。

    R39-P0-048：execution contract 是唯一权威（``forward_impact``）。contract
    解析失败 / 返回未知 = UNBOUNDED（``None``）—— 全量重算。绝不回退到
    operator-name 启发式（ts_ 前缀 / ewm 等）猜测的缩短窗口：unknown 必须
    保守地全量重算。production 下 contract 解析失败直接抛错（fail-closed）。
    """
    production = _production_mode()
    try:
        from runtime.execution_contract import forward_impact

        impact = forward_impact(op, attrs, production=production)
    except Exception:
        if production:
            raise
        return None
    if impact is not None:
        return int(impact)
    return None


def _parse_date(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


# ---------------------------------------------------------------------------
# DAG walk / propagation
# ---------------------------------------------------------------------------
def _walk_nodes(plan: Any) -> list[tuple[str, str, dict[str, Any], tuple[str, ...], Any]]:
    """展平 plan 为 (node_id, op, attrs, child_ids, column_identity)。

    R39-P0-045：synthetic node id 必须对每个对象恒定。旧实现对 shared DAG
    node 第一次访问分配 ``n1/n2/...``，第二次访问同一对象却返回 ``f"n{id(node)}"``
    —— 同一个对象两个 id，shared child 的 impact 传播边被丢。现在用一个
    ``id(object) -> canonical_assigned_node_id`` 映射保证同一对象永远一个 id。
    """
    out: list[tuple[str, str, dict[str, Any], tuple[str, ...], Any]] = []
    id_to_assigned: dict[int, str] = {}
    counter: list[int] = [0]

    def walk(node: Any) -> str:
        node_key = id(node)
        if node_key in id_to_assigned:
            return id_to_assigned[node_key]
        counter[0] += 1
        my_id = getattr(node, "node_id", None) or f"n{counter[0]}"
        id_to_assigned[node_key] = my_id
        child_ids = tuple(walk(c) for c in (getattr(node, "inputs", ()) or ()))
        attrs = dict(getattr(node, "attrs", None) or {})
        op = str(getattr(node, "op", "") or "")
        col_ident = _column_identity(node) if op == "column" else None
        out.append((my_id, op, attrs, child_ids, col_ident))
        return my_id

    walk(plan)
    return out


def compute_change_impact(
    plan: Any,
    *,
    field: str,
    changed_start: str,
    changed_end: str | None = None,
    column_identity: Any = None,
    calendar: Any = None,
) -> list[AffectedWindow]:
    """把 ``(field, [changed_start, changed_end])`` 沿 plan DAG 传播。

    ``column_identity``（R39-P0-046）：可传 :class:`ColumnIdentity` 或 dict，
    描述「哪个 source 的哪一列变化」。缺省退化为仅按 ``field`` 名字匹配
    （向后兼容）。``calendar``（R39-P0-047）：可传
    :class:`storage.trading_calendar.TradingCalendar` 或 :class:`CalendarSnapshot`，
    缺省退化为 pandas BDay（research）。

    返回每个受影响节点的最小输出区间。根（最后一个节点）的窗口即因子输出上
    必须重算的区间。
    """
    nodes = _walk_nodes(plan)
    start = _parse_date(changed_start)
    end = _parse_date(changed_end) or start
    target = _coerce_column_identity(column_identity, field=field)
    calendar_snapshot = _coerce_calendar(calendar)
    affected: dict[str, AffectedWindow] = {}

    def _propagate(
        node_id: str,
        op: str,
        attrs: dict[str, Any],
        child_ids: tuple[str, ...],
        col_ident: Any,
    ) -> None:
        # 子节点影响并集作为本节点的「输入受影响区间」。
        in_start: Any = None
        in_end: Any | None = None
        in_unbounded = False
        for cid in child_ids:
            w = affected.get(cid)
            if w is None:
                continue
            if w.start is not None and (in_start is None or w.start < in_start):
                in_start = w.start
            if w.end is None:
                in_unbounded = True
            elif in_end is None or w.end > in_end:
                in_end = w.end
        if op == "column":
            # R39-P0-046: exact source-identity match (never just name).
            if col_ident is not None and target.matches(col_ident):
                affected[node_id] = AffectedWindow(node_id, op, str(start), str(end))
            return
        if op == "literal":
            return
        impact = _own_impact(op, attrs)
        if in_start is None and op != "column":
            return  # 上游无变化
        out_start = _parse_date(in_start or start)
        if in_unbounded or impact is None:
            affected[node_id] = AffectedWindow(
                node_id, op, str(out_start.date()), None, unbounded=True
            )
        else:
            out_end = _forward_offset(
                _parse_date(in_end or end), impact, calendar=calendar_snapshot
            )
            affected[node_id] = AffectedWindow(
                node_id, op, str(out_start.date()), str(out_end.date())
            )

    # 传播：nodes 已是后序（children 先于 parent），正序迭代即 children 先算。
    for node_id, op, attrs, child_ids, col_ident in nodes:
        _propagate(node_id, op, attrs, child_ids, col_ident)
    return [w for w in affected.values() if w.op != "column"]


def affected_root_window(
    plan: Any,
    *,
    field: str,
    changed_start: str,
    changed_end: str | None = None,
    column_identity: Any = None,
    calendar: Any = None,
) -> AffectedWindow | None:
    """root 输出上必须重算的窗口（Change Impact 摘要）。"""
    affected = compute_change_impact(
        plan,
        field=field,
        changed_start=changed_start,
        changed_end=changed_end,
        column_identity=column_identity,
        calendar=calendar,
    )
    if not affected:
        return None
    return affected[-1]


__all__ = [
    "AffectedWindow",
    "ColumnIdentity",
    "CalendarSnapshot",
    "compute_change_impact",
    "affected_root_window",
]
