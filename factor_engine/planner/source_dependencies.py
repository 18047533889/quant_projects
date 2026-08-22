"""执行前(preflight)二级源依赖清单。

``compute_data_scope`` 只绑定 anchor source 与执行语义，不绑定计划中
``column`` 节点携带的二级 SourceRef 依赖（StockIncome/StockBalance 等）。
本模块在 plan 侧递归收集这些 SourceRef，输出规范 JSON 清单及其 sha256 前缀，
供 preflight / plan 缓存键把二级依赖纳入作用域。

只 import ``api.source_ref`` 与 ``planner.logical_plan``，避免循环依赖。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from api.source_ref import SourceRefSpec, decode_source_ref
from planner.logical_plan import PlanNode

# 与 api.source_ref 的 _PREFIX 保持一致；不 import 私有名以避免耦合。
_SOURCE_REF_PREFIX = "__fe_source_ref_v1__"

#: R20-062..066: ``*_ALL`` whole-market 标签的市场族映射。``ASHARE_ALL`` 属于
#: A 股族；``US_MASSIVE_ALL`` 属于 US 族。仅当标签族与声明 market 属同一族时
#: 才视为 whole market —— ``ASHARE_ALL`` + market="US" 绝不判为 whole market。
_MARKET_FAMILIES: dict[str, frozenset[str]] = {
    "ASHARE": frozenset({"A", "ASHARE", "CN", "CN_A"}),
    "US": frozenset({"US", "USA", "US_MASSIVE", "NASDAQ", "NYSE"}),
    "HK": frozenset({"HK", "HKG"}),
}

_UNIVERSE_ALL_MARKERS = frozenset({"", "ALL", "UNIVERSE_ALL"})


def _ref_canonical(spec: SourceRefSpec) -> str:
    """把 SourceRefSpec 折叠为规范 JSON 字符串（字段排序、键排序）。

    R10-P0-018: the canonical payload includes ``dialect`` /
    ``dialect_version`` — two tables with the same columns but different
    semantic dialect versions must NOT share a source-dependency hash.  The
    field's *contract* identity (PIT policy, unit, role) is keyed by
    ``(table, field)`` inside the field catalog and is covered by the field
    name itself; the per-field catalog hash is intentionally NOT duplicated
    here (no parallel contract truth source).
    """
    return json.dumps(
        {
            "table": spec.table,
            "field": spec.field,
            "params": dict(sorted(spec.params, key=lambda kv: kv[0])),
            "transform": spec.transform,
            "transform_params": dict(sorted(spec.transform_params, key=lambda kv: kv[0])),
            "dialect": spec.dialect,
            "dialect_version": spec.dialect_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_source_dependency_manifest(plan: PlanNode) -> tuple[str, ...]:
    """递归遍历计划，收集所有 SourceRef 二级依赖的规范 JSON 清单。

    仅处理 ``op == "column"`` 且 ``attrs["name"]`` 以 ``__fe_source_ref_v1__``
    开头的节点；普通列跳过。返回去重后排序的 ``tuple[str, ...]``。

    Round-8 audit #392（三态语义的 plan 侧配合）：若名字带 source-ref 前缀但
    payload 损坏（``decode_source_ref`` 抛 ``ValueError``），**不吞异常**——向
    调用方传播，调用方应把它当作不可缓存/需显式失败处理。
    """
    found: set[str] = set()

    def walk(node: PlanNode) -> None:
        if node.op == "column":
            name = node.attrs.get("name")
            if isinstance(name, str) and name.startswith(_SOURCE_REF_PREFIX):
                spec = decode_source_ref(name)
                found.add(_ref_canonical(spec))
        for child in node.inputs or ():
            walk(child)

    walk(plan)
    return tuple(sorted(found))


def source_dependency_hash(plan: PlanNode) -> str:
    """返回 source 依赖清单的 sha256 前缀哈希（与 data_scope 同为 24 位）。"""
    manifest = build_source_dependency_manifest(plan)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# R20-062..066: universe membership hash
# ---------------------------------------------------------------------------


class UniverseMembershipUnresolvedError(ValueError):
    """命名 universe 无法按 as-of date 解析出实际 membership（R20-062..066）。

    production 下命名 universe 必须解析为实际成员列表，不能只做字符串比较；
    解析失败即 fail-closed（不把未解析 membership 当空/全市场进入缓存键）。
    """


def resolve_universe_membership(
    universe: str | None,
    as_of: str | None = None,
    *,
    resolver: Any | None = None,
) -> tuple[str, ...]:
    """按 as-of date 解析命名 universe 的实际 membership（R20-062..066）。

    返回去重排序的成员 tuple。``resolver`` 为可调用
    ``(universe, as_of) -> Iterable[str]``；缺省时尝试 ``market.universe`` 的
    标准 resolver（``resolve_universe_membership`` 或 ``universe_members``）。
    解析失败抛 ``UniverseMembershipUnresolvedError`` —— 命名 universe 必须
    解析出真实成员，不能只字符串比较。
    """
    if resolver is not None:
        members = resolver(str(universe or ""), as_of)
    else:
        try:
            from market.universe import resolve_universe_members  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - resolver may live elsewhere
            try:
                from market.universe import universe_members  # type: ignore[attr-defined]

                resolve_universe_members = universe_members
            except Exception:
                resolve_universe_members = None
        if resolve_universe_members is None:
            raise UniverseMembershipUnresolvedError(
                f"cannot resolve membership for named universe {universe!r} "
                f"as-of {as_of!r}: no membership resolver available"
            )
        try:
            members = resolve_universe_members(str(universe or ""), as_of)
        except Exception as exc:  # pragma: no cover - resolver-specific failure
            raise UniverseMembershipUnresolvedError(
                f"universe membership resolution failed for {universe!r} "
                f"as-of {as_of!r}: {exc}"
            ) from exc
    out: list[str] = []
    for m in members or ():
        label = str(m).strip()
        if label and label not in out:
            out.append(label)
    return tuple(sorted(out))


def universe_membership_hash(
    universe: str | None,
    as_of: str | None = None,
    *,
    membership: Sequence[str] | None = None,
    resolver: Any | None = None,
) -> str:
    """实际 resolved membership 的 sha256 前缀哈希（R20-062..066）。

    非空 ``instrument_filter`` 与 ``factor.universe=CSI300`` 即使名字相同，
    成员列表也可能不同；缓存/CSE/身份必须绑定**实际成员**，不能只比较
    universe 字符串。``*_ALL`` 全市场标签先由 :func:`is_whole_market_label`
    判定，全市场时返回确定性 ``"*_ALL"`` 哈希（不解析成员）。

    ``membership`` 显式传入时优先（调用方已解析好）；否则用 ``resolver`` 或
    标准 market resolver 按 ``as_of`` 解析。
    """
    if is_whole_market_label(universe, market=None):
        return _whole_market_membership_hash(universe)
    if membership is None:
        membership = resolve_universe_membership(universe, as_of, resolver=resolver)
    payload = {
        "universe": str(universe or ""),
        "as_of": str(as_of or ""),
        "membership": sorted(str(m) for m in membership),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _whole_market_membership_hash(universe: str | None) -> str:
    """whole-market 标签的确定性成员哈希（带 market-family 一致性，R20-062..066）。"""
    raw = json.dumps(
        {"whole_market": True, "label": str(universe or "ALL").upper()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _market_family(market: str | None) -> str | None:
    """返回 market 声明所属的标准族（``A`` / ``US`` / ``HK``），未知返回 None。"""
    mkt = str(market or "").strip().upper()
    if not mkt:
        return None
    for family, aliases in _MARKET_FAMILIES.items():
        if mkt == family or mkt in aliases:
            return family
    # 未知 market：按前缀保守匹配（US_NASDAQ -> US 族）。
    for family, aliases in _MARKET_FAMILIES.items():
        if mkt.startswith(family) or any(mkt.startswith(a) for a in aliases):
            return family
    return None


def is_whole_market_label(universe: str | None, market: str | None = None) -> bool:
    """``*_ALL`` whole-market 判定（必须验证 market family 一致，R20-062..066）。

    - ``ALL`` / 空 → 全市场；
    - ``universe == market``（如 ``A`` + ``A``）→ 全市场；
    - ``*_ALL`` 结尾 → 仅当标签族与 ``market`` 属同一族（或未声明 market）时
      才全市场。``ASHARE_ALL`` + market="US" → **不是** whole market
      （fail-closed）。
    """
    univ = str(universe or "").strip().upper()
    if univ in _UNIVERSE_ALL_MARKERS:
        return True
    mkt = str(market or "").strip().upper()
    if mkt and univ == mkt:
        return True
    if univ.endswith("_ALL"):
        family = univ[: -len("_ALL")]
        if not mkt:
            return True
        mkt_family = _market_family(mkt)
        if mkt_family is None:
            return False
        if family == mkt_family or family in _MARKET_FAMILIES.get(mkt_family, frozenset()):
            return True
        # 复合族（``US_NASDAQ`` / ``US_MASSIVE``）以市场族为前缀 → 一致。
        if family.startswith(mkt_family) or mkt_family.startswith(family):
            return True
        # 标签族是标准族别名（如 ASHARE）而 market 属该族（如 A）→ 一致。
        for alias_family, aliases in _MARKET_FAMILIES.items():
            if family == alias_family and mkt_family == alias_family:
                return True
        return False
    return False
