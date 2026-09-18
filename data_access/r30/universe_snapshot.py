"""data_access.r30.universe_snapshot —— R30-P1-009 UniverseSnapshot。

股票池（universe）同一名字（如 CSI300）在**不同 ST 处理 / 停牌处理 / 上市天数 /
可交易规则**下是不同集合——只用名字当身份会串池。本模块把 universe 身份折叠成
稳定快照：

    - ``members``：成分股集合（排序后进 digest，顺序无关）；
    - ``membership_policy_version`` / ``tradability_policy_version``：成分
      口径 / 可交易规则的版本——同一名字、同一成分，规则变了也是不同 universe；
    - ``source_snapshot``：成分来源（universe 数据集/快照）的版本标识。

``from_store`` 可经 ``store._resolve_universe_instruments``（私有但存在，防御性
getattr）解析成分，或直接接收 ``members``。不修改任何既有文件。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from data_access.core.exceptions import DataAccessError, SourceResolutionError
from data_access.r30._shared import stable_digest_full
from data_access.r30.calendar_snapshot import canonical


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return repr(value)


def _norm_members(members: Iterable[Any] | None) -> tuple[str, ...]:
    """把成分集合规范化成排序、去重、str 化的元组。"""
    if not members:
        return ()
    return tuple(sorted({str(m) for m in members if m is not None}))


def _market_of(store: Any, universe_id: str) -> str | None:
    """从 store 推断 universe 数据集所属市场（防御性，取不到 None）。"""
    for attr in ("market", "_market"):
        val = getattr(store, attr, None)
        if val:
            return str(val)
    return None


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _normalise_effective_bound(value: Any, name: str) -> str | None:
    """Validate and canonicalise an inclusive snapshot effective bound."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, str):
        if _ISO_DATE_RE.fullmatch(value):
            try:
                return date.fromisoformat(value).isoformat()
            except ValueError as exc:
                raise ValueError(f"{name} must be a valid ISO date or timezone-aware datetime") from exc
        if not _ISO_DATETIME_RE.fullmatch(value):
            raise ValueError(f"{name} must be a valid ISO date or timezone-aware datetime")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid ISO date or timezone-aware datetime") from exc
    else:
        raise TypeError(f"{name} must be an ISO string, date, datetime, or None")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} datetime must include an explicit timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _bound_instant(value: str) -> datetime:
    """Map date and datetime forms onto one deterministic UTC timeline."""
    if _ISO_DATE_RE.fullmatch(value):
        return datetime.combine(date.fromisoformat(value), datetime.min.time(), timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass(frozen=True)
class UniverseSnapshot:
    """一个股票池（universe）的稳定身份快照。

    同名的股票池，只要成分（members）或两个 policy_version 任一不同，
    ``snapshot_id`` 就不同。``members`` 只是可追溯信息，**不参与**对象相等性
    （快照身份由 digest 决定）。
    """

    universe_id: str
    market: str | None
    source_snapshot: Any
    membership_policy_version: str
    tradability_policy_version: str
    snapshot_id: str
    members: tuple[str, ...] = field(default=(), compare=False, repr=False)
    effective_start: str | None = None
    effective_end: str | None = None

    # ---- 构建 ----

    @classmethod
    def build(
        cls,
        universe_id: str,
        market: str | None,
        members: Iterable[Any] | None,
        membership_policy_version: str,
        tradability_policy_version: str,
        source_snapshot: Any = None,
        effective_start: Any = None,
        effective_end: Any = None,
    ) -> "UniverseSnapshot":
        """从成分 + 两个 policy 版本 + 来源快照构建 UniverseSnapshot。

        ``snapshot_id = stable_digest_full(universe_id, market, 排序后 members,
        两个 policy_version, source_snapshot)``。
        """
        universe_id = str(universe_id)
        members_t = _norm_members(members)
        start = _normalise_effective_bound(effective_start, "effective_start")
        end = _normalise_effective_bound(effective_end, "effective_end")
        if (start is None) != (end is None):
            raise ValueError("effective_start and effective_end must be provided together")
        if start is not None and _bound_instant(start) > _bound_instant(end):
            raise ValueError("effective_start must be <= effective_end")
        snapshot_id = stable_digest_full(
            canonical(universe_id),
            canonical(market),
            canonical(members_t),
            canonical(membership_policy_version),
            canonical(tradability_policy_version),
            canonical(source_snapshot),
        ) if start is None else stable_digest_full(
            canonical(universe_id), canonical(market), canonical(members_t),
            canonical(membership_policy_version), canonical(tradability_policy_version),
            canonical(source_snapshot), canonical(start), canonical(end),
        )
        return cls(
            universe_id=universe_id,
            market=str(market) if market is not None else None,
            source_snapshot=source_snapshot,
            membership_policy_version=str(membership_policy_version),
            tradability_policy_version=str(tradability_policy_version),
            snapshot_id=snapshot_id,
            members=members_t,
            effective_start=start,
            effective_end=end,
        )

    @classmethod
    def from_store(
        cls,
        store: Any,
        universe_id: str,
        members: Iterable[Any] | None = None,
        *,
        market: str | None = None,
        membership_policy_version: str = "1",
        tradability_policy_version: str = "1",
        time_range: tuple[Any, Any] | None = None,
        instruments: Sequence[str] | None = None,
        source_snapshot: Any = None,
        effective_start: Any = None,
        effective_end: Any = None,
    ) -> "UniverseSnapshot":
        """从 store 解析成分（优先私有 ``_resolve_universe_instruments``）并构建。

        ``members`` 显式给出时直接用；否则调用
        ``store._resolve_universe_instruments(universe_id, time_range, instruments)``
        （私有但存在，防御性 getattr）。取不到 market 时尝试从 store 推断。
        ``effective_start`` / ``effective_end`` 只能由调用方基于可信、已审计的
        成分来源显式提供；本方法绝不从请求 ``time_range`` 推断历史有效区间。
        """
        resolved: Iterable[Any] | None = members
        if resolved is None:
            resolve = getattr(store, "_resolve_universe_instruments", None)
            if callable(resolve):
                try:
                    out = resolve(universe_id, time_range, instruments)
                    resolved = out if out is not None else ()
                except DataAccessError:
                    raise
                except Exception as exc:
                    raise SourceResolutionError(
                        f"failed to resolve universe {universe_id!r}"
                    ) from exc
            else:
                resolved = ()
        if market is None:
            market = _market_of(store, universe_id)
        return cls.build(
            universe_id=universe_id,
            market=market,
            members=resolved,
            membership_policy_version=membership_policy_version,
            tradability_policy_version=tradability_policy_version,
            source_snapshot=source_snapshot,
            effective_start=effective_start,
            effective_end=effective_end,
        )

    # ---- 访问 ----

    @staticmethod
    def membership_digest(members: Iterable[Any]) -> str:
        """仅成分集合的稳定 digest（顺序无关）。"""
        return stable_digest_full(*_norm_members(members))

    @property
    def member_count(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "universe_id": self.universe_id,
            "market": self.market,
            "source_snapshot": _jsonable(self.source_snapshot),
            "membership_policy_version": self.membership_policy_version,
            "tradability_policy_version": self.tradability_policy_version,
            "membership_digest": self.membership_digest(self.members),
            "member_count": self.member_count,
            "members": list(self.members),
            "effective_start": self.effective_start,
            "effective_end": self.effective_end,
        }


__all__ = [
    "UniverseSnapshot",
]
