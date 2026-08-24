from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from factor_engine.util.logging_utils import get_logger

from .datasource import DataSource


logger = get_logger("factor_engine.storage.composite_source")

# Round-11 §35 (plan A): ``current_only`` joins a snapshot source to the anchor
# on EXACTLY matching rows only — the snapshot is never asof-ffilled into the
# historical panel.  Semantically identical to ``exact`` at execution, but the
# distinct name carries the SnapshotOnlySourcePolicy intent (a snapshot source
# must only ever be consumed current-only).
_ALLOWED_JOIN_METHODS = frozenset({"exact", "current_only", "asof_backward", "forward_fill"})


class CompositeJoinPolicyError(ValueError):
    """P1-05: production composite requires an explicit join policy.

    A non-anchor source must declare its join alignment explicitly (via the
    ``joins`` spec, or a LogicalTableContract / COS contract carried by the
    child source); silently defaulting to ``asof_backward`` in production is
    too aggressive and can invent history / create look-ahead.
    """


class SnapshotVerificationError(RuntimeError):
    """P1-06: a child snapshot could not be verified (refresh failed).

    Production fail-closed: old composite cache must never stay trusted when
    the underlying snapshot cannot be confirmed.
    """


@dataclass(frozen=True)
class SnapshotState:
    """A child source's snapshot observation in the composite manifest (P0-29).

    ``version`` is the child's snapshot token (``None`` when the child exposes
    no token, or when its refresh/read failed).  ``verified`` records whether
    the snapshot was CONFIRMED during this observation: ``False`` (a failed
    refresh or an unreadable token) means the snapshot is UNUSABLE and must
    NEVER be treated as "no update" — a ``None==None`` comparison must not
    silently reuse a stale cache.  ``observed_at`` is a monotonic timestamp
    taken at construction (informational; never part of equality across
    barriers — the barrier compares ``version`` + ``verified`` only).
    """

    version: str | None
    verified: bool = True
    observed_at: float = field(default_factory=time.monotonic)


class CompositeSnapshotVerificationError(RuntimeError):
    """P1-07: child snapshot tokens moved during a composite read.

    The cross-source logical read epoch is not coherent (e.g. A@snap10 /
    B@snap11) — the caller retries once or surfaces the error.
    """


class CompositeContractUnavailableError(RuntimeError):
    """P1-09: COS contract lookup failed while judging PIT-sensitivity.

    Production fail-closed: a contract that cannot be queried must NOT be
    treated as "probably safe".
    """


@dataclass(frozen=True)
class CompositeJoinSpec:
    """组合数据源单源 join 对齐规格。

    ``explicit_join``（P1-05）标记该源是否由配置显式声明了 join 策略——
    production 下非锚点源必须显式声明，否则 build 时抛 ``CompositeJoinPolicyError``；
    research 才允许缺省回退 ``asof_backward``。

    参数:
        无
    """
    method: str = "asof_backward"
    tolerance: str | None = None
    explicit_join: bool = False


@dataclass
class CompositeJoinReport:
    """组合 join 对齐统计报告。

    Review-8 #474: ``matched_rows``（key 匹配数）与 ``value_valid_rows``（值有效
    数）是两回事 —— ``key 命中但 source 值本身是 null`` 的行不得混入
    ``key_unmatched``。 报告区分四类：

    * ``key_matched_rows`` — asof/exact 找到了该 instrument 在锚定时刻的源行；
    * ``key_unmatched_rows`` — 没有可用的源行（anchor_rows - key_matched）；
    * ``value_valid_rows`` — 对齐后值非 null；
    * ``value_null_rows`` — 对齐后值为 null（key 命中但源值 null，或 key 未命中）。

    ``matched_rows`` / ``unmatched_rows`` 保留为 value 语义的向后兼容别名
    （== value_valid_rows / value_null_rows）。
    """
    source: str
    column: str
    canonical_name: str
    method: str
    anchor_rows: int
    key_matched_rows: int
    key_unmatched_rows: int
    value_valid_rows: int
    value_null_rows: int

    def to_dict(self) -> dict[str, Any]:
        """to_dict。

        参数:
            无

        返回:
            dict[str, Any]
        """
        return {
            "source": self.source,
            "column": self.column,
            "canonical_name": self.canonical_name,
            "method": self.method,
            "anchor_rows": self.anchor_rows,
            "key_matched_rows": self.key_matched_rows,
            "key_unmatched_rows": self.key_unmatched_rows,
            "value_valid_rows": self.value_valid_rows,
            "value_null_rows": self.value_null_rows,
            # Backward-compatible aliases: matched == value-valid.
            "matched_rows": self.value_valid_rows,
            "unmatched_rows": self.value_null_rows,
        }


class CompositeSnapshotBarrier:
    """P1-07: 组合跨源读的原子一致性屏障。

    捕获所有 child 当前 snapshot token → 读全部数据 → revalidate（读前 vs 读后
    一致）→ 不变 commit / 变则抛 ``CompositeSnapshotVerificationError`` 让调用方
    整体 retry 一次。杜绝「refresh child A → refresh child B → load A → load B」
    期间底层快照漂移导致 A@snap10 / B@snap11 的非同一逻辑读 epoch。

    用法::

        barrier = CompositeSnapshotBarrier(source)
        barrier.begin()
        data = ... read ...
        barrier.revalidate()   # 抛 CompositeSnapshotVerificationError 于变化
    """

    def __init__(self, source: "CompositeDataSource") -> None:
        self._source = source
        self._before: tuple[tuple[str, SnapshotState], ...] | None = None

    def begin(self) -> None:
        """读前捕获所有 child 的当前 snapshot token（不刷新，只读 token）。"""
        self._before = self._source._child_snapshot_manifest(refresh=False)

    def revalidate(self) -> None:
        """读后 revalidate：token 变化 → 清缓存并抛验证错误（fail-closed）。

        比较 ``version`` + ``verified``（忽略 ``observed_at``；P0-29 下任一侧
        ``verified=False`` 也视为「变化」——不确认的 snapshot 不得当「无更新」）。

        R24-081: on success the manifest recorded is EXACTLY the epoch verified
        for this read (``after`` == ``before``) — a later refresh that advances
        the source is a separate observation and never silently rewrites the
        read's epoch.
        """
        if self._before is None:
            return
        after = self._source._child_snapshot_manifest(refresh=False)
        if self._source._snapshot_manifest_changed(self._before, after):
            self._source._clear_caches()
            self._source._snapshot_manifest = after
            raise CompositeSnapshotVerificationError(
                "composite child snapshot changed during load "
                f"(before={self._before!r} after={after!r}); "
                "refusing to serve a non-atomic cross-source read"
            )
        # Record the VERIFIED epoch (not a fresh re-read) so manifest == data.
        self._source._snapshot_manifest = after


class CompositeDataSource(DataSource):
    """多数据源按锚点对齐的统一列空间。
    
    参数:
        anchor_source: 锚点数据源名称（可选）
        anchor_column: 锚点列引用（可选）
        sources: 子数据源映射（可选）
        joins: 非锚点源的 join 配置（可选）
        aliases: 列名别名映射（可选）
        allow_unqualified_anchor_columns: 见函数签名（可选）
    """

    def __init__(
        self,
        *,
        anchor_source: str,
        anchor_column: str,
        sources: Mapping[str, DataSource],
        joins: Mapping[str, Any] | None = None,
        aliases: Mapping[str, str] | None = None,
        allow_unqualified_anchor_columns: bool = True,
        production: bool | None = None,
    ) -> None:
        """初始化实例。

        参数:
            anchor_source: 锚点数据源名称（可选）
            anchor_column: 锚点列引用（可选）
            sources: 子数据源映射（可选）
            joins: 非锚点源的 join 配置（可选）
            aliases: 列名别名映射（可选）
            allow_unqualified_anchor_columns: 见函数签名（可选）
            production: 父级 production authority（P1-08）。由
                FactorEngine / DataSourceBuildContext 注入；``None`` 时回退到
                「任一子源 production=True」的旧推导（直接构造的向后兼容）。

        返回:
            无
        """
        if not isinstance(anchor_source, str) or not anchor_source.strip():
            raise ValueError("Composite data source requires a non-empty anchor_source")
        if not isinstance(anchor_column, str) or not anchor_column.strip():
            raise ValueError("Composite data source requires a non-empty anchor_column")
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError("Composite data source requires a non-empty sources mapping")

        self.anchor_source = anchor_source.strip()
        self.anchor_column = anchor_column.strip()
        self.sources = {str(name): source for name, source in sources.items()}
        if self.anchor_source not in self.sources:
            raise ValueError(
                f"anchor_source '{self.anchor_source}' not found in composite sources"
            )

        # P1-08: effective production authority comes from the parent build
        # context; a child ``production=False`` must NOT downgrade a parent
        # production run.  ``None`` (direct construction) falls back to the old
        # child-derived rule for backward compatibility.
        self._production_authority = None if production is None else bool(production)
        if self._production_authority is None:
            self._effective_production = any(
                bool(getattr(src, "production", False))
                for src in self.sources.values()
            )
        else:
            self._effective_production = self._production_authority

        self.allow_unqualified_anchor_columns = bool(allow_unqualified_anchor_columns)
        self.aliases = self._normalize_aliases(aliases or {})
        self.joins = self._normalize_joins(joins or {})
        self._column_cache: dict[str, Any] = {}
        self._anchor_index_cache = None
        self._join_reports: list[CompositeJoinReport] = []
        #: R10 #44: combined child snapshot manifest.  ``None`` until the first
        #: load records a baseline; afterwards a change in any child snapshot id
        #: invalidates ``_column_cache`` / ``_anchor_index_cache``.
        self._snapshot_manifest: tuple[tuple[str, SnapshotState], ...] | None = None
        self._validate_join_policy()

    def execution_spec(self) -> dict[str, Any] | None:
        """返回可重建（``storage.factory.build_data_source``）的 canonical 配置。

        #收官轮 P0：递归序列化全部子源（含 join 契约 / aliases / production
        authority），供 lineage / semantic identity / full definition / 事件增量
        rebuild 复用——Composite 子源的 temporal join 语义必须原样还原。

        #收官轮 P1：任一 child 不是 ``DataSource`` 或无法产生完整 ``execution_spec``
        时，production 抛 ``UnreconstructableDataSource``；research 返回 ``None``。
        旧实现会伪造 ``{"type": "data_access"}``（无 ``dataset``）——既不能 rebuild，
        又把「无法恢复 source contract」伪装成「有 source config」。
        """
        from .datasource import DataSource, clean_execution_spec

        from factor_engine.storage.exceptions import UnreconstructableDataSource

        child_specs: dict[str, Any] = {}
        for name, source in self.sources.items():
            spec = source.execution_spec() if isinstance(source, DataSource) else None
            if not isinstance(spec, dict):
                if self._effective_production:
                    raise UnreconstructableDataSource(
                        f"Composite child {name!r}（type={type(source).__name__}）无法"
                        f"序列化完整 source contract：execution_spec() 返回 "
                        f"{type(spec).__name__}。production 禁止伪造残缺 "
                        f"{{'type': 'data_access'}} spec——事件增量 rebuild 无法还原"
                        f"该子源的真实读取语义。"
                    )
                return None
            child_specs[str(name)] = spec
        return clean_execution_spec(
            {
                "type": "composite",
                "anchor": self.anchor_source,
                "anchor_column": self.anchor_column,
                "sources": child_specs,
                "joins": {
                    str(name): clean_execution_spec(
                        {
                            "method": spec.method,
                            "tolerance": spec.tolerance,
                        }
                    )
                    for name, spec in self.joins.items()
                },
                "aliases": dict(self.aliases),
                "allow_unqualified_anchor_columns": self.allow_unqualified_anchor_columns,
                "production": self._effective_production,
            }
        )

    @staticmethod
    def _normalize_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
        """_normalize_aliases。
        
        参数:
            aliases: 列名别名映射
        
        返回:
            dict[str, str]
        """
        normalized: dict[str, str] = {}
        for alias, target in aliases.items():
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Composite data source alias names must be non-empty strings")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Composite data source alias targets must be non-empty strings")
            normalized[alias.strip()] = target.strip()
        return normalized

    def _normalize_joins(
        self,
        joins: Mapping[str, Any],
    ) -> dict[str, CompositeJoinSpec]:
        """_normalize_joins。
        
        参数:
            joins: 非锚点源的 join 配置
        
        返回:
            dict[str, CompositeJoinSpec]
        """
        unexpected = sorted(set(joins) - set(self.sources))
        if unexpected:
            joined = ", ".join(unexpected)
            raise ValueError(f"Unknown composite join source(s): {joined}")

        specs: dict[str, CompositeJoinSpec] = {}
        for source_name in self.sources:
            if source_name == self.anchor_source:
                if source_name in joins:
                    raise ValueError("Composite join spec must not be provided for anchor source")
                continue
            specs[source_name] = self._parse_join_spec(source_name, joins.get(source_name))
        return specs

    def _validate_join_policy(self) -> None:
        """P1-05: production 下非锚点源必须显式声明 join 策略。

        ``asof_backward`` 是 research 的方便缺省；production 里缺省太激进
        （可能给 snapshot/事件源发明历史、造成前视）。build/compile 阶段直接
        fail-closed，而不是等到 load 才暴露。
        """
        if not self._effective_production:
            return
        missing = [
            name for name, spec in self.joins.items() if not spec.explicit_join
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise CompositeJoinPolicyError(
                "production composite 要求每个非锚点源显式声明 join policy；"
                f"以下源未声明（不能回退默认 asof_backward）: {joined}"
            )

    def _clear_caches(self) -> None:
        """清空组合源列/锚点缓存（P1-06/P1-07 共用）。"""
        self._column_cache.clear()
        self._anchor_index_cache = None

    def _parse_join_spec(
        self,
        source_name: str,
        raw: Any,
    ) -> CompositeJoinSpec:
        """_parse_join_spec。
        
        参数:
            source_name: 见函数签名
            raw: 见函数签名
        
        返回:
            CompositeJoinSpec
        """
        if raw is None:
            return CompositeJoinSpec()
        if isinstance(raw, str):
            method = raw
            tolerance = None
        elif isinstance(raw, Mapping):
            method = raw.get("method", raw.get("strategy", raw.get("align", "asof_backward")))
            tolerance = raw.get("tolerance")
        else:
            raise TypeError(
                f"Composite join config for source '{source_name}' must be a string or mapping"
            )

        normalized_method = self._normalize_join_method(method)
        if tolerance is not None and normalized_method == "exact":
            raise ValueError(
                f"Composite join source '{source_name}' uses exact alignment and cannot set tolerance"
            )
        # P1-05: a configured join (string or mapping) is an explicit declaration;
        # only ``joins[name]`` missing entirely leaves ``explicit_join=False``.
        return CompositeJoinSpec(
            method=normalized_method,
            tolerance=tolerance,
            explicit_join=True,
        )

    @staticmethod
    def _normalize_join_method(method: Any) -> str:
        """_normalize_join_method。
        
        参数:
            method: 见函数签名
        
        返回:
            str
        """
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Composite join method must be a non-empty string")
        lowered = method.strip().lower()
        aliases = {
            "asof": "asof_backward",
            "backward": "asof_backward",
            "ffill": "forward_fill",
        }
        normalized = aliases.get(lowered, lowered)
        if normalized not in _ALLOWED_JOIN_METHODS:
            allowed = ", ".join(sorted(_ALLOWED_JOIN_METHODS))
            raise ValueError(
                f"Unsupported composite join method '{method}'. Allowed: {allowed}"
            )
        if normalized == "forward_fill":
            warnings.warn(
                "Composite join method 'forward_fill' 已废弃，语义同 asof_backward；"
                "请改用 asof_backward。",
                DeprecationWarning,
                stacklevel=4,
            )
            normalized = "asof_backward"
        return normalized

    def collect_join_reports(self, *, clear: bool = True) -> list[dict[str, Any]]:
        """返回组合 join 统计（可选写入 lineage.extra）。

        参数:
            clear: 见函数签名（可选）

        返回:
            list[dict[str, Any]]
        """
        reports = [r.to_dict() for r in self._join_reports]
        if clear:
            self._join_reports.clear()
        return reports

    def _child_snapshot_manifest(
        self,
        *,
        refresh: bool = True,
    ) -> tuple[tuple[str, SnapshotState], ...]:
        """组合子源快照清单（R10 #44 + #收官轮 P0 Integration + P1-06/P1-08 + P0-29）。

        ``refresh=True``：**先调用每个子源的 ``refresh_snapshot()``**（proactive
        刷新，TTL 内短路；DataAccessSource 的廉价路径走 manifest token），再读
        统一 ``snapshot_token``（``_manifest_token or data_snapshot_id``）—— 否则
        Composite 永远只看 ``data_snapshot_id`` 而看不到 manifest 级变化，底层
        dataset A→B 后仍命中自己的旧缓存（cache-of-cache coherence bug）。
        ``refresh=False``：只读当前 token，不推进刷新（P1-07 读前后 revalidate 用）。

        每个子源产出 ``SnapshotState``：``version`` 是 token（可 None），
        ``verified=False`` 表示本次观测无法确认（refresh 抛错 / token 读取失败）。
        refresh 失败（P1-06）：production 直接抛 ``SnapshotVerificationError``
        （fail-closed —— 两次失败 manifest 相同会让旧 cache 继续被信任，必须拒绝）；
        research 清缓存 + warning，并记录 ``verified=False`` —— P0-29 保证
        ``None==None`` 永不等于「无更新」（``_invalidate_if_snapshot_changed``
        对 ``verified=False`` 一律判「已变化 / 不可用」）。没有 ``refresh_snapshot``
        的子源退化为只读 ``data_snapshot_id``（None）。
        """
        production = self._effective_production
        manifest: list[tuple[str, SnapshotState]] = []
        for name, source in self.sources.items():
            verified = True
            if refresh:
                refresher = getattr(source, "refresh_snapshot", None)
                if callable(refresher):
                    try:
                        refresher()
                    except Exception as exc:  # refresh 失败 → 无法确认
                        if production:
                            self._clear_caches()
                            raise SnapshotVerificationError(
                                f"composite child source {name!r} refresh_snapshot "
                                f"failed ({type(exc).__name__}: {exc}); production "
                                "fail-closed — cannot verify snapshot, old cache "
                                "must not be trusted"
                            ) from exc
                        # P1-06: research 下 clear cache + warning。
                        logger.warning(
                            "组合源子源 %r refresh_snapshot 失败（research 清缓存、"
                            "不信任旧 snapshot）: %s",
                            name,
                            exc,
                        )
                        self._clear_caches()
                        verified = False
            snapshot = getattr(source, "snapshot_token", None)
            if snapshot is None:
                snapshot = getattr(source, "data_snapshot_id", None)
            if callable(snapshot):
                try:
                    snapshot = snapshot()
                except Exception:  # best-effort manifest: a child that cannot
                    # report a snapshot is unverifiable (unusable), never
                    # "confirmed None".
                    snapshot = None
                    verified = False
            # R24-079/080: a child with NO snapshot token (and no acceptable
            # alternative proof — content hash / tx id / MVCC / manifest digest)
            # is UNVERIFIABLE.  ``version=None, verified=True`` must never be
            # recorded: in production, no token → verified=False so ``None==None``
            # is never mistaken for "no update".  Research keeps the legacy
            # lenient reading for back-compat.
            if snapshot is None and production:
                verified = False
            manifest.append((name, SnapshotState(version=snapshot, verified=verified)))
        return tuple(manifest)

    @staticmethod
    def _snapshot_manifest_changed(
        old: tuple[tuple[str, SnapshotState], ...],
        current: tuple[tuple[str, SnapshotState], ...],
    ) -> bool:
        """True when the composite snapshot barrier must invalidate / retry.

        Compares only ``version`` + ``verified`` — never ``observed_at`` (each
        observation carries a fresh monotonic timestamp, so raw tuple equality
        would falsely report a drift on every call).

        P0-29: a ``verified=False`` on EITHER side — or in the new snapshot — is
        always "changed/unusable".  A child whose refresh failed exposes
        ``version=None``; treating ``None==None`` as "no update" would silently
        reuse a stale cache, so any unverified observation forces invalidation.
        """
        if len(old) != len(current):
            return True
        old_by_name = dict(old)
        for name, new_state in current:
            old_state = old_by_name.get(name)
            if old_state is None:
                return True
            if not old_state.verified or not new_state.verified:
                return True
            if old_state.version != new_state.version:
                return True
        return False

    def _invalidate_if_snapshot_changed(self) -> None:
        """子源快照版本变化时清除列/锚点缓存（R10 #44 + #收官轮 P0 + P0-29）。

        在公共 load 方法顶部调用：**主动 refresh 每个子源** 后比较最新
        snapshot_token 清单；任一子源 token 变化（或任何一侧 ``verified=False``，
        P0-29 —— 包括 refresh 失败 / token 读取失败）→ 旧列缓存与锚点索引缓存
        全部失效，下次读取强制重建——Composite 自己的缓存不能比底层数据更长寿，
        ``None==None`` 也绝不等于「无更新」。
        """
        current = self._child_snapshot_manifest()
        if self._snapshot_manifest is not None and self._snapshot_manifest_changed(
            self._snapshot_manifest, current
        ):
            logger.info(
                "组合源子源快照变化，清除列/锚点缓存 old=%s new=%s",
                self._snapshot_manifest,
                current,
            )
            self._clear_caches()
        self._snapshot_manifest = current

    def _record_snapshot_manifest(self) -> None:
        """load 完成后记录最新快照清单。

        子源首次读取后快照 id 从 ``None`` 变为真实值 —— 这是本组合源自己触发
        的读取，不是外部快照变化，必须在 load 末尾刷新清单，否则下一次 load
        会误判为变化而每次清空缓存。

        R24-081/082: the recorded manifest MUST equal the epoch verified for the
        read.  If the source advanced between the verified read (``revalidate``)
        and this record (data=A, manifest=B), recording B would let a stale A
        cache survive the next ``B == B`` check.  Detect the drift, invalidate
        and refuse instead — the caller retries.
        """
        current = self._child_snapshot_manifest()
        if (
            self._snapshot_manifest is not None
            and self._snapshot_manifest_changed(self._snapshot_manifest, current)
        ):
            if self._effective_production:
                self._clear_caches()
                raise CompositeSnapshotVerificationError(
                    "composite source advanced between the verified read and the "
                    "manifest record (data=A, manifest=B) — refusing to record a "
                    "mismatched epoch; invalidate and retry (R24-082)"
                )
            # Research: a permanently-unverifiable source (e.g. failing refresh)
            # has no coherent epoch; record the observation and let the next
            # load's invalidation gate handle it.
        self._snapshot_manifest = current

    def _expand_alias(self, name: str) -> str:
        """_expand_alias。
        
        参数:
            name: 逻辑列名
        
        返回:
            str
        """
        current = name
        visited: set[str] = set()
        while current in self.aliases:
            if current in visited:
                raise ValueError(f"Circular composite alias detected at '{current}'")
            visited.add(current)
            current = self.aliases[current]
        return current

    @staticmethod
    def _canonical_name(source_name: str, column_name: str) -> str:
        """_canonical_name。
        
        参数:
            source_name: 见函数签名
            column_name: 数据列名
        
        返回:
            str
        """
        return f"{source_name}.{column_name}"

    def _resolve_reference(self, name: str) -> tuple[str, str, str]:
        """_resolve_reference。
        
        参数:
            name: 逻辑列名
        
        返回:
            tuple[str, str, str]
        """
        if not isinstance(name, str) or not name.strip():
            raise KeyError("Composite column name must be a non-empty string")

        expanded = self._expand_alias(name.strip())
        if "." in expanded:
            source_name, column_name = expanded.split(".", 1)
            source_name = source_name.strip()
            column_name = column_name.strip()
            if not source_name or not column_name:
                raise KeyError(f"Invalid composite column reference: {expanded}")
            if source_name not in self.sources:
                available = ", ".join(sorted(self.sources))
                raise KeyError(
                    f"Unknown composite source '{source_name}' for column '{name}'. "
                    f"Available sources: {available}"
                )
            return source_name, column_name, self._canonical_name(source_name, column_name)

        if self.allow_unqualified_anchor_columns:
            return (
                self.anchor_source,
                expanded,
                self._canonical_name(self.anchor_source, expanded),
            )

        raise KeyError(
            f"Composite column '{name}' must include a source prefix or alias mapping"
        )

    def _get_anchor_index(self):
        """_get_anchor_index。
        
        参数:
            无
        
        返回:
            无
        """
        import pandas as pd

        self._invalidate_if_snapshot_changed()
        if self._anchor_index_cache is not None:
            return self._anchor_index_cache

        source_name, column_name, canonical_name = self._resolve_reference(self.anchor_column)
        if source_name != self.anchor_source:
            raise ValueError(
                "Composite anchor_column must resolve to the configured anchor_source"
            )

        anchor_series = self.load_column(canonical_name)
        if not isinstance(anchor_series.index, pd.MultiIndex):
            raise ValueError("Composite anchor column must use a MultiIndex index")

        self._anchor_index_cache = anchor_series.index
        return self._anchor_index_cache

    @staticmethod
    def _parse_tolerance(value: str | None):
        """_parse_tolerance。

        Review-8 #475: 拒绝负 tolerance —— ``pd.to_timedelta`` 能解析负时长，
        负 tolerance 会反转 asof 语义（丢弃匹配）。

        参数:
            value: 缓存值

        返回:
            无
        """
        if value is None:
            return None

        import pandas as pd

        tolerance = pd.to_timedelta(value, errors="coerce")
        if pd.isna(tolerance):
            raise ValueError(f"Invalid composite join tolerance: {value}")
        if tolerance < pd.Timedelta(0):
            raise ValueError(
                f"Composite join tolerance must be >= 0, got {value!r}"
            )
        return tolerance

    @staticmethod
    def _align_exact(anchor_index, series):
        """_align_exact。
        
        参数:
            anchor_index: 见函数签名
            series: MultiIndex Series
        
        返回:
            无
        """
        return series.reindex(anchor_index)

    def _align_asof_backward(self, anchor_index, series, *, tolerance: str | None = None):
        """_align_asof_backward。

        Review-8 #476: 源中 ``(timestamp, instrument)`` 重复行没有 revision
        ordering 时必须 fail-closed（不能依赖原始行序静默选一行）。

        返回 ``(aligned_series, key_matched_rows)``：key 命中数由 sentinel 列
        判定，与源值是否为 null 解耦（#474）。

        参数:
            anchor_index: 见函数签名
            series: MultiIndex Series
            tolerance: 见函数签名（可选）

        返回:
            无
        """
        import pandas as pd

        if len(anchor_index) == 0:
            return pd.Series(dtype=series.dtype, index=anchor_index, name=series.name), 0
        if len(series) == 0:
            return pd.Series(dtype=series.dtype, index=anchor_index, name=series.name), 0

        anchor_frame = anchor_index.to_frame(index=False)
        anchor_frame.columns = ["timestamp", "instrument"]
        anchor_frame["_row_id"] = range(len(anchor_frame))

        source_frame = series.rename("value").reset_index()
        source_frame.columns = ["timestamp", "instrument", "value"]
        source_frame["__fe_src_present__"] = 1.0

        # Fail-closed on ambiguous duplicate source keys (#476).
        dups = source_frame.duplicated(subset=["timestamp", "instrument"])
        if dups.any():
            raise ValueError(
                f"Composite asof source has {int(dups.sum())} duplicate "
                f"(timestamp, instrument) rows without a revision ordering; "
                f"refusing to silently pick one."
            )

        merged = pd.merge_asof(
            anchor_frame.sort_values(["timestamp", "instrument"]),
            source_frame.sort_values(["timestamp", "instrument"]),
            on="timestamp",
            by="instrument",
            direction="backward",
            allow_exact_matches=True,
            tolerance=self._parse_tolerance(tolerance),
        ).sort_values("_row_id")

        key_matched_rows = int(merged["__fe_src_present__"].notna().sum())
        out = pd.Series(merged["value"].to_numpy(), index=anchor_index, name=series.name)
        out.index = out.index.set_names(["timestamp", "instrument"])
        return out, key_matched_rows

    def _record_join_report(
        self,
        *,
        source_name: str,
        column_name: str,
        canonical_name: str,
        join_spec: CompositeJoinSpec,
        aligned,
        key_matched_rows: int,
    ) -> None:
        """_record_join_report。

        Review-8 #474: ``key_matched`` 由 join 判定（asof sentinel / exact 全部
        命中），``value_valid`` 由对齐后值的非 null 判定 —— 两者不再混淆。

        参数:
            source_name: 见函数签名（可选）
            column_name: 数据列名（可选）
            canonical_name: 见函数签名（可选）
            join_spec: 见函数签名（可选）
            aligned: 见函数签名（可选）
            key_matched_rows: 见函数签名（可选）

        返回:
            无
        """
        import pandas as pd

        anchor_rows = len(aligned)
        key_matched_rows = int(key_matched_rows or 0)
        value_valid_rows = int(pd.notna(aligned).sum()) if anchor_rows else 0
        self._join_reports.append(
            CompositeJoinReport(
                source=source_name,
                column=column_name,
                canonical_name=canonical_name,
                method=join_spec.method,
                anchor_rows=anchor_rows,
                key_matched_rows=key_matched_rows,
                key_unmatched_rows=max(0, anchor_rows - key_matched_rows),
                value_valid_rows=value_valid_rows,
                value_null_rows=max(0, anchor_rows - value_valid_rows),
            )
        )

    @staticmethod
    def _is_pit_sensitive_source(source: Any, *, production: bool) -> bool:
        """源是否 PIT 语义敏感（不能交给 Composite 自行 merge_asof）。

        R24-084..087: the FIRST authority is the source's declared
        ``temporal_contract()``.  A source that does not implement it, or whose
        contract is ``unknown``, is PIT-sensitive (fail-closed) — NEVER "safe by
        omission" because a ``dataset`` attribute is missing.  The legacy
        dataset-name / COS-contract heuristics remain only as a back-compat
        fallback for sources that have not yet declared a contract.

        P1-09: COS 契约查询失败在 production 下 fail-closed（抛
        ``CompositeContractUnavailableError``），不能把「无法证明」当成
        「probably safe」；research 才允许告警后按非敏感回退。
        """
        contract_method = getattr(source, "temporal_contract", None)
        if callable(contract_method):
            try:
                tc = contract_method()
            except Exception as exc:
                if production:
                    raise CompositeContractUnavailableError(
                        f"source {type(source).__name__} temporal_contract() "
                        f"failed ({type(exc).__name__}: {exc}); production "
                        "fail-closed — cannot prove PIT-sensitivity"
                    ) from exc
                logger.warning(
                    "source %s temporal_contract() failed; research 放行: %s",
                    type(source).__name__,
                    exc,
                )
                tc = None
            if tc is not None and getattr(tc, "join_capability", "unknown") != "unknown":
                # R24-087: detection from the contract, not a dataset blacklist.
                return bool(getattr(tc, "pit_sensitive", True))
        # Legacy heuristic fallback (pre-R24 sources without a contract).
        if not hasattr(source, "dataset"):
            return True  # R24-084: no dataset ≠ not sensitive → fail-closed
        dataset = str(getattr(source, "dataset", "") or "")
        try:
            from .data_access_source import _NO_KNOWLEDGE_TIME_FUNDAMENTALS

            if dataset in _NO_KNOWLEDGE_TIME_FUNDAMENTALS:
                return True
        except ImportError:
            pass
        read_mode = str(getattr(source, "read_mode", "panel") or "panel").lower()
        if read_mode in {"event", "pit"}:
            return True
        try:
            from data_access.cos_contract import get_cos_contract

            contract = get_cos_contract(dataset)
        except Exception as exc:
            # P1-09: 契约查询失败 → production fail-closed。
            if production:
                raise CompositeContractUnavailableError(
                    f"dataset={dataset!r} COS contract lookup failed "
                    f"({type(exc).__name__}: {exc}); production fail-closed — "
                    "cannot prove PIT-sensitivity"
                ) from exc
            logger.warning(
                "dataset=%s COS contract lookup failed; research 放行（无法证明 "
                "PIT-sensitivity 按非敏感处理）: %s",
                dataset,
                exc,
            )
            contract = None
        if contract is not None and (
            str(getattr(contract, "pit_policy", "") or "") == "strict"
        ):
            return True
        return False

    def _enforce_join_authority(
        self,
        *,
        source_name: str,
        join_spec: CompositeJoinSpec,
    ) -> None:
        """#收官轮 P0（Integration）：PIT 敏感源禁止 Composite 自行 merge_asof。

        边界：普通 D1/S1 同频数据 Composite ``exact`` 保留；PIT / E1 / E2 /
        RAW_EVENT / revision / period-selection 必须下沉到 DataAccess
        read_joined / SourceRef certified PIT resolver，Composite 不能再把
        period_end 数据当 decision-time 做 backward asof（前视泄漏）。
        production（P1-08：取父级 effective production authority，不依赖子源
        ``.production``）fail-closed；research 告警放行。
        """
        source = self.sources[source_name]
        if join_spec.method in {"exact", "current_only"}:
            # current_only executes exact-align (no merge_asof), so it never
            # creates a look-ahead on a PIT-sensitive source.
            return
        if not self._is_pit_sensitive_source(
            source, production=self._effective_production
        ):
            return
        production = self._effective_production
        if production:
            raise ValueError(
                f"Composite 非 exact 对齐 source={source_name!r}（dataset="
                f"{getattr(source, 'dataset', '?')!r}）是 PIT 敏感源（event/pit/"
                "无知识时钟财务/E2 strict-pit），禁止 Composite 自行 merge_asof——"
                "必须下沉到 DataAccess read_joined / SourceRef certified PIT "
                "resolver。production fail-closed。"
            )
        warnings.warn(
            f"Composite 非 exact 对齐 source={source_name!r}（dataset="
            f"{getattr(source, 'dataset', '?')!r}）是 PIT 敏感源，asof 拼接有前视"
            "风险（period_end 无 knowledge-time）；production 将拒绝。请改用 "
            "DataAccess read_joined / SourceRef certified PIT resolver。",
            stacklevel=4,
        )

    def _align_to_anchor(
        self,
        series,
        join_spec: CompositeJoinSpec,
        *,
        source_name: str,
        column_name: str,
        canonical_name: str,
    ):
        """_align_to_anchor。

        参数:
            series: MultiIndex Series
            join_spec: 见函数签名
            source_name: 见函数签名（可选）
            column_name: 数据列名（可选）
            canonical_name: 见函数签名（可选）

        返回:
            无
        """
        self._enforce_join_authority(source_name=source_name, join_spec=join_spec)
        anchor_index = self._get_anchor_index()
        key_matched_rows: int
        if join_spec.method in {"exact", "current_only"}:
            # Round-11 §35: ``current_only`` joins a snapshot source on EXACTLY
            # matching rows only (no asof, no forward-fill) — the snapshot value
            # never leaks into the historical panel.
            # R10 #43: reindex 把每个 anchor key 都对齐（未命中者置 NaN），但
            # key 命中数必须按「anchor key 在源索引中真实存在」计数，而不是
            # anchor 全长 —— 源里根本没有的 key 不能算 key-matched。
            aligned = self._align_exact(anchor_index, series)
            key_matched_rows = len(anchor_index.intersection(series.index))
        elif join_spec.method in {"asof_backward", "forward_fill"}:
            aligned, key_matched_rows = self._align_asof_backward(
                anchor_index, series, tolerance=join_spec.tolerance
            )
        else:
            raise ValueError(f"Unsupported composite join method: {join_spec.method}")
        self._record_join_report(
            source_name=source_name,
            column_name=column_name,
            canonical_name=canonical_name,
            join_spec=join_spec,
            aligned=aligned,
            key_matched_rows=key_matched_rows,
        )
        return aligned

    _SNAPSHOT_BARRIER_ATTEMPTS = 2

    def _load_with_snapshot_barrier(self, load_once):
        """P1-07: 快照一致性 retry 外壳。token 漂移 → 整体重试一次，仍漂移则抛。

        参数:
            load_once: 一次完整的 load 调用（返回列数据）
        """
        for attempt in range(1, self._SNAPSHOT_BARRIER_ATTEMPTS + 1):
            try:
                return load_once()
            except CompositeSnapshotVerificationError:
                if attempt >= self._SNAPSHOT_BARRIER_ATTEMPTS:
                    raise
                logger.warning(
                    "组合源 child snapshot 在 load 期间变化，整体 retry "
                    "attempt=%d",
                    attempt + 1,
                )
        raise AssertionError("unreachable")  # pragma: no cover

    def load_column(self, name: str):
        """load_column。

        参数:
            name: 逻辑列名

        返回:
            无
        """
        return self._load_with_snapshot_barrier(lambda: self._load_column_once(name))

    def _load_column_once(self, name: str):
        """单列一次读取（P1-07 barrier 包裹的原子读）。"""
        self._invalidate_if_snapshot_changed()
        if name in self._column_cache:
            logger.debug("命中组合列缓存: %s", name)
            return self._column_cache[name]

        barrier = CompositeSnapshotBarrier(self)
        barrier.begin()
        source_name, column_name, canonical_name = self._resolve_reference(name)
        if canonical_name in self._column_cache:
            series = self._column_cache[canonical_name]
            self._column_cache[name] = series
            barrier.revalidate()
            return series

        if source_name == self.anchor_source:
            series = self.sources[source_name].load_column(column_name)
        else:
            join_spec = self.joins[source_name]
            logger.info(
                "对齐组合列 '%s': source=%s, method=%s",
                canonical_name,
                source_name,
                join_spec.method,
            )
            series = self._align_to_anchor(
                self.sources[source_name].load_column(column_name),
                join_spec,
                source_name=source_name,
                column_name=column_name,
                canonical_name=canonical_name,
            )

        barrier.revalidate()
        self._column_cache[canonical_name] = series
        self._column_cache[name] = series
        if source_name == self.anchor_source and self.allow_unqualified_anchor_columns:
            self._column_cache.setdefault(column_name, series)
        self._record_snapshot_manifest()
        return series

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """按源批量读取并对齐：同源多列一次 ``load_columns``（DataAccessSource 子源
        会合并成一次 ``store.read``），非锚点列再逐个按 join 方法对齐到锚点索引
        （锚点索引已缓存，merge_asof 不重复读锚点）。P1-07 barrier 包裹。

        参数:
            names: 逻辑列名列表

        返回:
            dict[str, Any]
        """
        return self._load_with_snapshot_barrier(
            lambda: self._load_columns_once(names)
        )

    def _load_columns_once(self, names: list[str]) -> dict[str, Any]:
        """批量读取一次（P1-07 barrier 包裹的原子读）。"""
        self._invalidate_if_snapshot_changed()
        out: dict[str, Any] = {}
        missing: list[str] = []
        for name in names:
            if name in self._column_cache:
                out[name] = self._column_cache[name]
            else:
                missing.append(name)
        if not missing:
            return out

        barrier = CompositeSnapshotBarrier(self)
        barrier.begin()

        refs_by_source: dict[str, list[tuple[str, str, str]]] = {}
        for name in missing:
            source_name, column_name, canonical_name = self._resolve_reference(name)
            if canonical_name in self._column_cache:
                series = self._column_cache[canonical_name]
                self._column_cache[name] = series
                out[name] = series
                continue
            refs_by_source.setdefault(source_name, []).append(
                (name, column_name, canonical_name)
            )

        for source_name, refs in refs_by_source.items():
            is_anchor = source_name == self.anchor_source
            join_spec = None if is_anchor else self.joins[source_name]
            columns = list(dict.fromkeys(r[1] for r in refs))
            if not is_anchor:
                logger.info(
                    "组合源批读 source=%s method=%s cols=%d",
                    source_name,
                    join_spec.method,
                    len(columns),
                )
            batch = self._load_batch(source_name, columns)
            for name, column_name, canonical_name in refs:
                series = batch.get(column_name)
                if series is None:
                    raise KeyError(
                        f"composite source '{source_name}' did not return column '{column_name}'"
                    )
                if not is_anchor:
                    series = self._align_to_anchor(
                        series,
                        join_spec,
                        source_name=source_name,
                        column_name=column_name,
                        canonical_name=canonical_name,
                    )
                self._column_cache[canonical_name] = series
                self._column_cache[name] = series
                if is_anchor and self.allow_unqualified_anchor_columns:
                    self._column_cache.setdefault(column_name, series)
                out[name] = series

        barrier.revalidate()
        self._record_snapshot_manifest()
        return out

    def _load_batch(self, source_name: str, columns: list[str]) -> dict[str, Any]:
        """从子源批量取列；子源没有 ``load_columns`` 时退回逐列 ``load_column``。"""
        source = self.sources[source_name]
        loader = getattr(source, "load_columns", None)
        if callable(loader):
            return loader(columns)
        return {c: source.load_column(c) for c in columns}

    def prefetch_columns(self, names: list[str]) -> None:
        """prefetch_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        self.load_columns(names)

    def prefetch_panels(self, names: list[str]) -> None:
        """prefetch_panels。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        self.prefetch_columns(names)
