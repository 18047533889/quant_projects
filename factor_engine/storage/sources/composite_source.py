from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from logging_utils import get_logger

from .datasource import DataSource


logger = get_logger("storage.composite_source")

# Round-11 §35 (plan A): ``current_only`` joins a snapshot source to the anchor
# on EXACTLY matching rows only — the snapshot is never asof-ffilled into the
# historical panel.  Semantically identical to ``exact`` at execution, but the
# distinct name carries the SnapshotOnlySourcePolicy intent (a snapshot source
# must only ever be consumed current-only).
_ALLOWED_JOIN_METHODS = frozenset({"exact", "current_only", "asof_backward", "forward_fill"})


@dataclass(frozen=True)
class CompositeJoinSpec:
    """组合数据源单源 join 对齐规格。
    
    参数:
        无
    """
    method: str = "asof_backward"
    tolerance: str | None = None


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
    ) -> None:
        """初始化实例。
        
        参数:
            anchor_source: 锚点数据源名称（可选）
            anchor_column: 锚点列引用（可选）
            sources: 子数据源映射（可选）
            joins: 非锚点源的 join 配置（可选）
            aliases: 列名别名映射（可选）
            allow_unqualified_anchor_columns: 见函数签名（可选）
        
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

        self.allow_unqualified_anchor_columns = bool(allow_unqualified_anchor_columns)
        self.aliases = self._normalize_aliases(aliases or {})
        self.joins = self._normalize_joins(joins or {})
        self._column_cache: dict[str, Any] = {}
        self._anchor_index_cache = None
        self._join_reports: list[CompositeJoinReport] = []
        #: R10 #44: combined child snapshot manifest.  ``None`` until the first
        #: load records a baseline; afterwards a change in any child snapshot id
        #: invalidates ``_column_cache`` / ``_anchor_index_cache``.
        self._snapshot_manifest: tuple[tuple[str, str | None], ...] | None = None

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
        return CompositeJoinSpec(method=normalized_method, tolerance=tolerance)

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

    def _child_snapshot_manifest(self) -> tuple[tuple[str, str | None], ...]:
        """组合子源快照清单（R10 #44 + #收官轮 P0 Integration）。

        **先调用每个子源的 ``refresh_snapshot()``**（proactive 刷新，TTL 内短路；
        DataAccessSource 的廉价路径走 manifest token），再读统一 ``snapshot_token``
        （``_manifest_token or data_snapshot_id``）—— 否则 Composite 永远只看
        ``data_snapshot_id`` 而看不到 manifest 级变化，底层 dataset A→B 后仍
        命中自己的旧缓存（cache-of-cache coherence bug）。

        refresh 无法确认时：production 子源 fail-closed（视为变化、清缓存，
        下次 load 强制重读，不再信旧 cache）；research 告警后按 unknown 处理。
        没有 ``refresh_snapshot`` 的子源退化为只读 ``data_snapshot_id``（None）。
        """
        manifest: list[tuple[str, str | None]] = []
        for name, source in self.sources.items():
            refresher = getattr(source, "refresh_snapshot", None)
            if callable(refresher):
                try:
                    refresher()
                except Exception as exc:  # refresh 失败 → 无法确认 → fail-closed
                    production = bool(getattr(source, "production", False))
                    if production:
                        logger.warning(
                            "组合源子源 %r refresh_snapshot 失败（%s: %s）；production "
                            "fail-closed：清缓存强制重读，不信任旧 snapshot。",
                            name,
                            type(exc).__name__,
                            exc,
                        )
                        manifest.append((name, None))
                        continue
                    logger.warning(
                        "组合源子源 %r refresh_snapshot 失败（research 按 unknown）: %s",
                        name,
                        exc,
                    )
            snapshot = getattr(source, "snapshot_token", None)
            if snapshot is None:
                snapshot = getattr(source, "data_snapshot_id", None)
            if callable(snapshot):
                try:
                    snapshot = snapshot()
                except Exception:  # best-effort manifest: a child that cannot
                    # report a snapshot is treated as unknown (None).
                    snapshot = None
            manifest.append((name, snapshot))
        return tuple(manifest)

    def _invalidate_if_snapshot_changed(self) -> None:
        """子源快照版本变化时清除列/锚点缓存（R10 #44 + #收官轮 P0）。

        在公共 load 方法顶部调用：**主动 refresh 每个子源** 后比较最新
        snapshot_token 清单；任一子源 token 变化（或 production 下 refresh
        无法确认）→ 旧列缓存与锚点索引缓存全部失效，下次读取强制重建——
        Composite 自己的缓存不能比底层数据更长寿。
        """
        current = self._child_snapshot_manifest()
        if self._snapshot_manifest is not None and current != self._snapshot_manifest:
            logger.info(
                "组合源子源快照变化，清除列/锚点缓存 old=%s new=%s",
                self._snapshot_manifest,
                current,
            )
            self._column_cache.clear()
            self._anchor_index_cache = None
        self._snapshot_manifest = current

    def _record_snapshot_manifest(self) -> None:
        """load 完成后记录最新快照清单。

        子源首次读取后快照 id 从 ``None`` 变为真实值 —— 这是本组合源自己触发
        的读取，不是外部快照变化，必须在 load 末尾刷新清单，否则下一次 load
        会误判为变化而每次清空缓存。
        """
        self._snapshot_manifest = self._child_snapshot_manifest()

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
    def _is_pit_sensitive_source(source: Any) -> bool:
        """源是否 PIT 语义敏感（不能交给 Composite 自行 merge_asof）。

        #收官轮 P0（Integration）：以下三类源的 temporal 对齐必须由 DataAccess
        read_joined / SourceRef certified PIT resolver 完成，Composite 只允许
        ``exact`` 对齐（源自己已经把 temporal 语义对齐好）：
          * ``read_mode in {event, pit}`` 的 DataAccessSource；
          * 已知无 knowledge-time 的 cleaned 财务源（``_NO_KNOWLEDGE_TIME_FUNDAMENTALS``，
            period_end 无 filing_date，asof 拼接有真实前视风险）；
          * COS 契约 ``pit_policy=="strict"`` 的 E2 源（availability/filing_date 时钟）。
        """
        if not hasattr(source, "dataset"):
            return False
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
            if contract is not None and (
                str(getattr(contract, "pit_policy", "") or "") == "strict"
            ):
                return True
        except Exception:
            pass
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
        production（子源 production=True）fail-closed；research 告警放行。
        """
        source = self.sources[source_name]
        if join_spec.method in {"exact", "current_only"}:
            # current_only executes exact-align (no merge_asof), so it never
            # creates a look-ahead on a PIT-sensitive source.
            return
        if not self._is_pit_sensitive_source(source):
            return
        production = bool(getattr(source, "production", False))
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

    def load_column(self, name: str):
        """load_column。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        self._invalidate_if_snapshot_changed()
        if name in self._column_cache:
            logger.debug("命中组合列缓存: %s", name)
            return self._column_cache[name]

        source_name, column_name, canonical_name = self._resolve_reference(name)
        if canonical_name in self._column_cache:
            series = self._column_cache[canonical_name]
            self._column_cache[name] = series
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

        self._column_cache[canonical_name] = series
        self._column_cache[name] = series
        if source_name == self.anchor_source and self.allow_unqualified_anchor_columns:
            self._column_cache.setdefault(column_name, series)
        self._record_snapshot_manifest()
        return series

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """按源批量读取并对齐：同源多列一次 ``load_columns``（DataAccessSource 子源
        会合并成一次 ``store.read``），非锚点列再逐个按 join 方法对齐到锚点索引
        （锚点索引已缓存，merge_asof 不重复读锚点）。

        参数:
            names: 逻辑列名列表

        返回:
            dict[str, Any]
        """
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
