from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from logging_utils import get_logger

from .datasource import DataSource


logger = get_logger("storage.composite_source")

_ALLOWED_JOIN_METHODS = frozenset({"exact", "asof_backward", "forward_fill"})


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
    
    参数:
        无
    """
    source: str
    column: str
    canonical_name: str
    method: str
    anchor_rows: int
    matched_rows: int
    unmatched_rows: int

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
            "matched_rows": self.matched_rows,
            "unmatched_rows": self.unmatched_rows,
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
        
        参数:
            anchor_index: 见函数签名
            series: MultiIndex Series
            tolerance: 见函数签名（可选）
        
        返回:
            无
        """
        import pandas as pd

        if len(anchor_index) == 0:
            return pd.Series(dtype=series.dtype, index=anchor_index, name=series.name)
        if len(series) == 0:
            return pd.Series(dtype=series.dtype, index=anchor_index, name=series.name)

        anchor_frame = anchor_index.to_frame(index=False)
        anchor_frame.columns = ["timestamp", "instrument"]
        anchor_frame["_row_id"] = range(len(anchor_frame))

        source_frame = series.rename("value").reset_index()
        source_frame.columns = ["timestamp", "instrument", "value"]

        merged = pd.merge_asof(
            anchor_frame.sort_values(["timestamp", "instrument"]),
            source_frame.sort_values(["timestamp", "instrument"]),
            on="timestamp",
            by="instrument",
            direction="backward",
            allow_exact_matches=True,
            tolerance=self._parse_tolerance(tolerance),
        ).sort_values("_row_id")

        out = pd.Series(merged["value"].to_numpy(), index=anchor_index, name=series.name)
        out.index = out.index.set_names(["timestamp", "instrument"])
        return out

    def _record_join_report(
        self,
        *,
        source_name: str,
        column_name: str,
        canonical_name: str,
        join_spec: CompositeJoinSpec,
        aligned,
    ) -> None:
        """_record_join_report。
        
        参数:
            source_name: 见函数签名（可选）
            column_name: 数据列名（可选）
            canonical_name: 见函数签名（可选）
            join_spec: 见函数签名（可选）
            aligned: 见函数签名（可选）
        
        返回:
            无
        """
        import pandas as pd

        anchor_rows = len(aligned)
        matched_rows = int(pd.notna(aligned).sum()) if anchor_rows else 0
        self._join_reports.append(
            CompositeJoinReport(
                source=source_name,
                column=column_name,
                canonical_name=canonical_name,
                method=join_spec.method,
                anchor_rows=anchor_rows,
                matched_rows=matched_rows,
                unmatched_rows=max(0, anchor_rows - matched_rows),
            )
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
        anchor_index = self._get_anchor_index()
        if join_spec.method == "exact":
            aligned = self._align_exact(anchor_index, series)
        elif join_spec.method in {"asof_backward", "forward_fill"}:
            aligned = self._align_asof_backward(
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
        )
        return aligned

    def load_column(self, name: str):
        """load_column。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
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
