from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from logging_utils import get_logger

from .datasource import DataSource


logger = get_logger("storage.composite_source")

_ALLOWED_JOIN_METHODS = frozenset({"exact", "asof_backward", "forward_fill"})


@dataclass(frozen=True)
class CompositeJoinSpec:
    method: str = "asof_backward"
    tolerance: str | None = None


class CompositeDataSource(DataSource):
    """将多个数据源对外暴露为统一列空间，并按锚点源索引对齐。"""

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

    @staticmethod
    def _normalize_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
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
        return normalized

    def _expand_alias(self, name: str) -> str:
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
        return f"{source_name}.{column_name}"

    def _resolve_reference(self, name: str) -> tuple[str, str, str]:
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
        if value is None:
            return None

        import pandas as pd

        tolerance = pd.to_timedelta(value, errors="coerce")
        if pd.isna(tolerance):
            raise ValueError(f"Invalid composite join tolerance: {value}")
        return tolerance

    @staticmethod
    def _align_exact(anchor_index, series):
        return series.reindex(anchor_index)

    def _align_asof_backward(self, anchor_index, series, *, tolerance: str | None = None):
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

    def _align_to_anchor(self, series, join_spec: CompositeJoinSpec):
        anchor_index = self._get_anchor_index()
        if join_spec.method == "exact":
            return self._align_exact(anchor_index, series)
        if join_spec.method in {"asof_backward", "forward_fill"}:
            return self._align_asof_backward(anchor_index, series, tolerance=join_spec.tolerance)
        raise ValueError(f"Unsupported composite join method: {join_spec.method}")

    def load_column(self, name: str):
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
            )

        self._column_cache[canonical_name] = series
        self._column_cache[name] = series
        if source_name == self.anchor_source and self.allow_unqualified_anchor_columns:
            self._column_cache.setdefault(column_name, series)
        return series