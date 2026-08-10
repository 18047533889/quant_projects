"""通过 ``data_access.get_store()`` 读取登记数据集并绑定快照缓存。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable

import pandas as pd

from logging_utils import get_logger
from workspace_paths import quant_projects_root

from .datasource import DataSource
from .field_plan import (
    NormalizedFieldPlan,
    _market_from_dataset,
    plan_from_catalog_field,
    plan_from_field_spec,
)

logger = get_logger("storage.data_access_source")


class _SourceThreadState(threading.local):
    """Per-thread overrides for shared ``DataAccessSource`` instance state.

    R20-107..113：lazy scan 的 ``_lazy_scan`` / ``read_auto`` 是共享 mutable state，
    但 ``run_many_parallel`` 的每个 worker 线程都会 ``enable_lazy_scan(True)`` 并在
    finally 里恢复。直接改实例属性会让 restore 交错（A 恢复成 B 的前值）。改用
    per-thread override：每个线程的 enable/restore 只影响自己线程的视图，主线程
    的 base 值永远不被 worker 污染。

    R20-114..118：``_factor_engine_lqtp_wrapper`` 也从「改共享 inner 的实例属性」
    改为 per-thread 注册表，并发 context 不再互相覆盖 pointer。
    """

    def __init__(self) -> None:
        self.lazy_overrides: "weakref.WeakKeyDictionary[Any, dict[str, bool]]" = (
            weakref.WeakKeyDictionary()
        )
        self.logical_wrappers: "weakref.WeakKeyDictionary[Any, Any]" = (
            weakref.WeakKeyDictionary()
        )


_source_tls = _SourceThreadState()


def register_logical_wrapper(inner: Any, wrapper: Any) -> None:
    """Per-thread 注册 inner → logical wrapper（替代 ``setattr(inner, ...)``）。

    供 :mod:`backend.context` 使用，避免并发 context 在共享 inner 上互相覆盖
    ``_factor_engine_lqtp_wrapper``。
    """
    if wrapper is None:
        _source_tls.logical_wrappers.pop(inner, None)
    else:
        _source_tls.logical_wrappers[inner] = wrapper


def _logical_wrapper_for(inner: Any) -> Any | None:
    return _source_tls.logical_wrappers.get(inner)


class DataAccessColumnPreflightError(ValueError):
    """A logical formula dependency is not a valid physical column request."""


class MissingDataDependencyError(DataAccessColumnPreflightError):
    """A formula requires another logical dataset or an explicitly derived field."""


class UnknownFieldSemanticError(DataAccessColumnPreflightError):
    """A requested field has no contract in the field registry for this dataset.

    Raised only when the source runs with ``strict_unknown_fields`` (production).
    Research mode may explicitly allow raw physical columns instead.
    """


class FieldNormalizationError(DataAccessColumnPreflightError):
    """A registered field could not be normalized to its canonical unit/scale."""


class CatalogUnavailable(DataAccessColumnPreflightError):
    """SemanticFieldCatalog could not be reached (API-level failure)."""


class CatalogNotConfigured(DataAccessColumnPreflightError):
    """SemanticFieldCatalog is not configured/available for this dataset."""


class CatalogResolutionError(DataAccessColumnPreflightError):
    """SemanticFieldCatalog resolution failed with an unexpected error."""


class UnknownField(DataAccessColumnPreflightError):
    """The field is genuinely not present in the FE field registry (P0-31).

    Raised by ``_field_spec`` in production (fail-closed) instead of returning
    ``None`` and silently falling back to a generic untyped column.
    """


class FieldRegistryUnavailable(DataAccessColumnPreflightError):
    """The FE field registry could not be loaded / queried (P0-31).

    An infrastructure failure (import error, registry lookup exception) is
    distinct from a genuinely missing field: it must not be treated as
    "no contract" in production.
    """


class FieldSpecInvalid(DataAccessColumnPreflightError):
    """A field spec is present but malformed (P0-31).

    Raised in production when a resolved spec cannot be used as a contract.
    """


class CatalogCorrupt(DataAccessColumnPreflightError):
    """SemanticFieldCatalog data is corrupt or unreadable."""


class FieldMiningGateError(DataAccessColumnPreflightError):
    """A requested field is not mining_allowed and cannot be used as a mined input.

    Round-7 WS-E #279 production hard gate.
    """


class HistoricalSnapshotBackfillError(DataAccessColumnPreflightError):
    """A ``current_snapshot_only`` field cannot backfill a historical window.

    Round-7 WS-E #280: historical auto-mining over a snapshot-only field leaks
    the current snapshot into the past unless the operator opts into
    ``snapshot_now_only``.
    """


class FourLayerPITError(DataAccessColumnPreflightError):
    """At least one of the four PIT eligibility layers failed.

    Round-7 WS-E #282: field ∧ table ∧ dataset ∧ operator must all allow PIT for
    a production PIT-eligible read.
    """


class HistoricalCoverageError(DataAccessColumnPreflightError):
    """A field's historical coverage over the requested window is below threshold.

    Round-7 WS-E #315: a partial-history field over a search window must be
    coverage-gated (reject or flag) when coverage drops below the threshold.
    """


@dataclass(frozen=True)
class HistoricalCoverageContract:
    """Coverage contract for a partial-history field (round-7 WS-E #315).

    Attributes:
        field: the logical field this contract describes.
        first_valid_date: ISO date at which the field begins to have data.
        coverage_ratio: overall non-null coverage in [0, 1] over the declared
            history.
        coverage_by_year: per-year coverage ratio, keyed by int year.
        coverage_by_stock: per-stock coverage ratio, keyed by instrument.
        coverage_by_date: per-date coverage ratio, keyed by ISO date.
        threshold: default minimum coverage ratio for ``covers_window``.
        min_stock_coverage_threshold: a stock counts as covered when its
            per-stock coverage is at least this value (R9-P0-024).
        min_stock_coverage_quantile: at least this fraction of stocks must be
            covered, otherwise the per-stock gate flags the contract
            (R9-P0-024, default 90%).
        coverage_date_threshold: minimum per-date coverage for dates inside the
            requested window (R9-P0-024, default equals ``threshold``).
    """

    field: str
    first_valid_date: str | None = None
    coverage_ratio: float = 1.0
    coverage_by_year: dict[int, float] = _dc_field(default_factory=dict)
    coverage_by_stock: dict[str, float] = _dc_field(default_factory=dict)
    coverage_by_date: dict[str, float] = _dc_field(default_factory=dict)
    threshold: float = 0.7
    min_stock_coverage_threshold: float = 0.5
    min_stock_coverage_quantile: float = 0.9
    coverage_date_threshold: float = 0.7
    # R24-071/072: requested-window coverage vs full-history coverage are SEPARATE
    # evidence.  When the caller requests a window, the overall gate uses
    # ``requested_window_coverage`` (if declared) instead of the full-history
    # ratio — a poor 2010s full-history must not veto a well-covered 2025-2026
    # window unless the policy explicitly requires a full-history minimum.
    requested_window_coverage: float | None = None
    require_full_history_minimum: bool = False
    # R17-006: coverage identity must be market-aware.  The same logical field
    # (e.g. ``market_cap``) has different providers/coverage in A-share vs US;
    # keying coverage contracts only by ``field`` let a later-registered market
    # overwrite the other's contract.
    market: str | None = None
    provider_id: str | None = None
    dataset: str | None = None
    universe_id: str | None = None
    # R24-068..070: the coverage contract identity is a joint key
    # (market, dataset, provider, field_id/concept_id, timeframe, universe_id,
    # source_version).  A revenue vs US revenue never share; US quarterly vs US
    # TTM never share.
    timeframe: str | None = None
    source_version: str | None = None

    def covers_window(
        self,
        start: str | None = None,
        end: str | None = None,
        *,
        threshold: float | None = None,
        universe_stocks: set[str] | None = None,
        require_full_history_minimum: bool | None = None,
    ) -> bool:
        """True when the field's coverage meets ``threshold`` over [start, end]."""
        return not self._evaluate(
            start=start, end=end, threshold=threshold, universe_stocks=universe_stocks,
            require_full_history_minimum=require_full_history_minimum,
        )

    def violations(
        self,
        start: str | None = None,
        end: str | None = None,
        *,
        threshold: float | None = None,
        universe_stocks: set[str] | None = None,
        require_full_history_minimum: bool | None = None,
    ) -> list[str]:
        """Return a human-readable list of coverage violations (empty = OK)."""
        return self._evaluate(
            start=start, end=end, threshold=threshold, universe_stocks=universe_stocks,
            require_full_history_minimum=require_full_history_minimum,
        )

    def _evaluate(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        threshold: float | None = None,
        universe_stocks: set[str] | None = None,
        require_full_history_minimum: bool | None = None,
    ) -> list[str]:
        """Single shared coverage evaluator (R9-P0-023/024).

        Both :meth:`covers_window` and :meth:`violations` delegate here so the
        requested window is applied identically: per-year and per-date coverage
        are evaluated only for dates/years inside [start, end] (all years/dates
        when no window is given), and per-stock coverage is gated by a quantile
        rule so a high aggregate cannot mask a tail of poorly covered stocks.

        R24-073/074: when ``universe_stocks`` is supplied, the per-stock gate
        evaluates ONLY the stocks in the CURRENT requested universe (e.g.
        CSI300 / S&P500) — a stock outside the campaign universe is never
        allowed to drag the verdict, and coverage evidence carries the universe
        identity via the contract key.
        """
        threshold = float(threshold if threshold is not None else self.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("coverage threshold must be in [0, 1]")
        problems: list[str] = []

        window_requested = bool(start is not None or end is not None)
        req_full = (
            self.require_full_history_minimum
            if require_full_history_minimum is None
            else bool(require_full_history_minimum)
        )
        # R24-071/072: with a requested window, gate on the requested-window
        # coverage (when declared) unless the policy explicitly requires a
        # full-history minimum.  A poor 2010s full-history must not veto a
        # well-covered 2025-2026 window.
        if window_requested and not req_full and self.requested_window_coverage is not None:
            ratio = float(self.requested_window_coverage)
        else:
            ratio = float(self.coverage_ratio)
        if ratio < threshold:
            problems.append(f"overall coverage {ratio:.3f} < threshold {threshold:.3f}")

        year_start: int | None = None
        year_end: int | None = None
        ts_start = None
        ts_end = None
        if start is not None:
            try:
                ts_start = pd.Timestamp(start).normalize()
                year_start = int(ts_start.year)
            except (ValueError, TypeError):
                problems.append(f"unparseable window start {start!r}")
        if end is not None:
            try:
                ts_end = pd.Timestamp(end).normalize()
                year_end = int(ts_end.year)
            except (ValueError, TypeError):
                problems.append(f"unparseable window end {end!r}")

        if start is not None and self.first_valid_date:
            try:
                if pd.Timestamp(start).normalize() < pd.Timestamp(self.first_valid_date).normalize():
                    problems.append(
                        f"window start {start} precedes first_valid_date {self.first_valid_date}"
                    )
            except (ValueError, TypeError):
                problems.append(f"unparseable window start {start!r}")

        # Per-year coverage restricted to the requested window (R9-P0-023):
        # a year outside [start, end] must not veto a window that is covered.
        for year, year_ratio in (self.coverage_by_year or {}).items():
            if year_start is not None and int(year) < year_start:
                continue
            if year_end is not None and int(year) > year_end:
                continue
            if float(year_ratio) < threshold:
                problems.append(f"year {year} coverage {float(year_ratio):.3f} < {threshold:.3f}")

        # Per-date coverage restricted to the requested window (R9-P0-024).
        date_threshold = float(self.coverage_date_threshold)
        if not 0.0 <= date_threshold <= 1.0:
            raise ValueError("coverage_date_threshold must be in [0, 1]")
        for date, date_ratio in (self.coverage_by_date or {}).items():
            if ts_start is not None:
                try:
                    if pd.Timestamp(date).normalize() < ts_start:
                        continue
                except (ValueError, TypeError):
                    continue
            if ts_end is not None:
                try:
                    if pd.Timestamp(date).normalize() > ts_end:
                        continue
                except (ValueError, TypeError):
                    continue
            if float(date_ratio) < date_threshold:
                problems.append(f"date {date} coverage {float(date_ratio):.3f} < {date_threshold:.3f}")

        # Per-stock quantile gate (R9-P0-024): at least ``min_stock_coverage_quantile``
        # of stocks must have per-stock coverage >= ``min_stock_coverage_threshold``.
        by_stock = self.coverage_by_stock or {}
        if universe_stocks is not None:
            # R24-073/074: coverage verdict is computed over the CURRENT
            # requested universe, never the contract's full stock list.
            by_stock = {
                k: v for k, v in by_stock.items() if k in universe_stocks
            }
        if by_stock:
            stock_threshold = float(self.min_stock_coverage_threshold)
            quantile = float(self.min_stock_coverage_quantile)
            if not 0.0 <= stock_threshold <= 1.0:
                raise ValueError("min_stock_coverage_threshold must be in [0, 1]")
            if not 0.0 <= quantile <= 1.0:
                raise ValueError("min_stock_coverage_quantile must be in [0, 1]")
            covered = sum(
                1 for _ratio in by_stock.values() if float(_ratio) >= stock_threshold
            )
            total = len(by_stock)
            frac = covered / total if total else 1.0
            if frac < quantile:
                problems.append(
                    f"only {covered}/{total} stocks in the requested universe "
                    f"({frac:.1%}) have per-stock coverage >= {stock_threshold:.3f}; "
                    f"required >= {quantile:.1%}"
                )
        return problems


def assert_historical_coverage(
    contract: HistoricalCoverageContract | None,
    *,
    start: str | None = None,
    end: str | None = None,
    threshold: float | None = None,
    required: bool = False,
    context: str = "",
    universe_stocks: set[str] | None = None,
    require_full_history_minimum: bool | None = None,
) -> None:
    """Raise :class:`HistoricalCoverageError` when ``contract`` does not cover the window.

    R17-007: for a PARTIAL/SPARSE/CURRENT_ONLY provider, coverage evidence is an
    execution requirement — a missing contract is ``COVERAGE_EVIDENCE_MISSING``
    (fail closed) when ``required=True`` (production), instead of silently
    passing ("no evidence == qualified").

    R24-073/074: ``universe_stocks`` restricts the per-stock gate to the CURRENT
    campaign universe.  R24-071/072: ``require_full_history_minimum`` opts into
    a full-history gate even when a window is requested.
    """
    if contract is None:
        if required:
            raise HistoricalCoverageError(
                f"{context}coverage evidence MISSING for requested window "
                f"[{start}, {end}] (partial/sparse/current-only provider requires "
                "a declared HistoricalCoverageContract; COVERAGE_EVIDENCE_MISSING)"
            )
        return
    problems = contract.violations(
        start=start, end=end, threshold=threshold,
        universe_stocks=universe_stocks,
        require_full_history_minimum=require_full_history_minimum,
    )
    if problems:
        raise HistoricalCoverageError(
            f"field {contract.field!r} historical coverage over "
            f"window=[{start}, {end}] fails: {'; '.join(problems)}"
        )


#: Coverage contracts keyed by a JOINT identity — R24-068..070:
#: (market, dataset, provider, field/concept, timeframe, universe_id,
#: source_version).  A revenue vs US revenue never share; US quarterly vs US TTM
#: never share.  Legacy ``(field, market)`` / ``(field, "any")`` keys are kept as
#: a backward-compatible fallback.
_COVERAGE_CONTRACTS: dict[tuple, HistoricalCoverageContract] = {}


def _coverage_key(
    field: str,
    market: str | None,
    *,
    dataset: str | None = None,
    provider_id: str | None = None,
    timeframe: str | None = None,
    universe_id: str | None = None,
    source_version: str | None = None,
) -> tuple:
    # R24-068: joint key.  ``None`` components become "any".
    return (
        str(field).strip(),
        str(market or "any").strip().lower(),
        str(dataset or "any").strip(),
        str(provider_id or "any").strip(),
        str(timeframe or "any").strip(),
        str(universe_id or "any").strip(),
        str(source_version or "any").strip(),
    )


def register_coverage_contract(
    contract: HistoricalCoverageContract,
    *,
    market: str | None = None,
    provider_id: str | None = None,
    dataset: str | None = None,
    universe_id: str | None = None,
    timeframe: str | None = None,
    source_version: str | None = None,
) -> HistoricalCoverageContract:
    """Register a :class:`HistoricalCoverageContract` for coverage gating (#315).

    R17-006 + R24-068: contracts are keyed by a JOINT identity
    ``(market, dataset, provider, field, timeframe, universe_id, source_version)``
    so A/US / quarterly/TTM coverage never collide.  ``market`` defaults from
    the contract itself; the optional context is recorded in the dict key.
    """
    if contract is None or not getattr(contract, "field", None):
        raise ValueError("coverage contract requires a field name")
    eff_market = market or getattr(contract, "market", None)
    _COVERAGE_CONTRACTS[
        _coverage_key(
            contract.field, eff_market,
            dataset=dataset or getattr(contract, "dataset", None),
            provider_id=provider_id or getattr(contract, "provider_id", None),
            timeframe=timeframe or getattr(contract, "timeframe", None),
            universe_id=universe_id or getattr(contract, "universe_id", None),
            source_version=source_version or getattr(contract, "source_version", None),
        )
    ] = contract
    return contract


def get_coverage_contract(
    field: str,
    *,
    market: str | None = None,
    dataset: str | None = None,
    provider_id: str | None = None,
    timeframe: str | None = None,
    universe_id: str | None = None,
    source_version: str | None = None,
) -> HistoricalCoverageContract | None:
    """Look up a coverage contract for a field (R24-068 joint key).

    Exact joint-key match first; falls back to progressively coarser keys so
    pre-R24 registrations keep working, but the most specific registration wins.
    """
    base = (str(field).strip(), str(market or "any").strip().lower())
    specifics = (dataset, provider_id, timeframe, universe_id, source_version)
    for i in range(len(specifics) + 1):
        narrowed = list(specifics)
        for j in range(i, len(specifics)):
            narrowed[j] = None
        key = (
            base[0], base[1],
            *[str(v or "any") for v in narrowed],
        )
        if key in _COVERAGE_CONTRACTS:
            return _COVERAGE_CONTRACTS[key]
    # Legacy fallback keys.
    if (base[0], base[1]) in _COVERAGE_CONTRACTS:
        return _COVERAGE_CONTRACTS[(base[0], base[1])]
    return _COVERAGE_CONTRACTS.get((str(field).strip(), "any"))


def _ensure_data_access_importable() -> None:
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_store():
    _ensure_data_access_importable()
    from data_access import get_store
    return get_store()


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return max(1, value)


def _default_data_cache_budget() -> int:
    """DataAccessSource 数据缓存默认预算：显式 env > 进程预算 × 15%。

    Phase 5 R7：8GB 默认写死对 8GB 服务器是灾难、对 512GB 服务器又太保守。
    改为跟随 ``ExecutionResourcePlan.process_budget_bytes`` 自动缩放。
    """
    raw = os.environ.get("FACTOR_ENGINE_DATA_CACHE_MAX_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError as exc:
            raise ValueError(f"FACTOR_ENGINE_DATA_CACHE_MAX_BYTES must be an integer") from exc
    try:
        from runtime.resource_governor import ExecutionResourcePlan

        plan = ExecutionResourcePlan.auto()
        return max(128 * 1024 * 1024, int(plan.process_budget_bytes * 0.15))
    except Exception:
        return 4 * 1024 * 1024 * 1024


def _is_clean_catalog_miss(exc: Exception) -> bool:
    """True when the catalog reported a genuine "field not in catalog" miss.

    ``store.resolve_fields`` raises ``ValidationError`` when a logical name is
    not in the catalog and not in any dataset schema.  That is a clean miss (the
    requested field is simply unknown) and should fall through to the FE
    FIELD_REGISTRY.  Everything else from the catalog API is an availability /
    corruption / resolution error (P0-12).
    """
    from data_access.core.exceptions import ValidationError

    if not isinstance(exc, ValidationError):
        return False
    msg = str(exc)
    return "SemanticFieldCatalog" in msg and ("未在" in msg or "不在" in msg)


def _catalog_config_error_kind(exc: Exception) -> type[DataAccessColumnPreflightError]:
    """Classify a failure to *load* the SemanticFieldCatalog (config layer)."""
    from data_access.core.exceptions import DataAccessError, ValidationError

    if isinstance(exc, ValidationError):
        msg = str(exc)
        if "不存在" in msg or "PyYAML" in msg or "依赖" in msg:
            return CatalogNotConfigured
        return CatalogCorrupt
    if isinstance(exc, DataAccessError):
        return CatalogUnavailable
    return CatalogCorrupt


def _catalog_resolution_error_kind(exc: Exception) -> type[DataAccessColumnPreflightError]:
    """Classify a failure of ``store.resolve_fields`` itself (resolution layer)."""
    from data_access.core.exceptions import DataAccessError

    if isinstance(exc, DataAccessError):
        return CatalogUnavailable
    return CatalogResolutionError


#: #收官轮 P0：无知识时钟的「cleaned」财务数据集。``fundamentals_*`` 的
#: time_column=period_end、``financials_ratios`` 的 date 语义未证明是 knowledge/
#: availability 日期——asof 拼接有真实前视风险（Q1 period_end=03-31 实际
#: filing=05-05，04-01 的模型就「看到」了 Q1 财报）。production PIT 禁止；正确
#: 源是 us_stock_balance/income/cashflow（E2，filing_date 为 availability_column，
#: period_end 为 period_column）。
_NO_KNOWLEDGE_TIME_FUNDAMENTALS = frozenset({
    "fundamentals_balance_sheet",
    "fundamentals_income_statement",
    "fundamentals_cash_flow_statement",
    "financials_ratios",
})


def _strict_instrument_filter(value: Any) -> list[str] | None:
    """#收官轮 P0：FE 入口复用 DataAccess 的 ``strict_sequence`` 契约。

    ``None`` → 全市场；``list/tuple/set[str]``（成员语义）→ 规范化 list；裸
    ``str/bytes``（会被逐字符拆成 ``["A","A","P","L"]`` 的 silent semantic
    inversion）、``dict``（退化成键列表）、``generator``、非 str 元素 → 构造期
    即拒绝。这样 adapter 层不再把非法输入「洗成合法但错误」的 list 交给 DataAccess。
    """
    if value is None:
        return None
    try:
        from data_access.read.predicate import strict_sequence
    except ImportError:  # pragma: no cover - DA 不可导入时退化为手动校验
        strict_sequence = None
    if strict_sequence is not None:
        normalized = strict_sequence(
            value,
            name="instrument_filter",
            element_type=str,
            allow_none=True,
        )
        return None if normalized is None else list(normalized)
    # 手动 fallback（DA 未安装）：拒绝 str/bytes/dict/generator + 非 str 元素。
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"instrument_filter 必须是序列（list/tuple/set），收到 str/bytes {value!r}。"
            "字符串会被误当成字符序列逐项过滤，是典型的 silent semantic inversion。"
        )
    if isinstance(value, dict):
        raise ValueError(f"instrument_filter 不接受 mapping/dict，收到 {value!r}")
    import collections.abc

    if isinstance(value, collections.abc.Iterator):
        raise ValueError(f"instrument_filter 不接受 generator/iterator，收到 {value!r}")
    out: list[str] = []
    for element in value:
        if not isinstance(element, str):
            raise ValueError(
                f"instrument_filter 元素必须是 str，收到 {type(element).__name__} {element!r}"
            )
        out.append(element)
    return out


class DataAccessSource(DataSource):
    """FactorEngine DataAccess source with snapshot-bound caches and preflight."""

    def __init__(
        self,
        *,
        dataset: str,
        fields: dict[str, str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        instrument_filter: list[str] | None = None,
        normalize_timestamp: bool | None = None,
        timestamp_unit: str | None = None,
        read_auto: bool | None = None,
        params: dict[str, Any] | None = None,
        semantic_filters: dict[str, Any] | None = None,
        read_mode: str = "panel",
        strict_unknown_fields: bool | None = None,
        run_mode: str | None = None,
        production: bool | None = None,
        enforce_mining_gate: bool = False,
        snapshot_now_only: bool = False,
        mining_coverage_threshold: float | None = None,
        pit_enforce: bool = False,
        snapshot_only: bool = False,
        snapshot_valid_at: str | None = None,
        snapshot_created_at: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.fields = dict(fields or {})
        # R9-P0-022: strictness must come from an explicit run_mode / production
        # policy, never self-inferred from the global environment.  ``None``
        # defaults to research (fail-open); callers that require production
        # admission must pass run_mode="production" / production=True explicitly.
        # ``strict_unknown_fields`` remains the most direct gate and wins when
        # explicitly provided (backward compatible).  The resolved policy is
        # stored on ``self.run_mode`` / ``self.production`` so wrapper sources
        # that build child ``DataAccessSource`` instances can propagate it.
        self.run_mode = str(run_mode).lower() if run_mode else None
        if production is None and self.run_mode is not None:
            from runtime.production_policy import is_production_mode

            production = bool(is_production_mode(self.run_mode))
        self.production = bool(production)
        # R10 #3: production is a FLOOR.  ``strict_unknown_fields`` remains the
        # most direct gate and an explicit True may make a research run stricter,
        # but an explicit ``strict_unknown_fields=False`` must NEVER downgrade a
        # production run back to fail-open.  (The old ``production =
        # bool(strict_unknown_fields)`` branch let the caller switch production
        # semantics off by passing False explicitly.)
        if self.production:
            self.strict_unknown_fields = True
        elif strict_unknown_fields is not None:
            self.strict_unknown_fields = bool(strict_unknown_fields)
        else:
            self.strict_unknown_fields = False
        # Round-7 WS-E #279/#280: opt-in production hard gates for mining/backfill
        # field contracts.  Default OFF so legitimate reads of structural columns
        # (e.g. a one_to_many weight that a factor aggregates before mining) keep
        # working; the AlphaMiner / production preflight enables them explicitly.
        self.enforce_mining_gate = bool(enforce_mining_gate)
        #: Round-7 WS-E #280 opt-in: only the current snapshot may be used, so a
        #: ``current_snapshot_only`` field is allowed even on a "now" window.
        self.snapshot_now_only = bool(snapshot_now_only)
        # R24-075..077: caller ``snapshot_now_only`` is INTENT, not proof.  The
        # source must carry its snapshot validity metadata so the runtime can
        # verify a requested decision range against the real snapshot validity.
        self.snapshot_valid_at: str | None = snapshot_valid_at
        self.snapshot_created_at: str | None = snapshot_created_at
        #: Round-11 §35 (plan A) SnapshotOnlySourcePolicy: the WHOLE source is a
        #: current-only snapshot (X0 sparse valuation/indicator).  Production
        #: historical mining over it hard-fails (see
        #: ``_enforce_field_contract_gates``); only current/research snapshot
        #: reads are permitted.
        self.snapshot_only = bool(snapshot_only)
        #: Round-7 WS-E #315: when set, partial-history fields whose coverage over
        #: the requested window is below this threshold are rejected.
        if mining_coverage_threshold is not None:
            threshold = float(mining_coverage_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("mining_coverage_threshold must be in [0, 1]")
            self.mining_coverage_threshold = threshold
        else:
            self.mining_coverage_threshold = None
        self.start_date = start_date
        self.end_date = end_date
        # #收官轮 P0：``[]`` 是**空股票池**（0 行），绝不能折叠成 ``None``——
        # DataAccess 本体已经修过 ``None != []``（[]→WHERE FALSE），这里不能再
        # 用 ``if instrument_filter else None`` 把它变回全市场。同时用 DA 的
        # ``strict_sequence`` 契约拒绝裸 str/bytes/dict/generator——``"AAPL"``
        # 不能再被 ``list(instrument_filter)`` 洗成 ``["A","A","P","L"]``。
        self.instrument_filter = _strict_instrument_filter(instrument_filter)
        #: #收官轮 P0：四层 PIT 门禁开关（engine ``config.pit.enforce`` / production
        #: 由 DataSourceBuildContext 注入；SourceRef child 必须继承父级策略）。
        self.pit_enforce = bool(pit_enforce)
        self.normalize_timestamp = normalize_timestamp
        self.timestamp_unit = timestamp_unit
        self.params = dict(params or {})
        self.semantic_filters = dict(semantic_filters or {})
        self.read_mode = str(read_mode or "panel").lower()
        self._validate_semantic_contract()
        # R20-107..113：构造期只写 base；per-thread lazy/read_auto override 走
        # 属性（``_lazy_scan`` / ``read_auto``），worker 线程的 enable/restore
        # 不污染共享实例的 base。
        self._read_auto_base = (
            bool(read_auto)
            if read_auto is not None
            else bool(self.params.pop("read_auto", False))
        )
        self._lazy_scan_base = bool(self.params.pop("lazy_scan", False))
        #: Unified field-resolution plans keyed by ``(logical_name,
        #: semantic_catalog_version)`` (P0-11, round-7 P0).  Produced by
        #: ``_resolve_columns`` / ``_ensure_field_plans`` and consumed by scale
        #: normalization so the catalog/registry unit contract is a single object.
        #: Keying by catalog version invalidates stale plans when the catalog's
        #: scale/mapping semantics change.
        self._field_plans: dict[tuple[str, str], NormalizedFieldPlan] = {}
        self._column_cache: OrderedDict[str, Any] = OrderedDict()
        self._panel_cache: OrderedDict[str, Any] = OrderedDict()
        self._lazy_bundle: Any | None = None
        self._data_snapshot_id: str | None = None
        #: 廉价的 manifest 版本 token（读 _manifest.json sidecar），用于 TTL 内
        #: 判断数据是否变化；与真实 DataSnapshot id 分开跟踪（格式不同，不能互比）。
        self._manifest_token: str | None = None
        self._snapshot_checked_at = 0.0
        self._snapshot_ttl_seconds = float(
            os.environ.get("FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS", "60") or 60
        )
        self._max_cache_columns = _positive_int_env(
            "FACTOR_ENGINE_DATA_CACHE_MAX_COLUMNS", 64
        )
        # #42 字节感知缓存上限。Phase 5 R7：默认不再固定 8GB，而是跟随进程预算
        # （process_budget × 15%），8GB 服务器上自然变小、512GB 服务器上自动变大；
        # 显式 env 仍可覆盖。
        self._max_cache_bytes = _default_data_cache_budget()
        self._cache_bytes = 0
        self._closed = False
        # R20-111：restore 失败 / 检测到运行态被破坏时置位。production 下任何
        # 后续读操作 ``_assert_healthy`` 直接 abort；research 至少记 warning。
        self._corrupted_state: str | None = None

    def _validate_semantic_contract(self) -> None:
        """Apply COS panel/event and required-filter policy at construction.

        ``semantic_filters``（美股财务 timeframe=quarterly、指数 IndexSymbol 等）
        是**行级过滤**——这里只做 contract 校验，真正的过滤由 read/scan 以
        ``filters=self.semantic_filters`` 传给 DataAccess，落到物理 WHERE。绝不
        塞进 ``params``（那是 ParametricDataset 路径参数）：StaticDataset 会把
        多余的 params 当调用方错误拒绝，导致 filter「过了门禁却从不真正过滤」。
        """
        try:
            _ensure_data_access_importable()
            from data_access.cos_contract import (
                get_cos_contract,
                resolve_event_clock,
                validate_event_filters,
                validate_panel_request,
            )
        except ImportError:
            return
        contract = get_cos_contract(self.dataset)
        # semantic_filters 是行级过滤，不塞进 params（路径参数）。冲突检测无条件
        # 运行——params 里同名但不同值 = 调用方自相矛盾（不能因为无契约就跳过）。
        for name, value in self.semantic_filters.items():
            existing = self.params.get(name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"conflicting semantic filter {name}: params={existing!r} filter={value!r}"
                )
        if self.read_mode in {"event", "pit"}:
            if contract is None:
                # 收官轮 P0：PIT/事件读取必须有 COS 契约证明公告/可知时钟——
                # 「无法证明」≠「安全」。production fail-closed；research 告警放行。
                if self.strict_unknown_fields:
                    raise FourLayerPITError(
                        f"dataset={self.dataset!r} 没有 COS 契约，无法证明 PIT/事件"
                        "知识时钟（filing_date/knowledge_time/availability_column）；"
                        "production 拒绝 PIT 读取。"
                    )
                logger.warning(
                    "dataset=%s 无 COS 契约，read_mode=%s 的知识时钟无法证明"
                    "（research 放行）",
                    self.dataset,
                    self.read_mode,
                )
                return
            resolve_event_clock(
                self.dataset,
                allow_effective_time=self.read_mode == "event",
            )
            validate_event_filters(contract, self.semantic_filters)
        elif self.read_mode == "panel":
            if contract is None:
                return
            # Round-11 §35 (plan A): a snapshot_only / snapshot_now_only source is
            # a current-only sparse snapshot (X0 valuation/indicator) — it is NOT
            # a complete daily panel, so the sparse read is allowed.  Historical
            # backfill remains blocked by the SnapshotOnlySourcePolicy gate in
            # ``_enforce_field_contract_gates``.
            validate_panel_request(
                self.dataset,
                semantic_filters=self.semantic_filters,
                allow_sparse=bool(self.snapshot_only or self.snapshot_now_only),
            )
        else:
            raise ValueError("read_mode must be panel, event, or pit")

    @property
    def data_snapshot_id(self) -> str | None:
        return self._data_snapshot_id

    @property
    def snapshot_token(self) -> str | None:
        """组合快照 token：manifest token（廉价变化检测）优先，其次 snapshot id。

        #收官轮 P0（Integration）：``refresh_snapshot()`` 的廉价路径只更新
        ``_manifest_token``（不每次 describe），而 Composite 若只比较
        ``data_snapshot_id`` 会**永远看不到 manifest 级变化**。暴露统一 token
        让 Composite 能真正检测「refresh 后 token 变了」。
        """
        return self._manifest_token or self._data_snapshot_id

    @property
    def lazy_scan(self) -> bool:
        return bool(self._lazy_scan)

    @property
    def _lazy_scan(self) -> bool:
        """Effective lazy-scan flag（R20-107..113：per-thread override → base）。"""
        overrides = _source_tls.lazy_overrides.get(self)
        if overrides is not None and "_lazy_scan" in overrides:
            return overrides["_lazy_scan"]
        return self._lazy_scan_base

    @_lazy_scan.setter
    def _lazy_scan(self, value: bool) -> None:
        # R20-111：restore 路径不经 ``_assert_open`` 直接回写；若 source 已关闭
        # 说明运行态被破坏 —— 记录 corrupted-state（research warning / production
        # abort），不再静默吞掉。
        if getattr(self, "_closed", False):
            self._mark_corrupted("lazy_scan restore on closed source")
        _source_tls.lazy_overrides.setdefault(self, {})["_lazy_scan"] = bool(value)

    @property
    def read_auto(self) -> bool:
        """Effective ``read_auto`` flag（R20-107..113：per-thread override → base）。"""
        overrides = _source_tls.lazy_overrides.get(self)
        if overrides is not None and "read_auto" in overrides:
            return overrides["read_auto"]
        return self._read_auto_base

    @read_auto.setter
    def read_auto(self, value: bool) -> None:
        if getattr(self, "_closed", False):
            self._mark_corrupted("read_auto restore on closed source")
        _source_tls.lazy_overrides.setdefault(self, {})["read_auto"] = bool(value)

    @property
    def _factor_engine_lqtp_wrapper(self) -> Any | None:
        """Per-thread logical wrapper（R20-114..118：不做共享 setattr）。

        ``lineage_service._logical_wrapper`` 用 ``getattr(data_source,
        "_factor_engine_lqtp_wrapper")`` 找回逻辑 wrapper —— 通过本属性读
        per-thread 注册表，并发 context 不再互相覆盖。
        """
        return _logical_wrapper_for(self)

    @_factor_engine_lqtp_wrapper.setter
    def _factor_engine_lqtp_wrapper(self, wrapper: Any) -> None:
        register_logical_wrapper(self, wrapper)

    def _mark_corrupted(self, reason: str) -> None:
        """R20-111：记录运行态被破坏（restore 失败等）。research 记 warning，
        production 下后续 ``_assert_healthy`` 直接 abort。"""
        self._corrupted_state = str(reason)
        logger.error("DataAccessSource corrupted-state: %s", reason)

    def _assert_healthy(self) -> None:
        """R20-111：production 下 corrupted-state 必须 abort，不能继续用坏 source。"""
        if self._corrupted_state is not None:
            from runtime.production_policy import is_production_mode

            if is_production_mode(self.run_mode):
                raise RuntimeError(
                    f"DataAccessSource is in corrupted state: {self._corrupted_state}"
                )
            logger.warning(
                "DataAccessSource corrupted-state (research continues): %s",
                self._corrupted_state,
            )

    def execution_spec(self) -> dict[str, Any]:
        """返回可重建（``storage.factory.build_data_source``）的 canonical 配置。

        #收官轮 P0：程序化 ``materialize()`` 未显式传 ``data_source_config`` 时，
        orchestrator 用本 spec 还原正在执行的真实 source contract（含
        ``instrument_filter`` 的 None/[] 区分），供 lineage / semantic identity /
        full definition / 事件增量 rebuild 复用。
        """
        from .datasource import clean_execution_spec

        return clean_execution_spec(
            {
                "type": "data_access",
                "dataset": str(self.dataset),
                "fields": dict(self.fields or {}),
                "start_date": self.start_date,
                "end_date": self.end_date,
                "instrument_filter": (
                    list(self.instrument_filter)
                    if self.instrument_filter is not None
                    else None
                ),
                "normalize_timestamp": self.normalize_timestamp,
                "timestamp_unit": self.timestamp_unit,
                "read_auto": self.read_auto,
                "params": dict(self.params or {}),
                "semantic_filters": dict(self.semantic_filters or {}),
                "read_mode": self.read_mode,
                "run_mode": self.run_mode,
                "production": self.production,
                "strict_unknown_fields": self.strict_unknown_fields,
                "enforce_mining_gate": self.enforce_mining_gate,
                "snapshot_now_only": self.snapshot_now_only,
                "mining_coverage_threshold": self.mining_coverage_threshold,
                "pit_enforce": self.pit_enforce,
                "snapshot_only": self.snapshot_only,
            }
        )

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("DataAccessSource is closed")
        # R20-111：production 下 corrupted-state 必须 abort；research 记 warning。
        self._assert_healthy()

    def _time_range(self) -> tuple[Any, Any] | None:
        if self.start_date is None and self.end_date is None:
            return None
        return (self.start_date, self.end_date)

    def _preflight_logical_columns(self, names: Iterable[str]) -> None:
        """Reject known semantic mistakes before they become DuckDB Binder errors.

        The dataset registry schema is intentionally *not* used as a hard gate
        here because the LQTP StockDailyBar mirror can contain real columns that
        lag the checked-in schema documentation.  We instead fail on two cases
        that are unambiguously wrong:

        * an active FactorEngine operator name was emitted as a bare column;
        * a known LQTP derived/multi-source dependency was emitted as a daily-bar
          physical column.
        """
        requested = [str(name) for name in names]
        from cleaned_operators.production_tiers import LQTP_SOURCE_DEPENDENT_NAMES

        derived = sorted(
            name for name in requested
            if name in LQTP_SOURCE_DEPENDENT_NAMES and name not in self.fields
        )
        if derived:
            raise MissingDataDependencyError(
                f"dataset={self.dataset!r} cannot satisfy derived/source-backed fields "
                f"{derived}; configure a composite/financial/valuation source or a "
                "versioned derived-field definition instead of querying StockDailyBar"
            )

        try:
            from backend.cleaned_bridge import ensure_cleaned_loaded
            from cleaned_operators.registry import OperatorRegistry

            ensure_cleaned_loaded()
            operator_names = set(OperatorRegistry.list_canonical()) | set(OperatorRegistry._aliases)
        except Exception:
            operator_names = set()
        mistaken_ops = sorted(
            name for name in requested
            if name in operator_names and name not in self.fields
        )
        if mistaken_ops:
            raise DataAccessColumnPreflightError(
                f"dataset={self.dataset!r} received operator name(s) as physical columns: "
                f"{mistaken_ops}. Expand the formula/template before data access."
            )

    def _resolve_catalog_fields(self, names: list[str]) -> dict[str, Any] | None:
        """Resolve logical fields via ``store.resolve_fields`` (SemanticFieldCatalog).

        Returns ``dict[logical_name, SemanticField]`` for the names the catalog
        actually resolved, or ``None`` when the catalog itself is unavailable and
        the caller should fall back to the FE FIELD_REGISTRY for ALL names.

        Round-7 P1 partial resolution: a clean per-name catalog miss skips only
        that name (the caller falls back to the FE registry per-name below), it
        does NOT fail the whole batch.  Results are keyed by logical name — never
        a positional ``zip``, which would silently truncate/reorder if the catalog
        result ever diverged from the request order (round-7 P0).

        P0-12 error hardening: catalog *failures* are classified —
          * not configured / corrupt / unavailable are FATAL in production
            (``strict_unknown_fields=True``) and warn + fall back in research;
          * a genuine "field not found in catalog" (clean miss) falls through to
            FIELD_REGISTRY so the registry stays the compatibility source of truth.
        """
        try:
            from data_access.read.semantic_catalog import get_semantic_catalog

            get_semantic_catalog()
        except Exception as exc:
            self._raise_or_fallback(
                _catalog_config_error_kind(exc), exc, "semantic catalog not available"
            )
            return None
        catalog_by_name: dict[str, Any] = {}
        for name in names:
            try:
                result = _get_store().resolve_fields([name], dataset=self.dataset)
            except Exception as exc:
                if _is_clean_catalog_miss(exc):
                    logger.debug(
                        "semantic catalog clean miss dataset=%s field=%s: %s",
                        self.dataset, name, exc,
                    )
                    continue
                self._raise_or_fallback(
                    _catalog_resolution_error_kind(exc), exc, "semantic catalog resolution failed"
                )
                continue
            if result is None:
                continue
            try:
                resolved = list(result)
            except TypeError:
                self._raise_or_fallback(
                    CatalogResolutionError,
                    TypeError(
                        f"catalog resolve_fields returned a non-iterable for dataset={self.dataset!r}"
                    ),
                    "semantic catalog resolution returned a non-iterable",
                )
                continue
            if len(resolved) != 1:
                self._raise_or_fallback(
                    CatalogResolutionError,
                    ValueError(
                        f"catalog resolve_fields returned {len(resolved)} rows for one "
                        f"request field={name!r} dataset={self.dataset!r}"
                    ),
                    "semantic catalog resolution length mismatch",
                )
                continue
            catalog_by_name[name] = resolved[0]
        return catalog_by_name

    @staticmethod
    def _semantic_catalog_version() -> str:
        """Deterministic version token over the loaded SemanticFieldCatalog.

        The catalog is the single source of truth for field scale/mapping
        semantics; keying the field-plan cache by this token invalidates stale
        plans whenever the catalog's field declarations change (round-7 P0).
        Returns ``"unavailable"`` when the catalog cannot be loaded — a stable
        token so a later catalog availability change still busts the cache.
        """
        try:
            from data_access.read.semantic_catalog import get_semantic_catalog

            catalog = get_semantic_catalog()
            payload: dict[str, Any] = {}
            for name, f in getattr(catalog, "_fields", {}).items():
                try:
                    payload[name] = f.to_dict()
                except Exception:
                    payload[name] = str(f)
            raw = json.dumps(payload, sort_keys=True, default=str)
            return hashlib.sha1(raw.encode("utf-8")).hexdigest()
        except Exception:
            return "unavailable"

    def _raise_or_fallback(
        self,
        exc_cls: type[DataAccessColumnPreflightError],
        exc: Exception,
        context: str,
    ) -> None:
        """Production raises a ``Catalog*`` error; research warns and falls back."""
        catalog_exc = exc_cls(f"{context} dataset={self.dataset!r}: {exc}")
        if self.strict_unknown_fields:
            raise catalog_exc from exc
        logger.warning("%s (research fallback) dataset=%s: %s", context, self.dataset, exc)

    def _field_spec(self, name: str, *, production: bool = False) -> Any:
        """Look up a registered FE ``FieldSpec`` for this dataset (registry).

        P0-31: failure classes are distinguished instead of a blanket
        ``except Exception: return None``:

        * ``UnknownField`` — the field is genuinely not in the registry;
        * ``FieldRegistryUnavailable`` — the registry itself could not be
          loaded / queried (infrastructure failure);
        * ``FieldSpecInvalid`` — a spec exists but is malformed.

        In production (``production=True``) these are RAISED (fail-closed) so a
        missing/invalid spec is never silently downgraded to an untyped column;
        in research mode they fall back to ``None`` (fail-open, raw columns
        allowed).
        """
        try:
            from fields.registry import (
                AmbiguousField,
                ResolvedField,
                UnknownField as UnknownFieldResult,
            )
        except Exception as exc:
            if production:
                raise FieldRegistryUnavailable(
                    f"field registry unavailable for {name!r} dataset={self.dataset!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return None
        try:
            # R17-001: resolve within the dataset's MARKET registry (dataset names
            # are market-prefixed: ashare_* / us_*), never the legacy A-share-only
            # registry.  A US dataset's fields must not be hit by A-share aliases.
            dataset_market = _market_from_dataset(self.dataset)
            if dataset_market:
                from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

                # Market registry returns a FieldSpec directly (or None).
                resolved = MULTI_MARKET_FIELD_REGISTRY.resolve_field(
                    dataset_market, name, table=self.dataset, strict=False
                )
                if resolved is None:
                    if production:
                        raise UnknownFieldSemanticError(
                            f"unknown field {name!r} in dataset {self.dataset!r}"
                        )
                    return None
                return resolved
            from fields import FIELD_REGISTRY

            resolved = FIELD_REGISTRY.resolve_field(name, table=self.dataset)
        except Exception as exc:
            if isinstance(exc, UnknownFieldSemanticError):
                raise
            if production:
                raise FieldRegistryUnavailable(
                    f"field registry lookup failed for {name!r} dataset={self.dataset!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return None
        if isinstance(resolved, UnknownFieldResult):
            if production:
                raise UnknownFieldSemanticError(
                    f"unknown field {name!r} in dataset {self.dataset!r}"
                )
            return None
        if isinstance(resolved, AmbiguousField):
            if production:
                raise UnknownFieldSemanticError(
                    f"ambiguous field {name!r} in dataset {self.dataset!r} "
                    f"(candidates: {', '.join(resolved.candidates)})"
                )
            return None
        spec = resolved.spec
        if spec is None:
            if production:
                raise FieldSpecInvalid(
                    f"field {name!r} in dataset {self.dataset!r} resolved to a null spec"
                )
            return None
        return spec

    def _build_field_plans(self, names: list[str]) -> dict[str, NormalizedFieldPlan]:
        """Build unified field plans (P0-11): catalog first, then FE registry.

        The returned plans are keyed by logical name and are the *single* object
        consumed by ``_resolve_columns`` (physical read) and by scale
        normalization, so the unit/scale contract comes from one source.
        """
        plans: dict[str, NormalizedFieldPlan] = {}
        # Round-7 P1 partial resolution: only the names the catalog resolved use
        # catalog plans; clean per-name misses fall back to the FE registry below.
        catalog_by_name = self._resolve_catalog_fields(names) or {}
        for name in names:
            f = catalog_by_name.get(name)
            if f is not None:
                plans[name] = plan_from_catalog_field(name, f)
                continue
            spec = self._field_spec(name, production=self.strict_unknown_fields)
            if spec is not None:
                plans[name] = plan_from_field_spec(name, spec)
                continue
            plans[name] = NormalizedFieldPlan(logical_concept=name, physical_fields=(name,))
        return plans

    def _ensure_field_plans(self, names: Iterable[str]) -> dict[str, NormalizedFieldPlan]:
        """Return cached plans for ``names``, building only the missing ones.

        Round-7 P0: the plan cache is keyed by ``(logical_name,
        semantic_catalog_version)`` so a catalog semantic change (scale / mapping /
        mining_allowed) invalidates the cached plans instead of silently serving
        stale normalization contracts.  ``clear_cache()`` drops the whole cache.
        """
        names = list(names)
        version = self._semantic_catalog_version()
        missing = [n for n in names if (n, version) not in self._field_plans]
        if missing:
            built = self._build_field_plans(missing)
            for n, plan in built.items():
                self._field_plans[(n, version)] = plan
        plans = {n: self._field_plans[(n, version)] for n in names}
        self._enforce_field_contract_gates(plans)
        # #收官轮 P0（Integration）：four-layer PIT gate 自动挂在主通道上——
        # ``pit_enforce`` 时不需要调用方记得手动 ``assert_four_layer_pit()``。
        # 与 mining/coverage gate 一样是 opt-in：engine 的 DataSourceBuildContext
        # 注入 ``config.pit.enforce``（production 强制 True）；UNKNOWN 层在
        # production（strict_unknown_fields）下由 ``assert_four_layer_pit``
        # fail-closed 拒绝，research 告警降级。
        if self.pit_enforce:
            self.assert_four_layer_pit(plans)
        return plans

    def _window_is_historical(self, asof: str | None = None) -> bool:
        """True when this source is configured for a historical (backfill) window.

        No ``start_date`` (full history) or a ``start_date`` strictly before the
        reference point counts as historical — the danger for a
        ``current_snapshot_only`` field is the current snapshot leaking into the
        past (round-7 WS-E #280).

        R24-078: the reference point is the ``asof`` decision context (source
        snapshot validity / decision date) when provided.  Machine
        ``datetime.now()`` is only the research fallback — it is never the
        authority for production.
        """
        if self.start_date is None:
            return True
        try:
            import pandas as pd

            if asof is not None:
                reference = pd.Timestamp(asof).normalize()
            elif self.snapshot_valid_at is not None:
                reference = pd.Timestamp(self.snapshot_valid_at).normalize()
            elif self.production and self.enforce_mining_gate:
                # Production historical-MINING cannot prove "historical" against
                # machine time — require an explicit decision context (R24-078:
                # machine datetime.now() is never authoritative for a mining gate).
                raise HistoricalSnapshotBackfillError(
                    "production mining cannot judge historical/current without a "
                    "decision context (asof) or source snapshot_valid_at "
                    "(R24-078 — machine datetime.now() is not authoritative)"
                )
            else:
                reference = pd.Timestamp.now().normalize()
            return bool(pd.Timestamp(self.start_date).normalize() < reference)
        except HistoricalSnapshotBackfillError:
            # R24-078: the production decision-context requirement must NOT be
            # swallowed by the generic date-parsing except below.
            raise
        except (ValueError, TypeError):
            return True

    def _enforce_field_contract_gates(
        self,
        plans: dict[str, NormalizedFieldPlan],
    ) -> None:
        """Apply the opt-in production field-contract hard gates (round-7 WS-E).

        #279 ``mining_allowed``: reject a ``mining_allowed=False`` field when the
            mining gate is enabled (raw read path of the mining preflight).
        #280 ``current_snapshot_only``: reject historical auto-mining over a
            snapshot-only field unless the operator opts into ``snapshot_now_only``.
        #315 coverage: reject a partial-history field whose declared coverage is
            below ``mining_coverage_threshold`` over the requested window.
        """
        if not (self.enforce_mining_gate or self.mining_coverage_threshold is not None):
            return
        # R24-075..077: ``snapshot_now_only`` is a caller INTENT.  In production
        # it is only honored when the source supplies its snapshot validity
        # metadata AND the requested window lies within that validity — a bare
        # caller boolean never self-proves.  This check runs BEFORE the
        # historical-window computation so the proof requirement is reported
        # first.
        if (
            self.snapshot_now_only
            and self.production
            and self.snapshot_valid_at is None
            and self.snapshot_created_at is None
        ):
            raise HistoricalSnapshotBackfillError(
                f"dataset {self.dataset!r} requested snapshot_now_only=True in "
                "production but the source declares no snapshot_valid_at / "
                "snapshot_created_at — a caller boolean is intent, not proof "
                "(R24-075..077)"
            )
        historical = self._window_is_historical()
        # Round-11 §35 (plan A): SnapshotOnlySourcePolicy — a snapshot_only source
        # is current-only; production historical mining is a hard fail (the
        # snapshot value must never backfill the historical panel).
        if self.snapshot_only and self.enforce_mining_gate and historical and not self.snapshot_now_only:
            raise HistoricalSnapshotBackfillError(
                f"dataset {self.dataset!r} is snapshot_only (current snapshot) and "
                f"cannot backfill a historical window (start_date={self.start_date!r}); "
                "production historical mining is not permitted over a snapshot-only "
                "source — use snapshot_now_only=True for current-snapshot research"
            )
        for name, plan in plans.items():
            if plan is None:
                continue
            if self.enforce_mining_gate and not plan.mining_allowed:
                raise FieldMiningGateError(
                    f"field {name!r} (table={plan.physical_dataset!r}) has "
                    "mining_allowed=False and cannot be used as a mined factor input"
                )
            if (
                self.enforce_mining_gate
                and plan.current_snapshot_only
                and historical
                and not self.snapshot_now_only
            ):
                raise HistoricalSnapshotBackfillError(
                    f"field {name!r} is current_snapshot_only and cannot backfill a "
                    f"historical window (start_date={self.start_date!r}); opt into "
                    "snapshot_now_only=True to use only the current snapshot"
                )
            if (
                self.mining_coverage_threshold is not None
                and plan.coverage in {"partial_history", "sparse_event"}
            ):
                self._gate_coverage(plan, name)

    def _gate_coverage(self, plan: NormalizedFieldPlan, name: str) -> None:
        """Coverage-gate a partial-history field over the source window (#315).

        The declared coverage contract is read from the module-level registry
        (populated by the search/factor preflight from the catalog).  A
        partial-history field with an active threshold but NO registered contract
        fails closed in production (``strict_unknown_fields``) and warns in
        research.
        """
        contract = get_coverage_contract(
            name, market=getattr(plan, "market", None)
        )
        if contract is None:
            if self.strict_unknown_fields:
                raise HistoricalCoverageError(
                    f"field {name!r} is {plan.coverage!r} but has no declared "
                    "HistoricalCoverageContract; coverage-gate cannot be satisfied "
                    "(COVERAGE_EVIDENCE_MISSING)"
                )
            return
        assert_historical_coverage(
            contract,
            start=self.start_date,
            end=self.end_date,
            threshold=self.mining_coverage_threshold,
            required=self.strict_unknown_fields,
            context=f"field {name!r} market={getattr(plan, 'market', None)!r}: ",
        )

    def assert_four_layer_pit(
        self,
        plans: dict[str, NormalizedFieldPlan] | None = None,
        *,
        operator_pit_allowed: bool = True,
        names: Iterable[str] | None = None,
    ) -> dict[str, NormalizedFieldPlan]:
        """Four-layer PIT eligibility preflight (round-7 WS-E #282).

        Combined eligibility = ``field_pit_allowed ∧ table_pit_allowed ∧
        dataset_pit_allowed ∧ operator_pit_allowed``.

        * ``field_pit_allowed`` — the field's ``strict_pit_allowed`` contract.
        * ``table_pit_allowed`` — the owning table's ``strict_pit_allowed``
          (tri-state: None = UNKNOWN, no TableSpec).
        * ``dataset_pit_allowed`` — the DataAccess COS contract's ``pit_policy``
          is strict (or the dataset supports PIT reads); tri-state, None =
          UNKNOWN (no COS contract / no knowledge-time cleaned fundamental).
        * ``operator_pit_allowed`` — caller-supplied operator-level flag.

        #收官轮 P0：UNKNOWN（无法证明）≠ 安全。``strict_unknown_fields``
        (production) 下任何 UNKNOWN 层直接拒绝；research 告警后降级 True 继续。

        Returns the (resolved) plans so callers can chain.  Raises
        :class:`FourLayerPITError` on the first field whose combined eligibility
        is false (including production-UNKNOWN).
        """
        if names is not None:
            names_list = list(names)
            if names_list:
                plans = self._ensure_field_plans(names_list)
        if not plans:
            return {}
        from pit_contract import four_layer_pit_allowed

        dataset_pit_allowed = self._dataset_pit_allowed()
        for name, plan in plans.items():
            if plan is None:
                continue
            # R17-005: tri-state at every layer.  ``field_pit_allowed`` is now
            # None=UNKNOWN (plan default) instead of coercing to True, and the
            # production gate treats UNKNOWN as reject — same as table/dataset.
            field_pit_allowed = getattr(plan, "strict_pit_allowed", None)
            table_pit_allowed = self._table_pit_allowed(plan)
            unknown_layers: list[str] = []
            if field_pit_allowed is None:
                unknown_layers.append("field_pit_allowed")
            if table_pit_allowed is None:
                unknown_layers.append("table_pit_allowed")
            if dataset_pit_allowed is None:
                unknown_layers.append("dataset_pit_allowed")
            if unknown_layers:
                if self.strict_unknown_fields:
                    raise FourLayerPITError(
                        f"field {name!r} PIT eligibility cannot be proven at "
                        f"layer(s): {', '.join(unknown_layers)} — no COS/TableSpec "
                        "contract proving PIT-safe. production fail-closed "
                        "(UNKNOWN == reject)."
                    )
                logger.warning(
                    "field %r PIT eligibility UNKNOWN at layer(s) %s dataset=%s "
                    "(research downgrade to allowed)",
                    name,
                    ", ".join(unknown_layers),
                    self.dataset,
                )
                # research 降级：UNKNOWN → True（无法证明 ≠ 拒绝，告警已发出；
                # 不降级的话 bool(None)=False 会让下方 conjunction 误判失败）。
                if field_pit_allowed is None:
                    field_pit_allowed = True
                if table_pit_allowed is None:
                    table_pit_allowed = True
                if dataset_pit_allowed is None:
                    dataset_pit_allowed = True
            from pit_contract import PITLayerVerdict

            combined, layers = four_layer_pit_allowed(
                field_pit_allowed=bool(field_pit_allowed),
                table_pit_allowed=bool(table_pit_allowed),
                dataset_pit_allowed=bool(dataset_pit_allowed),
                operator_pit_allowed=bool(operator_pit_allowed),
            )
            if not combined:
                # R24-049..051: a layer is failed unless PROVEN_TRUE — the enum
                # members are string enums (all truthy), so compare explicitly.
                failed = sorted(
                    k for k, v in layers.items()
                    if v != PITLayerVerdict.PROVEN_TRUE
                )
                raise FourLayerPITError(
                    f"field {name!r} PIT eligibility failed at layer(s): "
                    f"{', '.join(failed)}"
                )
        return plans

    def _dataset_pit_allowed(self) -> bool | None:
        """Dataset-layer PIT eligibility（三态：True / False / None=UNKNOWN）。

        * ``True``  — 已证明 PIT safe：``pit_policy=="strict"``（事件表带
          availability_column），或 dense/state_ready/minute panel 的精确
          equi-read（点即时）。
        * ``False`` — 已证明不安全：``effective_time_only`` / sparse / forbidden。
        * ``None``  — **无法证明**：无 COS 契约 / 异常 / 已知无知识时钟的 cleaned
          财务源（``_NO_KNOWLEDGE_TIME_FUNDAMENTALS``）。

        #收官轮 P0：旧代码在 contract 缺失/异常时返回 ``True``——「我不知道」
        被当成「我证明安全了」，把 fundamentals_*（period_end 无 filing_date）的
        前视风险掩盖掉。production 下 UNKNOWN 由 ``assert_four_layer_pit`` 拒绝。
        """
        if self.dataset in _NO_KNOWLEDGE_TIME_FUNDAMENTALS:
            logger.warning(
                "dataset=%s time_column 无知识时钟（period_end/date 语义未证明是 "
                "knowledge/availability 日期），不能证明 PIT safe；生产请改用 "
                "us_stock_balance/income/cashflow（filing_date 为 availability）",
                self.dataset,
            )
            return None
        try:
            from data_access.cos_contract import get_cos_contract

            contract = get_cos_contract(self.dataset)
        except Exception:
            return None
        if contract is None:
            return None
        pit = str(getattr(contract, "pit_policy", "not_applicable") or "")
        panel = str(getattr(contract, "panel_policy", "") or "")
        if pit == "strict":
            return True
        if panel in {"dense", "state_ready", "minute"} and pit in {
            "not_applicable", "unsupported",
        }:
            return True
        return False

    def _table_pit_allowed(self, plan: NormalizedFieldPlan) -> bool | None:
        """Table-layer PIT eligibility（三态；None=无法证明，无 TableSpec/异常）。

        R17-002: the table is resolved within the plan's MARKET (or inferred from
        the dataset name), never against the legacy A-share-only registry — a US
        table (e.g. ``StockValuationDaily``) must not be interpreted with A-share
        same-name semantics.  R17-005: the tri-state is preserved (None ==
        UNKNOWN -> production fail-closed).
        """
        if not plan.physical_dataset:
            return None
        try:
            if getattr(plan, "market", None):
                from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

                table_spec = MULTI_MARKET_FIELD_REGISTRY.registry_for(
                    plan.market
                ).resolve_table(str(plan.physical_dataset))
            else:
                from fields import FIELD_REGISTRY

                table_spec = FIELD_REGISTRY.resolve_table(str(plan.physical_dataset))
            if table_spec is None:
                return None
            return getattr(table_spec, "strict_pit_allowed", None)
        except Exception:
            return None

    def source_dependency_hash(self) -> str:
        """Deterministic hash over this source's field-plan contracts + snapshot.

        Round-7 WS-E #317: folds each resolved input's source identity (dataset,
        physical column, scale, mining/PIT contract) and the current data
        snapshot into a reproducibility hash for factor materialization /
        cache identity.
        """
        plans: dict[str, Any] = {}
        for (name, _version), plan in self._field_plans.items():
            plans[name] = {
                "source": plan.source,
                "physical": list(plan.physical_fields),
                "scale": float(plan.scale) if plan.scale is not None else None,
                "mining_allowed": bool(plan.mining_allowed),
                "coverage": plan.coverage,
                "strict_pit_allowed": bool(plan.strict_pit_allowed),
                "current_snapshot_only": bool(plan.current_snapshot_only),
            }
        payload = json.dumps(
            {
                "dataset": self.dataset,
                "snapshot_id": self._data_snapshot_id,
                "manifest_token": self._manifest_token,
                "plans": plans,
                "params": dict(self.params),
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _detect_source_frequency(ds) -> str | None:
        """Detect daily vs minute source grain from the dataset schema (P0-10).

        ``date`` time column → daily; ``timestamp``/``datetime`` → intraday
        (minute).  Returns ``None`` when undetectable — the caller then defaults
        to the daily ``(instrument, time)`` ordering contract.
        """
        schema = getattr(ds, "schema", None) or {}
        tcol = getattr(ds, "time_column", None)
        if tcol and tcol in schema:
            dtype = str(schema[tcol]).lower()
            if dtype == "date":
                return "daily"
            if "timestamp" in dtype or "datetime" in dtype:
                return "minute"
        return None

    def _resolve_columns(self, names: Iterable[str]) -> tuple[list[str], dict[str, str]]:
        names = list(names)
        self._preflight_logical_columns(names)
        # #9 SemanticFieldCatalog 是字段解析的单一事实源；``_ensure_field_plans``
        # 产出统一 NormalizedFieldPlan（catalog → FE FIELD_REGISTRY → raw），
        # 物理列读取与 scale 归一化共享同一个 plan 对象（P0-11）。
        plans = self._ensure_field_plans(names)

        physical: list[str] = []
        output_names: dict[str, str] = {}
        for name in names:
            src = self.fields.get(name)
            plan = plans.get(name)
            # Only a real catalog/registry plan maps to a physical column; a
            # ``raw`` plan (no registered contract) must stay unresolved so the
            # production fail-closed check below still fires for unknown fields.
            # Round-7 P0: a derived field (``plan.is_derived``) is NOT a plain
            # physical column — its ``primary_physical`` (if any) is the base read,
            # never the derived value, so it must not be mapped here.
            if (
                src is None
                and plan is not None
                and plan.source != "raw"
                and plan.primary_physical
                and not plan.is_derived
            ):
                src = plan.primary_physical
            if plan is not None and plan.is_derived and src is None:
                # Round-7 P0: a catalog derived field carries a derived expression;
                # production fails closed instead of silently reading the raw
                # physical column as the value.  Research keeps the lenient raw
                # fallback (the plan still carries the expression for awareness).
                if self.strict_unknown_fields:
                    raise MissingDataDependencyError(
                        f"dataset={self.dataset!r} field {name!r} is a catalog derived "
                        f"field ({plan.transform!r}); derived-expression lowering is "
                        "not wired, so it cannot be read as a plain physical column"
                    )
            if src is None and self.strict_unknown_fields and name not in self.fields:
                # Production fail-closed: a request that matches neither an explicit
                # alias mapping nor a registered field of this dataset has no unit /
                # PIT / temporal contract and must not silently pass through as a raw
                # physical column.  Research keeps the raw fallback via
                # strict_unknown_fields=False.
                raise UnknownFieldSemanticError(
                    f"dataset={self.dataset!r} has no registered field {name!r}; "
                    "set strict_unknown_fields=False (research) to allow the raw "
                    "physical column"
                )
            src = src or name
            physical.append(src)
            if src != name:
                output_names[src] = name
        return physical, output_names

    def _record_read_snapshot(self, snapshot_id: str | None) -> None:
        if not snapshot_id:
            return
        if self._data_snapshot_id and snapshot_id != self._data_snapshot_id:
            self.clear_cache(reset_snapshot=False)
        self._data_snapshot_id = snapshot_id
        self._snapshot_checked_at = time.monotonic()

    def _query_scoped_snapshot_token(self, store) -> str | None:
        """廉价 query-scoped snapshot token：优先读 ``_manifest.json`` sidecar。

        几十微秒级（不 glob 全量文件、不读 footer）。返回 ``None`` 表示没有可用
        manifest（调用方回退到 ``describe_dataset``）。
        """
        try:
            version = store.manifest_version(self.dataset, **self.params)
        except Exception:
            return None
        if not version.get("has_manifest") or not version.get("fresh"):
            return None
        dv = version.get("dataset_version")
        pv = version.get("partition_version")
        if not dv or not pv:
            return None
        return f"manifest:{dv}:{pv}"

    def refresh_snapshot(self, *, force: bool = False) -> str | None:
        """proactive 快照刷新（TTL 内短路）。

        **优先走廉价 manifest token**（``store.manifest_version``），避免每隔 TTL
        对整个 dataset 做昂贵 ``describe_dataset`` 再执行一次实际 read；只有没有
        manifest 时才回退 describe。返回最近一次的数据快照身份（生产路径只消费
        这里的缓存失效副作用）。
        """
        self._assert_open()
        now = time.monotonic()
        if (
            not force
            and self._data_snapshot_id is not None
            and now - self._snapshot_checked_at < self._snapshot_ttl_seconds
        ):
            return self._data_snapshot_id
        store = _get_store()
        token = self._query_scoped_snapshot_token(store)
        if token is not None:
            # 廉价路径：manifest 版本变了才清缓存（token 与真实 snapshot id 分开跟踪）
            if self._manifest_token is not None and token != self._manifest_token:
                logger.info(
                    "data_access manifest version changed dataset=%s old=%s new=%s; clearing caches",
                    self.dataset,
                    self._manifest_token,
                    token,
                )
                self.clear_cache(reset_snapshot=False)
            self._manifest_token = token
        else:
            # 无 manifest：回退全量 describe（旧行为，仅此路径昂贵）
            snapshot = store.describe_dataset(
                self.dataset,
                params=dict(self.params),
                instrument_filter=self.instrument_filter,
            )
            current = snapshot.snapshot_id
            if self._data_snapshot_id and current != self._data_snapshot_id:
                logger.info(
                    "data_access snapshot changed dataset=%s old=%s new=%s; clearing caches",
                    self.dataset,
                    self._data_snapshot_id,
                    current,
                )
                self.clear_cache(reset_snapshot=False)
            self._data_snapshot_id = current
        self._snapshot_checked_at = now
        return self._data_snapshot_id

    def revalidate_for_long_collect(self) -> None:
        """#收官轮 P0：polars-long 受控 collect 前的快照 revalidation。

        ``execute_polars_long_plan`` 在终端 collect 前调用（governed terminal）。
        manifest token / snapshot_id 在「scan 构建 LF」与「collect 执行」之间变化
        时，production fail-closed（执行内容 ≠ 计划快照）；research 清缓存后继续
        （lineage 不撒谎：新读会拿到新快照）。
        """
        self._assert_open()
        store = _get_store()
        token = self._query_scoped_snapshot_token(store)
        if token is not None:
            if self._manifest_token is not None and token != self._manifest_token:
                if self.strict_unknown_fields:
                    raise DataAccessColumnPreflightError(
                        f"dataset={self.dataset!r} 的 manifest 在扫描与 collect 之间"
                        "变化；production fail-closed（polars-long 执行内容 ≠ 计划快照）。"
                    )
                logger.warning(
                    "dataset=%s manifest 在 collect 前变化（research 清缓存继续）",
                    self.dataset,
                )
                self.clear_cache(reset_snapshot=False)
            self._manifest_token = token
            return
        # 无 manifest：回退 describe 快照 id（无法探测时不误伤——scan 边界已记录快照）
        try:
            snapshot = store.describe_dataset(
                self.dataset,
                params=dict(self.params),
                instrument_filter=self.instrument_filter,
            )
        except Exception:
            return
        current = snapshot.snapshot_id
        if self._data_snapshot_id and current != self._data_snapshot_id:
            if self.strict_unknown_fields:
                raise DataAccessColumnPreflightError(
                    f"dataset={self.dataset!r} 的 snapshot 在扫描与 collect 之间"
                    "变化；production fail-closed。"
                )
            self.clear_cache(reset_snapshot=False)
        self._data_snapshot_id = current

    def clear_cache(self, *, reset_snapshot: bool = True) -> None:
        # Round-7 P0: the field-plan cache is versioned by the semantic catalog and
        # must be dropped too, otherwise a catalog change (scale/mapping) keeps
        # serving stale normalization contracts.
        self._field_plans.clear()
        self._column_cache.clear()
        self._panel_cache.clear()
        self._cache_bytes = 0
        self._lazy_bundle = None
        if reset_snapshot:
            self._data_snapshot_id = None
            self._manifest_token = None
            self._snapshot_checked_at = 0.0

    def close(self) -> None:
        self.clear_cache()
        self._closed = True

    def _put_cache(self, cache: OrderedDict[str, Any], name: str, value: Any) -> None:
        if name in cache:
            self._cache_bytes -= self._series_bytes(cache[name])
        cache[name] = value
        cache.move_to_end(name)
        self._cache_bytes += self._series_bytes(value)
        # #42 先按字节上限淘汰，再按列数上限淘汰（1 列分钟 vs 1 列日频差异百倍）
        while len(cache) > 0 and self._cache_bytes > self._max_cache_bytes:
            _, evicted = cache.popitem(last=False)
            self._cache_bytes -= self._series_bytes(evicted)
        while len(cache) > self._max_cache_columns:
            _, evicted = cache.popitem(last=False)
            self._cache_bytes -= self._series_bytes(evicted)

    @staticmethod
    def _series_bytes(value: Any) -> int:
        """估算缓存对象的字节占用（Phase 5 R7：deep 感知，不再只看 nbytes）。"""
        from runtime.resource_governor import estimate_object_bytes

        return estimate_object_bytes(value)

    def column_cache_stats(self) -> dict[str, int]:
        return {
            "cached_columns": len(self._column_cache),
            "cached_panels": len(self._panel_cache),
            "max_cache_columns": self._max_cache_columns,
            "cache_bytes": self._cache_bytes,
            "max_cache_bytes": self._max_cache_bytes,
        }

    def enable_lazy_scan(self, enabled: bool = True) -> None:
        self._assert_open()
        enabled = bool(enabled)
        if enabled != self._lazy_scan:
            self.clear_cache(reset_snapshot=False)
        self._lazy_scan = enabled
        if enabled:
            self.read_auto = True

    def read_session(self) -> "DataSourceReadSession":
        from .read_session import DataSourceReadSession
        return DataSourceReadSession(self)

    def load_column(self, name: str):
        self.refresh_snapshot()
        if name in self._column_cache:
            self._column_cache.move_to_end(name)
            return self._column_cache[name]
        return self.load_columns([name])[name]

    def _adapter_options(self, store):
        from data_access.store import adapter_options_for_dataset

        ds = store.get_dataset(self.dataset)
        options = adapter_options_for_dataset(ds)
        normalize = (
            self.normalize_timestamp
            if self.normalize_timestamp is not None
            else bool(options.get("normalize_timestamp", False))
        )
        unit = self.timestamp_unit or options.get("timestamp_unit")
        return ds, normalize, unit

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        self.refresh_snapshot()
        needed = [name for name in names if name not in self._column_cache]
        if not needed:
            for name in names:
                self._column_cache.move_to_end(name)
            return {name: self._column_cache[name] for name in names}

        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        ds, normalize, unit = self._adapter_options(store)
        logger.info(
            "data_access dataset=%s columns=%s time_range=%s lazy=%s",
            self.dataset,
            needed,
            self._time_range(),
            self._lazy_scan,
        )

        if self._lazy_scan:
            # 显式 polars-lazy 优化（collect 前表达式仍在 DuckDB 内下推）
            from backend.polars_lazy import scan_dataset_columns

            fetched = scan_dataset_columns(
                store,
                self.dataset,
                physical_columns=physical,
                time_column=ds.time_column,
                instrument_column=ds.instrument_column,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                output_names=output_names or None,
                normalize_timestamp=normalize,
                timestamp_unit=unit,
                params=dict(self.params),
                bundle=self._lazy_bundle,
                # #收官轮 P0：read_mode / semantic_filters 贯穿到 DataAccess scan。
                mode=self.read_mode,
                filters=self.semantic_filters or None,
            )
            if self._lazy_bundle is not None:
                self._record_read_snapshot(self._lazy_bundle.snapshot_id)
        else:
            # 引擎/结果形态交给 DataAccess 成本路由（read_auto 语义下沉到 DataAccess）。
            # #9/#12 单位归一化由 DataAccess 输出层完成（SemanticFieldCatalog scale），
            # FactorEngine 不再二次修正 catalog 覆盖的字段。
            from data_access.read.adapters import arrow_table_to_multiindex_columns

            all_columns = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
            handle = store.read(
                self.dataset,
                columns=all_columns,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                # #收官轮 P0：read_mode 真正贯穿到 DataAccess ``mode=``（FE 声明的
                # panel/event/pit 语义不能让 DA 侧落成 mode="auto" 造成分层漂移）；
                # semantic_filters 以 ``filters=`` 落到物理 WHERE（不再是「过了门禁
                # 却没过滤」）。
                mode=self.read_mode,
                filters=self.semantic_filters or None,
                normalize_units=True,
                **self.params,
            )
            self._record_read_snapshot(getattr(handle.snapshot, "snapshot_id", None))
            fetched = arrow_table_to_multiindex_columns(
                handle.to_arrow(),
                timestamp_column=ds.time_column,
                instrument_column=ds.instrument_column,
                value_columns=physical,
                output_names=output_names or None,
                normalize_timestamp=normalize,
                timestamp_unit=unit,
            )

        # DataAccess exposes raw COS values for ``load_columns``.  FactorEngine
        # must use the semantic contract before caching a logical factor input;
        # otherwise A-share Return remains in 1/10000 units and silently
        # contaminates every downstream return-based operator.
        self._normalize_contract_columns(fetched, needed)
        for name in needed:
            self._put_cache(self._column_cache, name, fetched[name])
        return {name: self._column_cache[name] for name in names}

    def _normalize_contract_columns(self, fetched: dict[str, Any], names: list[str]) -> None:
        """Normalize every registered logical field before it enters the cache.

        #9/#12（P0-11）：scale 归一化读取 ``_ensure_field_plans`` 产出的**同一个**
        NormalizedFieldPlan 对象。catalog 覆盖的字段（``source == "catalog"``）已由
        ``store.read(normalize_units=True)`` 在 DataAccess 输出层归一化，eager 路径
        跳过避免二次乘 scale；lazy scan 路径补 catalog scale（否则 A 股 Return/10000
        lazy 路径是 raw 单位、eager 是 decimal，parity bug）。FE FIELD_REGISTRY 覆盖的
        字段（``source == "registry"``）按 plan.scale 归一化（长尾兼容）。

        Production is fail-closed: a unit/scale failure raises instead of
        silently caching the raw vendor value (which would contaminate every
        downstream operator with a 10000× or 100× error).
        """
        plans = self._ensure_field_plans(names)

        normalized: set[str] = set()
        try:
            for name in names:
                plan = plans.get(name)
                if plan is not None and plan.source == "catalog":
                    # Eager path: ``store.read(normalize_units=True)`` already applied
                    # the catalog scale, so nothing to do.  Lazy scan path
                    # (``store.scan`` / ``scan_polars``) does NOT apply the output-layer
                    # unit normalization, so apply the catalog scale here.
                    if self._lazy_scan and plan.is_scale_applicable:
                        fetched[name] = fetched[name] * float(plan.scale)
                    # #收官轮 P0：catalog 覆盖的字段一律进 ``normalized``。否则下方
                    # COS compatibility fallback 会把 A股 Return 再乘一次
                    # ``return_scale``，eager 路径变成 vendor BP × 0.0001(DA) ×
                    # 0.0001(FE) = 10000× 二次缩放。catalog 已覆盖的字段完全禁止
                    # 再进 COS scale fallback。
                    normalized.add(name)
                    continue
                if plan is not None and plan.source == "registry":
                    if name not in fetched or (
                        plan.physical_dataset
                        and plan.physical_dataset != self.dataset
                    ):
                        continue
                    scale = float(plan.scale) if plan.scale is not None else 1.0
                    if scale != 1.0:
                        fetched[name] = fetched[name] * scale
                    normalized.add(name)
                    continue
                # source == "raw"（无登记契约，research pass-through）→ 不归一化
        except Exception as exc:
            if self.strict_unknown_fields:
                raise FieldNormalizationError(
                    f"failed to normalize field contracts for dataset={self.dataset!r}: {exc}"
                ) from exc
            logger.warning("field normalization skipped for %s: %s", self.dataset, exc)

        # Compatibility for external DataAccess contracts that are not represented
        # in FactorEngine's field registry yet.
        try:
            from data_access.cos_contract import get_cos_contract, normalize_return_values
        except Exception:
            return
        contract = get_cos_contract(self.dataset)
        if contract is None or not contract.return_column:
            return
        for name in names:
            if name in normalized or name not in fetched:
                continue
            physical_name = self.fields.get(name)
            if physical_name is None:
                try:
                    # R17-001: resolve within the dataset's market registry.
                    _mkt = _market_from_dataset(self.dataset)
                    if _mkt:
                        from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

                        spec = MULTI_MARKET_FIELD_REGISTRY.resolve_field(
                            _mkt, name, table=self.dataset, strict=False
                        )
                    else:
                        from fields import FIELD_REGISTRY

                        spec = FIELD_REGISTRY.get(name, table=self.dataset)
                    if spec is not None and spec.dataset == self.dataset:
                        physical_name = spec.source_name
                except Exception:
                    physical_name = None
            physical_name = physical_name or name
            if physical_name != contract.return_column:
                continue
            if float(contract.return_scale) != 1.0:
                fetched[name] = normalize_return_values(fetched[name], self.dataset)

    def prefetch_columns(self, names: list[str]) -> None:
        if not names:
            return
        self.refresh_snapshot()
        if self._lazy_scan:
            self._prefetch_lazy_bundle(names)
        else:
            self.load_columns(names)

    def _prefetch_lazy_bundle(self, names: list[str]) -> None:
        needed = [name for name in names if name not in self._column_cache]
        if not needed:
            return
        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        ds, normalize, unit = self._adapter_options(store)
        from backend.polars_lazy import build_lazy_column_bundle

        if self._lazy_bundle is None:
            merged_physical = physical
            merged_output = output_names
        else:
            merged_physical = list(dict.fromkeys([*self._lazy_bundle.physical_columns, *physical]))
            merged_output = dict(self._lazy_bundle.output_names)
            merged_output.update(output_names)
            if merged_physical == list(self._lazy_bundle.physical_columns):
                fetched = self._lazy_bundle.materialize_columns(
                    physical, output_names=output_names or None
                )
                self._normalize_contract_columns(fetched, needed)
                for name in needed:
                    self._put_cache(self._column_cache, name, fetched[name])
                return

        self._lazy_bundle = build_lazy_column_bundle(
            store,
            self.dataset,
            physical_columns=merged_physical,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            output_names=merged_output or None,
            normalize_timestamp=normalize,
            timestamp_unit=unit,
            params=dict(self.params),
            # #收官轮 P0：read_mode / semantic_filters 贯穿到 DataAccess scan。
            mode=self.read_mode,
            filters=self.semantic_filters or None,
        )
        self._record_read_snapshot(self._lazy_bundle.snapshot_id)
        fetched = self._lazy_bundle.materialize_columns(
            physical, output_names=output_names or None
        )
        self._normalize_contract_columns(fetched, needed)
        for name in needed:
            self._put_cache(self._column_cache, name, fetched[name])

    def prefetch_panels(self, names: list[str]) -> None:
        self.refresh_snapshot()
        needed = [name for name in names if name not in self._panel_cache]
        if not needed:
            return
        batch = self.load_columns(needed)
        for name, series in batch.items():
            level = series.index.names[-1] or "instrument"
            self._put_cache(self._panel_cache, name, series.unstack(level=level))

    def load_column_panel(self, name: str):
        self.refresh_snapshot()
        if name in self._panel_cache:
            self._panel_cache.move_to_end(name)
            return self._panel_cache[name]
        series = self.load_column(name)
        level = series.index.names[-1] or "instrument"
        panel = series.unstack(level=level)
        self._put_cache(self._panel_cache, name, panel)
        return panel

    def dataset_axis_columns(self) -> tuple[str, str]:
        return _get_store().dataset_axis_columns(self.dataset)

    def scan_polars_long(self, columns: list[str]):
        """Scan a long Polars LazyFrame for ``columns``.

        Ordering guarantee (P0-10): the returned LazyFrame is sorted ascending by
        ``(instrument, time)`` for daily data — ``(instrument, time, session)`` for
        minute sources when a session column is available.  ``build_scan_polars_long``
        applies the sort at the scan boundary; this method re-asserts it after the
        scale-normalization step so the contract holds for every path.

        Unit normalization (P0-11): scale reads from the SAME ``NormalizedFieldPlan``
        produced by ``_resolve_columns`` (catalog → FE FIELD_REGISTRY → raw).
        """
        self.refresh_snapshot()
        physical, output_names = self._resolve_columns(columns)
        store = _get_store()
        ds = store.get_dataset(self.dataset)
        frequency = self._detect_source_frequency(ds)
        from backend.polars_lazy import build_scan_polars_long, enforce_source_ordering

        lf = build_scan_polars_long(
            store,
            self.dataset,
            logical_columns=columns,
            physical_columns=physical,
            output_names=output_names,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            params=dict(self.params),
            frequency=frequency,
            # #收官轮 P0：read_mode / semantic_filters 贯穿到 DataAccess scan_polars。
            mode=self.read_mode,
            filters=self.semantic_filters or None,
        )
        # scale normalization reads from the same plan object (P0-11)
        plans = self._ensure_field_plans(columns)
        try:
            import polars as pl

            expressions = []
            for name in columns:
                plan = plans.get(name)
                if plan is None or not plan.is_scale_applicable:
                    continue
                scale = float(plan.scale)
                if scale != 1.0:
                    expressions.append((pl.col(name).cast(pl.Float64) * scale).alias(name))
            if expressions:
                lf = lf.with_columns(expressions)
        except ImportError:
            pass
        # Scan-boundary ordering contract (P0-10): re-assert after the scale step.
        return enforce_source_ordering(
            lf,
            instrument_col="inst",
            time_col="ts",
            frequency=frequency,
        )

    def scan_index_long(self):
        self.refresh_snapshot()
        store = _get_store()
        ds = store.get_dataset(self.dataset)
        from backend.polars_lazy import build_scan_index_long

        return build_scan_index_long(
            store,
            self.dataset,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            params=dict(self.params),
            # #收官轮 P0：read_mode / semantic_filters 贯穿到 DataAccess scan。
            mode=self.read_mode,
            filters=self.semantic_filters or None,
        )
