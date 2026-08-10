"""R21-032..047 + R21-008..010: typed HTTP request/response models, the
immutable ``ValidatedFactorRequest``, and request size budgets.

R21-032..036: no more raw-dict payloads; ``extra="forbid"``, strict enums,
explicit length/cardinality caps (a ``"sync": "false"`` string is *not* truthy
under a Pydantic bool).  R21-037..043: parse/downstream size budgets are
applied pre-``ast.parse``.  R21-044..047: a cheap ``FactorCostEstimate`` gates
high-cost jobs before they enter the queue.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

    class BaseModel:  # type: ignore[no-redef]
        model_config = {}

    def ConfigDict(**kw):  # type: ignore[misc]
        return kw

    def Field(*args, **kw):  # type: ignore[misc]
        return None

    def field_validator(*args, **kw):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class JobType(str, Enum):
    compute = "compute"
    materialize = "materialize"


class RunMode(str, Enum):
    research = "research"
    production = "production"


class SurfaceType(str, Enum):
    daily = "daily"
    minute = "minute"
    lqtp = "lqtp"


class DialectType(str, Enum):
    native = "native"
    lqtp = "lqtp"


BACKEND_ALIASES = {
    "pandas", "pandas_modin", "polars", "polars_lazy", "polars_long",
    "polars_native", "long_polars", "auto_long", "hybrid_long",
    "duckdb_sql", "sql_pushdown", "duckdb_pushdown", "sql",
    "clickhouse_sql", "ch_sql", "clickhouse_pushdown",
    "auto", "hybrid", "debug",
}


class WriteTarget(str, Enum):
    local = "local"
    staging = "staging"
    production = "production"
    clickhouse = "clickhouse"


# ---------------------------------------------------------------------------
# Size budgets (R21-036..043)
# ---------------------------------------------------------------------------

DEFAULT_SIZE_BUDGET = {
    "max_formula_bytes": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_FORMULA_BYTES", "65536")),
    "max_formula_chars": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_FORMULA_CHARS", "65536")),
    "max_string_literal_bytes": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_STRING_LITERAL_BYTES", "8192")),
    "max_identifier_length": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_IDENTIFIER_LENGTH", "256")),
    "max_keyword_length": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_KEYWORD_LENGTH", "128")),
    "max_name_length": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_NAME_LENGTH", "128")),
    "max_idempotency_key_length": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_IDEMPOTENCY_KEY_LENGTH", "128")),
    "max_source_refs": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_SOURCE_REFS", "64")),
    "max_instrument_filter": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_INSTRUMENT_FILTER", "10000")),
    "max_metadata_items": int(__import__("os").environ.get("FACTOR_ENGINE_MAX_METADATA_ITEMS", "64")),
}


def size_budget() -> dict[str, int]:
    """Return the current size budget dict (env-configurable)."""
    return {k: int(__import__("os").environ.get("FACTOR_ENGINE_" + k[len("max_"):].upper(), str(v)))
            for k, v in DEFAULT_SIZE_BUDGET.items()}


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def enforce_size(self, budget: dict[str, int] | None = None) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Typed request models
# ---------------------------------------------------------------------------


class ComputeRequest(_RequestModel):
    """R21-034: strict request model — extra fields rejected, enums validated."""

    formula: Optional[str] = None
    dsl: Optional[str] = None
    name: Optional[str] = None
    surface: SurfaceType = SurfaceType.daily
    dialect: DialectType = DialectType.native
    dialect_version: Optional[str] = None
    freq: Optional[str] = None
    universe: Optional[list[str]] = None
    market: Optional[str] = None
    calendar: Optional[str] = None
    decision_time_policy: Optional[str] = None
    run_mode: RunMode = RunMode.research
    backend: Optional[str] = None
    data_source: Optional[dict[str, Any]] = None
    approved_source_profile_id: Optional[str] = None
    config_path: Optional[str] = None
    idempotency_key: Optional[str] = None
    sync: bool = False
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    factor: Optional[dict[str, Any]] = None
    formula_schema_version: Optional[str] = None
    max_cost_per_job: Optional[int] = None

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if str(v).strip().lower() not in BACKEND_ALIASES:
            raise ValueError(f"unsupported backend: {v!r}")
        return str(v).strip().lower()

    @field_validator("formula", "dsl")
    @classmethod
    def _strip(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return str(v).strip()

    @field_validator("name")
    @classmethod
    def _name_cap(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from security.factor_id import FactorIdError, validate_factor_id

        try:
            return validate_factor_id(v)
        except FactorIdError as exc:
            # R32-P0-035: 超长/非法必须报 validation error，绝不截断 ——
            # ``[:128]`` 会让两个不同 ID 静默碰撞。
            raise ValueError(str(exc)) from exc

    @field_validator("universe")
    @classmethod
    def _universe_cap(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        if len(v) > DEFAULT_SIZE_BUDGET["max_instrument_filter"]:
            raise ValueError("universe too large")
        return v

    def formula_text(self) -> str:
        return str(self.formula or self.dsl or "").strip()


class MaterializeRequest(_RequestModel):
    config_path: str
    run_mode: RunMode = RunMode.production
    factor_id: Optional[str] = None
    author: Optional[str] = None
    frequency: Optional[str] = None
    description: Optional[str] = None
    expression: Optional[str] = None
    write_target: WriteTarget = WriteTarget.production
    idempotency_key: Optional[str] = None
    sync: bool = False
    request_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("factor_id")
    @classmethod
    def _factor_id_cap(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from security.factor_id import FactorIdError, validate_factor_id

        try:
            return validate_factor_id(v)
        except FactorIdError as exc:
            raise ValueError(str(exc)) from exc


class ValidateRequest(_RequestModel):
    formula: Optional[str] = None
    dsl: Optional[str] = None
    surface: SurfaceType = SurfaceType.daily
    dialect: DialectType = DialectType.native
    dialect_version: Optional[str] = None
    run_mode: RunMode = RunMode.research
    data_source: Optional[dict[str, Any]] = None
    factor: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Immutable validated request (R21-008..010)
# ---------------------------------------------------------------------------

_DIGEST_FIELDS = (
    "canonical_formula",
    "surface",
    "dialect",
    "dialect_version",
    "frequency",
    "market",
    "universe",
    "calendar",
    "decision_time_policy",
    "production_policy",
    "catalog_generations",
    "complexity_budget",
    "source_profile",
)


@dataclass(frozen=True)
class ValidatedFactorRequest:
    """R21-008: the *only* object executed after validation.

    Constructed once during validation; execution never re-parses a raw dict.
    """

    canonical_formula: str
    surface: str
    dialect: str
    dialect_version: str | None
    frequency: str | None
    market: str | None
    universe: tuple[str, ...]
    calendar: str | None
    decision_time_policy: str | None
    production_policy: str
    catalog_generations: tuple[str, ...] = ()
    complexity_budget: str = "default"
    source_profile: str | None = None
    extra: dict[str, Any] = dc_field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "canonical_formula": self.canonical_formula,
            "surface": self.surface,
            "dialect": self.dialect,
            "dialect_version": self.dialect_version,
            "frequency": self.frequency,
            "market": self.market,
            "universe": list(self.universe),
            "calendar": self.calendar,
            "decision_time_policy": self.decision_time_policy,
            "production_policy": self.production_policy,
            "catalog_generations": list(self.catalog_generations),
            "complexity_budget": self.complexity_budget,
            "source_profile": self.source_profile,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def execution_digest_matches(self, other_digest: str) -> bool:
        return other_digest == self.digest()


def canonical_formula_digest(formula: str, *, budget: dict[str, int]) -> str:
    return hashlib.sha256(
        f"{formula}|{json.dumps(budget, sort_keys=True)}".encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Cost estimate (R21-044..047)
# ---------------------------------------------------------------------------

_COSTY_OPERATORS = {
    "corr", "cov", "rolling_corr", "ts_corr", "rank", "cs_rank", "beta",
    "regression", "rolling_beta", "ewm", "ts_ema", "modwt", "dwt", "kalman",
    "garch", "har", "hmm", "matrix_profile", "rqa", "permutation_entropy",
    "sample_entropy", "multifractal", "dmd", "pca", "ica", "knn", "te",
    "transfer_entropy", "hicks", "hsic", "glr", "cross_corr", "ccorr",
    "rolling_corr", "rolling_cov", "group_corr",
}

_OPERATOR_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")


@dataclass(frozen=True)
class FactorCostEstimate:
    """R21-044: cheap pre-queue cost estimate for a formula + execution scope."""

    formula_length: int
    ast_node_estimate: int
    operator_count: int
    high_cost_operator_count: int
    window_estimate: int
    history_estimate: int
    source_ref_count: int
    universe_size: int
    cells_estimate: int
    expected_memory_mb: float
    cost_score: int

    def exceeds(self, *, max_cost: int | None = None, max_cells: int | None = None,
                max_history: int | None = None, max_sources: int | None = None,
                max_expected_memory: int | None = None) -> tuple[bool, str]:
        if max_cost is not None and self.cost_score > max_cost:
            return True, f"cost {self.cost_score} > {max_cost}"
        if max_cells is not None and self.cells_estimate > max_cells:
            return True, f"cells {self.cells_estimate} > {max_cells}"
        if max_history is not None and self.history_estimate > max_history:
            return True, f"history {self.history_estimate} > {max_history}"
        if max_sources is not None and self.source_ref_count > max_sources:
            return True, f"sources {self.source_ref_count} > {max_sources}"
        if max_expected_memory is not None and self.expected_memory_mb > max_expected_memory:
            return True, f"memory {self.expected_memory_mb:.0f}MB > {max_expected_memory}MB"
        return False, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_length": self.formula_length,
            "ast_node_estimate": self.ast_node_estimate,
            "operator_count": self.operator_count,
            "high_cost_operator_count": self.high_cost_operator_count,
            "window_estimate": self.window_estimate,
            "history_estimate": self.history_estimate,
            "source_ref_count": self.source_ref_count,
            "universe_size": self.universe_size,
            "cells_estimate": self.cells_estimate,
            "expected_memory_mb": round(self.expected_memory_mb, 2),
            "cost_score": self.cost_score,
        }


def estimate_factor_cost(
    formula: str,
    *,
    universe_size: int = 1,
    history_days: int = 250,
    surface: str = "daily",
) -> FactorCostEstimate:
    """R21-044..047: formula-level cost estimate computed before queue admission."""
    calls = _OPERATOR_RE.findall(formula)
    op_count = len(calls)
    high = sum(1 for c in calls if c in _COSTY_OPERATORS)
    identifiers = len(_IDENTIFIER_RE.findall(formula))
    window_est = max(1, identifiers)
    minutes_factor = 48 if surface == "minute" else 1
    history_est = history_days * minutes_factor
    cells = max(1, universe_size) * history_est * max(1, op_count)
    expected_mb = 16 * max(1, universe_size) * (history_est / 250.0) * max(1, op_count) / 1024.0
    expected_mb = max(0.1, expected_mb)
    cost_score = int(
        len(formula) / 64
        + op_count * 10
        + high * 60
        + int(math.log2(max(1, universe_size))) * 40
        + (history_est / 250.0) * 20
    )
    return FactorCostEstimate(
        formula_length=len(formula),
        ast_node_estimate=identifiers + op_count * 2 + 1,
        operator_count=op_count,
        high_cost_operator_count=high,
        window_estimate=window_est,
        history_estimate=history_est,
        source_ref_count=formula.count("source_col(") + formula.count("SourceRef("),
        universe_size=max(1, universe_size),
        cells_estimate=cells,
        expected_memory_mb=expected_mb,
        cost_score=cost_score,
    )
