"""Additive batch foundations for automated mining campaigns.

This module owns orchestration metadata only. Operator admission remains in
:mod:`mining.direct_use`; compilation and execution remain owned by FactorEngine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from api.factor import Factor
from expr.base import Expr
from expr.cleaned_call import CleanedCall
from expr.column import ColumnRef
from expr.field import FieldRef


def compiler_generation() -> str:
    """Return a deterministic generation for compile-result invalidation."""
    from cleaned_operators.operator_policy import compute_operator_catalog_hash
    from fields import compute_field_catalog_hash

    payload = {
        "field_catalog": compute_field_catalog_hash(),
        "operator_catalog": compute_operator_catalog_hash(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"unsupported candidate semantic value: {type(value).__name__}")


def _expr_payload(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, FieldRef):
        return {
            "kind": "field",
            "name": expr.name,
            "field_id": expr.field_id,
            "canonical_name": expr.canonical_name,
            "table": expr.table,
            "source_name": expr.source_name,
            "catalog_hash": expr.catalog_hash,
        }
    if isinstance(expr, ColumnRef):
        return {"kind": "column", "name": expr.name}
    if isinstance(expr, CleanedCall):
        try:
            from cleaned_operators.registry import OperatorRegistry

            op = OperatorRegistry._aliases.get(expr.op, expr.op)
        except (ImportError, AttributeError, RuntimeError):
            op = expr.op
        return {
            "kind": "call",
            "op": op,
            "args": [_expr_payload(arg) for arg in expr.args],
            "kwargs": _stable_value(dict(expr.kwargs)),
        }
    if hasattr(expr, "value"):
        return {"kind": "literal", "value": _stable_value(getattr(expr, "value"))}
    raise TypeError(f"unsupported candidate expression: {type(expr).__name__}")


def candidate_semantic_hash(candidate: Factor | Expr) -> str:
    """Hash parsed candidate structure, excluding factor name and source text.

    This is an AST/parameter-normalized identity foundation, not a claim of
    algebraic equivalence. In particular, argument order is preserved.
    """
    expr = candidate.expr if isinstance(candidate, Factor) else candidate
    raw = json.dumps(
        _expr_payload(expr), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NegativeCompileKey:
    candidate_hash: str
    compiler_generation: str


@dataclass(frozen=True)
class NegativeCompileEntry:
    error_class: str
    message: str


class NegativeCompileCache:
    """Thread-safe in-memory cache for deterministic candidate failures."""

    def __init__(self) -> None:
        self._entries: dict[NegativeCompileKey, NegativeCompileEntry] = {}
        self._lock = RLock()

    def get(self, key: NegativeCompileKey) -> NegativeCompileEntry | None:
        with self._lock:
            return self._entries.get(key)

    def record(self, key: NegativeCompileKey, exc: BaseException) -> NegativeCompileEntry:
        entry = NegativeCompileEntry(type(exc).__name__, str(exc))
        with self._lock:
            self._entries[key] = entry
        return entry

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_DETERMINISTIC_COMPILE_ERRORS = (KeyError, TypeError, ValueError, SyntaxError)


@dataclass(frozen=True, order=True)
class DependencySignature:
    """Immutable scheduling signature derived from a compiled candidate."""

    sources: tuple[str, ...]
    fields: tuple[str, ...]
    history: int
    universe_semantics: str
    backend_capability: tuple[str, ...]
    primitive_families: tuple[str, ...]

    @property
    def source_first_key(self) -> tuple[Any, ...]:
        return (self.sources, self.fields, self.history, self.universe_semantics)


def dependency_signature(
    factor: Factor,
    analysis: Any,
    *,
    backend_capability: Iterable[str] = (),
) -> DependencySignature:
    """Build a dependency signature from the analyzer's authoritative result."""
    fields_by_id = getattr(analysis, "referenced_fields", {}) or {}
    sources = {
        str(getattr(spec, "source_name", "") or getattr(spec, "table", ""))
        for spec in fields_by_id.values()
    }
    sources.discard("")
    fields = set(getattr(analysis, "referenced_field_ids", set()) or ())
    fields.update(getattr(analysis, "referenced_columns", set()) or ())
    primitives: set[str] = set()

    def visit(expr: Expr) -> None:
        if isinstance(expr, CleanedCall):
            primitives.add(str(expr.op).split("_", 1)[0])
        for child in expr.children():
            visit(child)

    visit(factor.expr)
    hint = factor.semantic_identity
    universe = (
        getattr(hint, "universe_id", None) if hint is not None else None
    ) or factor.universe or ""
    return DependencySignature(
        sources=tuple(sorted(sources)),
        fields=tuple(sorted(str(value) for value in fields)),
        history=int(getattr(analysis, "lookback", 0) or 0),
        universe_semantics=str(universe),
        backend_capability=tuple(sorted(str(value) for value in backend_capability)),
        primitive_families=tuple(sorted(primitives)),
    )


def group_source_first(
    candidates: Sequence[tuple[Factor, DependencySignature]],
) -> tuple[tuple[tuple[Any, ...], tuple[Factor, ...]], ...]:
    """Group candidates deterministically by source/fields/history/universe."""
    grouped: dict[tuple[Any, ...], list[Factor]] = {}
    for factor, signature in candidates:
        grouped.setdefault(signature.source_first_key, []).append(factor)
    return tuple(
        (key, tuple(sorted(factors, key=lambda factor: factor.name)))
        for key, factors in sorted(grouped.items(), key=lambda item: item[0])
    )


@dataclass(frozen=True)
class MiningCampaignSnapshot:
    """Frozen identities under which every candidate is compared."""

    source_snapshot: str
    universe_snapshot: str
    compiler_generation: str

    def __post_init__(self) -> None:
        for name in ("source_snapshot", "universe_snapshot", "compiler_generation"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a non-empty pinned identity")


@dataclass
class MiningCampaignSession:
    """One pinned mining run sharing compiler, CSE, evaluator and source session."""

    engine: Any
    snapshot: MiningCampaignSnapshot
    evaluator: Callable[[str, Any, Mapping[str, Any]], Any] | None = None
    resource_lease: Any = None
    negative_cache: NegativeCompileCache = field(default_factory=NegativeCompileCache)
    _compiled: dict[str, tuple[Any, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def compiled(self) -> Mapping[str, tuple[Any, Any]]:
        return MappingProxyType(self._compiled)

    def compile_many(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool = True,
    ) -> dict[str, Any]:
        """Batch-compile candidates and cache deterministic singleton failures.

        ``FactorEngine.analyze_batch`` is used so successful candidates compile
        exactly once. If a batch fails deterministically it is bisected until the
        bad singleton is identified; known-bad candidates are O(1) rejects on
        later submissions in the same compiler generation.
        """
        pending: list[Factor] = []
        failures: dict[str, NegativeCompileEntry] = {}
        for factor in factors:
            key = NegativeCompileKey(
                candidate_semantic_hash(factor), self.snapshot.compiler_generation
            )
            cached = self.negative_cache.get(key)
            if cached is not None:
                failures[factor.name] = cached
            else:
                pending.append(factor)

        successful_batches: list[tuple[tuple[Factor, ...], Any, Mapping[str, Any]]] = []

        def compile_batch(batch: tuple[Factor, ...]) -> None:
            if not batch:
                return
            try:
                result = self.engine.analyze_batch(batch, enable_cse=enable_cse)
            except _DETERMINISTIC_COMPILE_ERRORS as exc:
                if len(batch) > 1:
                    middle = len(batch) // 2
                    compile_batch(batch[:middle])
                    compile_batch(batch[middle:])
                    return
                factor = batch[0]
                key = NegativeCompileKey(
                    candidate_semantic_hash(factor), self.snapshot.compiler_generation
                )
                failures[factor.name] = self.negative_cache.record(key, exc)
                return
            analyses = result.get("analyses", {})
            successful_batches.append((batch, result.get("dag"), analyses))
            for factor in batch:
                self._compiled[factor.name] = (result.get("dag"), analyses[factor.name])

        compile_batch(tuple(pending))
        survivors = tuple(
            factor for batch, _dag, _analyses in successful_batches for factor in batch
        )
        analyses = {
            factor.name: batch_analyses[factor.name]
            for batch, _dag, batch_analyses in successful_batches
            for factor in batch
        }
        # One successful batch preserves whole-campaign CSE. Multiple batches only
        # occur during deterministic-failure isolation and remain explicit here.
        plans = tuple(dag for _batch, dag, _analyses in successful_batches)
        plan: Any = plans[0] if len(plans) == 1 else plans
        return {
            "plan": plan,
            "factors": survivors,
            "analyses": MappingProxyType(analyses),
            "failures": MappingProxyType(failures),
        }

    def run_many_iter(self, factors: Sequence[Factor], **kwargs: Any) -> Iterator[Any]:
        """Delegate to the engine's streaming result API under this snapshot."""
        yield from self.engine.run_many_iter(factors, **kwargs)

    def evaluate_many_iter(
        self, factors: Sequence[Factor], **kwargs: Any
    ) -> Iterator[tuple[str, Any, Any]]:
        """Stream candidate blocks through the evaluator without mandatory lake IO."""
        snapshot = MappingProxyType(
            {
                "source_snapshot": self.snapshot.source_snapshot,
                "universe_snapshot": self.snapshot.universe_snapshot,
                "compiler_generation": self.snapshot.compiler_generation,
            }
        )
        for name, block, backend_path in self.run_many_iter(factors, **kwargs):
            value = (
                self.evaluator(name, block, snapshot)
                if self.evaluator is not None
                else block
            )
            yield name, value, backend_path


__all__ = [
    "DependencySignature",
    "MiningCampaignSession",
    "MiningCampaignSnapshot",
    "NegativeCompileCache",
    "NegativeCompileEntry",
    "NegativeCompileKey",
    "candidate_semantic_hash",
    "compiler_generation",
    "dependency_signature",
    "group_source_first",
]
