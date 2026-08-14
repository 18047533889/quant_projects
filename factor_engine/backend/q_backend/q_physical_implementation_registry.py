"""Evidence-backed registry for executable q physical implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class QPhysicalImplementation:
    canonical: str
    lowering_id: str
    parameter_domain_id: str | None = None
    compile_evidence: str | None = None
    runtime_evidence: str | None = None
    parity_evidence: str | None = None
    null_semantics_evidence: str | None = None
    performance_evidence: str | None = None
    implementation_hash: str | None = None
    q_version_range: str | None = None
    pykx_version_range: str | None = None
    notes: str = ""

    @property
    def has_full_evidence(self) -> bool:
        return all((
            self.compile_evidence,
            self.runtime_evidence,
            self.parity_evidence,
            self.parameter_domain_id,
            self.implementation_hash,
            self.q_version_range,
            self.pykx_version_range,
        ))

    @property
    def research_ready(self) -> bool:
        """Executable lowering is available for research use only."""
        return bool(self.lowering_id)

    @property
    def production_ready(self) -> bool:
        # Compile, runtime, and parity references must be independent artifacts.
        return bool(
            self.lowering_id
            and self.has_full_evidence
            and self.parameter_domain_id
            and self.implementation_hash
            and self.q_version_range
            and self.pykx_version_range
        )


class QPhysicalImplementationRegistry:
    """Registry populated only from a compiler's executable lowering map."""

    _DECLARED_TARGETS = frozenset({
        "add", "subtract", "multiply", "divide", "negate", "abs", "power", "sqrt", "log", "exp", "log1p", "expm1", "sign", "floor", "ceil", "round",
        "greater", "less", "greater_equal", "less_equal", "equal", "not_equal", "lag", "delta", "pct_change", "ts_returns", "ts_diff",
        "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max", "ts_count", "ts_median", "ts_product", "ts_var", "ts_cumsum", "ts_cumprod", "ts_cummax", "ts_cummin",
        "ts_skew", "ts_kurt", "ts_moment", "ts_argmax", "ts_argmin", "ts_argmax_age", "ts_argmin_age", "ts_days_since_high", "ts_days_since_low", "ts_new_high", "ts_new_low",
        "ts_max_drawdown", "ts_distance_to_high", "ts_distance_to_low", "ts_decay_linear", "ts_decay_exp", "ts_sum_decay", "ts_rank", "ts_zscore", "ts_demean", "ts_normalize", "ts_quantile", "ts_percentile", "ts_corr", "ts_cov", "ts_beta",
        "rank", "cs_rank", "cs_zscore", "cs_demean", "cs_normalize", "cs_quantile", "cs_percentile_rank", "cs_winsorize", "cs_clip", "cs_mean", "cs_std", "cs_median", "cs_var",
        "group_mean", "group_sum", "group_std", "group_median", "group_min", "group_max", "group_count", "where", "fillna", "ffill", "bfill", "clip", "replace",
        "vwap", "mean", "sum", "std", "min", "max", "median", "product", "var", "count_nonzero", "first", "last", "resample", "time_bucket", "true_range", "ema", "wma", "sma",
    })

    def __init__(
        self,
        *,
        lowerings: Mapping[str, str] | None = None,
        declared_targets: frozenset[str] | None = None,
    ) -> None:
        self._implementations: dict[str, QPhysicalImplementation] = {}
        # An explicitly-created registry is an empty fixture unless its caller
        # supplies declarations. Production bootstrap supplies declarations
        # from the compiler through ``build_q_physical_implementation_registry``.
        self._declared_targets = frozenset(
            frozenset() if declared_targets is None else declared_targets
        )
        for canonical, lowering_id in (lowerings or {}).items():
            self.register(QPhysicalImplementation(canonical=canonical, lowering_id=lowering_id))

    def declared_targets(self) -> frozenset[str]:
        return self._declared_targets

    def register(self, impl: QPhysicalImplementation) -> None:
        existing = self._implementations.get(impl.canonical)
        if existing is not None and existing.lowering_id != impl.lowering_id:
            raise ValueError(
                f"Cannot register {impl.canonical}: already registered with "
                f"lowering_id={existing.lowering_id}, attempted {impl.lowering_id}"
            )
        self._implementations[impl.canonical] = impl

    def get(self, canonical: str) -> QPhysicalImplementation | None:
        return self._implementations.get(canonical)

    def get_research_ready(self) -> frozenset[str]:
        if self.has_disagreement():
            return frozenset()
        return frozenset(name for name, impl in self._implementations.items() if impl.research_ready)

    def get_production_ready(self) -> frozenset[str]:
        if self.has_disagreement():
            return frozenset()
        return frozenset(name for name, impl in self._implementations.items() if impl.production_ready)

    def get_with_lowering(self) -> frozenset[str]:
        return frozenset(name for name, impl in self._implementations.items() if impl.lowering_id)

    def has_lowering(self, canonical: str) -> bool:
        return bool((impl := self.get(canonical)) and impl.lowering_id)

    def is_production_certified(self, canonical: str) -> bool:
        if self.has_disagreement():
            return False
        return bool((impl := self.get(canonical)) and impl.production_ready)

    def admission_disagreements(self) -> dict[str, list[str]]:
        declared = set(self.declared_targets())
        lowering = set(self.get_with_lowering())
        return {
            "declared_without_lowering": sorted(declared - lowering),
            "lowering_without_declaration": sorted(lowering - declared),
        }

    def has_disagreement(self) -> bool:
        return any(self.admission_disagreements().values())

    def get_missing_evidence(self) -> dict[str, list[str]]:
        missing: dict[str, list[str]] = {}
        for name, impl in self._implementations.items():
            gaps = []
            if not impl.compile_evidence:
                gaps.append("compile")
            if not impl.runtime_evidence:
                gaps.append("runtime")
            if not impl.parity_evidence:
                gaps.append("parity")
            if not impl.parameter_domain_id:
                gaps.append("parameter_domain")
            if not impl.implementation_hash:
                gaps.append("implementation_hash")
            if not impl.q_version_range:
                gaps.append("q_version_range")
            if not impl.pykx_version_range:
                gaps.append("pykx_version_range")
            if gaps:
                missing[name] = gaps
        return missing

    def gate_all_lowerings_have_evidence(self) -> tuple[bool, str]:
        missing = self.get_missing_evidence()
        disagreements = self.admission_disagreements()
        if self._implementations and not missing and not any(disagreements.values()):
            return True, "PASS: all lowerings have independent evidence"
        return False, f"FAIL: missing={list(missing)[:5]}, disagreements={disagreements}"


_REGISTRY: QPhysicalImplementationRegistry | None = None


def build_q_physical_implementation_registry(
    lowerings: Mapping[str, str],
    declared_targets: frozenset[str],
) -> QPhysicalImplementationRegistry:
    """Build without importing compiler/capability modules."""
    return QPhysicalImplementationRegistry(
        lowerings=lowerings,
        declared_targets=declared_targets,
    )


def install_q_physical_implementation_registry(
    registry: QPhysicalImplementationRegistry,
) -> QPhysicalImplementationRegistry:
    global _REGISTRY
    _REGISTRY = registry
    return registry


def get_q_physical_implementation_registry() -> QPhysicalImplementationRegistry:
    """Return compiler-derived authority, or fail closed when unavailable.

    The lazy dependency points from registry data to the compiler lowering
    source only after module initialization, so the compiler never imports a
    partially initialized registry.  Isolated fixtures can still install an
    explicit registry; an unpopulated explicit registry admits nothing.
    """
    global _REGISTRY
    if _REGISTRY is None:
        from backend.q_backend.q_compiler import get_q_compiler

        compiler = get_q_compiler()
        if _REGISTRY is None:
            _REGISTRY = QPhysicalImplementationRegistry(
                lowerings=compiler.executable_lowerings(),
                declared_targets=compiler.declared_targets(),
            )
    return _REGISTRY
