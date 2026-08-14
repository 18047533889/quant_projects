"""Q Physical Implementation Registry.

Q2-P0-006: Single source of truth for Q backend physical implementations.

This registry tracks the complete evidence chain for each operator:
- Canonical operator identity
- Lowering function ID
- Parameter domain certification
- Compile evidence (does it compile?)
- Runtime evidence (does it execute?)
- Parity evidence (does it match Pandas oracle?)
- Implementation hash for reproducibility
- Version ranges (q, pykx) for compatibility

This is the ultimate authority for production readiness decisions.
The old manual lists (_PHASE1_NATIVE_OPS) are deprecated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QPhysicalImplementation:
    """Physical implementation record for a Q operator.

    This is the evidence-backed proof that an operator can execute in Q production.
    """

    # Identity
    canonical: str  # Canonical operator name from OperatorSemanticRegistry
    lowering_id: str  # Lowering function identifier in q_compiler._operator_map

    # Parameter domain
    parameter_domain_id: str | None = None  # Reference to parameter certification

    # Evidence chain
    compile_evidence: str | None = None  # Compile test result reference
    runtime_evidence: str | None = None  # Runtime test result reference
    parity_evidence: str | None = None  # Parity test result reference

    # Reproducibility
    implementation_hash: str | None = None  # Hash of q code implementation
    q_version_range: str | None = None  # e.g., ">=4.0,<5.0"
    pykx_version_range: str | None = None  # e.g., ">=2.0,<3.0"

    # Metadata
    notes: str = ""

    @property
    def has_full_evidence(self) -> bool:
        """All three evidence passes completed."""
        return (
            self.compile_evidence is not None
            and self.runtime_evidence is not None
            and self.parity_evidence is not None
        )

    @property
    def production_ready(self) -> bool:
        """Ready for production: full evidence + parameter domain certified."""
        return self.has_full_evidence and self.parameter_domain_id is not None


class QPhysicalImplementationRegistry:
    """Registry of Q physical implementations.

    Single authority for Q backend production readiness.
    Replaces manual _PHASE1_NATIVE_OPS lists.
    """

    def __init__(self):
        self._implementations: dict[str, QPhysicalImplementation] = {}

    def register(self, impl: QPhysicalImplementation):
        """Register a physical implementation.

        Args:
            impl: Implementation record

        Raises:
            ValueError: If canonical already registered with different lowering
        """
        if impl.canonical in self._implementations:
            existing = self._implementations[impl.canonical]
            if existing.lowering_id != impl.lowering_id:
                raise ValueError(
                    f"Cannot register {impl.canonical}: already registered with "
                    f"lowering_id={existing.lowering_id}, attempted {impl.lowering_id}"
                )

        self._implementations[impl.canonical] = impl

    def get(self, canonical: str) -> QPhysicalImplementation | None:
        """Get implementation record by canonical name.

        Args:
            canonical: Canonical operator name

        Returns:
            Implementation record or None
        """
        return self._implementations.get(canonical)

    def get_production_ready(self) -> frozenset[str]:
        """Get operators ready for production.

        Returns:
            Frozenset of canonical names
        """
        return frozenset(
            name for name, impl in self._implementations.items()
            if impl.production_ready
        )

    def get_with_lowering(self) -> frozenset[str]:
        """Get operators with lowering (may not have full evidence).

        Returns:
            Frozenset of canonical names
        """
        return frozenset(self._implementations.keys())

    def get_missing_evidence(self) -> dict[str, list[str]]:
        """Get operators missing evidence passes.

        Returns:
            Mapping from canonical name to list of missing evidence types
        """
        missing = {}
        for name, impl in self._implementations.items():
            gaps = []
            if impl.compile_evidence is None:
                gaps.append("compile")
            if impl.runtime_evidence is None:
                gaps.append("runtime")
            if impl.parity_evidence is None:
                gaps.append("parity")
            if impl.parameter_domain_id is None:
                gaps.append("parameter_domain")

            if gaps:
                missing[name] = gaps

        return missing

    def gate_all_lowerings_have_evidence(self) -> tuple[bool, str]:
        """Hard gate: All lowerings must have full evidence.

        Returns:
            (passed, message)
        """
        missing = self.get_missing_evidence()
        if not missing:
            return True, "PASS: All lowerings have full evidence"

        return (
            False,
            f"FAIL: {len(missing)} lowerings lack evidence: "
            f"{list(missing.keys())[:5]}"
        )


# Global singleton
_REGISTRY: QPhysicalImplementationRegistry | None = None


def get_q_physical_implementation_registry() -> QPhysicalImplementationRegistry:
    """Get global Q physical implementation registry.

    Returns:
        Registry instance
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = QPhysicalImplementationRegistry()
        # TODO: Populate from evidence ledger
    return _REGISTRY
