"""Evidence and admission authority for q physical implementations."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
from backend.q_backend.q_physical_implementation_registry import (
    QPhysicalImplementationRegistry,
    get_q_physical_implementation_registry,
)


def _get_authority() -> QPhysicalImplementationRegistry:
    return get_q_physical_implementation_registry()

class QEvidencePass(str, Enum):
    DECLARED_NATIVE = "declared_native"
    LOWERING_EXISTS = "lowering_exists"
    COMPILE_PASS = "compile_pass"
    RUNTIME_PASS = "runtime_pass"
    PARITY_PASS = "parity_pass"

@dataclass(frozen=True)
class QCapabilityEvidence:
    canonical: str
    declared_native: bool
    lowering_exists: bool
    compile_pass: bool = False
    runtime_pass: bool = False
    parity_pass: bool = False
    parameter_domain_pass: bool = False
    implementation_hash_pass: bool = False
    q_version_range_pass: bool = False
    pykx_version_range_pass: bool = False
    notes: str = ""
    @property
    def production_safe(self) -> bool:
        return all((
            self.declared_native,
            self.lowering_exists,
            self.compile_pass,
            self.runtime_pass,
            self.parity_pass,
            self.parameter_domain_pass,
            self.implementation_hash_pass,
            self.q_version_range_pass,
            self.pykx_version_range_pass,
        ))
    @property
    def native_without_lowering(self) -> bool:
        return self.declared_native and not self.lowering_exists

def get_declared_native_ops() -> frozenset[str]:
    return _get_authority().declared_targets()

def get_lowering_exists_ops() -> frozenset[str]:
    return _get_authority().get_with_lowering()

def compute_q_capability_evidence() -> dict[str, QCapabilityEvidence]:
    registry = _get_authority()
    result = {}
    for op in sorted(set(registry.declared_targets()) | set(registry.get_with_lowering())):
        impl = registry.get(op)
        result[op] = QCapabilityEvidence(
            canonical=op,
            declared_native=op in registry.declared_targets(),
            lowering_exists=bool(impl and impl.lowering_id),
            compile_pass=bool(impl and impl.compile_evidence),
            runtime_pass=bool(impl and impl.runtime_evidence),
            parity_pass=bool(impl and impl.parity_evidence),
            parameter_domain_pass=bool(impl and impl.parameter_domain_id),
            implementation_hash_pass=bool(impl and impl.implementation_hash),
            q_version_range_pass=bool(impl and impl.q_version_range),
            pykx_version_range_pass=bool(impl and impl.pykx_version_range),
            notes="registry evidence",
        )
    return result

def get_q_native_without_lowering() -> frozenset[str]:
    return frozenset(get_declared_native_ops() - get_lowering_exists_ops())

def get_q_production_safe_ops() -> frozenset[str]:
    return _get_authority().get_production_ready()

def generate_capability_report() -> dict[str, Any]:
    registry = _get_authority()
    declared, lowering = get_declared_native_ops(), get_lowering_exists_ops()
    missing = get_q_native_without_lowering()
    safe = get_q_production_safe_ops()
    disagreements = registry.admission_disagreements()
    declared_without_lowering = disagreements["declared_without_lowering"]
    lowering_without_declaration = disagreements["lowering_without_declaration"]
    return {
        "declared_native_count": len(declared), "lowering_exists_count": len(lowering),
        "native_without_lowering_count": len(missing), "production_safe_count": len(safe),
        "native_without_lowering": sorted(missing),
        "declared_without_lowering": declared_without_lowering,
        "lowering_without_declaration": lowering_without_declaration,
        "missing_evidence": registry.get_missing_evidence(),
        "research_ready": bool(lowering) and not lowering_without_declaration,
        "production_ready": bool(lowering) and not declared_without_lowering and not lowering_without_declaration and safe == lowering,
    }

class QCapabilityGate:
    @staticmethod
    def gate_q_native_without_lowering():
        missing = get_q_native_without_lowering()
        return (not missing, "PASS: no declared native op lacks lowering" if not missing else f"FAIL: {len(missing)} declared native ops lack lowering: {sorted(missing)}")
    @staticmethod
    def gate_q_manual_authority_removed():
        # A compatibility declaration may exist, but must not be imported by admission code.
        import inspect
        from backend.q_backend import q_capability
        source = inspect.getsource(q_capability.QBackendCapability)
        ok = "_PHASE1_NATIVE_OPS" not in source
        return (ok, "PASS: declarations are not admission authority" if ok else "FAIL: legacy declaration used in admission")
    @staticmethod
    def gate_q_production_safe_single_definition():
        expected = frozenset(op for op, impl in get_q_physical_implementation_registry()._implementations.items() if impl.production_ready)
        actual = get_q_production_safe_ops()
        return (actual == expected, "PASS: registry production_ready is authoritative" if actual == expected else "FAIL: production admission disagreement")
    @staticmethod
    def gate_q_compiler_admission_uses_evidence():
        from backend.q_backend.q_compiler import get_q_compiler
        compiler = get_q_compiler()
        ok = all(compiler.has_lowering(op) == (op in get_lowering_exists_ops()) for op in set(get_lowering_exists_ops()) | set(get_declared_native_ops()))
        return (ok, "PASS: compiler lowering admission uses registry" if ok else "FAIL: compiler/registry disagreement")
    @staticmethod
    def gate_q_capability_single_authority():
        disagreements = _get_authority().admission_disagreements()
        declared_without_lowering = disagreements["declared_without_lowering"]
        lowering_without_declaration = disagreements["lowering_without_declaration"]
        ok = not declared_without_lowering and not lowering_without_declaration
        return (ok, "PASS: registry is canonical" if ok else f"FAIL: {disagreements}")

def run_all_q_capability_gates():
    return {"Q_NATIVE_WITHOUT_LOWERING": QCapabilityGate.gate_q_native_without_lowering(), "Q_MANUAL_AUTHORITY_REMOVED": QCapabilityGate.gate_q_manual_authority_removed(), "Q_PRODUCTION_SAFE_SINGLE_DEFINITION": QCapabilityGate.gate_q_production_safe_single_definition(), "Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY": QCapabilityGate.gate_q_compiler_admission_uses_evidence(), "Q_CAPABILITY_SINGLE_AUTHORITY": QCapabilityGate.gate_q_capability_single_authority()}
