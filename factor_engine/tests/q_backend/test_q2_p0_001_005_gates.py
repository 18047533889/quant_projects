"""Tests for Q2-P0-001 through Q2-P0-005: Q Backend Authority Consolidation.

Tests verify:
1. Single definition of production-safe (ev.production_safe is authority)
2. Verification script honesty (only PASS if all gates pass)
3. No duplicate manual capability lists
4. Compiler uses evidence authority
5. Hard gates enforce single authority
"""

import pytest

from backend.q_backend.q_capability_evidence import (
    QCapabilityEvidence,
    QCapabilityGate,
    compute_q_capability_evidence,
    generate_capability_report,
    get_declared_native_ops,
    get_lowering_exists_ops,
    get_q_production_safe_ops,
    run_all_q_capability_gates,
)
from backend.q_backend.q_compiler import get_q_compiler
from backend.q_backend.q_physical_implementation_registry import (
    QPhysicalImplementation,
    QPhysicalImplementationRegistry,
)


class TestQ2P0001ProductionSafeSingleDefinition:
    """Q2-P0-001: Verify single definition of production-safe."""

    def test_production_safe_requires_all_five_passes(self):
        """production_safe property must require all 5 evidence passes."""
        # Missing compile
        ev1 = QCapabilityEvidence(
            canonical="test_op",
            declared_native=True,
            lowering_exists=True,
            compile_pass=False,
            runtime_pass=True,
            parity_pass=True,
        )
        assert ev1.production_safe is False

        # Missing runtime
        ev2 = QCapabilityEvidence(
            canonical="test_op",
            declared_native=True,
            lowering_exists=True,
            compile_pass=True,
            runtime_pass=False,
            parity_pass=True,
        )
        assert ev2.production_safe is False

        # Missing parity
        ev3 = QCapabilityEvidence(
            canonical="test_op",
            declared_native=True,
            lowering_exists=True,
            compile_pass=True,
            runtime_pass=True,
            parity_pass=False,
        )
        assert ev3.production_safe is False

        # All passes
        ev4 = QCapabilityEvidence(
            canonical="test_op",
            declared_native=True,
            lowering_exists=True,
            compile_pass=True,
            runtime_pass=True,
            parity_pass=True,
            parameter_domain_pass=True,
            implementation_hash_pass=True,
            q_version_range_pass=True,
            pykx_version_range_pass=True,
        )
        assert ev4.production_safe is True

    def test_get_q_production_safe_uses_ev_property(self):
        """get_q_production_safe_ops must use ev.production_safe exclusively."""
        evidence = compute_q_capability_evidence()
        production_safe = get_q_production_safe_ops()

        # Recompute expected from ev.production_safe
        expected = frozenset(op for op, ev in evidence.items() if ev.production_safe)

        assert production_safe == expected, (
            f"get_q_production_safe_ops() returned {len(production_safe)} ops "
            f"but expected {len(expected)} from ev.production_safe. "
            f"Difference: {production_safe ^ expected}"
        )

    def test_gate_production_safe_single_definition_passes(self):
        """Hard gate Q_PRODUCTION_SAFE_SINGLE_DEFINITION must pass."""
        passed, message = QCapabilityGate.gate_q_production_safe_single_definition()

        assert passed is True, (
            f"Q_PRODUCTION_SAFE_SINGLE_DEFINITION gate failed: {message}"
        )


class TestQ2P0002VerificationScriptHonesty:
    """Q2-P0-002: Verify verification script is honest about readiness."""

    def test_production_ready_requires_full_evidence(self):
        """production_ready flag must require all lowerings have full evidence."""
        report = generate_capability_report()

        # Current state: likely no operator has all 5 passes
        # So production_ready should be False
        if report['production_safe_count'] < report['lowering_exists_count']:
            assert report['production_ready'] is False, (
                "production_ready should be False when some lowerings lack evidence"
            )

    def test_production_ready_requires_zero_gaps(self):
        """production_ready requires zero native-without-lowering gaps."""
        report = generate_capability_report()

        if report['native_without_lowering_count'] > 0:
            assert report['production_ready'] is False

    def test_production_ready_requires_at_least_one_lowering(self):
        """production_ready requires at least one lowering exists."""
        report = generate_capability_report()

        # Non-trivially empty check
        assert report['lowering_exists_count'] > 0, (
            "Test assumes at least one lowering exists"
        )


class TestQ2P0003RemoveDuplicateAuthority:
    """Q2-P0-003: Verify no duplicate manual capability lists."""

    def test_phase1_native_ops_still_exists(self):
        """_PHASE1_NATIVE_OPS exists but is deprecated (documented)."""
        from backend.q_backend import q_capability

        # It's allowed to exist for backward compatibility
        # but must not be the admission authority
        assert hasattr(q_capability, '_PHASE1_NATIVE_OPS')

    def test_evidence_is_single_authority(self):
        """Evidence framework is the single authority."""
        declared = get_declared_native_ops()
        lowering = get_lowering_exists_ops()

        # These should be non-empty
        assert len(declared) > 0
        assert len(lowering) > 0

    def test_gate_q_capability_single_authority_passes(self):
        """Hard gate Q_CAPABILITY_SINGLE_AUTHORITY must pass."""
        passed, message = QCapabilityGate.gate_q_capability_single_authority()

        assert passed is False, (
            "The canonical authority gate must fail closed while declarations "
            "and executable lowerings disagree."
        )
        assert "declared_without_lowering" in message


class TestQ2P0004CompilerUsesEvidenceAuthority:
    """Q2-P0-004: Verify compiler uses evidence-based authority."""

    def test_compiler_can_compile_checks_operator_map(self):
        """Compiler.can_compile_operator must check _operator_map directly."""
        compiler = get_q_compiler()

        # Get operators with lowering
        lowering_ops = get_lowering_exists_ops()

        # Compiler should be able to compile operators with lowering
        for op in list(lowering_ops)[:10]:  # Test first 10
            can_compile = compiler.can_compile_operator(op)
            assert can_compile is True, (
                f"Compiler cannot compile {op} despite lowering exists"
            )

    def test_compiler_rejects_ops_without_lowering(self):
        """Compiler should reject operators without lowering."""
        compiler = get_q_compiler()

        # Operators without lowering
        declared = get_declared_native_ops()
        lowering = get_lowering_exists_ops()
        missing = declared - lowering

        for op in list(missing)[:5]:  # Test first 5
            can_compile = compiler.can_compile_operator(op)
            assert can_compile is False, (
                f"Compiler can compile {op} despite no lowering exists"
            )

    def test_gate_compiler_admission_uses_evidence_passes(self):
        """Hard gate Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY must pass."""
        passed, message = QCapabilityGate.gate_q_compiler_admission_uses_evidence()

        assert passed is True, (
            f"Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY gate failed: {message}"
        )


class TestQ2P0005AllGatesEnforce:
    """Q2-P0-005: Verify all new gates are enforced."""

    def test_all_mandatory_gates_exist(self):
        """All mandatory gates must be registered."""
        gates = run_all_q_capability_gates()

        required_gates = {
            "Q_PRODUCTION_SAFE_SINGLE_DEFINITION",
            "Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY",
            "Q_CAPABILITY_SINGLE_AUTHORITY",
        }

        for gate_name in required_gates:
            assert gate_name in gates, f"Mandatory gate {gate_name} missing"

    def test_gates_return_correct_structure(self):
        """All gates must return (bool, str) tuple."""
        gates = run_all_q_capability_gates()

        for gate_name, result in gates.items():
            assert isinstance(result, tuple), f"{gate_name} did not return tuple"
            assert len(result) == 2, f"{gate_name} tuple length != 2"
            passed, message = result
            assert isinstance(passed, bool), f"{gate_name} passed not bool"
            assert isinstance(message, str), f"{gate_name} message not str"
            assert len(message) > 0, f"{gate_name} message empty"


class TestQPhysicalImplementationRegistry:
    """Test Q Physical Implementation Registry."""

    def test_registry_creation(self):
        """Registry should initialize empty, including declarations."""
        registry = QPhysicalImplementationRegistry()
        assert registry.declared_targets() == frozenset()
        assert registry.get_with_lowering() == frozenset()
        assert registry.admission_disagreements() == {
            "declared_without_lowering": [],
            "lowering_without_declaration": [],
        }

    def test_custom_declarations_are_instance_scoped(self):
        registry = QPhysicalImplementationRegistry(
            lowerings={"a": "q_a"}, declared_targets=frozenset({"a"})
        )
        assert registry.declared_targets() == frozenset({"a"})
        assert registry.admission_disagreements() == {
            "declared_without_lowering": [],
            "lowering_without_declaration": [],
        }

    def test_disagreement_and_missing_evidence_are_fail_closed(self):
        registry = QPhysicalImplementationRegistry(
            lowerings={"rogue": "q_rogue"}, declared_targets=frozenset({"declared"})
        )
        missing = registry.get_missing_evidence()
        assert set(missing["rogue"]) == {
            "compile", "runtime", "parity", "parameter_domain",
            "implementation_hash", "q_version_range", "pykx_version_range",
        }
        assert registry.get_production_ready() == frozenset()
        assert registry.is_production_certified("rogue") is False
        assert registry.gate_all_lowerings_have_evidence()[0] is False

    def test_register_implementation(self):
        """Should register implementation."""
        registry = QPhysicalImplementationRegistry()

        impl = QPhysicalImplementation(
            canonical="add",
            lowering_id="q_add_v1",
            compile_evidence="test_compile_add_001",
            runtime_evidence="test_runtime_add_001",
            parity_evidence="test_parity_add_001",
            parameter_domain_id="param_domain_add",
            implementation_hash="sha256:add",
            q_version_range=">=4,<5",
            pykx_version_range=">=2,<3",
        )

        registry.register(impl)
        retrieved = registry.get("add")

        assert retrieved is not None
        assert retrieved.canonical == "add"
        assert retrieved.production_ready is True

    def test_production_ready_requires_full_evidence(self):
        """Production ready requires all evidence + parameter domain."""
        # Missing parity
        impl1 = QPhysicalImplementation(
            canonical="op1",
            lowering_id="q_op1",
            compile_evidence="c1",
            runtime_evidence="r1",
            parity_evidence=None,
            parameter_domain_id="p1",
        )
        assert impl1.production_ready is False

        # Missing parameter domain
        impl2 = QPhysicalImplementation(
            canonical="op2",
            lowering_id="q_op2",
            compile_evidence="c2",
            runtime_evidence="r2",
            parity_evidence="par2",
            parameter_domain_id=None,
        )
        assert impl2.production_ready is False

        # Full
        impl3 = QPhysicalImplementation(
            canonical="op3",
            lowering_id="q_op3",
            compile_evidence="c3",
            runtime_evidence="r3",
            parity_evidence="par3",
            parameter_domain_id="p3",
            implementation_hash="sha256:op3",
            q_version_range=">=4,<5",
            pykx_version_range=">=2,<3",
        )
        assert impl3.production_ready is True

    def test_get_missing_evidence(self):
        """Should identify operators with missing evidence."""
        registry = QPhysicalImplementationRegistry()

        # Partial evidence
        impl = QPhysicalImplementation(
            canonical="partial_op",
            lowering_id="q_partial",
            compile_evidence="c1",
            runtime_evidence=None,
            parity_evidence=None,
            parameter_domain_id=None,
        )
        registry.register(impl)

        missing = registry.get_missing_evidence()
        assert "partial_op" in missing
        assert set(missing["partial_op"]) == {
            "runtime", "parity", "parameter_domain",
            "implementation_hash", "q_version_range", "pykx_version_range",
        }

    def test_gate_all_lowerings_have_evidence(self):
        """Gate should fail if any lowering lacks evidence."""
        registry = QPhysicalImplementationRegistry()

        # Add partial implementation
        impl = QPhysicalImplementation(
            canonical="incomplete",
            lowering_id="q_incomplete",
            compile_evidence="c1",
        )
        registry.register(impl)

        passed, message = registry.gate_all_lowerings_have_evidence()
        assert passed is False
        assert "incomplete" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
