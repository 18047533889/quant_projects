"""Tests for Q Backend Evidence-Based Capability Framework.

Q-P0-001: Verify evidence-based capability replaces manual authority
Q-P0-002: Verify Q_NATIVE_WITHOUT_LOWERING hard gate
"""

import pytest

from backend.q_backend.q_capability_evidence import (
    QCapabilityEvidence,
    QCapabilityGate,
    QEvidencePass,
    compute_q_capability_evidence,
    generate_capability_report,
    get_declared_native_ops,
    get_lowering_exists_ops,
    get_q_native_without_lowering,
    get_q_production_safe_ops,
    run_all_q_capability_gates,
)


class TestQCapabilityEvidence:
    """Test Q capability evidence framework."""

    def test_declared_native_ops_structure(self):
        """Declared native ops should be non-empty frozenset."""
        declared = get_declared_native_ops()
        assert isinstance(declared, frozenset)
        assert len(declared) > 0
        # Phase 1 target from README: 65+ operators
        assert len(declared) >= 65

    def test_lowering_exists_ops_structure(self):
        """Lowering exists should be non-empty frozenset."""
        lowering = get_lowering_exists_ops()
        assert isinstance(lowering, frozenset)
        assert len(lowering) > 0

    def test_lowering_is_subset_of_or_equal_declared(self):
        """All lowering implementations should be for declared native ops or less."""
        declared = get_declared_native_ops()
        lowering = get_lowering_exists_ops()

        # Lowering should not exceed declared (no rogue implementations)
        extra_lowerings = lowering - declared
        # Allow some extra since compiler might have internal operators
        # but main invariant: no production-safe op without declaration
        assert len(extra_lowerings) == 0, (
            f"Lowering implementations without native declaration: {extra_lowerings}"
        )

    def test_q_native_without_lowering_computation(self):
        """Q_NATIVE_WITHOUT_LOWERING should be declared - lowering."""
        declared = get_declared_native_ops()
        lowering = get_lowering_exists_ops()
        missing = get_q_native_without_lowering()

        assert missing == declared - lowering
        assert isinstance(missing, frozenset)

    def test_capability_evidence_structure(self):
        """Evidence computation should return complete records."""
        evidence = compute_q_capability_evidence()

        assert isinstance(evidence, dict)
        assert len(evidence) > 0

        # Check a known operator
        if "add" in evidence:
            ev = evidence["add"]
            assert isinstance(ev, QCapabilityEvidence)
            assert ev.canonical == "add"
            assert ev.declared_native is True
            assert ev.lowering_exists is True
            assert ev.native_without_lowering is False

    def test_capability_evidence_identifies_gaps(self):
        """Evidence should correctly identify native-without-lowering gaps."""
        evidence = compute_q_capability_evidence()
        missing = get_q_native_without_lowering()

        for op in missing:
            if op in evidence:
                ev = evidence[op]
                assert ev.native_without_lowering is True
                assert ev.declared_native is True
                assert ev.lowering_exists is False

    def test_production_safe_ops_excludes_gaps(self):
        """Production safe ops must have both declaration and lowering."""
        production_safe = get_q_production_safe_ops()
        missing = get_q_native_without_lowering()

        # No overlap between production safe and gaps
        assert production_safe & missing == frozenset()

        # All production safe ops have lowering
        lowering = get_lowering_exists_ops()
        assert production_safe.issubset(lowering)

    def test_generate_capability_report_structure(self):
        """Capability report should have required fields."""
        report = generate_capability_report()

        required_keys = {
            "declared_native_count",
            "lowering_exists_count",
            "native_without_lowering_count",
            "production_safe_count",
            "native_without_lowering",
            "production_ready",
        }
        assert set(report.keys()) == required_keys

        assert isinstance(report["declared_native_count"], int)
        assert isinstance(report["lowering_exists_count"], int)
        assert isinstance(report["native_without_lowering_count"], int)
        assert isinstance(report["production_safe_count"], int)
        assert isinstance(report["native_without_lowering"], list)
        assert isinstance(report["production_ready"], bool)

    def test_report_consistency(self):
        """Report counts should be consistent with actual sets."""
        report = generate_capability_report()
        declared = get_declared_native_ops()
        lowering = get_lowering_exists_ops()
        missing = get_q_native_without_lowering()

        assert report["declared_native_count"] == len(declared)
        assert report["lowering_exists_count"] == len(lowering)
        assert report["native_without_lowering_count"] == len(missing)
        assert len(report["native_without_lowering"]) == len(missing)

    def test_production_readiness_logic(self):
        """Production ready IFF Q_NATIVE_WITHOUT_LOWERING is empty."""
        report = generate_capability_report()
        missing_count = report["native_without_lowering_count"]

        if missing_count == 0:
            assert report["production_ready"] is True
        else:
            assert report["production_ready"] is False


class TestQCapabilityGates:
    """Test Q capability hard gates."""

    def test_gate_q_native_without_lowering_structure(self):
        """Gate should return (bool, str) tuple."""
        passed, message = QCapabilityGate.gate_q_native_without_lowering()

        assert isinstance(passed, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_gate_q_native_without_lowering_logic(self):
        """Gate should fail IFF Q_NATIVE_WITHOUT_LOWERING is non-empty."""
        missing = get_q_native_without_lowering()
        passed, message = QCapabilityGate.gate_q_native_without_lowering()

        if len(missing) == 0:
            assert passed is True
            assert "PASS" in message
        else:
            assert passed is False
            assert "FAIL" in message
            assert str(len(missing)) in message

    def test_gate_manual_authority_removed_structure(self):
        """Gate should return (bool, str) tuple."""
        passed, message = QCapabilityGate.gate_q_manual_authority_removed()

        assert isinstance(passed, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_run_all_gates_structure(self):
        """All gates should return dict of results."""
        results = run_all_q_capability_gates()

        assert isinstance(results, dict)
        assert "Q_NATIVE_WITHOUT_LOWERING" in results
        assert "Q_MANUAL_AUTHORITY_REMOVED" in results

        for gate_name, (passed, message) in results.items():
            assert isinstance(passed, bool)
            assert isinstance(message, str)


class TestQCapabilityEvidenceRecord:
    """Test QCapabilityEvidence dataclass."""

    def test_evidence_record_creation(self):
        """Should create evidence record with required fields."""
        ev = QCapabilityEvidence(
            canonical="add",
            declared_native=True,
            lowering_exists=True,
        )

        assert ev.canonical == "add"
        assert ev.declared_native is True
        assert ev.lowering_exists is True
        assert ev.compile_pass is False
        assert ev.runtime_pass is False
        assert ev.parity_pass is False

    def test_production_safe_requires_all_passes(self):
        """Production safe requires all 5 evidence passes."""
        # Missing some passes
        ev1 = QCapabilityEvidence(
            canonical="add",
            declared_native=True,
            lowering_exists=True,
            compile_pass=True,
            runtime_pass=False,
            parity_pass=False,
        )
        assert ev1.production_safe is False

        # All passes
        ev2 = QCapabilityEvidence(
            canonical="add",
            declared_native=True,
            lowering_exists=True,
            compile_pass=True,
            runtime_pass=True,
            parity_pass=True,
        )
        assert ev2.production_safe is True

    def test_native_without_lowering_detection(self):
        """Should detect native declaration without lowering."""
        ev1 = QCapabilityEvidence(
            canonical="missing_op",
            declared_native=True,
            lowering_exists=False,
        )
        assert ev1.native_without_lowering is True

        ev2 = QCapabilityEvidence(
            canonical="implemented_op",
            declared_native=True,
            lowering_exists=True,
        )
        assert ev2.native_without_lowering is False


class TestQEvidencePass:
    """Test evidence pass enum."""

    def test_evidence_pass_enum_values(self):
        """Should have all required pass types."""
        assert QEvidencePass.DECLARED_NATIVE.value == "declared_native"
        assert QEvidencePass.LOWERING_EXISTS.value == "lowering_exists"
        assert QEvidencePass.COMPILE_PASS.value == "compile_pass"
        assert QEvidencePass.RUNTIME_PASS.value == "runtime_pass"
        assert QEvidencePass.PARITY_PASS.value == "parity_pass"


class TestQCapabilityIntegration:
    """Integration tests for Q capability evidence."""

    def test_known_operators_have_evidence(self):
        """Known operators should have complete evidence records."""
        evidence = compute_q_capability_evidence()

        # Core arithmetic
        for op in ["add", "subtract", "multiply", "divide"]:
            assert op in evidence
            ev = evidence[op]
            assert ev.declared_native is True
            assert ev.lowering_exists is True
            assert ev.native_without_lowering is False

    def test_missing_lowering_operators_detected(self):
        """Operators missing lowering should be detected."""
        missing = get_q_native_without_lowering()

        # These are known to be declared but not implemented (from investigation)
        known_missing = {"vwap", "bfill", "group_count"}

        assert known_missing.issubset(missing), (
            f"Known missing operators not detected. "
            f"Missing: {missing}, Expected subset: {known_missing}"
        )

    def test_evidence_framework_is_comprehensive(self):
        """Evidence framework should cover all declared and implemented ops."""
        declared = get_declared_native_ops()
        lowering = get_lowering_exists_ops()
        evidence = compute_q_capability_evidence()

        all_ops = declared | lowering
        evidence_ops = set(evidence.keys())

        assert evidence_ops == all_ops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
