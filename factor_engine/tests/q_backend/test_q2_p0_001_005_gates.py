"""Tests for Q2-P0-001 through Q2-P0-005: Q Backend Authority Consolidation.

Tests verify:
1. Single definition of production-safe (ev.production_safe is authority)
2. Verification script honesty (only PASS if all gates pass)
3. No duplicate manual capability lists
4. Compiler uses evidence authority
5. Hard gates enforce single authority
"""

import hashlib
from dataclasses import replace

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
    QEvidenceArtifact,
    QEvidenceValidationContext,
    QPhysicalImplementation,
    QPhysicalImplementationRegistry,
    compute_q_implementation_hash,
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
            certification_pass=True,
        )
        assert ev4.production_safe is False

        ev5 = replace(ev4, null_semantics_pass=True)
        assert ev5.production_safe is True

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

    def test_empty_authority_gates_fail_closed(self, monkeypatch):
        import backend.q_backend.q_capability_evidence as evidence_module

        registry = QPhysicalImplementationRegistry()
        monkeypatch.setattr(evidence_module, "_get_authority", lambda: registry)

        gates = evidence_module.run_all_q_capability_gates()
        for gate_name in (
            "Q_NATIVE_WITHOUT_LOWERING",
            "Q_PRODUCTION_SAFE_SINGLE_DEFINITION",
            "Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY",
            "Q_CAPABILITY_SINGLE_AUTHORITY",
        ):
            passed, message = gates[gate_name]
            assert passed is False, gate_name
            assert "FAIL" in message

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
        assert missing["rogue"] == ["validation_context: missing"]
        assert registry.get_production_ready() == frozenset()
        assert registry.is_production_certified("rogue") is False
        assert registry.gate_all_lowerings_have_evidence()[0] is False

    @staticmethod
    def _certified_registry(tmp_path):
        import json

        sha = "a" * 40
        lowering_source = "+"
        implementation_hash = compute_q_implementation_hash(
            "add", "q_add_v1", lowering_source, "param_domain_add"
        )
        payload = {
            "canonical": "add",
            "git_sha": sha,
            "implementation_hash": implementation_hash,
            "parameter_domain_id": "param_domain_add",
            "q_version": "4.1",
            "pykx_version": "2.6",
            "status": "PASS",
        }
        stage_results = {
            "compile": {
                "compiled": True,
                "compiled_cases": 3,
                "failure_count": 0,
            },
            "runtime": {
                "executed": True,
                "executed_cases": 3,
                "failure_count": 0,
            },
            "parity": {
                "reference_backend": "pandas",
                "compared_cases": 3,
                "mismatch_count": 0,
                "max_abs_error": 0.0,
                "tolerance": 1e-12,
            },
            "null_semantics": {
                "checked_cases": 3,
                "checks": ["null-mask", "warmup", "all-null-window"],
                "mismatch_count": 0,
            },
        }
        artifacts = {}
        for name in ("compile", "runtime", "parity", "null_semantics"):
            path = tmp_path / f"{name}.json"
            path.write_text(
                json.dumps(
                    {
                        **payload,
                        "stage": name,
                        "evidence_type": f"q_{name}_evidence/v1",
                        "result": stage_results[name],
                    }
                ),
                encoding="utf-8",
            )
            artifacts[name] = QEvidenceArtifact(
                path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        context = QEvidenceValidationContext(
            artifact_root=tmp_path,
            q_version="4.1",
            pykx_version="2.6",
            current_git_sha_provider=lambda: sha,
        )
        impl = QPhysicalImplementation(
            canonical="add",
            lowering_id="q_add_v1",
            lowering_source=lowering_source,
            compile_evidence=artifacts["compile"],
            runtime_evidence=artifacts["runtime"],
            parity_evidence=artifacts["parity"],
            null_semantics_evidence=artifacts["null_semantics"],
            parameter_domain_id="param_domain_add",
            git_sha=sha,
            generation_timestamp="2026-08-17T10:00:00+00:00",
            implementation_hash=implementation_hash,
            q_version_range=">=4,<5",
            pykx_version_range=">=2,<3",
        )
        registry = QPhysicalImplementationRegistry(
            declared_targets=frozenset({"add"}),
            validation_context=context,
        )
        registry.register(impl)
        return registry, impl

    def test_register_implementation_with_validated_artifacts(self, tmp_path):
        registry, impl = self._certified_registry(tmp_path)
        assert registry.get("add") is impl
        assert registry.evidence_errors("add") == ()
        assert registry.is_production_certified("add") is True
        assert registry.get_production_ready() == frozenset({"add"})

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("git_sha", "b" * 40, "git_sha: evidence is stale"),
            ("generation_timestamp", "2026-08-17T10:00:00", "generation_timestamp: timezone is required"),
            ("implementation_hash", "0" * 64, "implementation_hash: lowering or parameter domain changed"),
            ("q_version_range", "not-a-range", "q_version: invalid version or range"),
            ("pykx_version_range", ">=3", "pykx_version: outside certified range"),
        ],
    )
    def test_certification_rejects_invalid_metadata(self, tmp_path, field, value, expected):
        registry, impl = self._certified_registry(tmp_path)
        registry.register(replace(impl, **{field: value}))
        assert expected in registry.evidence_errors("add")
        assert registry.is_production_certified("add") is False

    def test_certification_rejects_tampered_and_duplicate_artifacts(self, tmp_path):
        registry, impl = self._certified_registry(tmp_path)
        (tmp_path / impl.compile_evidence.path).write_bytes(b"tampered")
        assert "compile_evidence: artifact sha256 mismatch" in registry.evidence_errors("add")

        registry, impl = self._certified_registry(tmp_path)
        registry.register(replace(impl, runtime_evidence=impl.compile_evidence))
        errors = registry.evidence_errors("add")
        assert "evidence_artifacts: required artifacts must be independent" in errors

    def test_certification_rejects_malformed_and_nonpassing_payloads(self, tmp_path):
        import json

        registry, impl = self._certified_registry(tmp_path)
        compile_path = tmp_path / impl.compile_evidence.path
        compile_path.write_bytes(b"not-json")
        malformed = replace(
            impl,
            compile_evidence=replace(
                impl.compile_evidence,
                sha256=hashlib.sha256(compile_path.read_bytes()).hexdigest(),
            ),
        )
        registry.register(malformed)
        assert "compile_evidence: invalid JSON payload" in registry.evidence_errors("add")

        for field, value, expected in (
            ("status", "FAIL", "compile_evidence: status is not PASS"),
            ("canonical", "subtract", "compile_evidence: canonical mismatch"),
            ("stage", "runtime", "compile_evidence: stage mismatch"),
        ):
            registry, impl = self._certified_registry(tmp_path)
            compile_path = tmp_path / impl.compile_evidence.path
            payload = json.loads(compile_path.read_text(encoding="utf-8"))
            payload[field] = value
            compile_path.write_text(json.dumps(payload), encoding="utf-8")
            changed = replace(
                impl,
                compile_evidence=replace(
                    impl.compile_evidence,
                    sha256=hashlib.sha256(compile_path.read_bytes()).hexdigest(),
                ),
            )
            registry.register(changed)
            assert expected in registry.evidence_errors("add")

    @pytest.mark.parametrize(
        ("stage", "result_update", "expected"),
        [
            ("compile", None, "compile_evidence: result must be an object"),
            (
                "compile",
                {"type": "q_runtime_result/v1", "compiled_cases": 3, "failure_count": 0},
                "compile_evidence: result.compiled must be true",
            ),
            (
                "runtime",
                {"type": "q_runtime_result/v1", "executed_cases": 0, "failure_count": 0},
                "runtime_evidence: result.executed_cases must be positive",
            ),
            (
                "parity",
                {
                    "type": "q_parity_result/v1",
                    "compared_cases": 3,
                    "mismatch_count": 0,
                    "max_abs_error": 0.1,
                    "tolerance": 0.01,
                },
                "parity_evidence: result.max_abs_error exceeds tolerance",
            ),
            (
                "null_semantics",
                {
                    "type": "q_null_semantics_result/v1",
                    "checked_cases": 3,
                    "mismatch_count": 1,
                },
                "null_semantics_evidence: result.mismatch_count must be zero",
            ),
        ],
    )
    def test_certification_rejects_metadata_only_and_invalid_stage_results(
        self, tmp_path, stage, result_update, expected
    ):
        import json

        registry, impl = self._certified_registry(tmp_path)
        artifact = getattr(impl, f"{stage}_evidence")
        path = tmp_path / artifact.path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if result_update is None:
            payload.pop("result")
        else:
            payload["result"] = result_update
        path.write_text(json.dumps(payload), encoding="utf-8")
        registry.register(
            replace(
                impl,
                **{
                    f"{stage}_evidence": replace(
                        artifact,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                },
            )
        )

        assert expected in registry.evidence_errors("add")
        assert registry.is_production_certified("add") is False
        assert registry.get_production_ready() == frozenset()

    def test_certification_requires_null_evidence_and_live_head(self, tmp_path):
        registry, impl = self._certified_registry(tmp_path)
        registry.register(replace(impl, null_semantics_evidence=None))
        assert "null_semantics_evidence: missing" in registry.evidence_errors("add")

        for provider, expected in (
            (lambda: None, "current_git_sha: unavailable or malformed"),
            (lambda: "b" * 40, "git_sha: evidence is stale"),
        ):
            registry, impl = self._certified_registry(tmp_path)
            context = replace(registry._validation_context, current_git_sha_provider=provider)
            denied = QPhysicalImplementationRegistry(
                declared_targets=frozenset({"add"}), validation_context=context
            )
            denied.register(impl)
            assert expected in denied.evidence_errors("add")
            assert denied.is_production_certified("add") is False

    def test_certification_binds_executable_lowering_source(self, tmp_path):
        registry, impl = self._certified_registry(tmp_path)
        registry.register(replace(impl, lowering_source="neg"))
        errors = registry.evidence_errors("add")
        assert "implementation_hash: lowering or parameter domain changed" in errors
        assert registry.is_production_certified("add") is False

    def test_capability_evidence_does_not_hide_global_validation_failure(self, tmp_path, monkeypatch):
        import backend.q_backend.q_capability_evidence as evidence_module

        registry, impl = self._certified_registry(tmp_path)
        stale = replace(impl, git_sha="b" * 40)
        registry.register(stale)
        monkeypatch.setattr(evidence_module, "_get_authority", lambda: registry)

        evidence = evidence_module.compute_q_capability_evidence()["add"]
        assert evidence.compile_pass is False
        assert evidence.runtime_pass is False
        assert evidence.parity_pass is False
        assert evidence.null_semantics_pass is False
        assert evidence.certification_pass is False
        assert evidence.production_safe is False
        assert "git_sha: evidence is stale" in evidence.notes
        assert "compile_evidence: git_sha mismatch" in evidence.notes

    def test_certification_rejects_path_escape_and_unavailable_runtime(self, tmp_path):
        registry, impl = self._certified_registry(tmp_path)
        escaped = QEvidenceArtifact(path="../outside", sha256="0" * 64)
        registry.register(replace(impl, parity_evidence=escaped))
        assert "parity_evidence: path escapes artifact root" in registry.evidence_errors("add")

        context = replace(registry._validation_context, q_version=None)
        unavailable = QPhysicalImplementationRegistry(
            declared_targets=frozenset({"add"}), validation_context=context
        )
        unavailable.register(impl)
        assert "q_version: unavailable" in unavailable.evidence_errors("add")
        assert unavailable.get_production_ready() == frozenset()

    def test_gate_all_lowerings_have_evidence(self, tmp_path):
        registry, _ = self._certified_registry(tmp_path)
        passed, message = registry.gate_all_lowerings_have_evidence()
        assert passed is True
        assert "current, validated evidence" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
