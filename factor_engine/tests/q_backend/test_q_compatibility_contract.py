from __future__ import annotations

import hashlib

import pytest

import backend.q_backend as q_backend
from backend.q_backend.q_backend import get_q_backend
from backend.q_backend.q_compiler import QCompiler, get_q_compiler
from backend.q_backend.q_capability import QBackendCapability, QCapabilityLevel
from backend.q_backend.q_physical_implementation_registry import (
    QEvidenceArtifact,
    QEvidenceValidationContext,
    QPhysicalImplementation,
    QPhysicalImplementationRegistry,
    compute_q_implementation_hash,
    get_q_physical_implementation_registry,
    install_q_physical_implementation_registry,
)


def test_package_exports_public_backend_contract() -> None:
    expected = {
        "QBackend",
        "get_q_backend",
        "QBackendCapability",
        "QProcessManager",
        "QTypeAdapter",
        "QExecutor",
        "QCompiler",
        "QRegionPlan",
        "get_q_compiler",
        "check_q_availability",
        "is_q_available",
    }
    assert expected <= set(q_backend.__all__)
    assert all(hasattr(q_backend, name) for name in expected)


def test_rank_specialized_lowering_precedes_generic_dispatch() -> None:
    compiler = get_q_compiler()
    assert compiler.compile_operator("rank", ["x"]) == "rank x"
    assert compiler.compile_operator("cs_rank", ["x"]) == "({(iasc iasc x)%count x})[x]"
    assert compiler.compile_operator("divide", ["x", "y"]) == "x % y"


def test_research_and_production_backend_singletons_are_isolated() -> None:
    production = get_q_backend(production_mode=True, fallback_to_pandas=False)
    research = get_q_backend(production_mode=False, fallback_to_pandas=True)

    assert production is get_q_backend(production_mode=True, fallback_to_pandas=False)
    assert research is get_q_backend(production_mode=False, fallback_to_pandas=True)
    assert production is not research


def test_compiler_instances_own_their_lowering_authority() -> None:
    first = QCompiler()
    second = QCompiler()
    global_compiler = get_q_compiler()

    assert first.capability.registry is not second.capability.registry
    assert first.capability.registry is not global_compiler.capability.registry
    assert first.capability.registry.get_with_lowering() == frozenset(
        first.executable_lowerings()
    )
    assert second.capability.registry.get_with_lowering() == frozenset(
        second.executable_lowerings()
    )


def test_custom_compiler_registry_does_not_cross_contaminate() -> None:
    custom_registry = QPhysicalImplementationRegistry(
        lowerings={"custom_only": "q_custom"},
        declared_targets=frozenset({"custom_only"}),
    )
    custom = QCompiler(QBackendCapability(custom_registry))
    ordinary = QCompiler()

    assert custom.capability.registry is custom_registry
    assert custom.capability.registry.has_lowering("custom_only")
    assert not ordinary.capability.registry.has_lowering("custom_only")
    assert not get_q_compiler().capability.registry.has_lowering("custom_only")


def test_unsupported_quantiles_fail_closed_before_parameter_validation() -> None:
    compiler = get_q_compiler()
    for op_name in ("ts_quantile", "cs_quantile"):
        with pytest.raises(ValueError, match="not supported"):
            compiler.compile_operator(op_name, ["x"], {"window": 5, "q": 0.5})

    for op_name, params in (
        ("clip", {"lower": None, "upper": 1}),
        ("fillna", {"fill_value": "x"}),
    ):
        with pytest.raises(ValueError):
            compiler.compile_operator(op_name, ["x"], params)


def test_financial_row_order_lowerings_are_rejected() -> None:
    compiler = get_q_compiler()
    for op_name in ("fin_lag", "fin_delta"):
        with pytest.raises(ValueError, match="not supported"):
            compiler.compile_operator(op_name, ["x"], {"periods": 1})


def test_explicit_registry_installed_before_compiler_bootstrap_is_shared(monkeypatch) -> None:
    import backend.q_backend.q_capability as capability_module
    import backend.q_backend.q_compiler as compiler_module
    import backend.q_backend.q_physical_implementation_registry as registry_module

    monkeypatch.setattr(compiler_module, "_COMPILER", None)
    monkeypatch.setattr(capability_module, "_CAPABILITY_REGISTRY", None)
    monkeypatch.setattr(registry_module, "_REGISTRY", None)
    explicit = QPhysicalImplementationRegistry(
        lowerings={"evidence_only": "q_evidence"},
        declared_targets=frozenset({"evidence_only"}),
    )
    install_q_physical_implementation_registry(explicit)

    compiler = compiler_module.get_q_compiler()
    evidence_registry = get_q_physical_implementation_registry()
    capability = capability_module.get_q_capability()

    assert compiler.capability.registry is explicit
    assert evidence_registry is explicit
    assert capability.registry is explicit
    assert explicit.has_lowering("evidence_only")


def test_registry_replacement_after_compiler_bootstrap_is_rejected(monkeypatch) -> None:
    import backend.q_backend.q_compiler as compiler_module
    import backend.q_backend.q_physical_implementation_registry as registry_module

    monkeypatch.setattr(compiler_module, "_COMPILER", None)
    monkeypatch.setattr(registry_module, "_REGISTRY", None)
    compiler = compiler_module.get_q_compiler()
    replacement = QPhysicalImplementationRegistry(
        lowerings={"replacement": "q_replacement"},
        declared_targets=frozenset({"replacement"}),
    )
    monkeypatch.setattr(registry_module, "_REGISTRY", replacement)

    with pytest.raises(RuntimeError, match="replaced after compiler bootstrap"):
        compiler_module.get_q_compiler()
    assert compiler.capability.registry is not replacement


def _certified_compiler_with_global_disagreement(tmp_path) -> QCompiler:
    import json

    sha = "a" * 40
    lowering_source = "q_certified"
    implementation_hash = compute_q_implementation_hash(
        "certified", "q_certified", lowering_source, "domain"
    )
    payload = {
        "canonical": "certified",
        "git_sha": sha,
        "implementation_hash": implementation_hash,
        "parameter_domain_id": "domain",
        "q_version": "4.1",
        "pykx_version": "2.6",
        "status": "PASS",
    }
    stage_results = {
        "compile": {
            "compiled": True,
            "compiled_cases": 1,
            "failure_count": 0,
        },
        "runtime": {
            "executed": True,
            "executed_cases": 1,
            "failure_count": 0,
        },
        "parity": {
            "reference_backend": "pandas",
            "compared_cases": 1,
            "mismatch_count": 0,
            "max_abs_error": 0.0,
            "tolerance": 0.0,
        },
        "null_semantics": {
            "checked_cases": 1,
            "checks": ["null-mask", "warmup", "all-null-window"],
            "mismatch_count": 0,
        },
    }
    artifacts = {}
    for name, result in stage_results.items():
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    **payload,
                    "stage": name,
                    "evidence_type": f"q_{name}_evidence/v1",
                    "result": result,
                }
            ),
            encoding="utf-8",
        )
        artifacts[name] = QEvidenceArtifact(
            path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    certified = QPhysicalImplementation(
        canonical="certified",
        lowering_id="q_certified",
        lowering_source=lowering_source,
        compile_evidence=artifacts["compile"],
        runtime_evidence=artifacts["runtime"],
        parity_evidence=artifacts["parity"],
        null_semantics_evidence=artifacts["null_semantics"],
        parameter_domain_id="domain",
        git_sha=sha,
        generation_timestamp="2026-08-17T10:00:00+00:00",
        implementation_hash=implementation_hash,
        q_version_range=">=4",
        pykx_version_range=">=2",
    )
    registry = QPhysicalImplementationRegistry(
        lowerings={"certified": lowering_source},
        declared_targets=frozenset({"certified", "unrelated_gap"}),
        validation_context=QEvidenceValidationContext(
            artifact_root=tmp_path,
            q_version="4.1",
            pykx_version="2.6",
            current_git_sha_provider=lambda: sha,
        ),
    )
    registry.register(certified)
    compiler = QCompiler(QBackendCapability(registry))
    compiler._operator_map = {"certified": lowering_source}
    return compiler


def test_operator_certification_is_independent_of_unrelated_declaration_gaps(tmp_path) -> None:
    compiler = _certified_compiler_with_global_disagreement(tmp_path)
    registry = compiler.capability.registry

    assert registry.is_production_certified("certified") is True
    assert registry.has_disagreement() is True
    assert registry.get_production_ready() == frozenset()


def test_compiler_production_admission_requires_global_production_membership(tmp_path) -> None:
    compiler = _certified_compiler_with_global_disagreement(tmp_path)
    nodes = [{"id": "n1", "operator": "certified", "inputs": ["x"]}]

    assert compiler.capability.registry.is_production_certified("certified") is True
    assert compiler.capability.get_capability("certified", mode="production") is QCapabilityLevel.UNSUPPORTED
    assert compiler.is_production_certified("certified") is False
    assert compiler.validate_region(nodes, mode="production") == (False, ["certified"])


def test_physical_region_admission_is_fail_closed_in_production() -> None:
    compiler = get_q_compiler()
    nodes = [{"id": "n1", "operator": "add", "inputs": ["x", "y"]}]

    research_valid, _ = compiler.validate_region(nodes, mode="research")
    production_valid, unsupported = compiler.validate_region(nodes, mode="production")

    assert research_valid is True
    assert production_valid is False
    assert unsupported == ["add"]
    with pytest.raises(ValueError, match="production mode"):
        compiler.compile_region(
            "r1",
            nodes,
            input_tables=["input"],
            output_name="result",
        )
