from __future__ import annotations

import pytest

import backend.q_backend as q_backend
from backend.q_backend.q_backend import get_q_backend
from backend.q_backend.q_compiler import get_q_compiler


def test_package_exports_public_backend_contract() -> None:
    expected = {
        "QBackend",
        "get_q_backend",
        "QBackendCapability",
        "QProcessManager",
        "QTypeAdapter",
        "QExecutor",
        "QCompiler",
        "check_q_availability",
        "is_q_available",
    }
    assert expected <= set(q_backend.__all__)
    assert all(hasattr(q_backend, name) for name in expected)


def test_rank_specialized_lowering_precedes_generic_dispatch() -> None:
    compiler = get_q_compiler()
    assert compiler.compile_operator("rank", ["x"]) == "rank x"
    assert compiler.compile_operator("cs_rank", ["x"]) == "{iasc iasc x}"
    assert compiler.compile_operator("divide", ["x", "y"]) == "x / y"


def test_research_and_production_backend_singletons_are_isolated() -> None:
    production = get_q_backend(production_mode=True, fallback_to_pandas=False)
    research = get_q_backend(production_mode=False, fallback_to_pandas=True)

    assert production is get_q_backend(production_mode=True, fallback_to_pandas=False)
    assert research is get_q_backend(production_mode=False, fallback_to_pandas=True)
    assert production is not research


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
