from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.planner.canonicalize_params import validate_plan_params
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.operator_snapshot import build_runtime_operator_snapshot


M33_FREQUENCY = "ts_matrix_profile_motif_frequency"
M33_DISPERSION = "ts_matrix_profile_neighbor_dispersion"
M33_NOVELTY = "ts_matrix_profile_novelty"
M36_HSIC = "ts_residualized_hsic"


def _assert_stage_registry_source():
    import factor_engine.cleaned_operators.registry as registry_module

    expected = Path.cwd() / "factor_engine/cleaned_operators/registry.py"
    assert Path(registry_module.__file__).resolve() == expected.resolve()


def _row(snapshot, canonical):
    return next(row for row in snapshot["operators"] if row["canonical"] == canonical)


def _backend(row, backend):
    return next(binding for binding in row["backends"] if binding["backend"] == backend)


def _relations(binding):
    return {
        item["x-factor-engine-relation"]
        for item in binding["parameter_schema"]["allOf"]
    }


def _plan(canonical, **attrs):
    return PlanNode(op=canonical, attrs=attrs)


def test_m33_m36_relations_survive_catalog_and_public_agent_snapshot():
    _assert_stage_registry_source()
    ensure_cleaned_loaded()
    expected = {
        M33_FREQUENCY: {
            "subsequence_length <= history",
            "history <= window",
            "history >= subsequence_length + subsequence_length // 4",
            "history >= subsequence_length + subsequence_length // 4 + 2",
        },
        M33_DISPERSION: {
            "subsequence_length <= history",
            "history <= window",
            "history >= subsequence_length + subsequence_length // 4",
            "history >= subsequence_length + subsequence_length // 4 + 2",
        },
        M36_HSIC: {"purge_gap <= window // 2 - 6"},
    }
    snapshot = build_runtime_operator_snapshot(profile="runtime")
    for canonical, relations in expected.items():
        catalog = OperatorRegistry._catalog[canonical]
        assert {spec.expression for spec in catalog["relational_specs"]} == relations
        assert {
            spec["expression"] for spec in catalog["contract"]["relational_specs"]
        } == relations
        row = _row(snapshot, canonical)
        assert _relations(_backend(row, "pandas_numpy")) == relations
        assert _relations(_backend(row, "polars")) == relations


@pytest.mark.parametrize("canonical", [M33_FREQUENCY, M33_DISPERSION])
def test_m33_planner_enforces_three_candidate_support(canonical):
    common = {"window": 40, "subsequence_length": 20}
    for history in (25, 26):
        with pytest.raises(OperatorParameterError, match="requires at least three"):
            validate_plan_params(_plan(canonical, history=history, **common))
    validate_plan_params(_plan(canonical, history=27, **common))


def test_m33_novelty_keeps_one_candidate_boundary():
    common = {"window": 40, "subsequence_length": 20}
    validate_plan_params(_plan(M33_NOVELTY, history=25, **common))
    validate_plan_params(_plan(M33_NOVELTY, history=26, **common))


@pytest.mark.parametrize("window", [30, 31])
def test_m36_planner_enforces_even_and_odd_purge_boundaries(window):
    validate_plan_params(_plan(M36_HSIC, window=window, purge_gap=9))
    with pytest.raises(OperatorParameterError, match="leave at least six test rows"):
        validate_plan_params(_plan(M36_HSIC, window=window, purge_gap=10))


def test_public_snapshot_is_order_independent_when_adapter_registers_first():
    code = textwrap.dedent("""
        from pathlib import Path
        from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
        from factor_engine.cleaned_operators import registry as registry_module
        from factor_engine.cleaned_operators.base import OperatorMetadata, RelationalParamSpec, SeriesOperator
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.runtime.operator_snapshot import build_runtime_operator_snapshot
        registry_file = Path(registry_module.__file__).resolve()
        assert registry_file == (Path.cwd() / "factor_engine/cleaned_operators/registry.py").resolve()
        print(f"ADAPTER_FIRST_REGISTRY_FILE={registry_file}")
        ensure_cleaned_loaded()
        OperatorRegistry.thaw_for_bootstrap(registry_module._BOOTSTRAP_TOKEN)
        canonical = "_test_relation_adapter_first"
        class Adapter(SeriesOperator):
            metadata = OperatorMetadata(name=canonical, category="test", param_names=["x", "window"])
            def _calculate_series(self, x, window=2): return x
        class Reference(SeriesOperator):
            metadata = OperatorMetadata(
                name=canonical, category="test", param_names=["x", "window"],
                relational_specs=[RelationalParamSpec("window >= 2", "minimum")])
            def _calculate_series(self, x, window=2): return x
        OperatorRegistry.register(Adapter(), canonical=canonical, backend="polars")
        OperatorRegistry.register(Reference(), canonical=canonical, backend="pandas_numpy")
        row = next(r for r in build_runtime_operator_snapshot(profile="runtime")["operators"]
                   if r["canonical"] == canonical)
        expected = {"window >= 2"}
        for binding in row["backends"]:
            actual = {item["x-factor-engine-relation"]
                      for item in binding["parameter_schema"]["allOf"]}
            assert actual == expected, (binding["backend"], actual)
        print("ADAPTER_FIRST_OK")
    """)
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    print(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ADAPTER_FIRST_OK" in result.stdout
    assert "ADAPTER_FIRST_REGISTRY_FILE=" in result.stdout


def test_first_nonempty_relation_is_single_authority_on_conflict():
    code = textwrap.dedent("""
        import copy
        from pathlib import Path
        from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
        from factor_engine.cleaned_operators import registry as registry_module
        from factor_engine.cleaned_operators.base import OperatorMetadata, RelationalParamSpec, SeriesOperator
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        registry_file = Path(registry_module.__file__).resolve()
        assert registry_file == (Path.cwd() / "factor_engine/cleaned_operators/registry.py").resolve()
        print(f"CONFLICT_REGISTRY_FILE={registry_file}")
        ensure_cleaned_loaded()
        OperatorRegistry.thaw_for_bootstrap(registry_module._BOOTSTRAP_TOKEN)
        canonical = "_test_relation_conflict"
        def implementation(expression):
            class Implementation(SeriesOperator):
                metadata = OperatorMetadata(
                    name=canonical, category="test", param_names=["x", "window"],
                    relational_specs=[RelationalParamSpec(expression, expression)])
                def _calculate_series(self, x, window=2): return x
            return Implementation()
        OperatorRegistry.register(
            implementation("window >= 2"), canonical=canonical, backend="pandas_numpy")
        before_backends = tuple(OperatorRegistry._operators[canonical])
        before_get = OperatorRegistry.get(canonical, "polars")
        before_catalog = copy.deepcopy(OperatorRegistry._catalog[canonical])
        before_version = OperatorRegistry._version
        try:
            OperatorRegistry.register(
                implementation("window >= 3"), canonical=canonical, backend="polars")
        except ValueError as exc:
            assert "logical-contract divergence" in str(exc)
            assert tuple(OperatorRegistry._operators[canonical]) == before_backends
            assert OperatorRegistry.get(canonical, "polars") is before_get
            assert OperatorRegistry._catalog[canonical] == before_catalog
            assert OperatorRegistry._version == before_version
            print("CONFLICT_REJECTED")
        else:
            raise AssertionError("conflicting non-empty relation was accepted")
    """)
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    print(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONFLICT_REJECTED" in result.stdout
    assert "CONFLICT_REGISTRY_FILE=" in result.stdout
