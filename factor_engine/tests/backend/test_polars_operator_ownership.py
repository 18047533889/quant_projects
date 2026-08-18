"""Deterministic ownership checks for generated/native Polars surfaces."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).parents[2]
AUTO = ROOT / "cleaned_operators" / "auto_polars_all.py"
NATIVE = ROOT / "cleaned_operators" / "polars_native" / "ts_advanced_batch5.py"
GENERATOR = ROOT / "scripts" / "auto_generate_polars_bridges.py"


def _canonical_registrations(path: Path, canonicals: set[str]) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, list[str]] = {canonical: [] for canonical in canonicals}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Name) or decorator.func.id != "register_operator":
                continue
            values = {
                keyword.arg: keyword.value.value
                for keyword in decorator.keywords
                if keyword.arg in {"canonical", "backend"}
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            }
            canonical = values.get("canonical")
            if canonical in found and values.get("backend") == "polars":
                found[canonical].append(node.name)
    return found


def test_research_placeholders_have_one_canonical_polars_owner() -> None:
    canonicals = {"ts_markov_committor", "ts_local_lyapunov_exponent"}
    auto = _canonical_registrations(AUTO, canonicals)
    native = _canonical_registrations(NATIVE, canonicals)

    assert auto == {
        "ts_markov_committor": ["TsMarkovCommittorPolars"],
        "ts_local_lyapunov_exponent": ["TsLocalLyapunovExponentPolars"],
    }
    assert native == {canonical: [] for canonical in canonicals}


def test_km_equilibrium_distance_is_not_reintroduced_by_batch5() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    assert 'canonical="ts_km_equilibrium_distance"' not in source
    assert "class TsDeviationFromMean" not in source


def test_ts_corr_is_quarantined_from_generated_polars_surface() -> None:
    auto_source = AUTO.read_text(encoding="utf-8")
    assert 'canonical="ts_corr"' not in auto_source
    assert "class TsCorrPolars" not in auto_source

    generator_source = GENERATOR.read_text(encoding="utf-8")
    assert 'POLARS_NATIVE_CANONICALS = frozenset({"ts_corr"})' in generator_source
    tree = ast.parse(generator_source, filename=str(GENERATOR))
    quarantine = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "POLARS_NATIVE_CANONICALS"
            for target in node.targets
        )
    )
    assert isinstance(quarantine.value, ast.Call)
    assert isinstance(quarantine.value.func, ast.Name)
    assert quarantine.value.func.id == "frozenset"
    assert len(quarantine.value.args) == 1
    assert ast.literal_eval(quarantine.value.args[0]) == {"ts_corr"}


def test_generated_module_import_blocker_is_unrelated_to_ts_corr() -> None:
    try:
        importlib.import_module("cleaned_operators.auto_polars_all")
    except TypeError as exc:
        assert "abstract class AcfPolars" in str(exc)
    else:
        raise AssertionError("auto_polars_all unexpectedly imported despite abstract bridges")
