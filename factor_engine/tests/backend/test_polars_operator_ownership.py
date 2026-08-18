"""Deterministic ownership checks for generated/native Polars surfaces."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
AUTO = ROOT / "cleaned_operators" / "auto_polars_all.py"
NATIVE = ROOT / "cleaned_operators" / "polars_native" / "ts_advanced_batch5.py"


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
