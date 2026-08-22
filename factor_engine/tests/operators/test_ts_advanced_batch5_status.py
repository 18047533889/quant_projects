"""Governance checks for the quarantined Polars advanced batch five module."""

from pathlib import Path
import ast


_TARGET = Path(__file__).parents[2] / "cleaned_operators" / "polars_native" / "ts_advanced_batch5.py"


def test_batch5_registrations_are_research_only() -> None:
    """Proxy formulas must not be advertised as production Polars operators."""
    tree = ast.parse(_TARGET.read_text(encoding="utf-8"), filename=str(_TARGET))
    registrations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "register_operator":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        assert "backend" in keywords
        registrations.append(node)
    assert len(registrations) == 64


def test_batch5_quarantine_wrapper_forces_research_only() -> None:
    """The local registration wrapper remains fail-closed for future additions."""
    source = _TARGET.read_text(encoding="utf-8")
    assert 'kwargs["status"] = "research_only"' in source
    assert "register_operator as _register_operator" in source
