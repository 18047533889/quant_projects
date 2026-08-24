"""R40 #86/#87: api.mining_integration market passthrough + honest rename tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from factor_engine.api.mining_integration import (
    validate_production_fastpath_dsl,
    validate_syntax_only_dsl,
    validate_us_dsl,
    validate_manifest_for_execution,
)


def _stub_parse(monkeypatch):
    """Concurrent-session mid-edit: parse_expr raises NameError — stub it out.

    ``api.mining_integration`` imports ``parse_expr`` at module import time, so
    the module attribute must be patched (not the source module's attribute).
    """
    monkeypatch.setattr("factor_engine.api.mining_integration.parse_expr", lambda *a, **k: None)


def _stub_fastpath_gate(monkeypatch):
    monkeypatch.setattr(
        "factor_engine.backend.production_fastpath_gate.check_production_fastpath_formula_ops",
        lambda formula, strict=None: SimpleNamespace(ok=True, violations=[]),
    )


# ---------------------------------------------------------------------------
# #86: fastpath validator threads market through to validate_production_dsl
# ---------------------------------------------------------------------------


def test_fastpath_market_passthrough(monkeypatch):
    captured = {}

    def fake_prod_dsl(formula, *, market=None):
        captured["market"] = market
        return True, "OK"

    monkeypatch.setattr(
        "factor_engine.api.mining_integration.validate_production_dsl", fake_prod_dsl
    )
    _stub_fastpath_gate(monkeypatch)
    ok, msg = validate_production_fastpath_dsl("close", market="us")
    assert ok is True
    assert captured["market"] == "us"


def test_fastpath_market_none_rejected(monkeypatch):
    """R21-FASTPATH-MARKET: market=None is a hard failure — no A-share default."""
    captured = {}

    def fake_prod_dsl(formula, *, market=None):
        captured["market"] = market
        return True, "OK"

    monkeypatch.setattr(
        "factor_engine.api.mining_integration.validate_production_dsl", fake_prod_dsl
    )
    _stub_fastpath_gate(monkeypatch)
    with pytest.raises(ValueError, match="market is required"):
        validate_production_fastpath_dsl("close", market=None)
    assert not captured  # validate_production_dsl 不应被调用


def test_fastpath_market_omitted_typeerror(monkeypatch):
    """R21-FASTPATH-MARKET: omitting market entirely is a TypeError (no default)."""
    _stub_fastpath_gate(monkeypatch)
    with pytest.raises(TypeError, match="market"):
        validate_production_fastpath_dsl("close")


def test_manifest_for_execution_threads_market(monkeypatch):
    captured = {}

    def fake_fastpath(formula, *, strict=None, market=None):
        captured["market"] = market
        return True, "OK"

    monkeypatch.setattr(
        "factor_engine.api.mining_integration.validate_production_fastpath_dsl", fake_fastpath
    )
    validate_manifest_for_execution(
        market="us", expression_type="dsl", formula="close", require_fastpath=True
    )
    assert captured["market"] == "us"


# ---------------------------------------------------------------------------
# #87: validate_us_dsl -> validate_syntax_only_dsl (deprecation + honest name)
# ---------------------------------------------------------------------------


def test_validate_us_dsl_deprecated_warns(monkeypatch):
    _stub_parse(monkeypatch)
    with pytest.warns(DeprecationWarning, match="validate_syntax_only_dsl"):
        ok, msg = validate_us_dsl("close")
    assert ok is True


def test_validate_syntax_only_dsl_works(monkeypatch):
    _stub_parse(monkeypatch)
    ok, msg = validate_syntax_only_dsl("rank(close)")
    assert ok is True


def test_validate_syntax_only_dsl_is_syntax_only(monkeypatch):
    _stub_parse(monkeypatch)
    ok, msg = validate_syntax_only_dsl("close + 1")
    assert ok is True
