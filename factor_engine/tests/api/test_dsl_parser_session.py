import pytest

import factor_engine.api.dsl_parser as dsl


def _identity(value):
    return value


def test_parser_session_builds_allowlist_once_and_resets_budget(monkeypatch):
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return {"only": _identity}

    monkeypatch.setattr(dsl, "build_dsl_allowlist", build)
    monkeypatch.setattr(dsl, "_current_registry_token", lambda: (7, "finalized"))
    parser = dsl.DSLParser(
        surface="compat_research", dialect="lqtp",
        dialect_version="2026-07-19",
        budget=dsl.ComplexityBudget(max_ast_nodes=2),
    )
    parsed = parser.parse_many(["only(close)"] * 100)

    assert len(calls) == 1
    assert len(parsed) == 100
    assert all(expr.name == "close" for expr in parsed)
    with pytest.raises(dsl.DSLParseError, match="max_ast_nodes"):
        parser.parse("only(only(close))")
    # A failed formula must not spend the next formula's node budget.
    assert parser.parse("only(close)").name == "close"


def test_session_snapshot_is_frozen_but_parse_expr_has_no_stale_global_cache(monkeypatch):
    surface = {"old_op": _identity}

    def build(**_kwargs):
        return dict(surface)

    monkeypatch.setattr(dsl, "build_dsl_allowlist", build)
    monkeypatch.setattr(dsl, "_current_registry_token", lambda: (7, "finalized"))
    parser = dsl.DSLParser()
    surface.clear()
    surface["new_op"] = _identity

    assert parser.parse("old_op(close)").name == "close"
    with pytest.raises(dsl.DSLUnknownOperatorError, match="new_op"):
        parser.parse("new_op(close)")
    # Non-session parsing rebuilds the allowlist and observes the new surface.
    assert dsl.parse_expr("new_op(close)").name == "close"
    with pytest.raises(dsl.DSLUnknownOperatorError, match="old_op"):
        dsl.parse_expr("old_op(close)")


def test_session_allowlist_cannot_be_mutated(monkeypatch):
    monkeypatch.setattr(
        dsl, "build_dsl_allowlist", lambda **_kwargs: {"only": _identity}
    )
    monkeypatch.setattr(dsl, "_current_registry_token", lambda: (7, "finalized"))
    parser = dsl.DSLParser()

    with pytest.raises(TypeError):
        parser._allowed["injected"] = _identity
    with pytest.raises(dsl.DSLUnknownOperatorError, match="injected"):
        parser.parse("injected(close)")


def test_session_rejects_registry_change(monkeypatch):
    token = [7, "finalized"]
    monkeypatch.setattr(
        dsl, "build_dsl_allowlist", lambda **_kwargs: {"only": _identity}
    )
    monkeypatch.setattr(dsl, "_current_registry_token", lambda: tuple(token))
    parser = dsl.DSLParser()
    token[:] = [8, "building"]

    with pytest.raises(dsl.DSLParseError, match="registry changed"):
        parser.parse("only(close)")
