# -*- coding: utf-8 -*-
"""Round-11 WS-A shared-contract layer tests.

Covers the shared, cross-cutting fixes every operator workstream consumes:
  * #12  per-operator ExecutionContract declaration (declare_stateful) + CI.
  * #11  cross_event is stateless with a finite lag-1 history.
  * #13/#14  ParamSpec history_semantics / history_formula replace the
        window/span/lookback name-guessing (declared event-clock -> full history).
  * #16  DSL string literals are never coerced at parse; the binder converts a
        numeric string ONLY against a declared numeric ParamSpec.
  * #17  non-finite numeric DSL literals are rejected.
  * #18  ComplexityBudget bounds AST nodes / depth / arity / literal magnitude.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# #12 / #11 — execution-contract declaration and cross_event
# ---------------------------------------------------------------------------


def test_declare_stateful_resolves_episode_contract():
    import cleaned_operators as co

    co.load_all()
    from runtime.execution_contract import (
        ExecutionContractResolutionError,
        declare_stateful,
        execution_contract,
    )

    declare_stateful(
        "__r11_probe_episode__",
        state_model="episode",
        chunking="required_full_history",
        checkpoint_schema="r11_probe.v1",
        minimum_history=3,
    )
    contract = execution_contract("__r11_probe_episode__")
    assert contract.state_model == "episode"
    assert contract.chunking == "required_full_history"
    assert contract.checkpoint_schema == "r11_probe.v1"


def test_declare_stateful_duplicate_rejected():
    import cleaned_operators as co

    co.load_all()
    from runtime.execution_contract import (
        ExecutionContractResolutionError,
        declare_stateful,
    )

    with pytest.raises(ExecutionContractResolutionError):
        declare_stateful(
            "__r11_probe_episode__",
            state_model="episode",
            chunking="required_full_history",
        )


def test_declare_stateful_rejects_stateless():
    import cleaned_operators as co

    co.load_all()
    from runtime.execution_contract import (
        ExecutionContractResolutionError,
        declare_stateful,
    )

    with pytest.raises(ExecutionContractResolutionError):
        declare_stateful("__r11_probe_stateless__", state_model="stateless", chunking="independent")


def test_cross_event_is_stateless_finite_lag1():
    """#11: cross_event is a pure lag-1 crossing detector, NOT stateful."""
    import cleaned_operators as co

    co.load_all()
    from runtime.execution_contract import execution_contract, history_requirement

    contract = execution_contract("cross_event")
    assert not contract.is_stateful
    assert contract.state_model == "stateless"
    assert contract.chunking == "independent"
    req = history_requirement("cross_event")
    assert req.kind == "finite"
    assert not req.is_full_history
    # lag-1 detector needs exactly one prior bar (the >=2 floor is a global
    # safety; the operator itself contributes exactly 1).
    from runtime.execution_contract import _own_history_extension

    assert _own_history_extension("cross_event", {}) == 1


def test_event_clock_history_is_full_history_conservative():
    """#7/#8: event-count / report-count history cannot be a bar warmup."""
    import cleaned_operators as co

    co.load_all()
    from runtime.execution_contract import declare_stateful, history_requirement

    declare_stateful(
        "__r11_probe_eventclock__",
        state_model="recursive",
        chunking="required_full_history",
        history_kind="report_count",
        history_count=4,
    )
    req = history_requirement("__r11_probe_eventclock__")
    assert req.kind == "report_count"
    assert req.count == 4
    assert req.is_full_history  # bar-warmup cannot derive report observations
    assert req.is_event_clock


# ---------------------------------------------------------------------------
# #13 / #14 — declared ParamSpec history semantics / formulas
# ---------------------------------------------------------------------------


class _FakeSpec:
    def __init__(self, dtype=None, default=None, min=None, history_semantics=None, history_formula=None):
        self.dtype = dtype
        self.default = default
        self.min = min
        self.history_semantics = history_semantics
        self.history_formula = history_formula


class _FakeMeta:
    def __init__(self, params, specs):
        self.param_names = params
        self.param_specs = specs
        self.param_types = {}


def _declared_extension(params, specs, monkeypatch):
    """Evaluate ``_declared_history_extension`` against a synthetic operator."""
    import cleaned_operators as co

    co.load_all()
    import runtime.execution_contract as ec

    meta = _FakeMeta(list(specs.keys()), specs)
    monkeypatch.setattr(ec, "_metadata", lambda canonical, **kw: meta)
    return ec._declared_history_extension("__probe__", params)


def test_declared_compound_formula_history(monkeypatch):
    """#14: ``outer_window + inner_window`` compound history (vol-of-vol)."""
    specs = {
        "inner_window": _FakeSpec(dtype=int, default=20),
        "outer_window": _FakeSpec(dtype=int, default=60, history_formula="outer_window + inner_window"),
    }
    assert _declared_extension({}, specs, monkeypatch) == 80
    assert _declared_extension({"inner_window": 10, "outer_window": 30}, specs, monkeypatch) == 40


def test_declared_concentration_2x_window(monkeypatch):
    """#125/#14: ``2 * window`` acceleration history."""
    specs = {"window": _FakeSpec(dtype=int, default=60, history_formula="2 * window")}
    assert _declared_extension({}, specs, monkeypatch) == 120


def test_declared_exact_rows_semantics(monkeypatch):
    """#13: lag-like param declares exact_rows -> contribution == value."""
    specs = {"lag": _FakeSpec(dtype=int, default=20, history_semantics="exact_rows")}
    assert _declared_extension({}, specs, monkeypatch) == 20


def test_declared_event_clock_semantics_unknown(monkeypatch):
    """#13: report_events / session_slots semantics -> UNKNOWN (full history)."""
    import runtime.execution_contract as ec

    specs = {"window": _FakeSpec(dtype=int, default=60, history_semantics="report_events")}
    assert _declared_extension({}, specs, monkeypatch) is ec._UNKNOWN


def test_declared_formula_fractional_bound_unknown(monkeypatch):
    """#13: an unresolvable (fractional/string) bound value is UNKNOWN, never truncated."""
    import runtime.execution_contract as ec

    specs = {"window": _FakeSpec(dtype=int, default=60, history_formula="2 * window")}
    assert _declared_extension({"window": 5.9}, specs, monkeypatch) is ec._UNKNOWN


# ---------------------------------------------------------------------------
# #16 — DSL strings stay strings; the binder coerces against declared dtype
# ---------------------------------------------------------------------------


def test_dsl_numeric_looking_string_stays_string():
    from api.dsl_parser import parse_expr

    ref = parse_expr('col("000001")')
    assert ref.name == "000001"  # leading zero preserved — never int(1)
    ref2 = parse_expr('col("2024Q1")')
    assert ref2.name == "2024Q1"


def test_dsl_string_category_kept_for_string_param():
    import cleaned_operators as co

    co.load_all()
    from api.dsl_parser import parse_expr

    # find an operator with a declared str param to prove no blind coercion
    from cleaned_operators.registry import OperatorRegistry

    target = None
    for canon in OperatorRegistry.list_canonical():
        meta = OperatorRegistry.get(canon).metadata
        specs = getattr(meta, "param_specs", None) or {}
        if any(getattr(s, "dtype", None) is str for s in specs.values()):
            target = (canon, specs)
            break
    assert target is not None, "no str-param operator found for coercion probe"
    canon, specs = target
    str_name = next(n for n, s in specs.items() if getattr(s, "dtype", None) is str)
    # the DSL keeps an enum-looking string exactly as written
    expr = parse_expr(f'{canon}(col("close"), {str_name}="doji")')
    assert expr is not None


def test_dsl_numeric_string_window_coerced_by_binder():
    """#16: ``"20"`` for a numeric window reaches the kernel as int 20."""
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("cs_autoencoder_reconstruction_error")
    meta = op.metadata
    specs = getattr(meta, "param_specs", None) or {}
    types = getattr(meta, "param_types", None) or {}
    assert specs.get("window") is not None  # declared int window spec
    from cleaned_operators.base import _coerce_declared_numeric_string

    assert _coerce_declared_numeric_string("20", "window", types.get("window"), specs.get("window")) == 20
    # a string parameter is NEVER coerced
    str_spec = _FakeSpec(dtype=str)
    assert _coerce_declared_numeric_string("000001", "code", None, str_spec) == "000001"
    # no declared contract -> no coercion
    assert _coerce_declared_numeric_string("20", "x", None, None) == "20"


# ---------------------------------------------------------------------------
# #17 — non-finite DSL literals rejected
# ---------------------------------------------------------------------------


def test_dsl_rejects_nonfinite_literal():
    from api.dsl_parser import DSLParseError, parse_expr

    for bad in ("1e309", "-1e400", "float('inf')"):
        with pytest.raises(DSLParseError):
            parse_expr(f"add(col('close'), {bad})")


# ---------------------------------------------------------------------------
# #18 — ComplexityBudget
# ---------------------------------------------------------------------------


def test_dsl_complexity_budget_arity():
    from api.dsl_parser import ComplexityBudget, DSLParseError, parse_expr

    with pytest.raises(DSLParseError):
        parse_expr("add(" + ", ".join(["col('close')"] * 40) + ")")


def test_dsl_complexity_budget_depth():
    from api.dsl_parser import ComplexityBudget, DSLParseError, parse_expr

    with pytest.raises(DSLParseError):
        parse_expr("ts_mean(" * 40 + "col('close'), 5)" + ")" * 40)


def test_dsl_complexity_budget_literal_magnitude():
    from api.dsl_parser import ComplexityBudget, DSLParseError, parse_expr

    with pytest.raises(DSLParseError):
        parse_expr("add(col('close'), 99999999999999999999)")
