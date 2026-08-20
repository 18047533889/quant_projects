# -*- coding: utf-8 -*-
"""Round-13 R11-auditor defect fixes (P0-06 / P0-07 / P0-08 / P0-09).

Focused pytest wrapper for the four R11 auditor bugs:

  P0-06  ``_slice_range`` returns the true trailing half (``iloc[start:]``),
         which ``audit_stateful_contract_discovery`` now uses for the
         no-carried-state second-half chunk.
  P0-07  ``audit_default_parameter_history`` measures the operator's real warmup
         (first finite output row ``m``) instead of the vacuous ``rows >= 2``
         early-continue, and FAILs when ``history_requirement`` under-declares.
  P0-08  Operators the auditor cannot construct valid inputs for are tracked as
         UNAUDITED; ``main()`` fails closed for production candidates and only
         warns for research/experimental operators.
  P0-09  ``_compare_region`` rejects index/column misalignment in addition to
         shape mismatch.

The audit modules are loaded by ABSOLUTE FILE PATH under unique module names to
avoid the monorepo ``scripts`` package shadowing factor_engine's under pytest.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

_FE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_FE_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_audit = _load_module("scripts/audit_r11_longtail.py", "audit_r11_longtail_round13")
_harness = _load_module(
    "scripts/audit_all_factor_production.py", "audit_all_factor_production_round13"
)


# ---------------------------------------------------------------------------
# P0-06 — _slice_range returns the true trailing half
# ---------------------------------------------------------------------------
def test_slice_range_takes_second_half() -> None:
    frame = pd.DataFrame({"v": np.arange(10)}, index=list("abcdefghij"))
    second = _harness._slice_range(frame, 5, None)
    assert list(second.index) == list("fghij")
    assert list(second["v"]) == [5, 6, 7, 8, 9]
    # interior window
    middle = _harness._slice_range(frame, 2, 6)
    assert list(middle.index) == list("cdef")
    # Series slicing
    series = pd.Series(np.arange(10), index=list("abcdefghij"))
    assert list(_harness._slice_range(series, 5, None).index) == list("fghij")
    # non-DataFrame/Series passthrough
    assert _harness._slice_range(42, 1, None) == 42
    assert _harness._slice_range([1, 2, 3], 1, 2) == [1, 2, 3]


def test_stateful_discovery_uses_trailing_half(monkeypatch) -> None:
    """P0-06 end-to-end: the second-half chunk is built from ``iloc[half:]``.

    Monkeypatch the harness ``_slice`` used by the old (buggy) implementation
    and assert the fixed function never slices the leading half for the chunk
    execution.
    """
    calls: list[int] = []

    def spy_slice_range(value, start, end):
        calls.append(start)
        return _harness._slice_range(value, start, end)

    monkeypatch.setattr(_audit, "_slice_range", spy_slice_range)

    panels = _harness._panels(rows=60, columns=2)

    class _IdentityOp:
        """Stateless identity operator.

        The audit's full-run calls ``calculate(*_build_call(...))`` which passes
        ``(arguments_list, kwargs_dict)`` as two positional args; the chunk-run
        uses ``calculate(*pargs, **pkwargs)``.  Handle both and return the panel.
        """

        metadata = types.SimpleNamespace(param_names=("x",))

        def calculate(self, *args, **kwargs):
            if args and isinstance(args[0], list) and args[0] and isinstance(args[0][0], pd.DataFrame):
                return args[0][0]
            return args[0]

    import cleaned_operators as co
    import cleaned_operators.registry as reg

    monkeypatch.setattr(co, "load_all", lambda: None)
    monkeypatch.setattr(reg.OperatorRegistry, "get", lambda canonical, *a, **k: _IdentityOp())
    monkeypatch.setattr(_audit, "_build_call", lambda canonical, op, panels: ([panels["x"]], {}))

    errors = _audit.audit_stateful_contract_discovery(["id_op"], panels)
    assert not errors
    # The chunk slice must be the trailing half (start == 30 for 60 rows), not 0.
    assert calls, "expected _slice_range to be exercised"
    assert all(start >= 30 for start in calls), calls


# ---------------------------------------------------------------------------
# P0-07 — default-history auditor measures real warmup
# ---------------------------------------------------------------------------
class _UnderDeclaredOp:
    """Fake operator whose output only matures at row 3 (rows 0..2 all-NaN)."""

    metadata = types.SimpleNamespace(param_names=("x",))

    def calculate(self, x, **kwargs):
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        out.iloc[3:] = 1.0
        return out


def test_default_history_flags_underdeclared(monkeypatch) -> None:
    """When ``history_requirement`` declares 1 row but the kernel matures at
    row 3, the auditor must FAIL (declared < mature + 1)."""
    import cleaned_operators as co
    import cleaned_operators.registry as reg
    import runtime.execution_contract as ec
    from runtime.execution_contract import HistoryRequirement

    monkeypatch.setattr(co, "load_all", lambda: None)
    monkeypatch.setattr(reg.OperatorRegistry, "get", lambda canonical, *a, **k: _UnderDeclaredOp())
    monkeypatch.setattr(_audit, "_build_call", lambda canonical, op, panels: ([panels["x"]], {}))
    monkeypatch.setattr(
        ec, "history_requirement",
        lambda canonical, params, *a, **k: HistoryRequirement(kind="finite", rows=1),
    )

    panels = _harness._panels(rows=60, columns=2)
    errors = _audit.audit_default_parameter_history(["under_declared_op"], panels)
    assert any("C default-history" in e and "under_declared_op" in e for e in errors)


def test_default_history_pass_when_declared_covers_warmup(monkeypatch) -> None:
    """An operator whose declared history covers the real warmup is not flagged."""
    import cleaned_operators as co
    import cleaned_operators.registry as reg
    import runtime.execution_contract as ec
    from runtime.execution_contract import HistoryRequirement

    monkeypatch.setattr(co, "load_all", lambda: None)
    monkeypatch.setattr(reg.OperatorRegistry, "get", lambda canonical, *a, **k: _UnderDeclaredOp())
    monkeypatch.setattr(_audit, "_build_call", lambda canonical, op, panels: ([panels["x"]], {}))
    # declared == mature + 1 == 4 -> covered, no error.
    monkeypatch.setattr(
        ec, "history_requirement",
        lambda canonical, params, *a, **k: HistoryRequirement(kind="finite", rows=4),
    )

    panels = _harness._panels(rows=60, columns=2)
    errors = _audit.audit_default_parameter_history(["covered_op"], panels)
    assert not errors


# ---------------------------------------------------------------------------
# P0-08 — UNAUDITED production candidates fail closed
# ---------------------------------------------------------------------------
def _install_main_fakes(monkeypatch, *, fail_closed_result: bool, in_targets: bool):
    panels = _harness._panels(rows=60, columns=2)
    monkeypatch.setattr(_audit, "_canonical_sample", lambda rng, n_random=60: [])
    monkeypatch.setattr(_audit, "_curated_causal_panels", lambda: panels)
    targets = frozenset({"fake_prod_op"}) if in_targets else frozenset()
    monkeypatch.setattr(_audit, "factor_production_targets", lambda: targets)
    monkeypatch.setattr(_audit, "should_fail_closed", lambda canonical: fail_closed_result)

    def _prefix(canonicals, panels, *, unaudited=None):
        if unaudited is not None:
            _audit._mark_unaudited(unaudited, "fake_prod_op")
        return []

    monkeypatch.setattr(_audit, "audit_prefix_invariance", _prefix)
    monkeypatch.setattr(
        _audit, "audit_stateful_contract_discovery",
        lambda canonicals, panels, *, unaudited=None: [],
    )
    monkeypatch.setattr(
        _audit, "audit_default_parameter_history",
        lambda canonicals, panels, *, unaudited=None: [],
    )
    monkeypatch.setattr(_audit, "audit_unit_algebra", lambda: [])
    monkeypatch.setattr(sys, "argv", ["audit_r11_longtail.py"])
    return panels


def test_unaudited_production_candidate_fails_main(monkeypatch, capsys) -> None:
    """A production candidate that the auditor cannot construct inputs for must
    make ``main()`` exit non-zero (fail closed)."""
    _install_main_fakes(monkeypatch, fail_closed_result=False, in_targets=True)
    assert _audit.main() == 1
    captured = capsys.readouterr()
    assert "UNAUDITED production candidate: fake_prod_op" in (captured.err + captured.out)


def test_unaudited_research_operator_only_warns(monkeypatch, capsys) -> None:
    """A research/experimental operator that is UNAUDITED only warns and does
    not fail the audit."""
    _install_main_fakes(monkeypatch, fail_closed_result=True, in_targets=True)
    assert _audit.main() == 0
    captured = capsys.readouterr()
    assert "UNAUDITED production candidate" not in captured.err
    assert "not a production candidate" in captured.err


# ---------------------------------------------------------------------------
# P0-09 — _compare_region rejects misaligned axes
# ---------------------------------------------------------------------------
def test_compare_region_rejects_misaligned_axes() -> None:
    base = pd.DataFrame(
        np.arange(20).reshape(10, 2),
        index=pd.RangeIndex(10),
        columns=["a", "b"],
    )
    shifted_index = pd.DataFrame(
        np.arange(20).reshape(10, 2),
        index=pd.RangeIndex(1, 11),
        columns=["a", "b"],
    )
    assert not _audit._compare_region(base, shifted_index, 0)
    shifted_columns = pd.DataFrame(
        np.arange(20).reshape(10, 2),
        index=pd.RangeIndex(10),
        columns=["a", "c"],
    )
    assert not _audit._compare_region(base, shifted_columns, 0)
    # aligned equal frames still agree
    assert _audit._compare_region(base, base.copy(), 0)
