# -*- coding: utf-8 -*-
"""R11 round-2 state/condition family regression tests.

Covers three review items on the state/event/CTA family:

1. ConditionBool fail-closed validation (P0, "ConditionBool 全 state/event/CTA
   family").  Every operator whose input slot is a condition / event / reset /
   trigger (``ts_rank_if``, ``state_ewm_if``, ``event_refractory``,
   ``state_since_*``, ``state_since_trend_tstat``, ``ts_transition_count``,
   ``ts_time_since_change``, ``ts_event_spacing_*``, ``event_decay_asof`` with
   ``event_kind="bool"``) must reject a finite value outside {0, 1} (fail
   closed) and accept 0 / 1 / NaN.  NaN is "unknown" and handled per each
   operator's documented missing-state semantics — never silently coerced.

2. ``state_since_reduce`` split (P0).  The four honest canonicals
   ``state_since_sum`` / ``state_since_mean`` / ``state_since_count`` /
   ``state_since_last`` reproduce the old ``reduce(mode=...)`` values exactly,
   each with the correct ``output_unit`` (``same_as:x`` / ``same_as:x`` /
   ``count`` / ``same_as:x``) and unit-algebra tags.  ``state_since_reduce`` is
   retired as a resolving alias to ``state_since_sum``.

3. ``state_since_trend_tstat`` unit (P0).  A t-statistic is dimensionless; the
   metadata must say ``dimensionless`` (not ``level``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import the affected modules directly so this test is self-contained (it does
# not depend on ``load_all``'s whole-tree import succeeding).  When the full
# suite runs first, ``load_all`` has already imported these modules, so the
# imports below are cached no-ops and the registry lifecycle is unaffected.
import factor_engine.cleaned_operators.stateful.episode  # noqa: F401
import factor_engine.cleaned_operators.stateful.events  # noqa: F401
import factor_engine.cleaned_operators.stateful.sequential  # noqa: F401
import factor_engine.cleaned_operators.state_event  # noqa: F401

from factor_engine.cleaned_operators.registry import OperatorRegistry

# Operators whose condition / event / reset / trigger slot must be a
# ConditionBool.  ``state_since_*`` and ``state_since_trend_tstat`` read a
# reset slot; ``event_decay_asof`` reads an event slot only when
# ``event_kind="bool"``.
CONDITION_SLOT_CANONICALS = (
    "ts_rank_if",
    "state_ewm_if",
    "event_refractory",
    "state_since_sum",
    "state_since_mean",
    "state_since_count",
    "state_since_last",
    "state_since_trend_tstat",
    "ts_transition_count",
    "ts_time_since_change",
    "ts_event_spacing_mean",
    "ts_event_spacing_cv",
    "event_decay_asof",
)

INVALID_CONDITION_VALUES = (0.5, -1.0, 2.0)


def _bdate(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


def _x_panel(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(np.arange(1, n + 1, dtype=float), index=_bdate(n), columns=["A"])


def _cond_panel(first_value: float, n: int = 4) -> pd.DataFrame:
    base = [first_value, 0.0, 1.0, 0.0]
    values = (base * (n // len(base) + 1))[:n]
    return pd.DataFrame(values, index=_bdate(n), columns=["A"])


def _build_condition_call(canonical: str, cond: pd.DataFrame) -> tuple[tuple, dict]:
    """Build ``(args, kwargs)`` with the condition/event/reset panel in place."""
    x = _x_panel(len(cond))
    if canonical == "ts_rank_if":
        return (x, cond), {}
    if canonical == "state_ewm_if":
        return (x, cond), {}
    if canonical == "event_refractory":
        return (cond,), {}
    if canonical in (
        "state_since_sum", "state_since_mean", "state_since_count",
        "state_since_last",
    ):
        return (x, cond), {"min_episode": 1}
    if canonical == "state_since_trend_tstat":
        return (x, cond), {}
    if canonical == "ts_transition_count":
        return (cond,), {}
    if canonical == "ts_time_since_change":
        return (cond,), {}
    if canonical == "ts_event_spacing_mean":
        return (cond,), {}
    if canonical == "ts_event_spacing_cv":
        return (cond,), {}
    if canonical == "event_decay_asof":
        return (cond,), {"event_kind": "bool"}
    raise AssertionError(f"unhandled canonical {canonical!r}")


# ---------------------------------------------------------------------------
# (a) ConditionBool fail-closed validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("canonical", CONDITION_SLOT_CANONICALS)
@pytest.mark.parametrize("bad", INVALID_CONDITION_VALUES)
def test_condition_slot_rejects_non_condition_bool(canonical: str, bad: float):
    """A finite value outside {0, 1} must fail closed, never read as truthy."""
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, f"{canonical} missing pandas runtime"
    cond = _cond_panel(bad)
    args, kwargs = _build_condition_call(canonical, cond)
    with pytest.raises(ValueError, match="ConditionBool"):
        op.calculate(*args, **kwargs)


@pytest.mark.parametrize("canonical", CONDITION_SLOT_CANONICALS)
def test_condition_slot_accepts_condition_bool(canonical: str):
    """0 / 1 / NaN are legal; NaN is an unknown state, not an error."""
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    # first row NaN (unknown), then 0 / 1 / 0.
    cond = _cond_panel(np.nan)
    args, kwargs = _build_condition_call(canonical, cond)
    out = op.calculate(*args, **kwargs)
    assert out.shape == (len(cond), 1)


def test_event_decay_asof_marked_event_accepts_magnitude():
    """event_kind="marked" keeps the magnitude — 2.0 is a legitimate mark."""
    dates = _bdate(4)
    ev = pd.DataFrame([0.0, 2.0, 0.0, 0.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("event_decay_asof", "pandas_numpy")
    out = op.calculate(ev, half_life=10.0, event_kind="marked")
    assert out["A"].iloc[1] == pytest.approx(2.0)


def test_event_decay_asof_bool_event_rejects_magnitude():
    """event_kind="bool" is a ConditionBool — a 2.0 mark must fail closed."""
    dates = _bdate(4)
    ev = pd.DataFrame([0.0, 2.0, 0.0, 0.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("event_decay_asof", "pandas_numpy")
    with pytest.raises(ValueError, match="ConditionBool"):
        op.calculate(ev, half_life=10.0, event_kind="bool")


# ---------------------------------------------------------------------------
# (b) state_since_reduce split: value parity + honest output units
# ---------------------------------------------------------------------------


def _old_reduce(x: pd.DataFrame, reset: pd.DataFrame, mode: str, min_episode: int) -> pd.DataFrame:
    """Reference for the retired ``state_since_reduce(mode=...)`` kernel."""
    xv = x.to_numpy(dtype=float)
    rv = reset.to_numpy(dtype=float)
    min_e = max(1, int(min_episode))
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        acc = 0.0
        cnt = 0
        last = np.nan
        for row in range(rows):
            if not np.isfinite(rv[row, col]):
                out[row, col] = np.nan
                acc = 0.0
                cnt = 0
                last = np.nan
                continue
            if bool(rv[row, col] != 0.0):
                out[row, col] = np.nan
                acc = 0.0
                cnt = 0
                last = np.nan
                continue
            xt = xv[row, col]
            if not np.isfinite(xt):
                out[row, col] = np.nan
                continue
            acc = acc + float(xt)
            cnt = cnt + 1
            last = float(xt)
            if cnt < min_e:
                out[row, col] = np.nan
                continue
            if mode == "sum":
                out[row, col] = acc
            elif mode == "mean":
                out[row, col] = acc / float(cnt)
            elif mode == "count":
                out[row, col] = float(cnt)
            else:  # last
                out[row, col] = last
    return pd.DataFrame(out, index=x.index, columns=x.columns)


SPLIT_CANONICALS = {
    "state_since_sum": "sum",
    "state_since_mean": "mean",
    "state_since_count": "count",
    "state_since_last": "last",
}


def test_state_since_split_matches_old_reduce_values():
    n = 48
    rng = np.random.default_rng(11)
    x = pd.DataFrame(
        np.cumsum(rng.normal(0, 1.0, (n, 3)), axis=0),
        index=_bdate(n), columns=["A", "B", "C"],
    )
    reset = pd.DataFrame(
        (rng.random((n, 3)) > 0.9).astype(float), index=_bdate(n), columns=["A", "B", "C"]
    )
    reset.iloc[5, 0] = np.nan  # exercise the NaN-reset (unknown) path
    for canonical, mode in SPLIT_CANONICALS.items():
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        got = op.calculate(x, reset, min_episode=2)
        expected = _old_reduce(x, reset, mode, 2)
        np.testing.assert_allclose(
            got.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=1e-12
        ), f"{canonical} diverges from old reduce(mode={mode!r})"


def test_state_since_split_output_units():
    """Each split canonical declares the honest output unit / unit-algebra tag."""
    expected_units = {
        "state_since_sum": ("same_as:x", "unit:same_as:x"),
        "state_since_mean": ("same_as:x", "unit:same_as:x"),
        "state_since_count": ("count", "unit:count"),
        "state_since_last": ("same_as:x", "unit:same_as:x"),
    }
    for canonical, (output_unit, tag) in expected_units.items():
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        assert op.metadata.output_unit == output_unit, f"{canonical} output_unit"
        assert tag in op.metadata.tags, f"{canonical} missing {tag!r} tag"
    # the old dynamic-unit problem is gone: no split canonical pins "level".
    for canonical in SPLIT_CANONICALS:
        assert "unit:level" not in OperatorRegistry.get(canonical, "pandas_numpy").metadata.tags


def test_state_since_reduce_retired_as_alias_to_sum():
    op = OperatorRegistry.get("state_since_reduce", "pandas_numpy")
    assert op is not None, "state_since_reduce must still resolve"
    assert op.metadata.name == "state_since_sum"
    assert OperatorRegistry.resolve_canonical("state_since_reduce") == "state_since_sum"
    # a default-mode call (no mode kwarg) keeps working through the alias
    x = _x_panel(4)
    reset = _cond_panel(1.0)  # [1, 0, 1, 0]
    out = op.calculate(x, reset, min_episode=1)
    # row0 reset -> NaN; row1 2; row2 reset -> NaN; row3 4
    assert np.isnan(out["A"].iloc[0]) and out["A"].iloc[1] == pytest.approx(2.0)
    assert np.isnan(out["A"].iloc[2]) and out["A"].iloc[3] == pytest.approx(4.0)


def test_state_since_reduce_mode_kwarg_rejected_after_retire():
    """The mode-enum dimension is gone: a ``mode=`` call must be rejected."""
    op = OperatorRegistry.get("state_since_reduce", "pandas_numpy")
    x = _x_panel(4)
    reset = _cond_panel(1.0)
    with pytest.raises(ValueError, match="mode"):
        op.calculate(x, reset, mode="mean", min_episode=1)


def test_state_since_reset_condition_validated():
    """The split canonicals fail closed on a non-ConditionBool reset slot."""
    x = _x_panel(4)
    bad_reset = _cond_panel(0.5)
    for canonical in SPLIT_CANONICALS:
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        with pytest.raises(ValueError, match="ConditionBool"):
            op.calculate(x, bad_reset, min_episode=1)


# ---------------------------------------------------------------------------
# (c) state_since_trend_tstat is dimensionless
# ---------------------------------------------------------------------------


def test_state_since_trend_tstat_dimensionless():
    op = OperatorRegistry.get("state_since_trend_tstat", "pandas_numpy")
    assert op.metadata.output_unit == "dimensionless"
    assert "unit:dimensionless" in op.metadata.tags
    assert "unit:level" not in op.metadata.tags


def test_state_since_trend_tstat_reset_condition_validated():
    x = _x_panel(8)
    bad_reset = _cond_panel(-1.0, n=8)
    op = OperatorRegistry.get("state_since_trend_tstat", "pandas_numpy")
    with pytest.raises(ValueError, match="ConditionBool"):
        op.calculate(x, bad_reset)
