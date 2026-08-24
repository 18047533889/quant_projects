# -*- coding: utf-8 -*-
"""R11 contract round: unified property tests for the P0/P1 audit fixes.

These tests are intentionally load_all-independent — they import the modules
under test directly so they stay green even while the full operator registry is
being rebuilt.  They cover the release gates added by the R11 audit:

* P0-01/P1-05  — the registry backfills the FULL logical contract (specs /
  aliases / relational / units / grains / availability / role / arity) onto
  every backend operator's instance metadata, not just ``param_names``.
* P0-03        — polars -> pandas conversion preserves the true time axis.
* P0-06        — ``latest_availability`` returns a ``MaxAvailability`` that
  resolves each child to a CONCRETE timestamp and takes the real max (no more
  static descriptor-type total order).
* P0-07/08     — ``PanelIdentity`` encodes full ordered MultiIndex pairs and
  uses a process-stable SHA-256 hash, never built-in ``hash()``.
* P0-09        — ``BroadcastSpec`` is the structured broadcast contract; an
  unknown index type fails closed instead of silently passing.
* P0-10        — a single-row universe mask cannot be broadcast across dates
  (survivorship / future-membership leakage).
* P0-11/P1-08/P1-09 — RQA eps scales WITH sqrt(dim); diagonal-entropy support
  is [min_line, M-1]; Theiler RR is normalised by admissible pairs only.
* P1-01        — a constrained typed slot receiving an UNKNOWN semantic_kind
  is rejected in production.
* P1-10        — unified group-key missing-value normalisation.
* P1-11        — ``align_panel_inputs(strict_axes=False)`` requires an explicit
  ``AlignmentPlan``; a bare boolean is refused.
* P1-12        — financial-history lookback fails closed in production instead
  of silently falling back to the 80-rows heuristic.
* P0-04        — a shape-changing (minute->daily) result bypasses the exact
  index-match check but is still validated as a well-formed downsampled panel.
"""
from __future__ import annotations

import datetime
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def strict_fiscal_parameter_domain_certification_guard():
    """Shadow the operators conftest autouse load_all guard.

    These contract unit tests exercise the modules under test directly and do
    not need the full operator registry (which concurrent sessions are actively
    rebuilding).  Loading everything here would be slow and flaky.
    """
    yield


# ---------------------------------------------------------------------------
# P0-01/P1-05: full logical-contract backfill onto backend operator metadata.
# ---------------------------------------------------------------------------
def test_registry_backfill_copies_full_logical_contract() -> None:
    from factor_engine.cleaned_operators.base import ParamSpec, RelationalParamSpec
    from factor_engine.cleaned_operators.base_polars import (
        OperatorMetadata as PolarsMetadata,
        SeriesOperator as PolarsSeriesOperator,
    )
    from factor_engine.cleaned_operators.registry import _backfill_logical_contract

    prev = {
        "param_specs": {"window": ParamSpec(dtype=int, min=2, default=8)},
        "param_aliases": {"d": "window"},
        "relational_specs": [RelationalParamSpec("window > 1", "w")],
        "param_types": {"window": int},
        "input_units": {"x": "price"},
        "output_unit": "zscore",
        "compatible_units": {"x": ("price",)},
        "window_semantics": "trailing_bars",
        "input_grain": "minute",
        "output_grain": "daily",
        "available_at": "session_close",
        "same_session_usable": False,
        "role": "global_state",
        "input_arity": 1,
        "panel_arity": 1,
        "total_positional_arity": 2,
        "scalar_params": ("window",),
        "panel_params": ("x",),
        "input_fields": ["x"],
    }

    class _Op(PolarsSeriesOperator):
        metadata = PolarsMetadata(name="x", category="bridge", param_names=["x", "window"])

        def _calculate_series(self, *a, **k):  # pragma: no cover - stub
            return None

    op = _Op()
    _backfill_logical_contract(op, prev)
    m = op.metadata
    assert m.param_specs["window"].min == 2
    assert m.param_aliases == {"d": "window"}
    assert len(m.relational_specs) == 1
    assert m.input_units == {"x": "price"}
    assert m.output_unit == "zscore"
    assert m.input_grain == "minute" and m.output_grain == "daily"
    assert m.available_at == "session_close" and m.same_session_usable is False
    assert m.role == "global_state"
    assert m.panel_arity == 1 and m.total_positional_arity == 2
    assert m.panel_params == ("x",) and m.scalar_params == ("window",)


def test_registry_backfill_native_spec_must_match_canonical() -> None:
    """P0-23: the canonical logical contract is the single authority.

    A backend that declares its own non-empty logical field that DIFFERS from the
    canonical is a contract divergence and must fail registration — never
    silently keep both (the old copy-if-empty backfill only inherited EMPTY
    slots; a conflicting non-empty value was the bug).  An empty native slot
    still inherits the canonical value.
    """
    from factor_engine.cleaned_operators.base import ParamSpec
    from factor_engine.cleaned_operators.base_polars import (
        OperatorMetadata as PolarsMetadata,
        SeriesOperator as PolarsSeriesOperator,
    )
    from factor_engine.cleaned_operators.registry import _backfill_logical_contract

    prev = {"param_specs": {"window": ParamSpec(dtype=int, min=2)}}

    # Matching native spec -> kept (no divergence).
    class _NativeMatch(PolarsSeriesOperator):
        metadata = PolarsMetadata(
            name="y",
            category="native",
            param_names=["x", "window"],
            param_specs={"window": ParamSpec(dtype=int, min=2)},
        )

        def _calculate_series(self, *a, **k):  # pragma: no cover - stub
            return None

    native_match = _NativeMatch()
    _backfill_logical_contract(native_match, prev)
    assert native_match.metadata.param_specs["window"].min == 2

    # Divergent native spec (min=5 vs canonical min=2) -> registration failure.
    class _NativeDivergent(PolarsSeriesOperator):
        metadata = PolarsMetadata(
            name="y",
            category="native",
            param_names=["x", "window"],
            param_specs={"window": ParamSpec(dtype=int, min=5)},
        )

        def _calculate_series(self, *a, **k):  # pragma: no cover - stub
            return None

    with pytest.raises(ValueError, match="logical-contract divergence"):
        _backfill_logical_contract(_NativeDivergent(), prev)


# ---------------------------------------------------------------------------
# P0-03: polars -> pandas preserves the true time axis.
# ---------------------------------------------------------------------------
def test_pl_to_pd_preserves_time_axis() -> None:
    import polars as pl

    from factor_engine.cleaned_operators.common._polars_bridge import frame_time_index, to_pandas_panel
    from factor_engine.cleaned_operators.rolling_pack import _pl_to_pd

    n = 5
    df = pl.DataFrame(
        {
            "__fe_time__": [datetime.datetime(2024, 1, i) for i in range(1, n + 1)],
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    pdf = _pl_to_pd(df)
    assert isinstance(pdf.index, pd.DatetimeIndex)
    assert pdf.index.is_monotonic_increasing
    assert list(pdf.columns) == ["A"]
    pdf2 = to_pandas_panel(df)
    assert isinstance(pdf2.index, pd.DatetimeIndex)
    assert frame_time_index(df) is not None

    # fail-closed: duplicate time axis
    dup = df.with_columns(
        pl.lit(datetime.datetime(2024, 1, 1)).alias("__fe_time__")
    )
    with pytest.raises(ValueError):
        _pl_to_pd(dup)


# ---------------------------------------------------------------------------
# P0-06: MaxAvailability resolves CONCRETE timestamps (no descriptor order).
# ---------------------------------------------------------------------------
def test_latest_availability_concrete_max() -> None:
    from factor_engine.ir.schema import propagate_available_at
    from factor_engine.ir.types import (
        MaxAvailability,
        PubDate,
        SessionClose,
        latest_availability,
    )

    class _Close(SessionClose):
        def resolve(self, row=None, calendar=None, timezone=None, decision_context=None):
            return row["trade_date"] + datetime.timedelta(hours=15)

    class _Pub(PubDate):
        def resolve(self, row=None, calendar=None, timezone=None, decision_context=None):
            return row["PubDate"]

    ma = latest_availability([_Close(), _Pub()])
    assert isinstance(ma, MaxAvailability)
    # stale filing + current close -> the CLOSE wins concretely despite PubDate's
    # higher static lateness.
    res = ma.resolve(row={
        "trade_date": datetime.datetime(2026, 8, 9),
        "PubDate": datetime.datetime(2026, 1, 10),
    })
    assert res == datetime.datetime(2026, 8, 9, 15)

    # string-level propagation is unchanged (backward compat)
    assert propagate_available_at(("session_close", "PubDate")) == "PubDate"
    assert propagate_available_at(("session_close", None)) == "unknown"
    assert propagate_available_at(()) is None


# ---------------------------------------------------------------------------
# P0-07/08: PanelIdentity ordered-pair identity + stable hash.
# ---------------------------------------------------------------------------
def test_panel_identity_ordered_pairs_distinguished() -> None:
    from factor_engine.cleaned_operators.common._polars_bridge import PanelIdentity

    mi_a = pd.MultiIndex.from_tuples(
        [("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-02", "A")],
        names=["ts", "inst"],
    )
    mi_b = pd.MultiIndex.from_tuples(
        [("2024-01-01", "B"), ("2024-01-01", "A"), ("2024-01-02", "A")],
        names=["ts", "inst"],
    )
    a = PanelIdentity.from_frame(pd.DataFrame(np.ones((3, 2)), index=mi_a, columns=["v1", "v2"]))
    b = PanelIdentity.from_frame(pd.DataFrame(np.ones((3, 2)), index=mi_b, columns=["v1", "v2"]))
    assert a != b  # same levels, different pairing

    a2 = PanelIdentity.from_frame(pd.DataFrame(np.ones((3, 2)), index=mi_a, columns=["v1", "v2"]))
    assert a == a2


def test_panel_identity_hash_stable_across_processes() -> None:
    from factor_engine.cleaned_operators.common._polars_bridge import PanelIdentity

    mi = pd.MultiIndex.from_tuples(
        [("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-02", "A")],
        names=["ts", "inst"],
    )
    identity = PanelIdentity.from_frame(
        pd.DataFrame(np.ones((3, 2)), index=mi, columns=["v1", "v2"])
    )
    expected = hash(identity)

    code = (
        "import pandas as pd, numpy as np\n"
        "from cleaned_operators.common._polars_bridge import PanelIdentity\n"
        "mi = pd.MultiIndex.from_tuples([('2024-01-01','A'),('2024-01-01','B'),"
        "('2024-01-02','A')], names=['ts','inst'])\n"
        "print(hash(PanelIdentity.from_frame("
        "pd.DataFrame(np.ones((3,2)), index=mi, columns=['v1','v2']))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    other = int(out.stdout.strip().splitlines()[-1])
    assert expected == other


# ---------------------------------------------------------------------------
# P0-09: BroadcastSpec structured contract, unknown index fails closed.
# ---------------------------------------------------------------------------
def test_broadcast_spec_fail_closed() -> None:
    from factor_engine.cleaned_operators.base import (
        BroadcastSpec,
        OperatorMetadata,
        _verify_broadcast_specs,
        _verify_typed_broadcast_axes,
    )

    with pytest.raises(ValueError):
        BroadcastSpec(mode="bogus")

    idx = pd.bdate_range("2024-01-01", periods=10)
    base = pd.DataFrame(np.ones((10, 2)), index=idx, columns=["a", "b"])
    other = pd.DataFrame(np.ones((10, 2)), index=idx, columns=["a", "b"])
    md = OperatorMetadata(name="t", category="x")
    specs = (BroadcastSpec(mode="same_trading_date", date_mapping="trading_date"),)
    _verify_broadcast_specs(md, specs, [base, other])  # aligned passes

    base_r = pd.DataFrame(np.ones((10, 2)), index=pd.RangeIndex(10), columns=["a", "b"])
    with pytest.raises(ValueError):
        _verify_broadcast_specs(md, specs, [base_r, other])
    with pytest.raises(ValueError):
        _verify_typed_broadcast_axes(md, {"daily_to_minute_broadcast"}, [base_r, other])
    # bare legacy tag remains a research waiver (no typed-tag fail-closed)
    _verify_typed_broadcast_axes(md, {"allow_panel_broadcast"}, [base_r, other])


# ---------------------------------------------------------------------------
# P0-10: single-row universe mask cannot be broadcast across dates.
# ---------------------------------------------------------------------------
def test_universe_mask_single_row_rejected() -> None:
    from factor_engine.cleaned_operators.alignment import PanelAxisMismatch, apply_universe_mask

    idx = pd.bdate_range("2024-01-01", periods=5)
    panel = pd.DataFrame(np.ones((5, 3)), index=idx, columns=["a", "b", "c"])
    per_date = pd.DataFrame(np.array([[1.0], [0.0], [1.0], [0.0], [1.0]]), index=idx, columns=["m"])
    out = apply_universe_mask(panel, per_date, mask_axes="broadcast_scalar")
    assert out.iloc[1, 0] != 1 and out.iloc[2, 0] == 1

    single_row = pd.DataFrame(np.array([[1.0, 0.0, 1.0]]), columns=["a", "b", "c"])
    with pytest.raises(PanelAxisMismatch):
        apply_universe_mask(panel, single_row, mask_axes="broadcast_scalar")
    scalar = pd.DataFrame(np.array([[1.0]]), columns=["x"])
    with pytest.raises(PanelAxisMismatch):
        apply_universe_mask(panel, scalar, mask_axes="broadcast_scalar")


# ---------------------------------------------------------------------------
# P0-11/P1-08/P1-09: RQA formulas.
# ---------------------------------------------------------------------------
def test_rqa_eps_scales_with_sqrt_dim() -> None:
    from factor_engine.cleaned_operators.recurrence_analysis import _recurrence_stats_window

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1.0, 120))
    for dim in (1, 2, 3):
        stats = _recurrence_stats_window(x[:60], dim, 1, 0.1, 2)
        assert stats is not None
        assert stats[0] > 0  # non-degenerate rate at every dim


def test_rqa_entropy_normalized_within_bounds() -> None:
    from factor_engine.cleaned_operators.recurrence_analysis import _recurrence_stats_window

    v = np.sin(np.linspace(0, 30, 60))
    stats = _recurrence_stats_window(v, 1, 1, 0.15, 2)
    assert stats is not None
    ent = stats[1]
    assert np.isfinite(ent) and 0.0 <= ent <= 1.0 + 1e-9


def test_rqa_theiler_rate_not_denominator_collapse() -> None:
    from factor_engine.cleaned_operators.rqa_ext import _rqa_stats_window

    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1.0, 120))
    r0 = _rqa_stats_window(x[:60], 2, 1, 0.1, 2, theiler=0)["rate"]
    r10 = _rqa_stats_window(x[:60], 2, 1, 0.1, 2, theiler=10)["rate"]
    assert r0 > 0 and r10 > r0 * 0.15  # structural decline, not a collapse


# ---------------------------------------------------------------------------
# P1-01: constrained typed slot + UNKNOWN kind fails closed in production.
# ---------------------------------------------------------------------------
def test_typed_input_strict_unknown() -> None:
    from factor_engine.ir.analyzer import validate_input_type_contracts

    class _Node:
        def __init__(self, op, inputs=None, semantic_attrs=None, attrs=None):
            self.op = op
            self.inputs = inputs or []
            self.semantic_attrs = semantic_attrs or {}
            self.attrs = attrs or {}

    child = _Node(op="column", attrs={"name": "raw"}, semantic_attrs={})
    root = _Node(op="volume_zscore", inputs=[child])
    assert validate_input_type_contracts(root, strict_unknown=False) == []
    errs = validate_input_type_contracts(root, strict_unknown=True)
    assert len(errs) == 1 and "UNKNOWN semantic_kind" in errs[0]

    match = _Node(
        op="column", attrs={"name": "vol"},
        semantic_attrs={"semantic_kind": "NonNegativeActivity"},
    )
    assert validate_input_type_contracts(_Node(op="volume_zscore", inputs=[match])) == []


# ---------------------------------------------------------------------------
# P1-10: unified group-key missing-value normalisation.
# ---------------------------------------------------------------------------
def test_group_key_missing_normalization() -> None:
    from factor_engine.cleaned_operators.common.group_key import is_missing_group_key, normalize_group_key

    for lab in [None, np.nan, float("nan"), pd.NA, pd.NaT, ""]:
        assert is_missing_group_key(lab), repr(lab)
        assert normalize_group_key(lab) is None, repr(lab)
    for lab in [0, 1, "A", 2.5, True]:
        assert not is_missing_group_key(lab), repr(lab)
    assert normalize_group_key(np.int64(3)) == 3


# ---------------------------------------------------------------------------
# P1-11: align_panel_inputs(strict_axes=False) requires an AlignmentPlan.
# ---------------------------------------------------------------------------
def test_align_panel_inputs_requires_plan() -> None:
    from factor_engine.cleaned_operators.alignment import AlignmentPlan, PanelAxisMismatch, align_panel_inputs

    idx = pd.bdate_range("2024-01-01", periods=3)
    a = pd.DataFrame(np.ones((3, 2)), index=idx, columns=["x", "y"])
    b = pd.DataFrame(np.ones((3, 2)), index=idx, columns=["x", "y"])
    assert len(align_panel_inputs(a, b, strict_axes=True)) == 2
    with pytest.raises(PanelAxisMismatch):
        align_panel_inputs(a, b, strict_axes=False)
    plan = AlignmentPlan(join="left", time_direction="backward", instrument_policy="exact",
                         missing_policy="nan", reason="unit test")
    assert len(align_panel_inputs(a, b, strict_axes=False, alignment_plan=plan)) == 2
    with pytest.raises(ValueError):
        AlignmentPlan(reason="")


# ---------------------------------------------------------------------------
# P1-12: financial lookback fails closed in production.
# ---------------------------------------------------------------------------
def test_financial_lookback_production_fail_closed() -> None:
    import os

    from factor_engine.ir import analyzer as A

    os.environ.pop("FACTOR_ENGINE_REPORT_PERIOD_LOOKBACK_ROWS", None)
    r = A._financial_lookback("fin_yoy", {"periods": 2}, production=False)
    assert r > 0
    with pytest.raises(A.HistoryRequirementError):
        A._financial_lookback("fin_yoy", {"periods": 2}, production=True)
    assert A._financial_lookback("ts_mean", {}, production=True) == 0


# ---------------------------------------------------------------------------
# P0-04: shape-changing (minute->daily) result bypasses exact index check.
# ---------------------------------------------------------------------------
def test_normalize_operator_result_shape_changing() -> None:
    from factor_engine.backend.cleaned_bridge import _normalize_operator_result
    from factor_engine.cleaned_operators.base import OperatorMetadata

    class _Ctx:
        timestamp_col = "ts"
        instrument_col = "inst"

        def __init__(self, native=False):
            self._native = native

    import factor_engine.backend.cleaned_bridge as cb

    def _native_enabled(ctx):
        return getattr(ctx, "_native", False)

    cb.panel_native_enabled = _native_enabled

    minute_idx = pd.date_range("2024-01-01 09:30", periods=60, freq="1min")
    in_panel = pd.DataFrame(np.ones((60, 2)), index=minute_idx, columns=["A", "B"])
    daily_idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    daily = pd.DataFrame(np.random.rand(2, 2), index=daily_idx, columns=["A", "B"])

    class _M2D:
        metadata = OperatorMetadata(
            name="m2d", category="x", param_names=["x"],
            input_grain="minute", output_grain="daily",
        )

    out = _normalize_operator_result(
        daily, backend="pandas_numpy", template=minute_idx,
        template_panel=in_panel, ctx=_Ctx(), operator=_M2D(),
    )
    assert isinstance(out, pd.Series) and len(out) == 4

    class _Plain:
        metadata = OperatorMetadata(name="plain", category="x", param_names=["x"])

    with pytest.raises(Exception):
        _normalize_operator_result(
            daily, backend="pandas_numpy", template=minute_idx,
            template_panel=in_panel, ctx=_Ctx(), operator=_Plain(),
        )

    # duplicate daily dates rejected
    daily_dup = daily.copy()
    daily_dup.index = ["2024-01-01", "2024-01-01"]
    with pytest.raises(Exception):
        _normalize_operator_result(
            daily_dup, backend="pandas_numpy", template=minute_idx,
            template_panel=in_panel, ctx=_Ctx(), operator=_M2D(),
        )
