# -*- coding: utf-8 -*-
"""WS-B PanelSchema / PanelIdentity acceptance tests (review findings #235-#248).

Covers:
* #241  MissingNumeric := null OR NaN — pandas/Polars NaN/null/mixed parity.
* #242  strict panel alignment — never take the column intersection.
* #243/#244  PanelIdentity compared in the polars call validation path
        (shifted date axis / permuted stock columns fail loudly).
* #245  panel_to_polars preserves the DatetimeIndex as ``__fe_time__``.
* #246  strict_polars_long_fallback must read ctx.run_mode first (xfail —
        function lives in backend/polars_long_policy.py, owned by WS-A).
* #247  optional-polars ImportError swallow policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.panel_polars import panel_to_polars, polars_to_panel
from factor_engine.cleaned_operators.base_polars import OperatorMetadata, TwoVarOperator
from factor_engine.cleaned_operators.common._polars_bridge import (
    PanelIdentity,
    align_cols,
    missing_numeric_pandas,
    missing_numeric_polars,
    normalize_missing_numeric,
    polars_nan_to_null,
    should_skip_optional_import_error,
    strip_panel_metadata,
    verify_frames_share_identity,
)

pytest.importorskip("polars")


def _panel(n: int = 20, seed: int = 0, cols: tuple[str, ...] = ("A", "B"), start="2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        np.random.default_rng(seed).normal(0, 1, (n, len(cols))),
        index=dates, columns=list(cols),
    )


def _pl_wide(pdf: pd.DataFrame, time_col: str = "date") -> pl.DataFrame:
    return pl.DataFrame(
        {time_col: pl.Series(pd.DatetimeIndex(pdf.index)),
         **{c: pdf[c].to_numpy() for c in pdf.columns}}
    )


# ---------------------------------------------------------------------------
# #243 / #244 — PanelIdentity equality and enforcement
# ---------------------------------------------------------------------------
def test_panel_identity_same_passes():
    a = _panel(seed=1)
    b = _panel(seed=2)
    assert PanelIdentity.from_frame(a) == PanelIdentity.from_frame(b)
    verify_frames_share_identity("test", a, b)


def test_panel_identity_shifted_date_differs():
    a = _panel(seed=1)
    shifted = a.copy()
    shifted.index = a.index + pd.Timedelta(days=1)
    assert PanelIdentity.from_frame(a) != PanelIdentity.from_frame(shifted)
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("test", a, shifted)


def test_panel_identity_permuted_columns_differs():
    a = _panel(seed=1)
    permuted = a[["B", "A"]]
    assert PanelIdentity.from_frame(a) != PanelIdentity.from_frame(permuted)
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("test", a, permuted)


def test_panel_identity_polars_shifted_date_fails():
    a = _panel(seed=1)
    shifted = a.copy()
    shifted.index = a.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("test", _pl_wide(a), _pl_wide(shifted))


def test_panel_identity_polars_matches_pandas():
    a = _panel(seed=1)
    # A bridge-converted frame (pandas index -> __fe_time__) and a direct polars
    # frame (date column) over the same timeline share the same PanelIdentity.
    via_bridge = panel_to_polars(a)
    direct = _pl_wide(a)
    assert PanelIdentity.from_frame(via_bridge) == PanelIdentity.from_frame(direct)


class _DummyTwoVar(TwoVarOperator):
    metadata = OperatorMetadata(name="_dummy_two_var", category="test", param_names=[])
    def _calculate_series(self, x, y, **kw):
        return x


class _BroadcastTwoVar(TwoVarOperator):
    metadata = OperatorMetadata(
        name="_broadcast_two_var", category="test", param_names=[],
        tags=["daily_to_minute_broadcast"],
    )
    def _calculate_series(self, x, y, **kw):
        return x


def test_operator_prepare_call_enforces_identity():
    op = _DummyTwoVar()
    a = _panel(seed=1)
    b = _panel(seed=2)
    # Same identity -> passes.
    op._prepare_call((panel_to_polars(a), panel_to_polars(b)), {})
    # Shifted date axis on one input -> fails loudly.
    shifted = b.copy()
    shifted.index = b.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="PanelIdentity"):
        op._prepare_call((panel_to_polars(a), panel_to_polars(shifted)), {})
    # Permuted stock columns -> fails loudly.
    permuted = b[["B", "A"]]
    with pytest.raises(ValueError):
        op._prepare_call((panel_to_polars(a), panel_to_polars(permuted)), {})


def test_broadcast_tag_relaxes_strict_identity():
    # A declared typed broadcast (formal BroadcastSpec) is the only waiver.
    op = _BroadcastTwoVar()
    daily = _panel(n=2, seed=1)
    minute = _panel(n=20, seed=2)
    op._prepare_call((panel_to_polars(minute), panel_to_polars(daily)), {})


# ---------------------------------------------------------------------------
# #245 — panel_to_polars preserves the time index
# ---------------------------------------------------------------------------
def test_panel_to_polars_preserves_time_index():
    a = _panel(seed=1)
    pldf = panel_to_polars(a)
    assert "__fe_time__" in pldf.columns
    assert list(pldf["__fe_time__"].to_pandas()) == list(a.index)
    assert PanelIdentity.from_frame(pldf) == PanelIdentity.from_frame(a)


def test_panel_to_polars_idempotent():
    a = _panel(seed=1)
    pldf = panel_to_polars(a)
    assert panel_to_polars(pldf) is pldf


def test_polars_to_panel_restores_time_index():
    a = _panel(seed=1)
    pldf = panel_to_polars(a)
    out = polars_to_panel(pldf, template=a)
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(a.index)
    assert list(out.columns) == list(a.columns)


def test_strip_panel_metadata_removes_fe_time():
    a = _panel(seed=1)
    pldf = panel_to_polars(a)
    stripped = strip_panel_metadata(pldf)
    assert "__fe_time__" not in stripped.columns
    assert "A" in stripped.columns and "B" in stripped.columns


# ---------------------------------------------------------------------------
# #242 — strict panel alignment (no column intersection)
# ---------------------------------------------------------------------------
def test_align_cols_matching_returns_columns():
    a = _panel(seed=1)
    pa, pb = panel_to_polars(a), panel_to_polars(a.copy())
    assert align_cols(pa, pb) == ["A", "B"]


def test_align_cols_mismatched_columns_fails():
    a = _panel(seed=1)
    b = _panel(seed=2, cols=("A", "C"))
    pa, pb = panel_to_polars(a), panel_to_polars(b)
    with pytest.raises(ValueError, match="align_cols"):
        align_cols(pa, pb)


def test_align_cols_permuted_columns_fails():
    a = _panel(seed=1)
    b = a[["B", "A"]]
    pa, pb = panel_to_polars(a), panel_to_polars(b)
    with pytest.raises(ValueError, match="align_cols"):
        align_cols(pa, pb)


def test_align_cols_shifted_date_fails():
    a = _panel(seed=1)
    shifted = a.copy()
    shifted.index = a.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="PanelIdentity"):
        align_cols(panel_to_polars(a), panel_to_polars(shifted))


def test_composite_fastpath_pl_align_strict():
    try:
        from factor_engine.cleaned_operators.composite_fastpath import _pl_align
    except Exception as exc:  # registry may be mid-edit by WS-A
        pytest.skip(f"composite_fastpath import unavailable (concurrent registry): {exc}")
    a = _panel(seed=1)
    pa, pb = panel_to_polars(a), panel_to_polars(a.copy())
    assert _pl_align(pa, pb) == ["A", "B"]
    b = _panel(seed=2, cols=("A", "C"))
    with pytest.raises(ValueError):
        _pl_align(pa, panel_to_polars(b))


# ---------------------------------------------------------------------------
# #241 — MissingNumeric := null OR NaN parity
# ---------------------------------------------------------------------------
def test_missing_numeric_polars_nan_null_mixed():
    df = pl.DataFrame(
        {
            "x": [1.0, None, np.nan, 4.0],
            "y": [None, np.nan, np.nan, None],
        }
    )
    mask = missing_numeric_polars(df)
    assert mask["x"].to_list() == [False, True, True, False]
    assert mask["y"].to_list() == [True, True, True, True]


def test_missing_numeric_pandas():
    panel = pd.DataFrame({"x": [1.0, np.nan, None]})
    mask = missing_numeric_pandas(panel)
    assert mask["x"].tolist() == [False, True, True]


def test_missing_numeric_parity_pandas_vs_polars():
    panel = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, None, 5.0],
            "b": [np.nan, None, np.nan, np.nan, 5.0],
        }
    )
    pldf = pl.DataFrame(
        {
            "a": [1.0, None, 3.0, None, 5.0],
            "b": [None, None, None, None, 5.0],
        }
    )
    assert missing_numeric_pandas(panel).to_numpy().tolist() == missing_numeric_polars(pldf).to_numpy().tolist()


def test_polars_nan_to_null():
    df = pl.DataFrame({"a": [1.0, np.nan, None, 4.0]})
    out = polars_nan_to_null(df)
    assert out["a"].to_list() == [1.0, None, None, 4.0]


def test_normalize_missing_numeric_polars():
    df = pl.DataFrame({"a": [1.0, np.nan]})
    out = normalize_missing_numeric(df, backend="polars")
    assert out["a"].to_list() == [1.0, None]


# ---------------------------------------------------------------------------
# #247 — optional-polars ImportError swallow policy
# ---------------------------------------------------------------------------
def test_should_skip_optional_import_error_only_swallows_missing_polars():
    mod = "factor_engine.cleaned_operators.common.polars_ops"
    # Missing optional dependency polars -> safe to skip.
    assert should_skip_optional_import_error(
        ModuleNotFoundError("No module named 'polars'", name="polars"), mod
    ) is True
    # Missing module itself / parent package (planned surface) -> safe to skip.
    assert should_skip_optional_import_error(
        ModuleNotFoundError("No module named 'factor_engine.cleaned_operators.ts_model'", name="factor_engine.cleaned_operators.ts_model"),
        "factor_engine.cleaned_operators.ts_model.dynamic_regression",
    ) is True
    # A genuine code error during import -> fatal.
    assert should_skip_optional_import_error(ImportError("boom"), mod) is False
    # Missing non-optional dependency -> fatal.
    assert should_skip_optional_import_error(
        ModuleNotFoundError("No module named 'numpy'", name="numpy"), mod
    ) is False


# ---------------------------------------------------------------------------
# #246 — strict_polars_long_fallback must read ctx.run_mode first.
# R7-246 fixed in backend/polars_long_policy.py (WS-A): ctx.run_mode is
# authoritative, then env, then runtime_stats, then global.
# ---------------------------------------------------------------------------
def test_strict_polars_long_fallback_reads_ctx_run_mode():
    from types import SimpleNamespace
    from factor_engine.backend.polars_long_policy import strict_polars_long_fallback
    import os

    saved = os.environ.get("FACTOR_ENGINE_STRICT_POLARS_LONG")
    try:
        os.environ.pop("FACTOR_ENGINE_STRICT_POLARS_LONG", None)
        ctx = SimpleNamespace(run_mode="production", runtime_stats={})
        assert strict_polars_long_fallback(ctx) is True
        ctx2 = SimpleNamespace(run_mode="research", runtime_stats={})
        assert strict_polars_long_fallback(ctx2) is False
        # ctx.run_mode is authoritative over env override.
        os.environ["FACTOR_ENGINE_STRICT_POLARS_LONG"] = "1"
        assert strict_polars_long_fallback(ctx2) is False  # research ctx wins
        assert strict_polars_long_fallback(None) is True  # env wins w/o ctx
    finally:
        if saved is None:
            os.environ.pop("FACTOR_ENGINE_STRICT_POLARS_LONG", None)
        else:
            os.environ["FACTOR_ENGINE_STRICT_POLARS_LONG"] = saved
