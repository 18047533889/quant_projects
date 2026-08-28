# -*- coding: utf-8 -*-
"""PARITY-A — cross-sectional Polars NaN-propagation parity vs pandas.

Reference (pandas, correct):
  - cs_demean      : x.sub(x.mean(axis=1, skipna=True), axis=0)  — NaN cell stays NaN,
                     finite cell = finite - row-mean-of-finite
  - scale          : x.mul(to / x.abs().sum(axis=1).replace(0,1), axis=0) — NaN stays NaN,
                     finite cell stays finite
  - cs_fill_mean   : NaN cell filled with row mean of finite values (all-NaN row stays NaN)
  - cs_fill_median : NaN cell filled with row median of finite values

Bug pinned (PARITY-A): the Polars cross-sectional kernels treated float NaN as a
numeric value that participates in the row aggregate and collapses the whole row
to NaN (cs_demean / scale), or never matched the NaN cells for filling
(cs_fill_mean / median — `is_null()` does not match float NaN).

Verification path: these tests call the operator CLASSES directly (importing
the canonical pandas + polars kernels) rather than through ``load_all()`` —
``ensure_cleaned_loaded()`` currently fails at collection on a PRE-EXISTING
registration-audit arity mismatch in unrelated legacy polars kernels
(e.g. ``intraday_medrv``, ``candle_gap_atr``, ``ts_count_if``) that another
agent is repairing.  The direct-call path exercises the exact kernels PARITY-A
fixes and is the coordinator-sanctioned verification route.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("polars")
pytest.importorskip("pandas")

# --- direct class imports (the PARITY-A kernels) -----------------------------
from factor_engine.cleaned_operators.common.cross_sectional import (  # noqa: E402
    CrossSectionalDemean,
    CrossSectionalDemeanPolars,
    Scale,
    ScalePolars,
)
from factor_engine.cleaned_operators.layer_primitives import (  # noqa: E402
    pd_cs_fill_mean,
    pd_cs_fill_median,
    pl_cs_fill_mean,
    pl_cs_fill_median,
)


def _panel() -> tuple[pd.DataFrame, pl.DataFrame]:
    """Small panel with NaN cells in each row (the shared-grid shape)."""
    pdf = pd.DataFrame(
        {
            "AAA": [np.nan, 3.0],
            "BBB": [2.0, 6.0],
            "CCC": [4.0, np.nan],
        },
        index=[0, 1],
    )
    plf = pl.from_pandas(pdf.reset_index(drop=True))
    return pdf, plf


def _assert_parity(name: str, pd_out, pl_out, pdf: pd.DataFrame) -> None:
    a_pd = np.asarray(pd_out, dtype=float)
    a_pl = np.asarray(pl_out, dtype=float)
    # 1. NaN cells preserved positionally (NaN stays NaN for demean/scale).
    np.testing.assert_array_equal(
        np.isnan(a_pd),
        np.isnan(a_pl),
        err_msg=f"{name}: NaN mask differs from pandas reference",
    )
    # 2. finite cells must remain finite and equal the pandas reference exactly.
    orig_fin = ~np.isnan(pdf.to_numpy(dtype=float))
    np.testing.assert_allclose(
        a_pl[orig_fin],
        a_pd[orig_fin],
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"{name}: finite cells do not match pandas reference",
    )


def test_cs_demean_direct_nan_panel() -> None:
    pdf, plf = _panel()
    _assert_parity(
        "cs_demean",
        CrossSectionalDemean().calculate(pdf.copy()),
        CrossSectionalDemeanPolars().calculate(plf.clone()).to_pandas(),
        pdf,
    )


def test_scale_direct_nan_panel() -> None:
    pdf, plf = _panel()
    _assert_parity(
        "scale",
        Scale().calculate(pdf.copy()),
        ScalePolars().calculate(plf.clone()).to_pandas(),
        pdf,
    )


def test_scale_to_non_default_preserved() -> None:
    pdf, plf = _panel()
    _assert_parity(
        "scale(to=100)",
        Scale().calculate(pdf.copy(), to=100.0),
        ScalePolars().calculate(plf.clone(), to=100.0).to_pandas(),
        pdf,
    )


def test_cs_fill_mean_direct_nan_panel() -> None:
    pdf, plf = _panel()
    _assert_parity(
        "cs_fill_mean",
        pd_cs_fill_mean(pdf.copy()),
        pl_cs_fill_mean(plf.clone()).to_pandas(),
        pdf,
    )


def test_cs_fill_median_direct_nan_panel() -> None:
    pdf, plf = _panel()
    _assert_parity(
        "cs_fill_median",
        pd_cs_fill_median(pdf.copy()),
        pl_cs_fill_median(plf.clone()).to_pandas(),
        pdf,
    )


def test_fill_mean_cross_section_is_per_row() -> None:
    """Row 0 mean=(2+4)/2=3 -> AAA filled 3; Row 1 mean=(3+6)/2=4.5 -> CCC filled 4.5."""
    _, plf = _panel()
    out = pl_cs_fill_mean(plf.clone()).to_pandas()
    assert out.iloc[0]["AAA"] == pytest.approx(3.0)
    assert out.iloc[1]["CCC"] == pytest.approx(4.5)


def test_fill_median_cross_section_is_per_row() -> None:
    _, plf = _panel()
    out = pl_cs_fill_median(plf.clone()).to_pandas()
    assert out.iloc[0]["AAA"] == pytest.approx(3.0)  # median(2, 4)
    assert out.iloc[1]["CCC"] == pytest.approx(4.5)  # median(3, 6)


def test_fill_all_nan_row_stays_nan() -> None:
    """A row with no finite values must stay NaN (mean/median undefined)."""
    pdf = pd.DataFrame({"A": [np.nan, 1.0], "B": [np.nan, 2.0], "C": [np.nan, 3.0]})
    plf = pl.from_pandas(pdf)
    for pd_fn, pl_fn in ((pd_cs_fill_mean, pl_cs_fill_mean), (pd_cs_fill_median, pl_cs_fill_median)):
        r_pd = np.asarray(pd_fn(pdf.copy()), dtype=float)
        r_pl = np.asarray(pl_fn(plf.clone()).to_pandas(), dtype=float)
        assert np.isnan(r_pd[0, :]).all() and np.isnan(r_pl[0, :]).all()
        assert np.allclose(r_pd[1, :], r_pl[1, :], equal_nan=True)