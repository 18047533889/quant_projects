from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.fundamental.research_quality import (
    pd_accounting_comparability_score,
    pd_fiscal_asymmetric_timeliness,
)
from factor_engine.cleaned_operators.polars_native.research_quality_pit_native_20260930 import _compute


def _panels(rows=24):
    names = [f"S{i}" for i in range(6)]
    idx = pd.date_range("2022-01-01", periods=rows)
    fiscal = [f"{2020+i//4}Q{i%4+1}" for i in range(rows)]
    returns = np.asarray([[( (i*7+j*3)%13-6 )/100 for j in range(6)] for i in range(rows)], dtype=float)
    earnings = np.asarray([[0.02 + returns[i,j]*(1.2+j*.07) + ((i+j)%3)*.001 for j in range(6)] for i in range(rows)], dtype=float)
    industries = np.full((rows, 6), "I", dtype=object)
    if rows > 5:
        fiscal[5] = fiscal[4]
        earnings[5, 0], returns[5, 0] = .17, -.14  # a visible revision
    if rows > 12:
        earnings[12, 2] = np.nan  # missing value must not create/update that fiscal event
    frames = [pd.DataFrame(a, index=idx, columns=names) for a in (earnings, returns, industries)]
    periods = pd.DataFrame({n: fiscal for n in names}, index=idx)
    return names, frames[0], frames[1], frames[2], periods


def _polars(frame):
    result = pl.from_pandas(frame.reset_index(drop=True))
    return result.with_columns(pl.Series("date", frame.index))


@pytest.mark.parametrize("revision_policy", ["latest_available", "first_available"])
@pytest.mark.parametrize("explicit", [False, True])
def test_research_quality_pit_native_matches_authority(revision_policy, explicit):
    names, earnings, returns, industry, period_id = _panels()
    kwargs = {"revision_policy": revision_policy}
    if explicit:
        kwargs.update(periods=14, min_periods=10, min_peers=5)
    expected = pd_accounting_comparability_score(earnings, returns, industry, period_id, **kwargs)
    got = _compute({"scaled_earnings": _polars(earnings), "report_return": _polars(returns), "industry": _polars(industry), "period_id": _polars(period_id), **kwargs}, "accounting_comparability_score")
    np.testing.assert_allclose(got.select(names).to_numpy(), expected.to_numpy(), rtol=1e-11, atol=1e-12, equal_nan=True)
    assert got.get_column("date").to_list() == list(earnings.index)

    expected = pd_fiscal_asymmetric_timeliness(earnings, returns, period_id, **{k:v for k,v in kwargs.items() if k != "min_peers"})
    got = _compute({"scaled_earnings": _polars(earnings), "report_return": _polars(returns), "period_id": _polars(period_id), **{k:v for k,v in kwargs.items() if k != "min_peers"}}, "fiscal_asymmetric_timeliness")
    np.testing.assert_allclose(got.select(names).to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True)
    assert got.get_column("date").to_list() == list(earnings.index)


@pytest.mark.parametrize("canonical", ["accounting_comparability_score", "fiscal_asymmetric_timeliness"])
def test_research_quality_prefix_is_causal(canonical):
    names, earnings, returns, industry, period_id = _panels()
    cutoff = 18
    args = {"scaled_earnings": _polars(earnings), "report_return": _polars(returns), "period_id": _polars(period_id), "periods": 16, "min_periods": 12, "revision_policy": "latest_available"}
    if canonical == "accounting_comparability_score":
        args["industry"] = _polars(industry)
        args["min_peers"] = 5
    full = _compute(args, canonical).select(names).to_numpy()
    short = {k:(v.head(cutoff) if isinstance(v,pl.DataFrame) else v) for k,v in args.items()}
    prefix = _compute(short, canonical).select(names).to_numpy()
    np.testing.assert_allclose(full[:cutoff], prefix, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("canonical", ["accounting_comparability_score", "fiscal_asymmetric_timeliness"])
def test_registered_research_quality_native_matches_authority(canonical):
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.polars_backend_kind import canonical_polars_is_delegate

    load_all()
    names, earnings, returns, industry, period_id = _panels()
    panels = [earnings, returns]
    if canonical == "accounting_comparability_score":
        panels.append(industry)
    panels.append(period_id)
    native = OperatorRegistry.get(canonical, "polars", mode="any")
    reference = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    assert native is not None and reference is not None
    assert not canonical_polars_is_delegate(canonical, production_mode=False)
    actual = native.calculate(*[_polars(p) for p in panels])
    expected = reference.calculate(*panels)
    np.testing.assert_allclose(actual.select(names).to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True)
    assert actual.get_column("date").to_list() == list(earnings.index)
