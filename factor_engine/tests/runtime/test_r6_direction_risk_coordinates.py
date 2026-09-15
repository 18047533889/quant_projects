"""Stable absolute-coordinate identity for risk models at the final registry."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

@pytest.mark.parametrize("timezone",[None,"Asia/Hong_Kong"])
def test_excess_nanosecond_identity_and_slice(timezone):
    load_all()
    rng=np.random.default_rng(301)
    idx=pd.date_range("2025-01-01",periods=45,freq="ns",tz=timezone)
    x=pd.DataFrame({"A":rng.normal(size=45)},index=idx)
    y=x.shift(1).fillna(0)+pd.DataFrame({"A":rng.normal(size=45)*.1},index=idx)
    pop=OperatorRegistry.get("ts_best_lag_corr_excess","pandas_numpy",mode="research")
    op=OperatorRegistry.get("ts_best_lag_corr_excess","polars",mode="research")
    convert=lambda z:pl.from_pandas(z.rename_axis("date").reset_index())
    full=op.calculate(convert(y),convert(x),20,2)
    expected=pop.calculate(y,x,20,2)
    np.testing.assert_allclose(full["A"].to_numpy(),expected.A,equal_nan=True,atol=1e-12)
    part=op.calculate(convert(y.iloc[5:]),convert(x.iloc[5:]),20,2)
    np.testing.assert_allclose(part["A"].to_numpy()[22:],full["A"].to_numpy()[27:],equal_nan=True,atol=1e-12)
    with pytest.raises(ValueError,match="explicit date coordinate"):
        op.calculate(convert(y).drop("date"),convert(x).drop("date"),20,2)
    renamed=op.calculate(convert(y).rename({"date":"__fe_time__"}),convert(x).rename({"date":"__fe_time__"}),20,2)
    np.testing.assert_allclose(renamed["A"].to_numpy(),full["A"].to_numpy(),equal_nan=True,atol=1e-12)
