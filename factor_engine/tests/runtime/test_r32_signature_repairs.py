import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

def _arrays():
 x=np.linspace(-1,2,36); y=.4+1.7*x+.08*np.sin(np.arange(36)); y[9]=np.nan; x[17]=np.nan
 return pd.DataFrame({"A":y}),pd.DataFrame({"A":x})

def test_ridge_and_huber_signatures_identity_and_oracles():
 load_all();y,x=_arrays(); yp,xp=pl.from_pandas(y).with_columns(pl.Series("date",pd.date_range("2025-01-01",periods=36))),pl.from_pandas(x).with_columns(pl.Series("date",pd.date_range("2025-01-01",periods=36)))
 for name in ("ts_ridge_regression_in_sample_resid","ts_huber_regression_in_sample_resid"):
  params={"window":12,"min_periods":5};
  if "ridge" in name:params["alpha"]=.1
  po=OperatorRegistry.get(name,"pandas_numpy",mode="any").calculate(y,x,**params).to_numpy()[:,0]
  out=OperatorRegistry.get(name,"polars",mode="any").calculate(yp,xp,**params)
  assert out["date"].to_list()==yp["date"].to_list();np.testing.assert_allclose(out["A"].to_numpy(),po,rtol=1e-11,atol=1e-12,equal_nan=True)
  # Independent full-window ridge closed form at final row.
  if "ridge" in name:
   yy=y.A.to_numpy()[-12:];xx=x.A.to_numpy()[-12:];ok=np.isfinite(yy)&np.isfinite(xx);xc=xx[ok]-xx[ok].mean();yc=yy[ok]-yy[ok].mean();slope=(xc@yc)/(xc@xc+.1);intercept=yy[ok].mean()-slope*xx[ok].mean();expect=yy[-1]-(intercept+slope*xx[-1])
   assert abs(out["A"][-1]-expect)<1e-12

def test_quantile_prior_signature_order_and_current_row_exclusion():
 load_all();y,x=_arrays();params={"window":12,"q":.4,"min_periods":5}; op=OperatorRegistry.get("ts_quantile_regression_coeff_prior","polars",mode="any")
 base=op.calculate(pl.from_pandas(y),pl.from_pandas(x),**params)["A"].to_numpy(); changed=y.copy();changed.iloc[25,0]+=1000; alt=op.calculate(pl.from_pandas(changed),pl.from_pandas(x),**params)["A"].to_numpy()
 assert base[25]==alt[25]
 po=OperatorRegistry.get("ts_quantile_regression_coeff_prior","pandas_numpy",mode="any").calculate(y,x,**params).to_numpy()[:,0]
 np.testing.assert_allclose(base,po,rtol=1e-10,atol=1e-11,equal_nan=True)
