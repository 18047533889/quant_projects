"""Native liquidity-beta callables do not contain dormant delegate branches."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.polars_backend_kind import _kernel_is_delegate


@pytest.mark.parametrize("canonical", ["ts_market_liquidity_beta", "ts_industry_liquidity_beta"])
def test_liquidity_beta_native_callable_and_authority_parity(canonical):
    load_all()
    rng = np.random.default_rng(731)
    x = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("ABC"))
    y = pd.DataFrame(rng.normal(size=(80, 3)).cumsum(axis=0), columns=list("ABC"))
    x.iloc[12, 0] = np.nan
    y.iloc[20, 1] = np.inf
    native = OperatorRegistry.get(canonical, "polars", mode="any")
    authority = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    assert not _kernel_is_delegate(native)
    actual = native.calculate(pl.from_pandas(x), pl.from_pandas(y), window=10)
    expected = authority.calculate(x, y, window=10)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(),
                               rtol=1e-9, atol=1e-9, equal_nan=True)
