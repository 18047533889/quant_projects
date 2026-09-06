import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.return_decomp import OpenCloseReturn
from factor_engine.cleaned_operators.common.polars_state_event import open_close_return


def _calculate(backend, opening, closing, basis):
    if backend == "pandas":
        return OpenCloseReturn()._calculate_series(pd.DataFrame({"A": opening}), pd.DataFrame({"A": closing}), price_basis=basis).to_numpy()
    return open_close_return(pl.DataFrame({"A": opening}), pl.DataFrame({"A": closing}), price_basis=basis).to_numpy()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("basis", ["CONTINUOUS", "RAW", "raw"])
def test_declared_basis_and_positive_finite_prices(backend, basis):
    opening = [10., -10., 0., np.inf, np.nan, 10.]
    closing = [12., 12., 12., 12., 12., -12.]
    actual = _calculate(backend, opening, closing, basis)
    np.testing.assert_allclose(actual[:, 0], [.2, np.nan, np.nan, np.nan, np.nan, np.nan], rtol=0, atol=1e-14)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("basis", ["unknown", "", 42, True, "EITHER", "RETURN"])
def test_unknown_or_nonprice_basis_is_rejected(backend, basis):
    with pytest.raises(ValueError, match="price basis|price_basis"):
        _calculate(backend, [10.], [12.], basis)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_untyped_security_columns_require_explicit_basis(backend):
    with pytest.raises(ValueError, match="price basis|price_basis"):
        _calculate(backend, [10.], [12.], None)
