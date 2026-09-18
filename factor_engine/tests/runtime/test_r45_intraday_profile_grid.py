"""Daily profile outputs retain identities through unavailable warmup estimates."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = ["intra_return_profile_cosine", "intra_volume_profile_cosine",
         "intra_amount_profile_cosine", "intra_volume_profile_jsd",
         "intra_amount_profile_jsd", "intra_profile_earth_mover_distance"]


def _input(days):
    times = pd.DatetimeIndex([pd.Timestamp("2024-01-02") + pd.Timedelta(days=d, minutes=570+m)
                             for d in range(days) for m in range(4)])
    values = np.tile([1., 2., 3., 4.], days)
    return pl.DataFrame({"QuoteTime": times, "B": values, "A": values * 2})


@pytest.mark.parametrize("name", NAMES)
def test_warmup_and_valid_constant_profile_keep_grid(name):
    load_all()
    frame = _input(4)
    op = OperatorRegistry.get(name, "polars", mode="any")
    out = op.calculate(frame, window=2)
    assert out.columns == ["date", "B", "A"]
    assert out.height == 4
    numbers = out.select("B", "A").to_numpy()
    assert np.isnan(numbers[:2]).all()
    np.testing.assert_allclose(numbers[2:], 1.0 if "cosine" in name else 0.0, atol=1e-12)
    short = op.calculate(frame.head(4), window=2)
    assert short.shape == (1, 3)
    assert np.isnan(short.select("B", "A").to_numpy()).all()
    empty = op.calculate(frame.head(0), window=2)
    assert empty.shape == (0, 3)


@pytest.mark.parametrize("name", NAMES)
def test_entirely_missing_symbol_is_not_dropped(name):
    load_all()
    frame = _input(4).with_columns(pl.lit(float("nan")).alias("A"))
    out = OperatorRegistry.get(name, "polars", mode="any").calculate(frame, window=2)
    assert out.columns == ["date", "B", "A"]
    assert out.height == 4
    assert np.isnan(out["A"].to_numpy()).all()
    assert np.isfinite(out["B"].to_numpy()[2:]).all()


@pytest.mark.parametrize("name", NAMES)
def test_missing_day_ages_history_instead_of_compressing_clock(name):
    load_all()
    frame = _input(5).with_columns(
        pl.when(pl.int_range(pl.len()).is_between(8, 11)).then(float("nan")).otherwise(pl.col("A")).alias("A")
    )
    actual = OperatorRegistry.get(name, "polars", mode="any").calculate(frame, window=2)
    assert np.isnan(actual["A"].to_numpy()[2:5]).all()
    assert np.isfinite(actual["B"].to_numpy()[2:]).all()
