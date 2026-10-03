from __future__ import annotations

import random
from decimal import Decimal, localcontext

import numpy as np
import polars as pl
import pytest

from factor_engine.backend.long_stable_zscore import (
    finite_anchor_centered_zscore_long,
)


def _decimal_row_oracle(values, *, ddof=1, constant=0.0):
    with localcontext() as ctx:
        ctx.prec = 1200
        if any(value is not None and np.isinf(value) for value in values):
            return [
                float("nan") if value is None or np.isnan(value) else constant
                for value in values
            ]
        finite = [
            Decimal.from_float(float(value))
            for value in values
            if value is not None and np.isfinite(value)
        ]
        if len(finite) <= ddof or len(finite) < 2 or min(finite) == max(finite):
            return [
                float("nan") if value is None or np.isnan(value) else constant
                for value in values
            ]
        mean = sum(finite) / len(finite)
        variance = sum((value - mean) ** 2 for value in finite) / (
            Decimal(len(finite)) - Decimal(str(ddof))
        )
        standard_deviation = variance.sqrt()
        return [
            (
                float((Decimal.from_float(float(value)) - mean) / standard_deviation)
                if value is not None and np.isfinite(value)
                else float("nan")
            )
            for value in values
        ]


@pytest.mark.parametrize("ddof", [0, 1, 2, 4])
def test_grouped_long_matches_decimal_and_fp_reference(ddof):
    frame = pl.DataFrame(
        {
            "date": ["d1"] * 7 + ["d2"] * 7,
            "asset": [f"a{i}" for i in range(7)] * 2,
            "value": [1.0, 2.0, 3.0, None, float("nan"), 8.0, 4.0]
            + [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "keep": list(range(14)),
        }
    )
    output = finite_anchor_centered_zscore_long(
        frame, value_col="value", group_col="date", ddof=ddof, constant_value=7.0
    )
    assert output.columns == frame.columns
    assert output["date"].equals(frame["date"])
    assert output["asset"].equals(frame["asset"])
    assert output["keep"].equals(frame["keep"])

    from factor_preprocess.backends.zscore_polars import (
        finite_anchor_centered_zscore_polars,
    )

    fp_reference = finite_anchor_centered_zscore_polars(
        frame,
        value_col="value",
        group_col="date",
        ddof=ddof,
        constant_value=7.0,
    ).to_numpy()
    np.testing.assert_allclose(
        output["value"].to_numpy(), fp_reference, rtol=5e-12, atol=5e-12, equal_nan=True
    )
    for date in ("d1", "d2"):
        mask = [value == date for value in frame["date"].to_list()]
        raw = [value for value, keep in zip(frame["value"].to_list(), mask) if keep]
        expected = _decimal_row_oracle(raw, ddof=ddof, constant=7.0)
        actual = [value for value, keep in zip(output["value"].to_list(), mask) if keep]
        np.testing.assert_allclose(
            actual, expected, rtol=5e-12, atol=5e-12, equal_nan=True
        )


def test_tiny_time_large_universe_permuted_extreme_sparse_inf_and_name_collisions():
    count = 5461
    maximum = np.finfo(np.float64).max
    records = []
    for index in range(count):
        normal = float((index % 37) - 18) / 7.0 + (index % 5) * 0.125
        extreme = (
            maximum if index == 0 else -maximum if index == 1 else float(index % 17 - 8)
        )
        sparse = (
            1.0
            if index == 0
            else (
                2.0
                if index == 100
                else (
                    float("nan")
                    if index == 200
                    else float("inf") if index == 300 else 4.0 if index == 301 else None
                )
            )
        )
        for date, value in (
            ("normal", normal),
            ("extreme", extreme),
            ("sparse", sparse),
        ):
            records.append(
                {
                    "date": date,
                    "asset": f"asset-{index:05d}",
                    "value": value,
                    "__fe_fa_long_mean": f"preserved-{index}",
                    "__fe_fa_long_missing": index,
                }
            )
    random.Random(20261003).shuffle(records)
    frame = pl.DataFrame(records)
    output = finite_anchor_centered_zscore_long(
        frame, value_col="value", group_col="date", ddof=1, constant_value=-3.5
    )
    assert output.columns == frame.columns
    assert output["date"].equals(frame["date"])
    assert output["asset"].equals(frame["asset"])
    assert output["__fe_fa_long_mean"].equals(frame["__fe_fa_long_mean"])
    assert output["__fe_fa_long_missing"].equals(frame["__fe_fa_long_missing"])

    for date in ("normal", "extreme", "sparse"):
        selected = [
            i for i, value in enumerate(frame["date"].to_list()) if value == date
        ]
        source_values = [frame["value"][i] for i in selected]
        actual_values = [output["value"][i] for i in selected]
        expected = _decimal_row_oracle(source_values, ddof=1, constant=-3.5)
        np.testing.assert_allclose(
            actual_values, expected, rtol=8e-12, atol=8e-12, equal_nan=True
        )


def test_empty_frame_dtype_and_large_ddof_shortcut():
    empty = pl.DataFrame(
        schema={"date": pl.String, "asset": pl.String, "value": pl.Int64}
    )
    result = finite_anchor_centered_zscore_long(
        empty, value_col="value", group_col="date"
    )
    assert result.height == 0
    assert result.schema["value"] == pl.Float64
    frame = pl.DataFrame({"date": ["d", "d"], "asset": ["a", "b"], "value": [1.0, 2.0]})
    result = finite_anchor_centered_zscore_long(
        frame,
        value_col="value",
        group_col="date",
        ddof=10**1000,
        constant_value=9.0,
    )
    assert result["value"].to_list() == [9.0, 9.0]


@pytest.mark.parametrize("ddof", [-1, -0.25, float("nan"), float("inf"), True, "0.5"])
def test_rejects_invalid_ddof(ddof):
    frame = pl.DataFrame({"date": ["d"], "value": [1.0]})
    with pytest.raises(ValueError):
        finite_anchor_centered_zscore_long(
            frame, value_col="value", group_col="date", ddof=ddof
        )


@pytest.mark.parametrize("ddof", [0.25, 0.5, 0.75])
def test_fractional_ddof_matches_decimal_across_groups_and_preserves_rows(ddof):
    maximum = np.finfo(np.float64).max
    offset = 1e16
    records = [
        ("normal", "n0", -3.5),
        ("normal", "n1", 0.25),
        ("normal", "n2", 4.0),
        ("normal", "n3", None),
        ("normal", "n4", float("nan")),
        ("extreme", "e0", -maximum),
        ("extreme", "e1", 0.0),
        ("extreme", "e2", maximum),
        ("offset", "o0", offset),
        ("offset", "o1", offset + 2.0),
        ("offset", "o2", offset + 6.0),
        ("sparse_inf", "s0", 1.0),
        ("sparse_inf", "s1", None),
        ("sparse_inf", "s2", float("inf")),
        ("sparse_inf", "s3", 3.0),
    ]
    random.Random(20261003 + int(ddof * 100)).shuffle(records)
    frame = pl.DataFrame(
        {
            "date": [row[0] for row in records],
            "asset": [row[1] for row in records],
            "value": [row[2] for row in records],
            "keep": list(range(len(records))),
        }
    )
    output = finite_anchor_centered_zscore_long(
        frame,
        value_col="value",
        group_col="date",
        ddof=ddof,
        constant_value=-3.25,
    )

    assert output.columns == frame.columns
    assert output.schema["value"] == pl.Float64
    for column in frame.columns:
        if column != "value":
            assert output[column].equals(frame[column])
    for date in dict.fromkeys(frame["date"].to_list()):
        positions = [i for i, group in enumerate(frame["date"].to_list()) if group == date]
        raw = [frame["value"][i] for i in positions]
        actual = [output["value"][i] for i in positions]
        expected = _decimal_row_oracle(raw, ddof=ddof, constant=-3.25)
        np.testing.assert_allclose(
            actual, expected, rtol=8e-12, atol=8e-12, equal_nan=True
        )


@pytest.mark.parametrize("constant", [float("inf"), float("nan"), 1 + 2j, True])
def test_rejects_invalid_constant(constant):
    frame = pl.DataFrame({"date": ["d"], "value": [1.0]})
    with pytest.raises(ValueError):
        finite_anchor_centered_zscore_long(
            frame, value_col="value", group_col="date", constant_value=constant
        )


def test_rejects_invalid_columns_and_boolean_values():
    frame = pl.DataFrame({"date": ["d"], "value": [True]})
    with pytest.raises(TypeError):
        finite_anchor_centered_zscore_long(frame, value_col="value", group_col="date")
    with pytest.raises(ValueError):
        finite_anchor_centered_zscore_long(frame, value_col="missing", group_col="date")
    with pytest.raises(ValueError):
        finite_anchor_centered_zscore_long(frame, value_col="date", group_col="date")
