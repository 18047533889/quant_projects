import math

import pyarrow as pa

from status_industry_features import aggregate_industry_features, to_pyarrow_table


def test_aggregates_by_date_and_industry_and_keeps_empty_group():
    rows = [
        {"date": "2024-01-02", "industry": "A", "ret_1d": 0.1, "ma_gap_20d": 1.0, "market_cap": 1.0},
        {"date": "2024-01-02", "industry": "A", "ret_1d": -0.1, "ma_gap_20d": -1.0, "market_cap": 3.0},
        {"date": "2024-01-02", "industry": "", "ret_1d": float("nan"), "ma_gap_20d": float("nan")},
    ]
    result = aggregate_industry_features(rows)
    a = next(row for row in result if row["industry"] == "A")
    empty = next(row for row in result if row["industry"] == "")
    assert a["advancing_ratio"] == 0.5
    assert a["above_ma_ratio"] == 0.5
    assert a["return_dispersion"] == 0.1
    assert math.isnan(empty["advancing_ratio"])
    assert empty["n_members"] == 1


def test_arrow_table_and_weighted_ratio():
    table = pa.table({
        "observation_date": ["d", "d"], "sw_l1": ["A", "A"],
        "ret_1d": [1.0, -1.0], "above_ma_20": [1.0, 0.0], "market_cap": [1.0, 3.0],
    })
    row = aggregate_industry_features(table, columns={"industry": "sw_l1"}, weight_field="market_cap")[0]
    assert row["advancing_ratio_weighted"] == 0.25
    assert row["above_ma_ratio_weighted"] == 0.25
    assert row["leadership_hhi"] == 1.0


def test_no_future_membership_inference_and_arrow_output():
    rows = aggregate_industry_features([
        {"date": 1, "industry": "A", "ret_1d": 0.1},
        {"date": 2, "industry": "B", "ret_1d": 0.1},
    ])
    assert {(x["observation_date"], x["industry"]) for x in rows} == {(1, "A"), (2, "B")}
    assert to_pyarrow_table(rows).num_rows == 2
