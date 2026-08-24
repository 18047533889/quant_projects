import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / ".local" / "lib" / "python3.12" / "site-packages"))
sys.path.insert(0, str(Path(__file__).parents[1] / "jobs"))

import datetime as dt
import math

import numpy as np
import pyarrow as pa

from ashare_feature_pipeline import (
    INDUSTRY_FEATURE_IDS,
    INDUSTRY_FIELD_MAP,
    build_daily_features,
    formula_hash,
    load_registry,
    metadata_for_registry,
    rolling_product,
    rolling_std,
    rolling_max,
    rolling_sum,
)


def test_registry_is_complete():
    rows = load_registry()
    assert len(rows) == 587
    assert len({row.canonical_id for row in rows}) > 500
    assert sum(row.exec_bucket == "NOW_BUILD" for row in rows) == 144


def test_formula_hash_is_stable():
    assert formula_hash("  x + 1 ") == hashlib.sha256(b"x + 1").hexdigest()


def test_rolling_product_requires_full_window():
    values = rolling_product(__import__("numpy").array([0.1, 0.2, 0.3]), 2)
    assert __import__("numpy").isnan(values[0])
    assert abs(values[1] - 0.32) < 1e-12


def test_rolling_std_and_max_boundaries():
    np = __import__("numpy")
    values = np.array([1.0, 2.0, 3.0])
    assert np.isnan(rolling_std(values, 2)[0])
    result = rolling_max(values, 2)
    assert np.isnan(result[0])
    assert result[1:].tolist() == [2.0, 3.0]


def test_daily_features_include_downside_breadth_and_liquidity():
    pa = __import__("pyarrow")
    np = __import__("numpy")
    from ashare_feature_pipeline import build_daily_features

    dates = [dt.date(2024, 1, 1)] * 4
    table = pa.table({
        "TradeDate": dates,
        "Symbol": ["A", "B", "C", "D"],
        "Open": [10.0, 10.0, 10.0, 10.0],
        "High": [11.0, 10.5, 10.2, 10.1],
        "Low": [9.0, 9.5, 9.8, 9.9],
        "Close": [10.5, 9.5, 10.1, 10.0],
        "PreClose": [10.0] * 4,
        "Volume": [100.0, 200.0, 0.0, 400.0],
        "Amount": [1000.0, 2000.0, 0.0, 4000.0],
        "Return": [500.0, -500.0, 100.0, 0.0],
        "IsSuspend": [False] * 4,
    })
    features, _ = build_daily_features(table, [])
    assert features["advance_ratio"].to_pylist() == [0.5] * 4
    assert features["decline_ratio"].to_pylist() == [0.25] * 4
    assert features["zero_volume_share"].to_pylist() == [0.25] * 4
    assert features["amount_top10_share"].to_pylist() == [1.0] * 4
    assert np.isnan(features["downside_vol_ratio"].to_numpy()[0])


def _sample_daily_and_industry() -> tuple[pa.Table, pa.Table]:
    """Two dates x four symbols, two sw_l1 industries, with 20d MA gaps."""
    dates = [dt.date(2024, 1, 1)] * 4 + [dt.date(2024, 1, 2)] * 4
    symbols = ["A", "B", "C", "D"] * 2
    daily = pa.table({
        "TradeDate": dates,
        "Symbol": symbols,
        "Open": [10.0] * 8,
        "High": [11.0, 10.5, 10.2, 10.1, 11.0, 10.5, 10.2, 10.1],
        "Low": [9.0, 9.5, 9.8, 9.9, 9.0, 9.5, 9.8, 9.9],
        "Close": [10.5, 9.5, 10.1, 10.0, 10.6, 9.6, 10.2, 10.1],
        "PreClose": [10.0] * 8,
        "Volume": [100.0, 200.0, 0.0, 400.0, 100.0, 200.0, 0.0, 400.0],
        "Amount": [1000.0, 2000.0, 0.0, 4000.0, 1000.0, 2000.0, 0.0, 4000.0],
        "Return": [500.0, -500.0, 100.0, 0.0, 600.0, -400.0, 200.0, 100.0],
        "IsSuspend": [False] * 8,
    })
    industry = pa.table({
        "TradeDate": dates,
        "Symbol": symbols,
        "IndustrySource": ["sw_l1"] * 8,
        "IndustryName": ["银行I", "银行I", "有色I", "有色I", "银行I", "银行I", "有色I", "有色I"],
        "IndustryCode": ["801780"] * 4 + ["801050"] * 4,
    })
    return daily, industry


def test_industry_features_appended_when_stock_industry_available():
    daily, industry = _sample_daily_and_industry()
    features, quality = build_daily_features(daily, [], [industry])
    assert all(name in features.column_names for name in INDUSTRY_FIELD_MAP)
    assert quality["industry_source_used"] is True
    assert set(quality["industry_computed_fields"]) == set(INDUSTRY_FIELD_MAP)
    advancing = features["industry_advancing_ratio"].to_pylist()
    # Per-date market-level columns are broadcast to every row of that date.
    for start in (0, 4):
        assert all(value == advancing[start] for value in advancing[start:start + 4])
    # Day 1: 银行I mean 0.0 (advance), 有色I mean 0.05 (advance) -> 0.5.
    assert advancing[0] == 0.5
    # Day 2: 银行I mean 0.10, 有色I mean 0.15 -> 1.0.
    assert advancing[4] == 1.0
    dispersion = features["industry_return_dispersion_5d"].to_pylist()
    assert all(math.isfinite(dispersion[start]) for start in (0, 4))
    assert dispersion[0] != dispersion[4]  # different cross-industry values per date
    hhi = features["industry_leadership_hhi"].to_pylist()
    assert 0.0 < hhi[0] <= 1.0
    entropy = features["industry_leadership_entropy"].to_pylist()
    assert math.isfinite(entropy[0])


def test_industry_features_not_added_without_source():
    daily, _ = _sample_daily_and_industry()
    features, quality = build_daily_features(daily, [], None)
    assert not any(name.startswith("industry_") for name in features.column_names)
    assert quality["industry_source_used"] is False
    assert quality["industry_computed_fields"] == []


def test_industry_features_use_pit_membership_snapshot():
    daily, industry = _sample_daily_and_industry()
    # Day 1 symbols map to industries; day 2 has a different mapping for A only.
    renamed = industry.set_column(
        industry.column_names.index("IndustryName"),
        "IndustryName",
        pa.array(["银行I", "银行I", "有色I", "有色I", "食品饮料I", "银行I", "有色I", "有色I"]),
    )
    features, _ = build_daily_features(daily, [], [renamed])
    advancing = features["industry_advancing_ratio"].to_pylist()
    # Day 1: 银行I mean 0.0, 有色I mean 0.05 -> 0.5.
    assert advancing[0] == 0.5
    # Day 2: 食品饮料I (A, 0.06), 银行I (B, -0.04), 有色I (C+D, 0.015) -> 2/3.
    assert advancing[4] == 2 / 3
    assert features["industry_leadership_hhi"].to_pylist()[4] != features["industry_leadership_hhi"].to_pylist()[0]


def test_metadata_includes_landed_industry_features():
    registry = load_registry()
    meta = metadata_for_registry(registry, [{"key": "k", "rows": 1}])
    fields = [entry["field_name"] for entry in meta]
    landed = [
        "industry_advancing_ratio",
        "industry_above_ma20_ratio",
        "industry_return_dispersion_5d",
        "industry_return_dispersion_20d",
        "industry_leadership_hhi",
        "industry_leadership_entropy",
    ]
    assert INDUSTRY_FEATURE_IDS.issubset(landed)
    for name in landed:
        entry = next(item for item in meta if item["field_name"] == name)
        assert entry["exec_bucket"] == "LANDED"
        assert entry["market"] == "CN-A-share"
        assert len(entry["formula_hash"]) == 64
    assert "industry_leadership_entropy" in fields
    # LANDED F23 rows the pipeline cannot build stay out of the manifest.
    assert "industry_rotation_speed" not in fields
    # Every emitted field appears exactly once.
    assert len(fields) == len(set(fields))


def test_rolling_sum_requires_full_window():
    np = __import__("numpy")
    values = rolling_sum(np.array([1.0, 2.0, 3.0]), 2)
    assert np.isnan(values[0])
    assert values[1:].tolist() == [3.0, 5.0]


def test_registry_contains_stock_daily_buildable_priority_features():
    rows = load_registry()
    ids = {row.feature_id for row in rows if row.family in {"F01", "F02", "F03", "F04"}}
    assert {"downside_vol_ratio", "advance_ratio", "amihud_illiq", "amount_hhi"} <= ids


def test_canary_manifest_has_integrity_fields():
    manifest = Path("/home/sunhaiwei/quantsociety/runs/status-canary-20160104-08-r3/manifests/feature_manifest.json")
    data = json.loads(manifest.read_text())
    assert data["registry_version"] == "587"
    assert data["output"]["rows"] > 0
    assert len(data["output"]["sha256"]) == 64
    assert data["quality"]["publishable_rows"] > 0
    assert data["memory"]["available_bytes"] > data["memory"]["total_bytes"] * 0.2
