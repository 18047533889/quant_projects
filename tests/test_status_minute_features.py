"""Tests for status_minute_features (F06 minute-bar features)."""
import datetime as dt
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "jobs"))

from status_minute_features import (
    CN_TZ,
    UTC_TZ,
    FEATURE_IDS,
    close_30m_ret,
    group_intraday_features,
    in_continuous_session,
    intraday_features,
    intraday_reversal,
    intraday_rv,
    intraday_trend_score,
    intraday_vol_of_vol,
    last_hour_amount_share,
    minute_of_day,
    morning_afternoon_spread,
    open_30m_ret,
    prepare_bars,
    to_asia_shanghai,
    volume_u_shape,
    vwap_path_pressure,
)

DATE = dt.date(2024, 1, 2)
SESSION_MINUTES = list(range(571, 691)) + list(range(781, 901))


def _utc_minute(mod: int) -> dt.datetime:
    local = dt.datetime.combine(DATE, dt.time()) + dt.timedelta(minutes=mod)
    return local.replace(tzinfo=CN_TZ).astimezone(UTC_TZ)


def make_bars(
    closes,
    volumes=None,
    amounts=None,
    opens=None,
    minutes=None,
):
    rows = []
    minutes = minutes or SESSION_MINUTES
    for i, mod in enumerate(minutes):
        c = closes[i]
        v = volumes[i] if volumes is not None else 100
        a = amounts[i] if amounts is not None else c * v
        o = opens[i] if opens is not None else c
        rows.append(
            {
                "QuoteTime": _utc_minute(mod),
                "Symbol": "TEST.SZ",
                "TradeDate": DATE,
                "Open": o,
                "High": max(o, c),
                "Low": min(o, c),
                "Close": c,
                "Volume": v,
                "Amount": a,
                "Vwap": a / v if v else 0.0,
            }
        )
    return rows


def make_day(closes):
    return make_bars(closes)


def _n(d, minute):
    return [r for r in d if minute_of_day(to_asia_shanghai(r["QuoteTime"])) == minute][0]


# ---------------------------------------------------------------------------
# Timezone and session handling
# ---------------------------------------------------------------------------


def test_to_asia_shanghai_converts_utc_to_shanghai():
    assert to_asia_shanghai(dt.datetime(2016, 1, 4, 1, 31, tzinfo=dt.timezone.utc)).hour == 9
    assert to_asia_shanghai(dt.datetime(2016, 1, 4, 1, 31, tzinfo=dt.timezone.utc)).minute == 31
    # Naive datetimes are interpreted as UTC.
    assert to_asia_shanghai(dt.datetime(2016, 1, 4, 7, 0)).hour == 15
    # Epoch milliseconds.
    millis = int(dt.datetime(2016, 1, 4, 1, 31, tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert to_asia_shanghai(millis).minute == 31


def test_session_bounds_and_filtering():
    assert in_continuous_session(9 * 60 + 31)
    assert in_continuous_session(11 * 60 + 30)
    assert in_continuous_session(13 * 60 + 1)
    assert in_continuous_session(15 * 60)
    assert not in_continuous_session(9 * 60 + 30)  # 09:30 is auction, not continuous
    assert not in_continuous_session(11 * 60 + 31)  # lunch break
    assert not in_continuous_session(13 * 60)  # 13:00 is auction
    assert not in_continuous_session(15 * 60 + 1)  # after close

    # A 09:30 bar (auction) is dropped even if present in the source.
    pre_auction = make_bars([10.0] * 240)
    pre_auction.append(
        {
            "QuoteTime": _utc_minute(570),
            "Symbol": "TEST.SZ",
            "TradeDate": DATE,
            "Open": 999.0,
            "High": 999.0,
            "Low": 999.0,
            "Close": 999.0,
            "Volume": 100,
            "Amount": 99900.0,
            "Vwap": 999.0,
        }
    )
    prepared = prepare_bars(pre_auction)
    assert len(prepared.times) == 240
    assert 999.0 not in prepared.close


def test_prepare_bars_sorts_by_time_and_detects_nontradable():
    day = make_day([10.0] * 240)
    reversed_rows = list(reversed(day))
    prepared = prepare_bars(reversed_rows)
    assert [m for m in prepared.minutes] == sorted(prepared.minutes.tolist())
    assert len(prepared.times) == 240
    assert prepared.day_open == 10.0
    assert prepared.total_volume == 24000.0

    suspended = make_bars([10.0] * 240, volumes=[0] * 240, amounts=[0.0] * 240)
    assert prepare_bars(suspended).total_volume == 0.0
    assert prepare_bars([]) is None


# ---------------------------------------------------------------------------
# Individual features on hand-computable days
# ---------------------------------------------------------------------------


def test_open_30m_ret_uses_vwap_over_0930_1000():
    closes = [11.0] * 30 + [10.0] * 210
    amounts = [1100.0] * 30 + [1000.0] * 210
    opens = [10.0] + [10.0] * 239
    day = make_bars(closes, amounts=amounts, opens=opens)
    # VWAP(09:31-10:00) = 1100/100 = 11; day open = 10 -> +10%.
    assert open_30m_ret(day) == pytest.approx(0.1)


def test_close_30m_ret_uses_last_30_minutes():
    closes = [10.0] * 239 + [10.5]
    assert close_30m_ret(make_day(closes)) == pytest.approx(0.05)


def test_morning_afternoon_spread_and_reversal():
    closes = [10.0] * 240
    closes[119] = 10.2  # 11:30
    closes[120] = 10.2  # 13:01
    closes[239] = 10.3  # 15:00
    day = make_day(closes)
    afternoon = 10.3 / 10.2 - 1.0
    assert morning_afternoon_spread(day) == pytest.approx(0.02 - afternoon)
    assert intraday_reversal(day) == pytest.approx(afternoon - 0.02)
    assert intraday_reversal(day) == pytest.approx(-morning_afternoon_spread(day))


def test_intraday_rv_uses_5_minute_log_return_blocks():
    closes = [10.0 + 0.1 * i for i in range(10)]
    day = make_bars(closes, minutes=SESSION_MINUTES[:10])
    block_ends = [closes[4], closes[9]]
    expected = (math.log(block_ends[1] / block_ends[0])) ** 2
    assert intraday_rv(day) == pytest.approx(expected)


def test_intraday_trend_score():
    flat = make_day([10.0] * 240)
    assert intraday_trend_score(flat) == 0.0
    # Perfectly log-linear path -> R^2 = 1, score = slope.
    closes = [10.0 * math.exp(0.01 * i) for i in range(240)]
    assert intraday_trend_score(make_day(closes)) == pytest.approx(0.01, rel=1e-3)


def test_volume_u_shape():
    assert volume_u_shape(make_day([10.0] * 240)) == pytest.approx(0.5)
    # Volume only in the first hour (09:31-10:30): share = 1.0.
    volumes = [0] * 240
    for i in range(60):
        volumes[i] = 100
    assert volume_u_shape(make_bars([10.0] * 240, volumes=volumes)) == pytest.approx(1.0)


def test_last_hour_amount_share():
    assert last_hour_amount_share(make_day([10.0] * 240)) == pytest.approx(0.25)
    amounts = [0.0] * 240
    for i in range(180, 240):  # 14:01-15:00
        amounts[i] = 1000.0
    assert last_hour_amount_share(make_bars([10.0] * 240, amounts=amounts)) == pytest.approx(1.0)


def test_vwap_path_pressure():
    assert vwap_path_pressure(make_day([10.0] * 240)) == 0.0
    # Rising closes with amount = close*volume -> close stays above cumulative VWAP.
    closes = [10.0 + i / 240.0 for i in range(240)]
    day = make_day(closes)  # amounts default to close*100
    expected = sum(i / 480.0 for i in range(240)) / (10.0 + 239.0 / 240.0)
    assert vwap_path_pressure(day) == pytest.approx(expected, rel=1e-6)


def test_intraday_vol_of_vol():
    assert intraday_vol_of_vol(make_day([10.0] * 240)) == 0.0
    # Volatility regime change mid-day produces a nonzero vol-of-vol.
    closes = [10.0] * 200 + [10.0 + (i % 2) * 0.2 for i in range(200, 240)]
    assert intraday_vol_of_vol(make_day(closes)) > 0.0


# ---------------------------------------------------------------------------
# Edge cases and aggregation
# ---------------------------------------------------------------------------


def test_suspended_stock_returns_nan_for_every_feature():
    suspended = make_bars([10.0] * 240, volumes=[0] * 240, amounts=[0.0] * 240)
    for name in FEATURE_IDS.values():
        assert math.isnan(intraday_features(suspended)[name])


def test_intraday_features_keys_match_registry_canonical_ids():
    result = intraday_features(make_day([10.0] * 240))
    assert set(result) == set(FEATURE_IDS.values())
    assert set(FEATURE_IDS) == {
        "open_30m_ret",
        "close_30m_ret",
        "morning_afternoon_spread",
        "intraday_rv",
        "intraday_trend_score",
        "intraday_reversal",
        "volume_u_shape",
        "last_hour_amount_share",
        "vwap_path_pressure",
        "intraday_vol_of_vol",
    }


def test_group_intraday_features_groups_by_symbol_and_date():
    rows = []
    rows += make_bars([10.0] * 240, volumes=[100] * 240)
    rows += make_bars([11.0] * 240, volumes=[200] * 240)
    for row in rows:
        row["Symbol"] = "A.SZ" if row["Volume"] == 100 else "B.SZ"
    groups = group_intraday_features(rows)
    assert len(groups) == 2
    symbols = {g["symbol"] for g in groups}
    assert symbols == {"A.SZ", "B.SZ"}
    for g in groups:
        assert g["trade_date"] == DATE
        assert set(FEATURE_IDS.values()) <= set(g)


def test_constant_day_feature_snapshot():
    values = intraday_features(make_day([10.0] * 240))
    assert values["open30_ret"] == pytest.approx(0.0)
    assert values["close30_ret"] == pytest.approx(0.0)
    assert values["morning_afternoon_gap"] == pytest.approx(0.0)
    assert values["intraday_rv"] == pytest.approx(0.0)
    assert values["intraday_trend_score"] == pytest.approx(0.0)
    assert values["intraday_reversal"] == pytest.approx(0.0)
    assert values["volume_u_shape"] == pytest.approx(0.5)
    assert values["last_hour_amount_share"] == pytest.approx(0.25)
    assert values["vwap_path_pressure"] == pytest.approx(0.0)
    assert values["intraday_vol_of_vol"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Real COS schema sanity check (skipped when sample file is absent)
# ---------------------------------------------------------------------------

_SAMPLE = Path(__file__).parents[1] / "runs" / "_minute_schema_probe" / "minute.parquet"


@pytest.mark.skipif(not _SAMPLE.exists(), reason="real COS minute-bar sample not present")
def test_real_sample_produces_plausible_values():
    import pyarrow.parquet as pq

    table = pq.read_table(_SAMPLE)
    rows = [r for r in table.to_pylist() if r["Symbol"] == "000001.SZ"]
    prepared = prepare_bars(rows)
    assert len(prepared.times) == 240
    features = intraday_features(rows)
    assert features["volume_u_shape"] == pytest.approx(0.5, abs=0.25)
    assert 0.0 <= features["last_hour_amount_share"] <= 1.0
    assert abs(features["open30_ret"]) < 0.2
    assert features["intraday_rv"] > 0.0
    assert math.isfinite(features["intraday_trend_score"])
    assert math.isfinite(features["vwap_path_pressure"])
