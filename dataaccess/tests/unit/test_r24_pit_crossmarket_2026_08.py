# -*- coding: utf-8 -*-
"""R24 财务 PIT（§31 T-P01..T-P11）+ 跨市场单位/定义（§32 T-X01..T-X07）测试。"""
from __future__ import annotations

import datetime as _dt

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.semantic_catalog import (
    UnitSpec,
    cross_market_compatible,
    get_semantic_catalog,
    reset_semantic_catalog,
)
from data_access.read.session_calendar import (
    MarketCalendar,
    compile_available_from,
    reset_calendars,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_semantic_catalog()
    reset_calendars()
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    yield
    reset_semantic_catalog()
    reset_calendars()


# ---------------------------------------------------------------------------
# T-P07 / T-P08 — date_label vs instant（两条不同代码路径）
# ---------------------------------------------------------------------------
def test_tp07_us_filing_date_label_no_utc_shift():
    """filing_date=2024-05-10 00:00, date_label → semantic date 仍是 2024-05-10，
    不能变纽约 2024-05-09 20:00。"""
    from data_access.read.session_calendar import _to_local_date_time

    knowledge = _dt.datetime(2024, 5, 10, 0, 0)
    d, t = _to_local_date_time(
        knowledge, "America/New_York", time_representation="date_label"
    )
    assert d == _dt.date(2024, 5, 10)  # 不提前一天
    # instant 路径才做时区换算
    d2, t2 = _to_local_date_time(knowledge, "America/New_York", time_representation="instant")
    assert d2 == _dt.date(2024, 5, 9)  # UTC 00:00 → 前一日纽约


def test_tp08_true_utc_instant_shift():
    """FactNews.published_utc instant：UTC → America/New_York。"""
    knowledge = _dt.datetime(2024, 5, 10, 12, 0)
    d, t = _to_local_date_time(knowledge, "America/New_York", time_representation="instant")
    assert d == _dt.date(2024, 5, 10) and t == _dt.time(8, 0)


def _to_local_date_time(*a, **kw):
    from data_access.read.session_calendar import _to_local_date_time as f

    return f(*a, **kw)


# ---------------------------------------------------------------------------
# T-P06 / T-P02 — US date-only filing conservative next session；A 下个 session
# ---------------------------------------------------------------------------
def test_tp06_us_filing_next_session():
    """filing_date=Friday 2024-05-10, next_session_open：
    周五盘中不可用；下一真实交易时段开盘可用。"""
    cal = MarketCalendar(
        "us",
        trading_days=[
            _dt.date(2024, 5, 10),  # Fri
            _dt.date(2024, 5, 13),  # Mon
        ],
        source="explicit",
    )
    k = _dt.datetime(2024, 5, 10, 0, 0)
    # same-day open 不可用：next_session_open → 下一交易时段
    avail = compile_available_from(
        k,
        "next_session_open",
        calendar=cal,
        time_representation="date_label",
        strict=True,
    )
    assert avail.date() == _dt.date(2024, 5, 13)  # 保守到下一真实交易时段
    # same_day 才允许当天（旧错误语义的对照组）
    same = compile_available_from(k, "same_day", calendar=cal, strict=True)
    assert same == k


def test_tp02_a_pubdate_next_session():
    """PubDate=T，decision=T 不可用；decision=下一真实交易日可用（真实日历）。"""
    cal = MarketCalendar(
        "ashare",
        trading_days=[
            _dt.date(2024, 4, 30),  # T
            _dt.date(2024, 5, 6),   # 下一交易日（跳过五一假期/周末）
        ],
        source="explicit",
    )
    k = _dt.datetime(2024, 4, 30, 0, 0)
    avail = compile_available_from(
        k, "next_trading_day", calendar=cal, time_representation="date_label", strict=True
    )
    assert avail == _dt.date(2024, 5, 6)


# ---------------------------------------------------------------------------
# T-P11 — calendar missing → production hard fail（P0-PIT6）
# ---------------------------------------------------------------------------
def test_tp11_calendar_missing_strict_hard_fail(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    k = _dt.datetime(2024, 5, 10, 0, 0)
    with pytest.raises(ValidationError, match="日历不可用"):
        compile_available_from(
            k, "next_trading_day", calendar=None, strict=True
        )
    # research 显式降级才允许（无 strict）
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    out = compile_available_from(k, "next_trading_day", calendar=None, strict=False)
    assert out == k


# ---------------------------------------------------------------------------
# T-P09 — timeframe exactly-one
# ---------------------------------------------------------------------------
def test_tp09_timeframe_exactly_one():
    from data_access.cos_contract import (
        COS_DATASET_CONTRACTS,
        validate_event_filters,
    )

    contract = COS_DATASET_CONTRACTS["us_stock_balance"]
    assert contract.filter_cardinalities == (("timeframe", "exactly_one"),)
    # missing → 拒绝（required_filters 先拦：必须显式提供 timeframe）
    with pytest.raises(ValidationError, match="timeframe"):
        validate_event_filters(contract, None)
    # multiple → 拒绝
    with pytest.raises(ValidationError, match="exactly_one"):
        validate_event_filters(contract, {"timeframe": ["quarterly", "annual"]})
    # invalid → 拒绝
    with pytest.raises(ValidationError):
        validate_event_filters(contract, {"timeframe": "foo"})
    # 恰好一个 enum → 通过
    out = validate_event_filters(contract, {"timeframe": "quarterly"})
    assert out["timeframe"] == "quarterly"


# ---------------------------------------------------------------------------
# T-P04 — A UpdateTime hindsight（pit_fidelity 控制，不用 UpdateTime 冒充历史 PIT）
# ---------------------------------------------------------------------------
def test_tp04_ashare_update_time_is_not_revision_availability():
    from data_access.cos_contract import COS_DATASET_CONTRACTS

    contract = COS_DATASET_CONTRACTS["ashare_stock_balance"]
    # revision_columns 只是 dedup tiebreaker
    assert contract.revision_columns == ("UpdateTime",)
    # revision_availability_time = None：COS 无历史 revision vintage
    assert contract.revision_availability_time is None
    # pit_fidelity：不能承诺 full bitemporal revision-vintage PIT
    assert contract.pit_fidelity == "knowledge_date_pit"
    # knowledge 仍是 PubDate（不是 UpdateTime）
    assert contract.availability_column == "PubDate"


# ---------------------------------------------------------------------------
# T-X01 / T-X02 — Return / ROE 单位归一化 parity
# ---------------------------------------------------------------------------
def test_tx01_return_parity():
    cat = get_semantic_catalog()
    a_ret = cat.resolve_one("return_bp", market="ashare")
    us_ret = cat.resolve_one("ret", market="us")
    # A Return bp → decimal 0.0001；US Ret 已是 decimal
    assert a_ret.scale == 0.0001
    assert us_ret.scale is None or us_ret.scale == 1.0


def test_tx02_roe_parity():
    cat = get_semantic_catalog()
    a_roe = cat.resolve_one("roe", market="ashare")
    us_roe = cat.resolve_one("return_on_equity", market="us")
    assert a_roe.scale == 0.01  # % → ratio
    assert us_roe.scale is None or us_roe.scale == 1.0  # 已 decimal


# ---------------------------------------------------------------------------
# T-X03 — money 跨市场禁止静默混合
# ---------------------------------------------------------------------------
def test_tx03_money_cross_market_rejected():
    cat = get_semantic_catalog()
    a_mcap = cat.resolve_one("market_cap", market="ashare")   # CNY
    us_mcap = cat.resolve_one("market_cap", market="us")      # USD
    ok, reason = cross_market_compatible(a_mcap, us_mcap)
    assert not ok
    assert "FX" in reason or "币种" in reason


# ---------------------------------------------------------------------------
# T-X04 — US dividend currency
# ---------------------------------------------------------------------------
def test_tx04_dividend_currency():
    cat = get_semantic_catalog()
    local = cat.resolve_one("cash_dividend_local", market="us")
    usd = cat.resolve_one("cash_dividend_usd", market="us")
    # local：value + currency，cross_market_comparable=false
    assert local.currency_column == "currency"
    assert local.cross_market_comparable is False
    assert local.requires_fx is True
    # usd：currency=USD（其他币种 unavailable/filtered）
    assert usd.currency == "USD"


# ---------------------------------------------------------------------------
# T-X05 — flow semantics 机器可读
# ---------------------------------------------------------------------------
def test_tx05_flow_semantics():
    cat = get_semantic_catalog()
    a_rev = cat.resolve_one("operating_revenue", market="ashare")
    us_rev = cat.resolve_one("revenue", market="us")
    assert a_rev.flow_semantics == "cumulative_ytd_flow"
    assert us_rev.flow_semantics == "single_period_flow"
    # 必须通过 canonical quarterization 后才可比（这里只证明语义不同源）
    assert a_rev.flow_semantics != us_rev.flow_semantics


# ---------------------------------------------------------------------------
# T-X06 — consolidated vs attributable 是两个概念
# ---------------------------------------------------------------------------
def test_tx06_net_income_attribution_separate():
    cat = get_semantic_catalog()
    cons = cat.resolve_one("net_income_consolidated", market="us")
    attr = cat.resolve_one("net_income_attributable", market="us")
    assert cons.physical_name == "consolidated_net_income_loss"
    assert attr.physical_name == "net_income_loss_attributable_common_shareholders"
    assert cons.physical_name != attr.physical_name


# ---------------------------------------------------------------------------
# T-X07 — instrument market namespace
# ---------------------------------------------------------------------------
def test_tx07_instrument_market_scope():
    from data_access.cos_contract import market_scoped_instrument

    assert market_scoped_instrument("ashare", "000001.SZ") == "ashare:000001.SZ"
    assert market_scoped_instrument("us", "AAPL") == "us:AAPL"
    # 无市场 → 不猜，保持裸 instrument
    assert market_scoped_instrument(None, "000001.SZ") == "000001.SZ"


# ---------------------------------------------------------------------------
# T-P10 — US revised filing（filing_date vintage PIT）
# ---------------------------------------------------------------------------
def test_tp10_us_revised_filing_availability():
    """同一 period，filing t1 之后 t2：decision 在 t1 与 t2 之间 → 用 t1；
    decision 在 t2 之后 → 用 t2。这里验证 revision_availability_time=filing_date
    的契约声明（vintage PIT 由 join 消费）。"""
    from data_access.cos_contract import COS_DATASET_CONTRACTS

    contract = COS_DATASET_CONTRACTS["us_stock_balance"]
    assert contract.revision_availability_time == "filing_date"
    assert contract.pit_fidelity == "vintage_pit"
    # 契约层面保证：knowledge_time=filing_date 是真披露时点，支持逐 filing PIT
    assert contract.availability_column == "filing_date"
