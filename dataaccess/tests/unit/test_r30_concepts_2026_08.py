# -*- coding: utf-8 -*-
"""R30-P0-012（Canonical ConceptId + 一等 UnitType）+ R30-P1-003..010 测试。

纯 Python，无真实 store：只测概念对象 / 单位代数 / 规格类型 / 身份折叠。

覆盖：
    - ConceptId 构造 / 解析 / descendant / hash 相等；
    - UnitType 加减乘规则；Money[CNY]+Money[USD] 无 fx 抛 UnitError、有 fx 通过；
    - FrequencySpec parse / label / requires_aggregation；
    - GrainSpec join_cardinality 四种；
    - MarketProfile.resolve ASHARE/US；
    - PriceBasisSpec label；
    - InstrumentIdentity stable_id vs display（ticker rename）；
    - RevisionFidelity.from_pit_fidelity 映射；
    - SemanticColumnVersion semantic epoch；
    - AggregationRecipeIdentity 相同 VWAP 不同 policy → 不同 identity。
"""
from __future__ import annotations

import pytest

from data_access.r30.concepts import (
    ASHARE,
    ConceptId,
    MarketContext,
    UnitError,
    UnitType,
    US,
    check_addable,
    check_multiply,
)
from data_access.r30.specs import (
    ASHARE_PROFILE,
    AggregationRecipeIdentity,
    FrequencySpec,
    GrainSpec,
    InstrumentIdentity,
    InstrumentType,
    MarketProfile,
    PriceBasisSpec,
    RevisionFidelity,
    SemanticColumnVersion,
    US_PROFILE,
)


# ---------------------------------------------------------------------------
# ConceptId（R30-P0-012）
# ---------------------------------------------------------------------------


def test_concept_id_construct_str_eq_hash():
    c = ConceptId("price.close")
    assert str(c) == "price.close"
    assert c.family == "price"
    assert c.subpath == ("price", "close")
    assert c.to_logical_name() == "price.close"
    # 两种构造等价
    assert ConceptId("price", "close") == c
    assert hash(ConceptId("price", "close")) == hash(c)
    assert c.is_valid()


def test_concept_id_multi_segment_examples():
    for name in (
        "price.vwap",
        "return.close_to_close",
        "valuation.market_cap",
        "financial.revenue",
        "financial.net_income.consolidated",
        "financial.net_income.attributable",
        "financial.total_assets",
        "fundamental.roe",
    ):
        c = ConceptId(name)
        assert str(c) == name
        assert c.family == name.split(".")[0]
        assert c.subpath == tuple(name.split("."))


def test_concept_id_descendant():
    assert ConceptId("financial.net_income.consolidated").is_descendant_of(
        ConceptId("financial.net_income")
    )
    assert ConceptId("financial.net_income.consolidated").is_descendant_of("financial")
    assert ConceptId("price.vwap").is_descendant_of("price")
    # 非祖先 / 自身不算后代
    assert not ConceptId("price.close").is_descendant_of(ConceptId("financial.revenue"))
    assert not ConceptId("price.close").is_descendant_of(ConceptId("price.close"))


def test_concept_id_from_logical_name():
    c = ConceptId.from_logical_name("price.vwap")
    assert c == ConceptId("price.vwap")
    assert ConceptId.from_logical_name("a.b.c") == ConceptId("a.b.c")
    # 非法 → None 不抛
    assert ConceptId.from_logical_name("bad name") is None
    assert ConceptId.from_logical_name("Price.Close") is None
    assert ConceptId.from_logical_name("single") is None  # 无点号
    assert ConceptId.from_logical_name(None) is None
    assert ConceptId.from_logical_name("price.close ") == ConceptId("price.close")


def test_concept_id_invalid_raises():
    with pytest.raises(UnitError):
        ConceptId("bad-name")
    with pytest.raises(UnitError):
        ConceptId("Price.Close")
    with pytest.raises(UnitError):
        ConceptId("single")


def test_concept_id_is_descendant_of_str_vs_self():
    assert ConceptId("financial.net_income.consolidated").is_descendant_of(
        "financial.net_income"
    )


# ---------------------------------------------------------------------------
# UnitType（R30-P0-012）
# ---------------------------------------------------------------------------


def test_unit_type_str_and_parse():
    assert str(UnitType.money("CNY")) == "Money[CNY]"
    assert str(UnitType.price("USD")) == "Price[USD]"
    assert str(UnitType.return_()) == "Return"
    assert str(UnitType.ratio()) == "Ratio"
    assert str(UnitType.shares()) == "Shares"
    assert str(UnitType.volume()) == "Volume"
    assert str(UnitType.count()) == "Count"
    assert str(UnitType.days()) == "Days"

    assert UnitType.parse("Money[CNY]") == UnitType.money("CNY")
    assert UnitType.parse("money[cny]") == UnitType.money("CNY")  # 大小写不敏感
    assert UnitType.parse("Price[USD]") == UnitType.price("USD")
    assert UnitType.parse("Return") == UnitType.return_()
    assert UnitType.parse("ratio") == UnitType.ratio()
    with pytest.raises(UnitError):
        UnitType.parse("Money[CNY")


def test_unit_type_add_rules():
    # 同 kind 同币种可加
    assert check_addable(UnitType.money("CNY"), UnitType.money("CNY")) == UnitType.money("CNY")
    # 无量纲可加
    assert check_addable(UnitType.ratio(), UnitType.ratio()) == UnitType.ratio()
    # 不同 kind 不可加
    with pytest.raises(UnitError):
        check_addable(UnitType.ratio(), UnitType.money("CNY"))


def test_unit_type_money_cross_currency_needs_fx():
    with pytest.raises(UnitError) as ei:
        check_addable(UnitType.money("CNY"), UnitType.money("USD"))
    assert "FX" in str(ei.value)
    # 有 FX contract → 通过，以 self 币种为基准
    res = check_addable(UnitType.money("CNY"), UnitType.money("USD"), fx_contract=True)
    assert res == UnitType.money("CNY")


def test_unit_type_multiply_rules():
    # Return × Price → Price 等价
    assert check_multiply(UnitType.return_(), UnitType.price("USD")) == UnitType.price("USD")
    assert check_multiply(UnitType.price("USD"), UnitType.return_()) == UnitType.price("USD")
    # Ratio × Ratio → Ratio
    assert check_multiply(UnitType.ratio(), UnitType.ratio()) == UnitType.ratio()
    # Money / Shares → Price
    assert check_multiply(UnitType.money("USD"), UnitType.shares()) == UnitType.price("USD")
    assert check_multiply(UnitType.shares(), UnitType.money("USD")) == UnitType.price("USD")
    # Ratio 缩放度量单位
    assert check_multiply(UnitType.money("USD"), UnitType.ratio()) == UnitType.money("USD")
    assert check_multiply(UnitType.ratio(), UnitType.shares()) == UnitType.shares()


def test_unit_type_multiply_incompatible_conservative():
    assert not UnitType.money("CNY").can_multiply(UnitType.money("USD"))
    with pytest.raises(UnitError):
        check_multiply(UnitType.money("CNY"), UnitType.money("USD"))
    assert UnitType.return_().can_multiply(UnitType.price("USD")) is True


def test_unit_type_requires_fx():
    assert UnitType.money("CNY").requires_fx(UnitType.money("USD")) is True
    assert UnitType.money("CNY").requires_fx(UnitType.money("CNY")) is False
    assert UnitType.price("CNY").requires_fx(UnitType.price("USD")) is True
    assert UnitType.ratio().requires_fx(UnitType.ratio()) is False
    assert UnitType.money("CNY").requires_fx(UnitType.ratio()) is False


def test_unit_type_from_to_unit_spec_roundtrip():
    from data_access.read.semantic_catalog import UnitSpec

    spec = UnitType.money("CNY").to_unit_spec()
    assert spec.dimension == "money"
    assert spec.currency == "CNY"
    assert UnitType.from_unit_spec(spec) == UnitType.money("CNY")

    spec2 = UnitType.ratio().to_unit_spec()
    assert UnitType.from_unit_spec(spec2) == UnitType.ratio()

    # 直接构造 UnitSpec 映射回（防御性：缺省字段回退 Ratio）
    raw = UnitSpec(dimension="money", currency="USD")
    assert UnitType.from_unit_spec(raw) == UnitType.money("USD")


def test_market_context_presets():
    assert ASHARE.market_id == "ashare"
    assert ASHARE.currency == "CNY"
    assert ASHARE.calendar_id == "ashare"
    assert US.market_id == "us"
    assert US.currency == "USD"
    mc = MarketContext(market_id="hk", currency="HKD")
    assert mc.market_id == "hk"
    assert mc.currency == "HKD"


# ---------------------------------------------------------------------------
# FrequencySpec（R30-P1-003）
# ---------------------------------------------------------------------------


def test_frequency_label_and_parse():
    assert FrequencySpec("minute", 5).label() == "5m"
    assert FrequencySpec("daily", 1).label() == "1d"
    assert FrequencySpec("weekly").label() == "weekly"
    assert FrequencySpec("daily", 5).label() == "5d"

    assert FrequencySpec.parse("5m") == FrequencySpec("minute", 5)
    assert FrequencySpec.parse("1d") == FrequencySpec("daily", 1)
    assert FrequencySpec.parse("daily") == FrequencySpec("daily")
    assert FrequencySpec.parse("weekly") == FrequencySpec("weekly")
    assert FrequencySpec.parse("15m") == FrequencySpec("minute", 15)

    with pytest.raises(UnitError):
        FrequencySpec("hourly")
    with pytest.raises(UnitError):
        FrequencySpec.parse("fortnightly")


def test_frequency_requires_aggregation():
    assert FrequencySpec("minute", 5).requires_aggregation(FrequencySpec("daily")) is True
    assert FrequencySpec("minute", 1).requires_aggregation("daily") is True
    assert FrequencySpec("daily").requires_aggregation(FrequencySpec("minute", 5)) is False
    assert FrequencySpec("minute", 5).requires_aggregation(FrequencySpec("minute", 15)) is True
    assert FrequencySpec("minute", 15).requires_aggregation(FrequencySpec("minute", 5)) is False
    assert FrequencySpec("minute", 5).requires_aggregation(FrequencySpec("minute", 5)) is False


# ---------------------------------------------------------------------------
# GrainSpec（R30-P1-004）
# ---------------------------------------------------------------------------


def test_grain_join_cardinality_four_cases():
    base = GrainSpec(("trade_date", "instrument"))
    assert base.join_cardinality(GrainSpec(("trade_date", "instrument"))) == "1:1"
    assert (
        base.join_cardinality(GrainSpec(("trade_date", "instrument", "period_end")))
        == "1:N"
    )
    assert (
        GrainSpec(("trade_date", "instrument", "period_end")).join_cardinality(base)
        == "N:1"
    )
    assert (
        base.join_cardinality(GrainSpec(("knowledge_date", "instrument")))
        == "N:N"
    )
    assert str(base) == "trade_date|instrument"
    assert GrainSpec("a|b") == GrainSpec(("a", "b"))


# ---------------------------------------------------------------------------
# MarketProfile（R30-P1-005）
# ---------------------------------------------------------------------------


def test_market_profile_resolve():
    assert MarketProfile.resolve("ashare") is ASHARE_PROFILE
    assert MarketProfile.resolve("us") is US_PROFILE
    assert MarketProfile.resolve("hk") is None

    assert ASHARE_PROFILE.timezone == "Asia/Shanghai"
    assert ASHARE_PROFILE.default_currency == "CNY"
    assert ASHARE_PROFILE.calendar_id == "ashare"
    assert US_PROFILE.timezone == "America/New_York"
    assert US_PROFILE.default_currency == "USD"
    assert US_PROFILE.calendar_id == "us"


# ---------------------------------------------------------------------------
# PriceBasisSpec（R30-P1-006）
# ---------------------------------------------------------------------------


def test_price_basis_label():
    assert PriceBasisSpec.RAW.label() == "raw"
    assert PriceBasisSpec.FORWARD_ADJUSTED.label() == "forward_adjusted"
    assert PriceBasisSpec.BACKWARD_ADJUSTED.label() == "backward_adjusted"
    assert PriceBasisSpec.TOTAL_RETURN.label() == "total_return"
    assert PriceBasisSpec.POINT_IN_TIME_ADJUSTED.label() == "point_in_time_adjusted"


# ---------------------------------------------------------------------------
# InstrumentIdentity（R30-P1-007）
# ---------------------------------------------------------------------------


def test_instrument_identity_stable_vs_display():
    pre = InstrumentIdentity(
        symbol="FB", security_id="META123", market="us", instrument_type=InstrumentType.EQUITY
    )
    post = InstrumentIdentity(
        symbol="META", security_id="META123", market="us", instrument_type=InstrumentType.EQUITY
    )
    # 美股 ticker rename：display 变化，stable security_id 不变
    assert pre.display_symbol() == "FB"
    assert post.display_symbol() == "META"
    assert pre.stable_id() == post.stable_id() == "META123"
    assert pre == pre
    assert pre != post  # symbol 不同 → 身份对象不同


# ---------------------------------------------------------------------------
# RevisionFidelity（R30-P1-008）
# ---------------------------------------------------------------------------


def test_revision_fidelity_from_pit_fidelity():
    assert (
        RevisionFidelity.from_pit_fidelity("knowledge_date_pit")
        == RevisionFidelity.KNOWLEDGE_DATE
    )
    assert (
        RevisionFidelity.from_pit_fidelity("vintage_pit")
        == RevisionFidelity.INGESTION_VINTAGE
    )
    assert (
        RevisionFidelity.from_pit_fidelity("vintage_pit", vendor_vintage=True)
        == RevisionFidelity.TRUE_VENDOR_VINTAGE
    )
    assert RevisionFidelity.from_pit_fidelity("effective_only") == RevisionFidelity.NONE
    assert RevisionFidelity.from_pit_fidelity("unsupported") == RevisionFidelity.NONE
    assert RevisionFidelity.from_pit_fidelity("unknown") == RevisionFidelity.NONE
    assert RevisionFidelity.from_pit_fidelity(None) == RevisionFidelity.NONE


# ---------------------------------------------------------------------------
# SemanticColumnVersion（R30-P1-009）
# ---------------------------------------------------------------------------


def test_semantic_column_version_epoch():
    v = SemanticColumnVersion(
        concept_id=ConceptId("financial.revenue"),
        physical_column="revenue",
        dtype="float64",
        unit=UnitType.money("CNY"),
        definition_version="v2",
        valid_from="2024-01-01",
        valid_to=None,
    )
    assert v.concept_id == ConceptId("financial.revenue")
    assert v.unit == UnitType.money("CNY")
    assert v.physical_column == "revenue"
    assert v.is_current_as_of("2025-01-01") is True
    assert v.is_current_as_of("2024-01-01") is True  # 左闭
    assert v.is_current_as_of("2023-01-01") is False


# ---------------------------------------------------------------------------
# AggregationRecipeIdentity（R30-P1-010）
# ---------------------------------------------------------------------------


def _make_vwap(**overrides) -> AggregationRecipeIdentity:
    base = dict(
        source_frequency=FrequencySpec("minute", 5),
        market_session="us",
        minute_window=5,
        timezone="America/New_York",
        halt_policy="strict",
        missing_bar_policy="ffill",
        early_close_policy="treat_as_regular",
        price_basis=PriceBasisSpec.RAW,
        aggregation_function="VWAP",
    )
    base.update(overrides)
    return AggregationRecipeIdentity(**base)


def test_aggregation_recipe_identity_distinguishes_policy():
    r1 = _make_vwap()
    r_same = _make_vwap()
    assert r1.identity() == r_same.identity()
    # 停牌处理不同 → identity 不同
    assert r1.identity() != _make_vwap(halt_policy="lenient").identity()
    # 价格基准不同 → identity 不同
    assert (
        r1.identity()
        != _make_vwap(price_basis=PriceBasisSpec.BACKWARD_ADJUSTED).identity()
    )
    # 计算时段 / 缺失 bar 策略不同 → identity 不同
    assert r1.identity() != _make_vwap(market_session="ashare").identity()
    assert r1.identity() != _make_vwap(missing_bar_policy="drop").identity()
    assert "VWAP" in r1.label()
