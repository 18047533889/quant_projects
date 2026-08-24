# -*- coding: utf-8 -*-
"""R40 #226-231: InstrumentNormalizer / SecurityMasterId / limit_ops PriceBasis
+tick grid / ReturnSemantic / AdjustmentPolicy / PriceGridContract."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.market.adjustment_policy import (
    AdjustmentPolicy,
    AdjustmentPolicyViolation,
    AdjustmentVintage,
    validate_adjustment_policy_for_production,
)
from factor_engine.market.instrument import InstrumentKey, InstrumentNormalizer
from factor_engine.market.price_basis import PriceBasis, PriceBasisMismatchError, validate_limit_ops_price_basis
from factor_engine.market.price_grid import PriceGridContract, price_grid_for_market
from factor_engine.market.return_semantic import (
    CorporateActionAdjustment,
    ReturnInterval,
    ReturnSemantic,
)
from factor_engine.market.security_master import SecurityMaster, SecurityMasterId, SymbolValidityInterval


def test_instrument_normalizer_ashare() -> None:
    """#226: 000001.SZ / SZ000001 / 000001 统一到 000001.SZ。"""
    n = InstrumentNormalizer.for_market("ashare")
    assert n.canonical_security_id("000001.SZ") == "000001.SZ"
    assert n.canonical_security_id("SZ000001") == "000001.SZ"
    assert n.canonical_security_id("000001") == "000001.SZ"
    assert n.canonical_security_id("sh600000") == "600000.SH"
    assert n.canonical_security_id("600000") == "600000.SH"
    assert n.display_ticker("000001.SZ") == "000001"
    assert n.provider_symbol("000001.SZ") == "000001"


def test_instrument_normalizer_us_ticker_case() -> None:
    """#226: aapl -> AAPL（uppercase canonical），保留 BRK.B。"""
    n = InstrumentNormalizer.for_market("us")
    assert n.canonical_security_id("aapl") == "AAPL"
    assert n.canonical_security_id("AAPL") == "AAPL"
    assert n.canonical_security_id("brk.b") == "BRK.B"
    assert InstrumentKey.normalized("us", "aapl").instrument == "AAPL"


def test_security_master_id_stable_across_rename() -> None:
    """#227: ticker rename 后 security_master_id 不变。"""
    sid = SecurityMasterId.generate("us", seed="LEH-OLD")
    master = SecurityMaster()
    master.register(
        sid,
        [
            SymbolValidityInterval("LEH", "2000-01-01", "2008-09-15"),
            SymbolValidityInterval("BARCLAYS-US", "2008-09-16", None),
        ],
    )
    before = master.resolve("us", "LEH", "2008-01-01")
    after = master.resolve("us", "BARCLAYS-US", "2010-01-01")
    assert before is not None and after is not None
    assert before == after == sid  # 同一永久主键
    assert master.symbols_for(sid) == ("BARCLAYS-US", "LEH")
    assert master.resolve("us", "LEH", "2009-01-01") is None  # 旧 symbol 已失效


def test_limit_operator_rejects_adjusted_prices_when_raw_required() -> None:
    """#228: 涨跌停算子要求 RAW，混入 ADJUSTED -> PriceBasisMismatchError。"""
    with pytest.raises(PriceBasisMismatchError):
        validate_limit_ops_price_basis({"close": PriceBasis.ADJUSTED})
    # RAW_OFFICIAL_LIMIT 与 RAW 兼容
    validate_limit_ops_price_basis(
        {"close": PriceBasis.RAW, "upper_limit": PriceBasis.RAW_OFFICIAL_LIMIT}
    )


def test_return_semantic_identity_changes_with_adjustment_policy() -> None:
    """#229: ReturnSemantic identity 随 adjustment / interval 变化。"""
    raw = ReturnSemantic(
        source_price_basis=PriceBasis.RAW,
        interval=ReturnInterval.CLOSE_TO_CLOSE,
        corporate_action_adjustment=CorporateActionAdjustment.RAW,
    )
    adj = ReturnSemantic(
        source_price_basis=PriceBasis.ADJUSTED,
        interval=ReturnInterval.CLOSE_TO_CLOSE,
        corporate_action_adjustment=CorporateActionAdjustment.SPLIT_DIV_ADJUSTED,
    )
    tr = ReturnSemantic(
        source_price_basis=PriceBasis.ADJUSTED,
        interval=ReturnInterval.CLOSE_TO_CLOSE,
        corporate_action_adjustment=CorporateActionAdjustment.TOTAL_RETURN,
    )
    assert raw.identity_hash() != adj.identity_hash()
    assert adj.identity_hash() != tr.identity_hash()
    assert adj.identity_hash() == ReturnSemantic().identity_hash()  # 默认 = 复权 close-to-close


def test_pit_adjustment_does_not_leak_future_events() -> None:
    """#230: PIT_ADJUSTED 用 knowledge_time 门控；RETROSPECTIVE 禁 production。"""
    vintage = AdjustmentVintage(as_of="2024-03-01", factor_version="v3")
    # 2024-02-15 时 v3 因子尚未可及 -> 不应使用
    assert vintage.available_at("2024-02-15") is False
    assert vintage.available_at("2024-03-01") is True
    with pytest.raises(AdjustmentPolicyViolation):
        validate_adjustment_policy_for_production(AdjustmentPolicy.RETROSPECTIVE_ADJUSTED)
    validate_adjustment_policy_for_production(AdjustmentPolicy.PIT_ADJUSTED)  # OK


def test_tick_tolerance_from_price_grid_contract_per_market() -> None:
    """#231: tick_tolerance 上限受 PriceGridContract 约束，默认从 grid 取。"""
    grid = price_grid_for_market("ashare")
    assert grid.base_tick == 0.01
    assert grid.validate_tick_tolerance(0.005) == 0.005
    with pytest.raises(ValueError):
        grid.validate_tick_tolerance(0.02)  # 超过 declared_tick_policy_bound
    # 自定义网格
    g = PriceGridContract(market="test", base_tick=0.001, declared_tick_policy_bound=0.002)
    assert g.default_tick_tolerance() == 0.002
    assert g.tick_size("000001", pd.Timestamp("2024-01-01")) == 0.001
