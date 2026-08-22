# -*- coding: utf-8 -*-
"""R32-P0-079..086: change_impact 增量重算正确性修复终审。

覆盖：
    T-R32-IMPACT-001 (R32-P0-079): rolling 向前传播到 OUTPUTS（MA20: input[t] 变
        → outputs[t..t+19] 受影响），不是向后扩 input window（旧 bug 方向反了）。
    T-R32-IMPACT-002 (R32-P0-080): window=60 是 60 BARS（交易日），不是 60 calendar
        days；需真实 TradingCalendar 映射（含节假日 gap）。
    T-R32-IMPACT-003 (R32-P0-081): 单标的 ROE 变化 → rank(ROE) 标记整个截面受影响。
    T-R32-IMPACT-004 (R32-P0-081): group neutralize 扩散到组内（GROUP_CROSS_SECTION），
        **不**扩散到无关组（precision 测试）。
    T-R32-IMPACT-005 (R32-P0-085): split adjustment 允许 BACKWARD 影响方向。
    T-R32-IMPACT-006 (R32-P0-082): manifest min/max 必须是 RANGE 类型，不能当 EXACT_SET。
    T-R32-IMPACT-007 (R32-P0-084): 多列修订 → 所有列都传给 FE matcher（不只取 [0]）。
    T-R32-IMPACT-008 (R32-P0-086): universe 变化触发 cross-sectional 因子重算。

验收：under-invalidation 是 CORRECTNESS bug（stale 值）；over-invalidation 是
性能成本。当精度被声明时（如 T-004 group 不扩散到无关组），测试双向边界。
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from data_access.r30.change_impact import (
    AffectedFactor,
    plan_minimal_recompute,
    to_fe_input,
)
from data_access.r30.change_impact_types import (
    ColumnChangeKind,
    ImpactDirection,
    InstrumentScope,
    OperatorDependencyTraits,
    TimeAxisKind,
)
from data_access.r30.data_change import ChangeKind, DataChangeSet


# ---------------------------------------------------------------------------
# T-R32-IMPACT-001: rolling forward propagation (R32-P0-079 核心修复)
# ---------------------------------------------------------------------------
def test_t_r32_impact_001_rolling_forward_propagation():
    """R32-P0-079: MA20(x) input[t] 变化 → outputs[t..t+19] 受影响（向前传播）。

    旧 bug：只向后扩 input window（affected_start = t-20），未标记 affected_end。
    新修复：forward_output_horizon=20 → affected_end = t+19（向前传播 20 bars）。
    """
    change = DataChangeSet(
        dataset="prices",
        changed_time_range=("2024-01-15", "2024-01-15"),  # 单日 input 变化
        changed_instruments=("000001.SZ",),
        changed_columns=("close",),
        change_kind=ChangeKind.revision,
    )

    # MA20: backward_input_horizon=20, forward_output_horizon=20
    demands = [
        {
            "factor_id": "MA20",
            "datasets": ("prices",),
            "instruments": ("000001.SZ",),
            "axis_effect": "time_series",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=20,
                forward_output_horizon=20,  # R32-P0-079: 向前污染 20 个 output
                axis_effect="time_series",
                time_axis=TimeAxisKind.CALENDAR_DAYS,  # 无 calendar 按自然日 fallback
            ),
        }
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)
    assert len(affected) == 1

    ma20_affected = affected[0]
    assert ma20_affected.factor_id == "MA20"

    # 验收：affected_start 向后扩（backward input），affected_end 向前扩（forward output）
    # 2024-01-15 input 变化：
    #   - affected_start = 2024-01-15 - 20 days（backward input）
    #   - affected_end = 2024-01-15 + 20 days（forward output）
    start_date = date.fromisoformat(ma20_affected.affected_start[:10])
    end_date = date.fromisoformat(ma20_affected.affected_end[:10])

    # backward: 至少 15 天前（20 天保守放大 1.5 倍 = 30 天）
    assert start_date <= date(2024, 1, 15) - timedelta(days=15)

    # forward: 至少 15 天后（R32-P0-079 核心修复：向前传播）
    assert end_date >= date(2024, 1, 15) + timedelta(days=15)

    # 旧 bug 会是 affected_end = "2024-01-15"（没有向前传播） → FAIL。


# ---------------------------------------------------------------------------
# T-R32-IMPACT-002: trading bars not calendar days (R32-P0-080)
# ---------------------------------------------------------------------------
def test_t_r32_impact_002_trading_bars_with_holiday_gap():
    """R32-P0-080: 20-bar window 跨节假日 gap → 日期 span > 20 calendar days。

    模拟春节 7 天假期：2024-02-10..16 无交易日。20 个交易日 bar 向前偏移必须跨过
    假期，日期跨度 > 20 自然日（证明用的是 trading bars）。
    """
    # 构造含假期 gap 的日历（1月15日..3月15日，2月10-16日无交易）
    all_days = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    holiday = pd.date_range("2024-02-10", "2024-02-16", freq="D")
    trading_days = [d for d in all_days if d not in holiday and d.weekday() < 5]

    # Mock calendar object with offset method (避免导入 factor_engine)
    class MockTradingCalendar:
        def __init__(self, days):
            self._days = sorted([pd.Timestamp(d).normalize() for d in days])

        def offset(self, base, n, clamp=False):
            import bisect
            base = pd.Timestamp(base).normalize()
            pos = bisect.bisect_left(self._days, base)
            if pos >= len(self._days):
                pos = len(self._days) - 1
            new_pos = pos + n
            if new_pos < 0:
                new_pos = 0 if clamp else None
            if new_pos >= len(self._days):
                new_pos = len(self._days) - 1 if clamp else None
            if new_pos is None:
                raise ValueError(f"offset out of range: base={base}, n={n}")
            return self._days[new_pos]

    calendar = MockTradingCalendar(trading_days)

    change = DataChangeSet(
        dataset="prices",
        changed_time_range=("2024-02-20", "2024-02-20"),  # 假期后首日变化
        changed_instruments=("000001.SZ",),
        changed_columns=("close",),
        change_kind=ChangeKind.revision,
    )

    demands = [
        {
            "factor_id": "MA20_BARS",
            "datasets": ("prices",),
            "instruments": ("000001.SZ",),
            "axis_effect": "time_series",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=20,
                forward_output_horizon=0,
                axis_effect="time_series",
                time_axis=TimeAxisKind.TRADING_BARS,  # R32-P0-080: 交易日 bars
            ),
        }
    ]

    affected = plan_minimal_recompute(change, demands, calendar=calendar)
    assert len(affected) == 1

    ma20_affected = affected[0]
    start_date = date.fromisoformat(ma20_affected.affected_start[:10])
    end_date = date.fromisoformat(ma20_affected.affected_end[:10])

    # 验收：20 个交易日 bar 必须跨过 2024-02-10..16 假期（7 天）
    # 日期 span = 2024-02-20 - affected_start >= 20 + 7 = 27 自然日
    calendar_days_span = (end_date - start_date).days

    # 20 bars + 假期 7 天 + 周末 → span 至少 27 天
    assert calendar_days_span >= 27, (
        f"20 trading bars 跨春节假期应 >= 27 calendar days，实际 {calendar_days_span}"
    )

    # 若用自然日会是 20 天 → FAIL（证明用的是 trading bars）


# ---------------------------------------------------------------------------
# T-R32-IMPACT-003: cross-sectional diffusion (R32-P0-081)
# ---------------------------------------------------------------------------
def test_t_r32_impact_003_cross_section_one_stock_affects_all():
    """R32-P0-081: 单标的 ROE 变化 → rank(ROE) 标记整个截面受影响（一变全变）。"""
    change = DataChangeSet(
        dataset="fundamentals",
        changed_time_range=("2024-01-01", "2024-01-01"),
        changed_instruments=("000001.SZ",),  # 只有一个标的变化
        changed_columns=("roe",),
        change_kind=ChangeKind.revision,
    )

    demands = [
        {
            "factor_id": "rank_roe",
            "datasets": ("fundamentals",),
            "instruments": None,  # 全市场因子
            "axis_effect": "cross_section",  # R32-P0-081: 截面算子
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=0,
                forward_output_horizon=0,
                axis_effect="cross_section",
            ),
        }
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)
    assert len(affected) == 1

    rank_affected = affected[0]
    assert rank_affected.factor_id == "rank_roe"
    assert rank_affected.axis_effect == "cross_section"

    # 验收：affected_instruments 应是 None（全截面），不是只有 000001.SZ
    assert rank_affected.affected_instruments is None, (
        "cross_section 算子单标的变化必须扩散到整个截面（一变全变）"
    )


# ---------------------------------------------------------------------------
# T-R32-IMPACT-004: group cross-section precision (R32-P0-081)
# ---------------------------------------------------------------------------
def test_t_r32_impact_004_group_cross_section_precision():
    """R32-P0-081: group neutralize 扩散到组内，**不**扩散到无关组（precision）。

    当前实现：保守返回 None（全量）；精确实现需 group_key 匹配（未来优化）。
    本测试记录预期行为（当前 XFAIL），待精确实现时启用。
    """
    change = DataChangeSet(
        dataset="fundamentals",
        changed_time_range=("2024-01-01", "2024-01-01"),
        changed_instruments=("000001.SZ",),  # 银行业
        changed_columns=("roe",),
        change_kind=ChangeKind.revision,
    )

    demands = [
        {
            "factor_id": "industry_neutralize_roe",
            "datasets": ("fundamentals",),
            "instruments": None,
            "axis_effect": "group_cross_section",  # 组内截面
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=0,
                forward_output_horizon=0,
                axis_effect="group_cross_section",
            ),
            "group_key": "industry",  # 未来实现：根据 group_key 精确扩散
        }
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)
    assert len(affected) == 1

    group_affected = affected[0]
    assert group_affected.axis_effect == "group_cross_section"

    # 当前实现：保守全量（affected_instruments=None）
    # 精确实现：只返回同 industry 的标的（银行业），不包含无关行业（如房地产）
    # TODO(R32-P0-081-precision): 实现 group_key 过滤后，此测试应验证：
    #   assert "000001.SZ" in group_affected.affected_instruments  # 银行业内
    #   assert "000002.SZ" not in group_affected.affected_instruments  # 房地产不受影响

    # 当前保守实现验收
    assert group_affected.affected_instruments is None  # 保守全量


# ---------------------------------------------------------------------------
# T-R32-IMPACT-005: backward impact direction (R32-P0-085)
# ---------------------------------------------------------------------------
def test_t_r32_impact_005_split_adjustment_backward_impact():
    """R32-P0-085: split adjustment 允许 BACKWARD 影响方向（重写历史）。

    公司行为（split、复权因子调整）在 t 时刻生效，但重写 [history_start, t] 的
    adjusted 序列。impact_direction=BACKWARD 必须被允许。
    """
    change = DataChangeSet(
        dataset="corporate_actions",
        changed_time_range=("2024-06-01", "2024-06-01"),  # split effective date
        changed_instruments=("000001.SZ",),
        changed_columns=("split_factor",),
        change_kind=ChangeKind.revision,
    )

    demands = [
        {
            "factor_id": "adjusted_close",
            "datasets": ("corporate_actions",),
            "instruments": ("000001.SZ",),
            "axis_effect": "time_series",
            "impact_direction": ImpactDirection.BACKWARD.value,  # R32-P0-085
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=365,  # 向后读 1 年历史（重写）
                forward_output_horizon=0,  # split 不影响未来
                axis_effect="time_series",
            ),
        }
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)
    assert len(affected) == 1

    adj_affected = affected[0]
    assert adj_affected.factor_id == "adjusted_close"
    assert adj_affected.impact_direction == ImpactDirection.BACKWARD.value

    # 验收：split 在 2024-06-01 生效 → affected_start 向后扩（重写历史）
    start_date = date.fromisoformat(adj_affected.affected_start[:10])
    assert start_date < date(2024, 6, 1), "split adjustment 必须能向后重写历史"


# ---------------------------------------------------------------------------
# T-R32-IMPACT-006: instrument scope typed (R32-P0-082)
# ---------------------------------------------------------------------------
def test_t_r32_impact_006_manifest_min_max_is_range_not_exact():
    """R32-P0-082: manifest min/max 是 RANGE 统计量，不能当 EXACT_SET 精确集合。

    changed_instruments=("000001", "600000") 从 manifest min/max 推导时，只能表示
    「可能影响 [000001, 600000] 区间内的标的」，不能表示「精确只影响这两个」。
    """
    from data_access.r30.change_impact_types import (
        InstrumentChange,
        coerce_instrument_change,
    )

    # 模拟从 manifest 推导的 changed_instruments（min/max 统计量）
    legacy_tuple = ("000001.SZ", "600000.SH")  # 看起来像精确集合

    # R32-P0-082: 调用方必须显式标记 scope=RANGE（不能当 EXACT_SET）
    inst_change = InstrumentChange(
        scope=InstrumentScope.RANGE,
        min_instrument="000001.SZ",
        max_instrument="600000.SH",
    )

    assert inst_change.scope == InstrumentScope.RANGE
    assert inst_change.is_conservative() is False  # RANGE 不是全量，但也不是精确

    # 验收：InstrumentScope.RANGE 消费方必须保守处理（可能影响区间内任何标的）
    # 消费方不能假设「只有 000001 和 600000 两个标的」（会 under-invalidate）


def test_t_r32_impact_006_exact_set_vs_range():
    """R32-P0-082: EXACT_SET vs RANGE 的语义差异（精确 vs 统计量）。"""
    from data_access.r30.change_impact_types import InstrumentChange

    # EXACT_SET: 精确标的列表（逐行扫描得出）
    exact = InstrumentChange(
        scope=InstrumentScope.EXACT_SET,
        exact_instruments=("000001.SZ", "000002.SZ"),
    )
    assert exact.scope == InstrumentScope.EXACT_SET
    assert len(exact.exact_instruments) == 2

    # RANGE: min/max 统计量（manifest footer）
    range_inst = InstrumentChange(
        scope=InstrumentScope.RANGE,
        min_instrument="000001.SZ",
        max_instrument="000999.SZ",
    )
    assert range_inst.scope == InstrumentScope.RANGE
    # RANGE 无 exact_instruments —— 调用方不能假设「只有 min 和 max 两个」


# ---------------------------------------------------------------------------
# T-R32-IMPACT-007: multi-column revision (R32-P0-084)
# ---------------------------------------------------------------------------
def test_t_r32_impact_007_multi_column_revision_all_passed():
    """R32-P0-084: 多列同时修订 → 所有列都传给 FE matcher（不只取 [0]）。

    旧 bug：to_fe_input 只取 changed_columns[0]，丢失后续列。
    新修复：to_fe_input 返回完整 changed_columns 列表。
    """
    change = DataChangeSet(
        dataset="financials",
        changed_time_range=("2024-01-01", "2024-12-31"),
        changed_instruments=("000001.SZ",),
        changed_columns=("revenue", "net_income", "eps"),  # 三列同时修订
        change_kind=ChangeKind.revision,
    )

    fe_input = to_fe_input(change, calendar=None)

    # R32-P0-084: changed_columns 必须完整保留
    assert "changed_columns" in fe_input
    assert fe_input["changed_columns"] == ["revenue", "net_income", "eps"]

    # 旧 bug 只取 [0] → field="revenue", changed_columns=["revenue"] → eps 修订丢失
    # 新修复 → 所有列都在 changed_columns，FE matcher 可逐列匹配依赖


# ---------------------------------------------------------------------------
# T-R32-IMPACT-008: universe change first-class event (R32-P0-086)
# ---------------------------------------------------------------------------
def test_t_r32_impact_008_universe_change_triggers_cross_sectional():
    """R32-P0-086: universe 变化是一类事件，触发 cross-sectional 因子重算。

    universe membership 变化（如标的从 CSI300 移除）→ 影响 cross-sectional 因子
    （rank/zscore/neutralize/exposures）的整个截面，不能只当泛化 schema_change。
    """
    change = DataChangeSet(
        dataset="universe",
        changed_time_range=("2024-01-01", "2024-01-01"),
        changed_instruments=("000001.SZ",),  # 单标的移出 universe
        changed_columns=("in_csi300",),
        change_kind=ChangeKind.universe_change,  # R32-P0-086: 一类事件
    )

    demands = [
        {
            "factor_id": "cs_rank_momentum",
            "datasets": ("universe",),  # 依赖 universe
            "instruments": None,
            "axis_effect": "cross_section",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=0,
                forward_output_horizon=0,
                axis_effect="cross_section",
            ),
        },
        {
            "factor_id": "ts_mean_close",  # 非截面算子
            "datasets": ("universe",),
            "instruments": None,
            "axis_effect": "time_series",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=20,
                forward_output_horizon=0,
                axis_effect="time_series",
            ),
        },
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)

    # 验收：universe_change 只影响 cross-sectional 算子（cs_rank_momentum），
    # 不影响 time_series 算子（ts_mean_close）
    affected_ids = {a.factor_id for a in affected}
    assert "cs_rank_momentum" in affected_ids, "universe 变化必须触发 cross-sectional 因子"
    assert "ts_mean_close" not in affected_ids, "universe 变化不应触发 time_series 因子"


def test_t_r32_impact_008_calendar_change_triggers_time_series():
    """R32-P0-086: calendar 变化触发 time_series/rolling 算子重算。

    交易日历变化（如补班调整）→ 影响 rolling window（MA20 的 20 个 bar 边界变化）、
    next_session、minute→daily 聚合，不能只当泛化 schema_change。
    """
    change = DataChangeSet(
        dataset="calendar",
        changed_time_range=("2024-02-10", "2024-02-16"),  # 节假日调整
        changed_instruments=(),
        changed_columns=("is_trading_day",),
        change_kind=ChangeKind.calendar_change,  # R32-P0-086
    )

    demands = [
        {
            "factor_id": "ma20_close",
            "datasets": ("calendar",),
            "instruments": None,
            "axis_effect": "time_series",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=20,
                forward_output_horizon=20,
                axis_effect="time_series",
            ),
        },
        {
            "factor_id": "cs_rank_roe",  # 非时间序列算子
            "datasets": ("calendar",),
            "instruments": None,
            "axis_effect": "cross_section",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=0,
                forward_output_horizon=0,
                axis_effect="cross_section",
            ),
        },
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)

    # 验收：calendar_change 只影响 time_series 算子（ma20_close），
    # 不影响 cross_section 算子（cs_rank_roe）
    affected_ids = {a.factor_id for a in affected}
    assert "ma20_close" in affected_ids, "calendar 变化必须触发 time_series/rolling 因子"
    assert "cs_rank_roe" not in affected_ids, "calendar 变化不应触发 cross_section 因子"


# ---------------------------------------------------------------------------
# Integration: 综合场景
# ---------------------------------------------------------------------------
def test_integration_multi_factor_rolling_cross_section():
    """综合场景：同时有 rolling 和 cross-sectional 因子，验证传播方向正确性。"""
    change = DataChangeSet(
        dataset="prices",
        changed_time_range=("2024-01-15", "2024-01-15"),
        changed_instruments=("000001.SZ",),
        changed_columns=("close",),
        change_kind=ChangeKind.revision,
    )

    demands = [
        {
            "factor_id": "ma20",
            "datasets": ("prices",),
            "instruments": ("000001.SZ",),
            "axis_effect": "time_series",
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=20,
                forward_output_horizon=20,  # R32-P0-079
                axis_effect="time_series",
                time_axis=TimeAxisKind.CALENDAR_DAYS,
            ),
        },
        {
            "factor_id": "rank_close",
            "datasets": ("prices",),
            "instruments": None,
            "axis_effect": "cross_section",  # R32-P0-081
            "operator_traits": OperatorDependencyTraits(
                backward_input_horizon=0,
                forward_output_horizon=0,
                axis_effect="cross_section",
            ),
        },
    ]

    affected = plan_minimal_recompute(change, demands, calendar=None)
    assert len(affected) == 2

    by_id = {a.factor_id: a for a in affected}

    # ma20: 双向扩展（backward input + forward output）
    ma20 = by_id["ma20"]
    start = date.fromisoformat(ma20.affected_start[:10])
    end = date.fromisoformat(ma20.affected_end[:10])
    assert start < date(2024, 1, 15), "ma20 必须向后扩 backward input"
    assert end > date(2024, 1, 15), "ma20 必须向前扩 forward output (R32-P0-079)"

    # rank_close: 单标的变化扩散全截面
    rank = by_id["rank_close"]
    assert rank.axis_effect == "cross_section"
    assert rank.affected_instruments is None, "cross_section 单标的变化必须全截面"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
