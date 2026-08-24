from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.storage.datasource import DataSource


class _Source(DataSource):
    def __init__(self):
        self.start_date = "2024-01-01"
        self.end_date = "2024-12-31"
        self.bar_freq = "1d"
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", "2024-12-31", freq="B"), ["A"]],
            names=["timestamp", "instrument"],
        )
        self.series = pd.Series(range(len(index)), index=index, dtype=float)

    def load_column(self, name):
        return self.series


def _incremental_plan():
    from factor_engine.runtime.incremental import build_incremental_plan

    return build_incremental_plan(
        factor_id="momentum",
        analysis_lookback=20,
        watermark={"end_date": "2024-10-31"},
        end_date="2024-12-31",
        lookback_extra=5,
        recompute_tail_bars=5,
        market="us",
        factor_freq="1d",
        source_bar_freq="1d",
    )


def _narrow_for_plan(plan):
    import factor_engine.runtime  # noqa: F401 - installs contract-preserving narrowing
    from factor_engine.storage.time_window import narrow_data_source_for_window

    return narrow_data_source_for_window(
        _Source(),
        start_date=plan.load_start,
        end_date=plan.load_end,
        bar_freq=plan.source_bar_freq,
    )


def test_plan_alone_does_not_certify_incremental_history():
    from factor_engine.runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )

    plan = _incremental_plan()
    assert plan.is_full_run is False
    with pytest.raises(ProductionPolicyViolation, match="incremental_history"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="plan-without-window",
        )


def test_matching_source_narrowing_issues_one_shot_history_certificate():
    from factor_engine.runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )

    plan = _incremental_plan()
    narrowed = _narrow_for_plan(plan)
    assert str(narrowed.start_date).startswith(str(plan.load_start.date()))
    assert str(narrowed.end_date).startswith(str(plan.load_end.date()))

    assert_production_run_flags(
        mode="production",
        input_dq_check=True,
        auto_warmup=False,
        pit_enforce=True,
        context="incremental",
    )
    with pytest.raises(ProductionPolicyViolation, match="incremental_history"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="reused-certificate",
        )


def test_mismatched_source_window_invalidates_pending_plan():
    import factor_engine.runtime  # noqa: F401
    from factor_engine.runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )
    from factor_engine.storage.time_window import narrow_data_source_for_window

    plan = _incremental_plan()
    narrow_data_source_for_window(
        _Source(),
        start_date=pd.Timestamp(plan.load_start) + pd.Timedelta(days=1),
        end_date=plan.load_end,
        bar_freq=plan.source_bar_freq,
    )
    with pytest.raises(ProductionPolicyViolation, match="incremental_history"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="mismatched-window",
        )


def test_direct_production_run_cannot_bypass_warmup():
    from factor_engine.runtime.incremental import clear_incremental_history_contract
    from factor_engine.runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )

    clear_incremental_history_contract()
    with pytest.raises(ProductionPolicyViolation, match="auto_warmup"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="direct",
        )


def test_full_history_incremental_plan_never_issues_tail_certificate():
    from factor_engine.runtime.incremental import (
        FULL_HISTORY_LOOKBACK_SENTINEL,
        build_incremental_plan,
    )
    from factor_engine.runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )

    plan = build_incremental_plan(
        factor_id="kama",
        analysis_lookback=FULL_HISTORY_LOOKBACK_SENTINEL,
        watermark={"end_date": "2024-10-31"},
        end_date="2024-12-31",
        market="us",
        factor_freq="1d",
        source_bar_freq="1d",
    )
    assert plan.is_full_run is True
    assert plan.full_history_required is True
    _narrow_for_plan(plan)
    with pytest.raises(ProductionPolicyViolation, match="auto_warmup"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="full-history-incremental",
        )
