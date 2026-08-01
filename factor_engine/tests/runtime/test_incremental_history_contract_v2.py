from __future__ import annotations

import pytest


def _incremental_plan():
    from runtime.incremental import build_incremental_plan

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


def test_incremental_plan_issues_one_shot_history_certificate():
    from runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )

    plan = _incremental_plan()
    assert plan.is_full_run is False
    assert plan.load_start is not None
    assert plan.output_start is not None

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


def test_direct_production_run_cannot_bypass_warmup():
    from runtime.incremental import consume_incremental_history_certificate
    from runtime.production_policy import (
        ProductionPolicyViolation,
        assert_production_run_flags,
    )

    consume_incremental_history_certificate()
    with pytest.raises(ProductionPolicyViolation, match="auto_warmup"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="direct",
        )


def test_full_history_incremental_plan_never_issues_tail_certificate():
    from runtime.incremental import (
        FULL_HISTORY_LOOKBACK_SENTINEL,
        build_incremental_plan,
    )
    from runtime.production_policy import (
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
    with pytest.raises(ProductionPolicyViolation, match="auto_warmup"):
        assert_production_run_flags(
            mode="production",
            input_dq_check=True,
            auto_warmup=False,
            pit_enforce=True,
            context="full-history-incremental",
        )
