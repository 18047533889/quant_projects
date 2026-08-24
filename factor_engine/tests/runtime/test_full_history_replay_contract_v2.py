from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from factor_engine.storage.datasource import DataSource


class _Source(DataSource):
    def __init__(self, *, full_history_start=None):
        self.start_date = "2024-01-01"
        self.end_date = "2024-12-31"
        self.params = {}
        if full_history_start is not None:
            self.params["full_history_start"] = full_history_start
        self.bar_freq = "1d"

    def load_column(self, name):
        raise AssertionError("warmup planning must not read columns")


@dataclass
class _Factor:
    name: str = "stateful"
    freq: str = "1d"
    universe: str = "us_equities"


class _Engine:
    def __init__(self, source, *, run_mode="production"):
        self.data_source = source
        self.run_mode = run_mode
        self.cache = None

    def with_data_source(self, source, *, fresh_cache=False):
        return _Engine(source, run_mode=self.run_mode)


def _stateful_analysis():
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.ir.analyzer import Analyzer

    factor = parse_factor(
        "KAMA(close,10,2,30)", name="kama", surface="extended"
    )
    return Analyzer().lower(factor.expr)


def test_full_history_analysis_encodes_incremental_sentinel():
    from factor_engine.runtime.incremental import FULL_HISTORY_LOOKBACK_SENTINEL

    analysis = _stateful_analysis()
    assert analysis.requires_full_history is True
    assert analysis.lookback >= FULL_HISTORY_LOOKBACK_SENTINEL


def test_incremental_watermark_is_ignored_without_checkpoint_restore():
    from factor_engine.runtime.incremental import build_incremental_plan

    analysis = _stateful_analysis()
    plan = build_incremental_plan(
        factor_id="kama",
        analysis_lookback=analysis.lookback,
        watermark={"end_date": "2024-10-31"},
        end_date="2024-12-31",
        factor_freq="1d",
        source_bar_freq="1d",
    )
    assert plan.full_history_required is True
    assert plan.is_full_run is True
    assert plan.window_mode == "full_history"
    assert plan.load_start is None
    assert plan.output_start is None


def test_production_full_history_requires_explicit_origin():
    from factor_engine.runtime.production_policy import ProductionPolicyViolation
    from factor_engine.runtime.warmup_service import prepare_run_warmup

    with pytest.raises(ProductionPolicyViolation, match="full_history_start"):
        prepare_run_warmup(
            _Engine(_Source()),
            _Factor(),
            _stateful_analysis(),
            auto_warmup=True,
            trim_warmup=True,
            market="us",
        )


def test_production_full_history_loads_from_declared_origin():
    from factor_engine.runtime.warmup_service import prepare_run_warmup

    warmup = prepare_run_warmup(
        _Engine(_Source(full_history_start="2000-01-03")),
        _Factor(),
        _stateful_analysis(),
        auto_warmup=True,
        trim_warmup=True,
        market="us",
    )
    assert warmup.full_history_required is True
    assert warmup.full_history_satisfied is True
    assert warmup.full_history_start == "2000-01-03"
    assert warmup.run_window is not None
    assert warmup.run_window.actual_load_start == "2000-01-03"
    assert warmup.run_window.requested_start == "2024-01-01"
    assert warmup.run_window.trim_output is True
