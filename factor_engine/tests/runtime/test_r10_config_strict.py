# -*- coding: utf-8 -*-
"""R10 #9/#10/#11/#12 regression tests: the config loader is strongly typed.

* #9  — unknown keys fail fast (``run: {mdoe: production}`` must raise, never
        silently fall back to research).
* #10 — ``"false"`` must parse to False, not Python-truthiness True.
* #11 — fractional integer config (``lookback_extra: 5.9``) is rejected, never
        truncated to 5.
* #12 — MarketExecutionSpec separates timezone/decision_time/bar_timestamp_role
        from the single ``calendar`` string.
"""
from __future__ import annotations

import pytest
import yaml

from factor_engine.runtime.config import MarketExecutionSpec, RunConfig, load_config


def _write(tmp_path, payload: dict) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(p)


def _base_factor() -> dict:
    return {
        "factor": {"name": "f", "expr": "ts_mean(col('close'), 3)"},
        "data_source": {"type": "parquet", "root": "/tmp/x",
                        "timestamp_col": "t", "instrument_col": "i"},
    }


def test_unknown_top_level_key_fails(tmp_path):
    cfg = _base_factor()
    cfg["run"] = {"mdoe": "production"}  # typo
    with pytest.raises(ValueError, match="unknown key"):
        load_config(_write(tmp_path, cfg))


def test_unknown_run_key_fails(tmp_path):
    cfg = _base_factor()
    cfg["run"] = {"mode": "production", "auto_warmup": True, "mdoe": "x"}
    with pytest.raises(ValueError, match="unknown key"):
        load_config(_write(tmp_path, cfg))


def test_false_string_parses_to_false(tmp_path):
    cfg = _base_factor()
    cfg["run"] = {"auto_warmup": "false"}
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded.run.auto_warmup is False


def test_true_string_parses_to_true(tmp_path):
    cfg = _base_factor()
    cfg["run"] = {"trim_warmup": "true"}
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded.run.trim_warmup is True


def test_bool_string_rejected(tmp_path):
    cfg = _base_factor()
    cfg["run"] = {"auto_warmup": "maybe"}
    with pytest.raises(ValueError, match="expected a boolean"):
        load_config(_write(tmp_path, cfg))


def test_fractional_integer_rejected(tmp_path):
    cfg = _base_factor()
    cfg["materialization"] = {"target": "local", "incremental": {"lookback_extra": 5.9}}
    with pytest.raises(ValueError, match="fractional integer"):
        load_config(_write(tmp_path, cfg))


def test_integral_float_accepted(tmp_path):
    cfg = _base_factor()
    cfg["materialization"] = {"target": "local", "incremental": {"lookback_extra": 5.0}}
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded.materialization.incremental.lookback_extra == 5


def test_market_execution_spec_independent(tmp_path):
    cfg = _base_factor()
    cfg["run"] = {
        "mode": "production",
        "market": "ashare",
        "calendar": "SSE",
        "timezone": "Asia/Shanghai",
        "decision_time": "close",
        "bar_timestamp_role": "bar_end",
    }
    loaded = load_config(_write(tmp_path, cfg))
    spec = loaded.run.market_execution
    assert isinstance(spec, MarketExecutionSpec)
    assert spec.market == "ashare"
    assert spec.calendar_id == "SSE"
    assert spec.timezone == "Asia/Shanghai"
    assert spec.decision_time == "close"
    assert spec.bar_timestamp_role == "bar_end"
    # market/calendar are still readable directly on RunConfig (no break).
    assert loaded.run.market == "ashare"
    assert loaded.run.calendar == "SSE"


def test_runconfig_property_shape():
    r = RunConfig(market="us", calendar="NYSE", timezone="America/New_York")
    assert r.market_execution.market == "us"
    assert r.market_execution.calendar_id == "NYSE"
    assert r.market_execution.timezone == "America/New_York"
