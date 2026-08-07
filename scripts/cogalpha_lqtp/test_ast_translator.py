# -*- coding: utf-8 -*-
"""Regression tests for CogAlpha Python -> factor_engine DSL translation fixes."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "factor_engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "factor_engine"))

from api.mining_integration import validate_factor_engine_dsl
from scripts.cogalpha_lqtp.ast_translator import (
    _rewrite_drawdown_duration,
    translate_python,
    dsl_to_lqtp,
)
from scripts.cogalpha_lqtp.python_to_dsl import convert_record


def _translate(code: str) -> str:
    res = translate_python(code, tools="")
    assert res.status == "ready", res.notes
    return res.dsl


def test_ewm_halflife_emits_ts_ema_span() -> None:
    """pandas ewm(halflife=N) must lower to daily ts_ema(x, span), not ewm(x, alpha)
    (FE's daily ewm is aliased to the span-based ts_ema)."""
    code = (
        "def f(df):\n"
        "    df_copy = df.copy()\n"
        "    raw = (df_copy['close'] / df_copy['open'] - 1)\n"
        "    smooth = raw.ewm(halflife=7, min_periods=7).mean()\n"
        "    return smooth.rename('f')\n"
    )
    dsl = _translate(code)
    assert "ts_ema(" in dsl
    assert "ewm(" not in dsl
    assert "rename" not in dsl
    ok, msg = validate_factor_engine_dsl(dsl, surface="daily")
    assert ok, msg


def test_series_where_receiver_call_not_mangled() -> None:
    """receiver that is a call expr (e.g. ts_pct(close, 1).where(...)) must not
    mangle into ts_pctwhere(...)."""
    code = (
        "def f(df):\n"
        "    df_copy = df.copy()\n"
        "    returns = df_copy['close'].pct_change()\n"
        "    downside = returns.where(returns < 0, 0)\n"
        "    return downside.rolling(21, min_periods=5).std(ddof=1)\n"
    )
    dsl = _translate(code)
    assert "ts_pctwhere" not in dsl
    assert "where(" in dsl
    ok, msg = validate_factor_engine_dsl(dsl, surface="daily")
    assert ok, msg


def test_bitand_route_is_lqtp_native() -> None:
    """& | ~ in translated DSL must parse on the daily surface and lower to
    LQTP-native infix and/or/not."""
    code = (
        "def f(df):\n"
        "    df_copy = df.copy()\n"
        "    breakout = df_copy['close'] > df_copy['close'].rolling(10, min_periods=10).max().shift(1)\n"
        "    vol_ok = df_copy['volume'] / df_copy['volume'].rolling(10, min_periods=10).mean() > 1\n"
        "    sig = (breakout & vol_ok) * df_copy['close'].pct_change(5)\n"
        "    return sig.rolling(5).sum()\n"
    )
    dsl = _translate(code)
    ok, msg = validate_factor_engine_dsl(dsl, surface="daily")
    assert ok, msg
    lqtp = dsl_to_lqtp(dsl)
    assert " and " in lqtp


def test_drawdown_duration_idiom_rewrites_to_fe_operator() -> None:
    """new_peak.cumsum() + groupby().cumcount() must rewrite to
    ts_current_drawdown_duration so the factor routes to local_dsl."""
    code = (
        "def f(df):\n"
        "    df_copy = df.copy()\n"
        "    rolling_high = df_copy['close'].rolling(window=252, min_periods=20).max()\n"
        "    drawdown = (rolling_high - df_copy['close']) / rolling_high\n"
        "    new_peak = df_copy['close'] >= rolling_high\n"
        "    group = new_peak.cumsum()\n"
        "    duration = new_peak.groupby(group).cumcount()\n"
        "    rank_dd = drawdown.rolling(252, min_periods=20).rank(pct=True)\n"
        "    rank_dur = duration.rolling(252, min_periods=20).rank(pct=True)\n"
        "    return rank_dd + rank_dur\n"
    )
    dsl = _translate(code)
    assert "ts_current_drawdown_duration" in dsl
    ok, msg = validate_factor_engine_dsl(dsl, surface="daily")
    assert ok, msg


def test_drawdown_rewrite_requires_group_resolution() -> None:
    """A groupby/cumcount that does NOT reference the pending cumsum group must be
    left untouched (hard pattern)."""
    code = (
        "def f(df):\n"
        "    df_copy = df.copy()\n"
        "    x = df_copy['close'].cumsum()\n"
        "    y = df_copy['volume'].groupby('z').cumcount()\n"
        "    return x + y\n"
    )
    rewritten = _rewrite_drawdown_duration(code)
    # No drawdown rewrite applied (group not from a peak cumsum).
    assert "ts_current_drawdown_duration" not in rewritten


def test_convert_record_all_parsed_factors_ready() -> None:
    """Every cogalpha parsed factor must convert to a ready DSL that validates on
    the daily surface (or be a deliberate LQTP route)."""
    import json

    parsed_path = ROOT / "data/cogalpha_lqtp_production/screening_reeval_parsed_factors.json"
    if not parsed_path.exists():
        pytest.skip("parsed factors fixture not present")
    records = json.loads(parsed_path.read_text(encoding="utf-8"))
    from scripts.cogalpha_lqtp.lqtp_dsl_compat import is_lqtp_native_dsl

    broken: list[str] = []
    for r in records:
        entry = convert_record(r)
        if entry.status != "ready":
            broken.append(f"{r['function_name']}:{entry.notes[:60]}")
            continue
        # lqtp_dsl routes carry the LQTP formula (infix and/or/not) — validate
        # LQTP-native, not the FE daily surface.  local_dsl routes carry FE DSL.
        if entry.eval_route == "lqtp_dsl":
            if not is_lqtp_native_dsl(entry.dsl):
                broken.append(f"{r['function_name']}:lqtp_invalid:{entry.dsl[:60]}")
        else:
            ok, msg = validate_factor_engine_dsl(entry.dsl, surface="daily")
            if not ok:
                broken.append(f"{r['function_name']}:dsl_invalid:{msg[:60]}")
    assert not broken, f"{len(broken)} factors fail: {broken[:8]}"
