# -*- coding: utf-8 -*-
"""Tests for the unified LQTP converter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "factor_engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "factor_engine"))

from scripts.cogalpha_lqtp.lqtp_converter import (
    apply_fitness_direction,
    convert_dsl,
    convert_python,
    convert_auto,
)


def test_dsl_renames_and_tanh() -> None:
    result = convert_dsl(
        "tanh(clip(cs_rank(ts_pct(close, 5)), -3, 3))",
        validate_fe=False,
    )
    assert "cap(" in result.lqtp_formula
    assert "rank(" in result.lqtp_formula
    assert "sigmoid" in result.lqtp_formula
    assert "tanh" not in result.lqtp_formula
    assert "clip" not in result.lqtp_formula
    assert result.status in {"ready", "review"}


def test_and_or_to_infix() -> None:
    result = convert_dsl(
        "where(and_(close > open, volume > 0), close, open)",
        validate_fe=False,
    )
    assert " and " in result.lqtp_formula
    assert "and_" not in result.lqtp_formula
    assert result.status == "ready"


def test_gaussian_is_review() -> None:
    result = convert_dsl(
        "cs_rank_gaussian(safe_div(volume, ema(volume, 20)))",
        validate_fe=False,
    )
    assert result.lqtp_formula.startswith("rank(")
    assert result.status == "review"


def test_atr_blocked() -> None:
    result = convert_dsl(
        "ATR(high, low, close, 14) * ts_pct(close, 5)",
        validate_fe=False,
    )
    assert result.status == "blocked"
    assert "ATR" in ",".join(result.fe_only_ops)


def test_python_simple_translates_or_blocks_cleanly() -> None:
    code = '''
def factor_mom(df):
    return (df["close"] / df["close"].shift(5) - 1).rename("f")
'''
    result = convert_python(code, name="factor_mom", validate_fe=False)
    assert result.source_kind == "python"
    assert result.status in {"ready", "review", "blocked"}


def test_auto_detects_python() -> None:
    code = "def f(df):\n    return df['close']\n"
    result = convert_auto(code, validate_fe=False)
    assert result.source_kind == "python"


def test_sanitize_large_int_for_clickhouse() -> None:
    from scripts.cogalpha_lqtp.lqtp_converter import sanitize_lqtp_clickhouse_literals

    fixed = sanitize_lqtp_clickhouse_literals(
        "cap(safe_div(volume, ema(volume, 20)), 0.000000000001, 1000000000000)"
    )
    assert "1000000000000" not in fixed
    assert "1e12" in fixed or "1e+12" in fixed
    assert "1e-12" in fixed


def test_apply_fitness_direction_negates() -> None:
    out, applied = apply_fitness_direction("rank(close)", -1)
    assert applied
    assert out == "-(rank(close))"
    keep, applied2 = apply_fitness_direction("-(rank(close))", 1)
    assert not applied2
    assert keep == "-(rank(close))"
