# -*- coding: utf-8 -*-
"""R24-174..181: the label layer owns a LabelIR with CONTROLLED forward
semantics; the feature DSL forbids forward ops; the default label formula is the
single authority and matches the runtime builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.label_pit import (
    LabelIR,
    LabelOp,
    build_forward_return_series,
    build_label_series,
    default_mining_label_config,
    parse_label_expr,
)


def _prices(n=10):
    return pd.DataFrame(
        np.arange(100.0, 100.0 + n), index=pd.date_range("2024-01-01", periods=n),
        columns=["A"],
    )


def test_default_config_formula_is_single_authority() -> None:
    # R24-177/178: config formula matches the runtime builder (forward, not
    # backward ts_pct).
    cfg = default_mining_label_config(horizon_bars=5)
    assert cfg["label_formula"] == "forward_return(close, 5)"
    ir = parse_label_expr(cfg["label_formula"])
    assert ir.horizon_bars == 5
    assert ir.op == LabelOp.FORWARD_RETURN


def test_backward_formula_rejected_as_label() -> None:
    # R24-174: a feature-style backward pct-change is not a label expression.
    with pytest.raises(ValueError, match="forward-label"):
        parse_label_expr("ts_pct(close, 5)")


def test_horizon_strict_positive_int() -> None:
    # R24-181: max(1, int(horizon)) is forbidden.
    with pytest.raises(ValueError, match="> 0"):
        LabelIR(op=LabelOp.FORWARD_RETURN, horizon_bars=0)
    with pytest.raises(ValueError, match="strict positive int"):
        build_forward_return_series(pd.Series([1.0]), horizon=2.5)


def test_build_label_forward_semantics() -> None:
    # R24-249 golden: t close 100, t+5 close 110 → label 0.10 (matching config).
    prices = _prices(6)
    prices["A"] = [100.0, 101.0, 102.0, 105.0, 108.0, 110.0]
    out = build_label_series(prices, ir=LabelIR(op=LabelOp.FORWARD_RETURN, horizon_bars=5))
    assert out["A"].iloc[0] == pytest.approx(0.10)
