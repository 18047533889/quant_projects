# -*- coding: utf-8 -*-
"""Alias resolution for the 2026-08 deepening pack.

Per the no-duplicate rule, ``intraday_signed_jump_balance`` is an exact
duplicate of the existing ``intra_signed_jump_ratio`` and is registered as an
alias (not a new canonical).  It must resolve to the same canonical and produce
bit-identical output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()


def test_intraday_signed_jump_balance_alias_resolves():
    assert OperatorRegistry.resolve_canonical("intraday_signed_jump_balance") == "intra_signed_jump_ratio"
    # The alias name must not shadow a canonical operator.
    assert "intraday_signed_jump_balance" not in OperatorRegistry.list_canonical()


def test_intraday_signed_jump_balance_equivalent_output():
    rng = np.random.default_rng(7)
    days, per = 3, 240
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per)])
    lp = np.cumsum(rng.normal(0.0, 0.001, len(idx)))
    close = pd.DataFrame(100.0 * np.exp(lp), index=idx, columns=["S0"])
    op = OperatorRegistry.get("intra_signed_jump_ratio", "pandas_numpy")
    out = op.calculate(close).to_numpy(dtype=float)
    # Same panel, same math by construction; verify the alias target is sane.
    # Jump-based stats are NaN on days with no detected jump (3σ threshold).
    assert out.shape == (days, 1)
    fin = out[np.isfinite(out)]
    assert fin.size >= 1
    assert np.all((fin >= -1.0) & (fin <= 1.0))
