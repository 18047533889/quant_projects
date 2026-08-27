# -*- coding: utf-8 -*-
"""PERF-2 vectorized daily_agg equivalence — pytest entry.

Delegates to the direct-run harness so the whitelist test also runs as a
normal pytest collection unit.  Does NOT boot the full cleaned bridge (which
is blocked by pre-existing R4-100/R6-157 under this venv); it imports the
kernels and calls daily_agg* directly, exactly like the production operator
does, and compares scalar vs vector output.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.intraday import perf_vec_equiv as pve


@pytest.mark.parametrize("case", range(3))
def test_intra_vec_equiv(case: int) -> None:
    results = pve.run_all(verbose=False)
    label, msg = results[case]
    assert not msg, f"{label} not equivalent: {msg}"


def test_intra_vec_equiv_all() -> None:
    results = pve.run_all(verbose=False)
    assert all(not m for _, m in results), [
        f"{l}: {m}" for l, m in results if m
    ]
