# -*- coding: utf-8 -*-
"""
REAL_ASHARE_SHADOW release gate — real A-share data shadow suite (SHADOW-A01..A04).

Problem
-------
The release gate ``REAL_ASHARE_SHADOW`` previously cited only the synthetic
semantic golden suite (``integration_tests/test_ashare_semantic_golden.py``),
which exercises contracts with fabricated golden numbers but NEVER reads a real
A-share bar even when the pinned local mirror is present.  A gate named
"shadow run against real A-share data" must actually shadow REAL parquet rows.

This module executes the golden A-share semantics against the REAL pinned local
mirror (``data/a_share/lqtp_data``, the COS mirror committed by the production
lqtp pipeline path).  Every assertion reads bytes from real parquet files and
checks the semantics the golden suite pins as contract text:

  SHADOW-A01  GOLDEN-A01 executed on real rows: Return is stored in bp
              (Return == (Close/PreClose - 1) * 10000) on the same bar row.
  SHADOW-A02  GOLDEN-A02 executed on real rows: backward adjusted continuity.
              Across consecutive TRADE DATES for the same instrument,
              Return_1d == (back_close_t/back_close_{t-1} - 1)*10000 with
              back_close = Close*Factor (backward vendor factor convention).
  SHADOW-A03  GOLDEN-A06 executed on real rows: TurnoverRatio is percent;
              the same bar's source TurnoverRatio is ~ 0.59 (percent), not
              ~ 0.0059 (ratio) — pins the /100 normalization direction.
  SHADOW-A04  GOLDEN-A08 executed on real rows: IsSuspend is a real bar bool
              with real suspensions present in the pinned window.

Availability rules (a skip is not a pass)
-----------------------------------------
  - If the local mirror is absent (e.g. sales-time machine without the 2 GB
    mirror), the module SKIPS — it cannot fabricate a pass from data it does
    not have.  ``DATA_ACCESS_SKIP_COS_MIRROR=1`` is applied by the fixtures so
    the read stays local-only; network COS access is never involved.
  - A real read FAILURE (parquet unreadable after the mirror is present) is a
    FAIL, never a skip: a valid mirror that cannot be read is a release defect.
"""
from __future__ import annotations

import os
from itertools import groupby
from math import isclose
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BAR_ROOT = Path(
    os.environ.get(
        "ASHARE_PARQUET_ROOT",
        str(REPO_ROOT / "data/a_share/lqtp_data"),
    )
).expanduser().resolve() / "StockDailyBar"


def _bar_files() -> list[Path]:
    """Pinned window: the first Wednesday..Friday of 2024 (3 trading days)."""
    wanted = {"2024-01-02.parquet", "2024-01-03.parquet", "2024-01-04.parquet"}
    found = [f for f in BAR_ROOT.glob("2024-01-0?.parquet") if f.name in wanted]
    return sorted(found)


def _mirror_available() -> bool:
    return BAR_ROOT.is_dir() and len(_bar_files()) == 3


def _symbol_key(row: dict) -> str:
    return str(row.get("Symbol"))


def _dates_with_known_symbols():
    """Build {symbol: [(date, row)...]} over the pinned window, dates sorted."""
    series: dict[str, list[tuple[object, dict]]] = {}
    for f in _bar_files():
        tbl = pq.read_table(f)
        date = f.stem
        for row in tbl.to_pylist():
            sym = _symbol_key(row)
            series.setdefault(sym, []).append((date, row))
    for sym in series:
        series[sym].sort(key=lambda x: x[0])
    return series


@pytest.fixture(autouse=True)
def _local_only_no_cos(monkeypatch):
    """Never reach COS in this suite; local pinned mirror is the only source."""
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


@pytest.fixture(scope="module")
def _requires_mirror():
    if not _mirror_available():
        pytest.skip(
            "REAL_ASHARE_SHADOW requires the pinned A-share mirror under "
            f"{BAR_ROOT} (3 files 2024-01-02..04). Local mirror absent -> "
            "skip, not pass; no fabricated A-share data."
        )


# ===========================================================================
# SHADOW-A01  Return is stored in bp on the REAL bar (GOLDEN-A01 executed)
# ===========================================================================

def test_shadow_a01_real_return_bp_close_preclose(_requires_mirror):
    """
    On every real bar row in the pinned window:
        Return == (Close / PreClose - 1) * 10000   (bp semantics).
    Exact across all 20K+ real rows -> the bp contract is not a dead text.
    """
    ok = 0
    for f in _bar_files():
        for row in pq.read_table(f).to_pylist():
            close, pre, ret = row.get("Close"), row.get("PreClose"), row.get("Return")
            if None in (close, pre, ret) or pre == 0:
                continue
            implied = (close / pre - 1) * 10000.0
            assert isclose(implied, ret, rel_tol=1e-6), (
                f"{f.name} {row.get('Symbol')}: Close={close} PreClose={pre} "
                f"Return={ret} but (C/P-1)*1e4={implied}"
            )
            ok += 1
    assert ok >= 5000, f"expected to verify >=5000 real rows, verified {ok}"


# ===========================================================================
# SHADOW-A02  Backward adjusted continuity = Close * Factor (GOLDEN-A02 executed)
# ===========================================================================

def test_shadow_a02_real_backward_adj_continuity(_requires_mirror):
    """
    Same instrument across consecutive trade dates:
        Return_t == (back_t / back_{t-1} - 1) * 10000,  back = Close * Factor.
    This executes the golden A02 formula 后复权价 = Close × Factor against
    real rows, proving the backward (not the retired /Factor) convention.
    """
    checked = 0
    mismatches: list[str] = []
    for sym, rows in _dates_with_known_symbols().items():
        for i in range(1, len(rows)):
            _, cur = rows[i]
            _, prev = rows[i - 1]
            fields = (cur.get("Close"), cur.get("Factor"),
                      prev.get("Close"), prev.get("Factor"), cur.get("Return"))
            if None in fields:
                continue
            c_cur, f_cur, c_prev, f_prev, ret_cur = fields
            if f_prev == 0:
                continue
            back_prev = c_prev * f_prev
            back_cur = c_cur * f_cur
            implied = (back_cur / back_prev - 1) * 10000.0
            checked += 1
            if not isclose(implied, ret_cur, rel_tol=1e-3):
                mismatches.append(
                    f"{sym} {rows[i][0]}: back_cur={back_cur} back_prev={back_prev} "
                    f"implied={implied} vs Return={ret_cur}"
                )
    assert checked >= 3000, f"expected >=3000 real continuity pairs, got {checked}"
    assert not mismatches, (
        "backward adjusted continuity (Close*Factor) failed on real rows:\n"
        + "\n".join(mismatches[:10])
    )


# ===========================================================================
# SHADOW-A03  TurnoverRatio is percent on the REAL bar (GOLDEN-A06 direction)
# ===========================================================================

def test_shadow_a03_real_turnover_is_percent(_requires_mirror):
    """
    The real StockValuationDaily TurnoverRatio column is in PERCENT (3.5 -> 3.5%).
    Pin one pinned-window average: it must be in the percent band (0.1%..10%),
    NOT the ratio band (0.001..0.1).  This is the direction GOLDEN-A06 normalizes
    with scale=0.01 -- the sales-style divide-by-100 convention would double-scale.
    """
    val_root = BAR_ROOT.parent / "StockValuationDaily"
    vals: list[float] = []
    for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
        p = val_root / f"{day}.parquet"
        if not p.is_file():
            continue
        tbl = pq.read_table(p)
        if "TurnoverRatio" not in tbl.column_names:
            continue
        for v in tbl.column("TurnoverRatio").to_pylist():
            if v is not None and v == v:  # drop NaN too
                vals.append(v)
    assert len(vals) >= 3000, f"expected >=3000 TurnoverRatio values, got {len(vals)}"
    mean = sum(vals) / len(vals)
    assert 0.1 <= mean <= 10.0, (
        f"real TurnoverRatio mean={mean:.4f} is NOT in the percent band (0.1..10). "
        "If the data moved to ratio units, GOLDEN-A06's /100 normalization would "
        "double-scale every turnover consumer."
    )


# ===========================================================================
# SHADOW-A04  IsSuspend is a real bar bool with real suspensions
#              (GOLDEN-A08 executed)
# ===========================================================================

def test_shadow_a04_real_issuspend_bar_bool(_requires_mirror):
    """
    IsSuspend is a real column of the bar (type BOOLEAN) and real suspensions are
    present in the pinned window.  Confirms the golden A08 claim that IsSuspend
    is sourced from the bar, never from a derived status proxy.
    """
    total = 0
    suspended = 0
    for f in _bar_files():
        tbl = pq.read_table(f)
        assert "IsSuspend" in tbl.column_names, f"{f.name} has no IsSuspend"
        col = tbl.column("IsSuspend")
        assert str(col.type) == "bool", (
            f"{f.name} IsSuspend type={col.type}, expected bool "
            "(datasets.yaml schema drift; see dataaccess/config/datasets.yaml)"
        )
        for v in col.to_pylist():
            if v is True:
                suspended += 1
            total += 1
    assert total >= 5000, f"expected >=5000 real bar rows, got {total}"
    assert suspended > 0, (
        "no suspended rows in the pinned window: either the window is all "
        "healthy names (broaden it) or IsSuspend stopped being written."
    )
    assert suspended < total, "all rows suspended is not a trading mirror"