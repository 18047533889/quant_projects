# -*- coding: utf-8 -*-
"""Auto batch materialization tests: all factors in, values land, zero knobs.

Validates the user contract for ``materialize_auto``:
1. every factor's values land (receipts complete, no writer errors);
2. the queue scales: multiple orders of magnitude of factors flow through the
   same all-default path with no manual resource tuning;
3. the automation is auditable: the run reports the resource snapshot it
   executed under;
4. ``write_target="staging"`` routes values through the data-access store.
"""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.expanduser("~/cos_data"))
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")
os.environ.setdefault("QUANTSOCIETY_WORKSPACE_DATA_ROOT",
                      tempfile.mkdtemp(prefix="r66_ws_"))

import factor_engine.cleaned_operators  # noqa: E402,F401
from factor_engine.cleaned_operators import load_all  # noqa: E402

with __import__("warnings").catch_warnings():
    __import__("warnings").simplefilter("ignore")
    load_all()

from factor_engine.api.dsl_parser import parse_factor  # noqa: E402
from factor_engine.backend.factory import build_backend  # noqa: E402
from factor_engine.runtime.auto_materialize import (
    _mem_available_bytes,  # noqa: E402
    materialize_auto,
    snapshot_auto_resources,
)
from factor_engine.storage.sources.data_access_source import DataAccessSource  # noqa: E402
from factor_engine.runtime.engine import FactorEngine  # noqa: E402

INSTRUMENTS = (
    [f"{c}.SZ" for c in ("000001", "000002", "000063", "000100", "000333",
                         "000651", "000725", "000858")]
    + [f"{c}.SH" for c in ("600000", "600016", "600030", "600036", "600104",
                           "600276", "600438", "600519")]
)

# three expression families x varied windows = heterogeneous factor queue
FAMILIES = (
    "ts_mean(close, {w})",
    "rank(ts_mean(volume, {w}))",
    "ts_corr(close, volume, {w})",
    "ts_zscore(close, {w})",
    "ts_std(close, {w})",
    "delta(close, {w})",
)


def _make_factors(n, tag):
    out = []
    for i in range(n):
        expr = FAMILIES[i % len(FAMILIES)].format(w=5 + (i % 12) * 5)
        out.append(parse_factor(expr, name=f"auto_{tag}_{i:04d}"))
    return out


def _auto_engine():
    src = DataAccessSource(
        dataset="ashare_stock_daily_adj",
        start_date="2024-06-01", end_date="2024-12-31",
        instrument_filter=list(INSTRUMENTS),
        run_mode="interactive_research", production=False, read_auto=False,
    )
    return FactorEngine(build_backend("polars"), src, run_mode="research")


def _assert_all_landed(res, n, tag):
    ids = res.get("factor_ids") or []
    assert len(ids) == n, f"expected {n} factor ids, got {len(ids)}"
    mats = res.get("materializations") or {}
    missing = [i for i in ids if i not in mats]
    assert not missing, f"unmaterialized factors ({tag}): {missing[:10]}"
    writer_errors = res.get("writer_errors") or []
    assert not writer_errors, f"writer errors ({tag}): {writer_errors[:5]}"
    receipt = res.get("write_receipt") or {}
    if receipt:
        items = receipt.get("items") or {}
        failed = [k for k, v in items.items()
                  if str((v or {}).get("state", "ok")).upper().rstrip(".")
                  not in ("WRITTEN", "OK", "COMMITTED")]
        assert not failed, f"receipt failures ({tag}): {failed[:10]}"


def test_auto_small_batch_lands_all_values():
    """All-default call: 12 factors, values land, receipts complete."""
    factors = _make_factors(12, "s")
    with tempfile.TemporaryDirectory() as tmp:
        res = materialize_auto(
            _auto_engine(), factors,
            write_target="local", lake_root=tmp,
        )
    _assert_all_landed(res, 12, "small")


def test_auto_multi_scale_queue():
    """Same all-default path across orders of magnitude: 12 / 60 / 240 factors."""
    for n in (12, 60, 240):
        factors = _make_factors(n, f"m{n}")
        with tempfile.TemporaryDirectory() as tmp:
            res = materialize_auto(
                _auto_engine(), factors,
                write_target="local", lake_root=tmp,
            )
        _assert_all_landed(res, n, f"scale{n}")


def test_auto_resource_summary_auditable():
    """The run must report the resource state it auto-derived under."""
    assert _mem_available_bytes() > 0
    factors = _make_factors(12, "aud")
    with tempfile.TemporaryDirectory() as tmp:
        res = materialize_auto(
            _auto_engine(), factors,
            write_target="local", lake_root=tmp,
        )
    auto = res.get("auto_resource") or {}
    assert auto.get("mode") == "auto"
    assert auto.get("mem_available_bytes", 0) > 0
    telemetry = res.get("resource_telemetry") or {}
    assert isinstance(telemetry, dict)


def test_staging_target_routes_via_data_access():
    """write_target="staging" must resolve to the data-access staging dataset."""
    from factor_engine.storage.materialize.write_targets import normalize_write_target
    norm = normalize_write_target("staging")
    assert norm["staging"] is True and norm["local"] is False
    factors = _make_factors(6, "stg")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        res = materialize_auto(_auto_engine(), factors, write_target="staging",
                               lake_root=tmp)
    mats = res.get("materializations") or {}
    assert len(mats) == 6
    for summary in mats.values():
        tgt = str(summary.get("write_target", "") or "")
        assert "staging" in tgt.lower() or "staging" in str(summary).lower(), (
            f"staging not routed: {str(summary)[:200]}")
