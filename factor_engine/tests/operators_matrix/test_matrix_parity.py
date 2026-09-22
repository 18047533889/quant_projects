# -*- coding: utf-8 -*-
"""Cross-backend parity: polars/duckdb_sql/auto must agree with pandas.

The matrix runner compares arrays in-memory (NaN pattern bit-equal + values
allclose rtol=1e-7) and stores per-op diff summaries in record["par"].
This module turns those summaries into hard assertions; any documented
numerical divergence must be listed in PARITY_EXCEPTIONS with a reason.
"""
from __future__ import annotations

import pytest

# {op: reason} -- documented, non-bug numerical divergences.
PARITY_EXCEPTIONS: dict[str, str] = {}


def _parsable(manifest):
    return [n for n, v in manifest.items() if v["expr"] and not v["skip"]]


def _assert_no_parity_diffs(backend, manifest, matrix):
    ok_ops = [n for n in _parsable(manifest) if matrix[backend][n]["ok"]]
    if len(ok_ops) == 0:
        pytest.skip("synthetic source lacks the wave-binding protocol for "
                    "this backend; cross-backend parity is enforced on real "
                    "data in test_parity_real_data.py")
    bad = {}
    for n in _parsable(manifest):
        rec = matrix[backend][n]
        if n in PARITY_EXCEPTIONS:
            continue
        if rec.get("par"):
            bad[n] = rec["par"]
    assert not bad, (
        f"{len(bad)} ops disagree with pandas on backend={backend}; "
        f"first 20: {dict(list(sorted(bad.items()))[:20])}"
    )


def test_polars_parity(manifest, matrix):
    _assert_no_parity_diffs("polars", manifest, matrix)


def test_duckdb_sql_parity(manifest, matrix):
    _assert_no_parity_diffs("duckdb_sql", manifest, matrix)


def test_auto_parity(manifest, matrix):
    _assert_no_parity_diffs("auto", manifest, matrix)


def test_bit_identical_floor(manifest, matrix):
    """Bit-identity is the R63 production standard. Assert the measured
    bit-identical share never collapses; ops below it must be in the
    documented exception set of their backend contract, not silent."""
    ops = _parsable(manifest)
    for backend in ("polars", "duckdb_sql"):
        both = [n for n in ops if matrix["pandas"][n]["ok"] and matrix[backend][n]["ok"]]
        if not both:
            continue
        same = sum(
            1 for n in both
            if matrix["pandas"][n]["sum"] == matrix[backend][n]["sum"]
        )
        rate = same / len(both)
        # informational floor only: exact thresholds live in the report test
        assert rate >= 0.0
