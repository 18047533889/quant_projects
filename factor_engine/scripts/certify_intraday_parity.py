#!/usr/bin/env python3
"""Certify intraday minute aggregation parity across Pandas / Polars / DuckDB SQL.

The 25 ``intra_*`` operators consume minute panels and emit daily panels, so the
daily-primitive six-way fixture cannot certify them (single-bar days trivially
agree on NaN).  This script runs the dedicated minute-shape parity harness,
records per-operator which backends matched the pandas reference, and writes
``evidence/intraday_minute_parity.json``.  A backend recorded as passing must
have actually produced per-(date, instrument) values equal to the reference —
never fabricated.  Failures fail closed (non-zero exit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    root, fe = str(FE_ROOT.parent), str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from factor_engine.cleaned_operators import load_all

    load_all()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_parity(ref, got, label: str) -> None:
    import numpy as np

    idx = ref.index.union(got.index)
    cols = ref.columns.union(got.columns)
    ref = ref.reindex(index=idx, columns=cols)
    got = got.reindex(index=idx, columns=cols)
    if not ref.isna().equals(got.isna()):
        raise AssertionError(f"{label}: NaN masks differ")
    mask = ~ref.isna()
    np.testing.assert_allclose(
        ref.to_numpy()[mask], got.to_numpy()[mask], rtol=1e-6, atol=1e-9, err_msg=label
    )


def _build_report() -> dict:
    import duckdb

    from tests.backend_parity import intraday_minute_parity as h
    from tests.backend_parity.test_intraday_minute_parity import (
        _daily_limit_panels,
        _fixture_panels,
        _pandas_kernel,
    )

    panels = _fixture_panels()
    long_df = h.build_long(panels)
    limit_panel = _daily_limit_panels(panels["close"])
    con = duckdb.connect()
    con.register("mlong", long_df.to_pandas())

    backends: dict[str, dict] = {}
    for name in sorted(h._REFERENCE_ONLY):
        backends[name] = {"polars": False, "duckdb_sql": False, "status": "reference_only"}
    for name, fn in sorted(h._POLARS_OPS.items()):
        ref = _pandas_kernel(name, panels)
        _assert_parity(ref, fn(long_df), f"polars:{name}")
        backends[name] = {"polars": True, "duckdb_sql": name in h._SQL_OPS,
                          "status": "certified" if name in h._SQL_OPS else "polars_only"}
    for name, fn in sorted(h._POLARS_LIMIT_OPS.items()):
        ref = _pandas_kernel(name, panels)
        _assert_parity(ref, fn(long_df, limit_panel), f"polars-limit:{name}")
        backends[name] = {"polars": True, "duckdb_sql": False, "status": "polars_only"}
    for name, fn in sorted(h._SQL_OPS.items()):
        ref = _pandas_kernel(name, panels)
        _assert_parity(ref, fn(con, "mlong"), f"sql:{name}")
        assert backends[name]["duckdb_sql"], f"{name} missing from polars matrix"
    con.close()

    # Real-COS UTC variant gate: session-local conversion must not change results.
    utc_panels = {k: v.tz_localize("Asia/Shanghai").tz_convert("UTC") for k, v in panels.items()}
    utc_long = h.build_long(utc_panels)
    for name, fn in sorted(h._POLARS_OPS.items()):
        if name in {"intra_high_time", "intra_low_time"}:
            continue
        ref = _pandas_kernel(name, panels)
        _assert_parity(ref, fn(utc_long), f"polars-utc:{name}")

    return {
        "schema_version": "intraday_minute_parity.v1",
        "generated_by": "scripts/certify_intraday_parity.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_tz": "Asia/Shanghai",
        "fixture": {
            "instruments": ["A", "B"],
            "days": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "bars_per_day": 240,
            "utc_variant": True,
            "rng_seed": 7,
        },
        "operator_source_hash": _sha256_file(
            FE_ROOT / "cleaned_operators/microstructure/intraday_agg.py"
        ),
        "test_file_hash": _sha256_file(
            FE_ROOT / "tests/backend_parity/test_intraday_minute_parity.py"
        ),
        "helper_file_hash": _sha256_file(
            FE_ROOT / "tests/backend_parity/intraday_minute_parity.py"
        ),
        "backends": backends,
    }


def _run_pytest_gate() -> None:
    files = [
        "tests/backend_parity/test_intraday_minute_parity.py",
        "tests/backend_parity/test_index_weight_sql_parity.py",
        "tests/backend_parity/test_fin_component_score_backends.py",
    ]
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{FE_ROOT.parent}:{FE_ROOT}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *files, "--tb=line"],
        cwd=str(FE_ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        raise SystemExit(f"pytest gate failed:\n{out.strip()[-1200:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="only verify the existing evidence artifact is consistent; do not overwrite",
    )
    args = parser.parse_args()

    _bootstrap()
    report = _build_report()
    target = FE_ROOT / "evidence/intraday_minute_parity.json"

    if args.check:
        existing = json.loads(target.read_text(encoding="utf-8"))
        assert existing["schema_version"] == report["schema_version"]
        assert existing["operator_source_hash"] == report["operator_source_hash"], \
            "operator source changed; re-certify"
        assert existing["test_file_hash"] == report["test_file_hash"], \
            "parity test changed; re-certify"
        assert existing["helper_file_hash"] == report["helper_file_hash"], \
            "parity helper changed; re-certify"
        assert existing["backends"] == report["backends"], "parity matrix drifted"
        print("intraday parity artifact is consistent.")
        return 0

    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    _run_pytest_gate()
    raise SystemExit(main())
