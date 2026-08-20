#!/usr/bin/env python3
"""CrossSectionalNonDegeneracy certification for TERMINAL_ALPHA operators.

For each operator whose MiningRole is ALPHA or ALPHA_HIGH_COST (terminal
allowed), generate a heterogeneous synthetic stock panel, compute the output,
and verify cross-sectional standard deviation is > epsilon on a reasonable
fraction of dates after warmup.

Operators whose all-market output is identical across every stock are
reclassified as GLOBAL_STATE.  Operators whose all output is identical within
a predefined group are reclassified as GROUP_STATE.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Thread safety: always set before numpy/polars import
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports (after path setup)
# ---------------------------------------------------------------------------

def _load_deps():
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from mining.operator_catalog import get_mining_operators, MiningRole
    from scripts.audit_all_factor_production import _panels, _build_call
    return {
        "load_all": load_all,
        "OperatorRegistry": OperatorRegistry,
        "get_mining_operators": get_mining_operators,
        "MiningRole": MiningRole,
        "_panels": _panels,
        "_build_call": _build_call,
    }


# ---------------------------------------------------------------------------
# Synthetic heterogeneity config
# ---------------------------------------------------------------------------
N_ROWS = 220          # trading days
N_COLS = 30           # number of synthetic stocks
EPSILON = 1e-12       # minimum cs_std to consider non-degenerate
MIN_PASS_FRACTION = 0.50  # at least 50% of dates must have cs_std > epsilon
WARMUP_FRACTION = 0.40   # skip first 40% of rows as warmup

# Synthetic group labels: first N_COLS//3 in group A, next in B, rest in C
_GROUP_LABELS = (
    ["A"] * (N_COLS // 3)
    + ["B"] * (N_COLS // 3)
    + ["C"] * (N_COLS - 2 * (N_COLS // 3))
)


def _make_panels() -> dict[str, pd.DataFrame]:
    """Create heterogeneous synthetic panel data with stock-specific drift."""
    idx = pd.date_range("2024-01-01", periods=N_ROWS, freq="D")
    rng = np.random.default_rng(2026_08_21)

    # Base price with per-stock drift
    drift = rng.uniform(-0.002, 0.005, size=N_COLS)
    noise = rng.standard_normal((N_ROWS, N_COLS)) * 0.02
    log_price = np.cumsum(drift[None, :] + noise, axis=0) + np.log(100.0)
    close = pd.DataFrame(np.exp(log_price), index=idx,
                         columns=[f"S{i:03d}" for i in range(N_COLS)])
    high = close * (1 + rng.uniform(0.001, 0.015, size=(N_ROWS, N_COLS)))
    low = close * (1 - rng.uniform(0.001, 0.015, size=(N_ROWS, N_COLS)))
    volume = pd.DataFrame(
        rng.integers(50_000, 5_000_000, size=(N_ROWS, N_COLS)).astype(float),
        index=idx, columns=close.columns,
    )
    amount = close * volume
    return {
        "x": close, "y": close, "close": close, "open": close,
        "high": pd.DataFrame(high, index=idx, columns=close.columns),
        "low": pd.DataFrame(low, index=idx, columns=close.columns),
        "volume": volume, "amount": amount, "vwap": close,
        "turnover": volume, "ret": close.pct_change(),
        "signal": close.pct_change(),
        "condition": (close.pct_change() > 0).astype(float),
        "weight": pd.DataFrame(1.0 / N_COLS, index=idx, columns=close.columns),
        "returns": close.pct_change(),
    }


def _cs_std(frame: pd.DataFrame) -> pd.Series:
    """Cross-sectional std for each row (date)."""
    return frame.astype(float).std(axis=1, ddof=1)


def _is_all_market_identical(frame: pd.DataFrame) -> bool:
    """True if every row has zero cross-sectional variation."""
    cs = _cs_std(frame.dropna())
    return bool((cs < EPSILON).all()) if len(cs) > 0 else True


def _is_group_identical(frame: pd.DataFrame) -> bool:
    """True if within each group label, cs_std is zero for all dates."""
    for grp in _GROUP_LABELS:
        pass
    # Simpler check: cs_std over whole frame is zero implies group identical too
    # But we need group-level: compute group cs_std and check all zero
    for label in set(_GROUP_LABELS):
        cols = [f"S{i:03d}" for i, g in enumerate(_GROUP_LABELS) if g == label]
        subset = frame[[c for c in cols if c in frame.columns]]
        if subset.shape[1] < 2:
            continue
        cs = _cs_std(subset.dropna())
        if len(cs) > 0 and not bool((cs < EPSILON).all()):
            return False
    return True


def _warmup_slice(series: pd.Series) -> pd.Series:
    n = len(series)
    start = int(n * WARMUP_FRACTION)
    return series.iloc[start:]


def certify_one(canonical: str, deps: dict[str, Any]) -> dict[str, Any]:
    """Certify a single operator for cross-sectional non-degeneracy."""
    OperatorRegistry = deps["OperatorRegistry"]
    _panels_fn = deps["_panels"]
    _build_call_fn = deps["_build_call"]

    result: dict[str, Any] = {
        "name": canonical,
        "status": "PASS",
        "cs_std_pass_dates": 0,
        "cs_std_total_dates": 0,
        "pass_fraction": 0.0,
        "reclassification": None,
        "notes": [],
    }

    try:
        op = OperatorRegistry.get(canonical, backend="pandas_numpy")
        if op is None:
            result["status"] = "SKIP"
            result["notes"].append("no pandas_numpy backend")
            return result

        panels = _make_panels()
        args, kwargs = _build_call_fn(canonical, op, panels)
        out = op.calculate(*args, **kwargs)

        # Normalize to DataFrame
        if isinstance(out, pd.Series):
            out = pd.DataFrame(out)
        if not isinstance(out, pd.DataFrame):
            result["status"] = "SKIP"
            result["notes"].append(f"output type {type(out).__name__} not DataFrame")
            return result

        # Ensure numeric
        try:
            out_float = out.astype(float)
        except (TypeError, ValueError):
            result["status"] = "SKIP"
            result["notes"].append("output not numeric")
            return result

        # Check all-market identical -> GLOBAL_STATE
        if _is_all_market_identical(out_float):
            result["status"] = "RECLASSIFY_GLOBAL"
            result["reclassification"] = "GLOBAL_STATE"
            result["notes"].append("all-market output is constant across stocks")
            return result

        # Check group identical -> GROUP_STATE
        if _is_group_identical(out_float):
            result["status"] = "RECLASSIFY_GROUP"
            result["reclassification"] = "GROUP_STATE"
            result["notes"].append("within-group output is constant across stocks")
            return result

        # Cross-sectional non-degeneracy check
        cs = _cs_std(out_float)
        cs_post_warmup = _warmup_slice(cs)
        total = int(cs_post_warmup.count())
        passing = int((cs_post_warmup > EPSILON).sum())
        frac = passing / total if total > 0 else 0.0

        result["cs_std_pass_dates"] = passing
        result["cs_std_total_dates"] = total
        result["pass_fraction"] = round(frac, 4)

        if frac < MIN_PASS_FRACTION:
            result["status"] = "FAIL"
            result["notes"].append(
                f"cs_std pass fraction {frac:.2%} < {MIN_PASS_FRACTION:.0%}"
            )
        else:
            result["notes"].append(f"cs_std pass fraction {frac:.2%}")

    except Exception as e:
        result["status"] = "ERROR"
        result["notes"].append(f"{type(e).__name__}: {str(e)[:120]}")

    return result


def run_certification() -> list[dict[str, Any]]:
    deps = _load_deps()
    deps["load_all"]()
    ops = deps["get_mining_operators"](
        admission="all",
        roles=[deps["MiningRole"].ALPHA, deps["MiningRole"].ALPHA_HIGH_COST],
    )
    results = []
    t0 = time.time()
    for op in ops:
        r = certify_one(op.canonical, deps)
        results.append(r)
        if r["status"] not in ("PASS", "SKIP"):
            print(f"  {r['status']:20s} {op.canonical}: {'; '.join(r['notes'])}", flush=True)
    elapsed = time.time() - t0
    print(f"\nCertified {len(results)} operators in {elapsed:.1f}s", flush=True)
    return results


def write_evidence(results: list[dict[str, Any]]) -> Path:
    import yaml

    evidence_dir = REPO_ROOT / "evidence" / "r2"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_path = evidence_dir / "R21-CS-NON-DEGENERACY.yaml"

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0,
              "RECLASSIFY_GLOBAL": 0, "RECLASSIFY_GROUP": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    counts["total"] = len(results)
    counts["evaluable"] = counts.get("PASS", 0) + counts.get("FAIL", 0)
    counts["non_degenerate_pass_rate"] = (
        round(counts.get("PASS", 0) / counts["evaluable"], 4)
        if counts["evaluable"] > 0 else 0.0
    )

    data = {
        "metadata": {
            "certification": "CrossSectionalNonDegeneracy",
            "target": "TERMINAL_ALPHA (MiningRole.ALPHA / ALPHA_HIGH_COST)",
            "n_stocks": N_COLS,
            "n_rows": N_ROWS,
            "epsilon": EPSILON,
            "min_pass_fraction": MIN_PASS_FRACTION,
            "warmup_fraction": WARMUP_FRACTION,
            "total_operators": len(results),
            "counts": counts,
        },
        "operators": [
            {k: v for k, v in r.items() if k != "notes"}
            | {"verification_notes": r.get("notes", [])}
            for r in results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Wrote evidence to {out_path}", flush=True)
    return out_path


def main() -> int:
    results = run_certification()
    write_evidence(results)

    # Summary
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    reclass_global = sum(1 for r in results if r["status"] == "RECLASSIFY_GLOBAL")
    reclass_group = sum(1 for r in results if r["status"] == "RECLASSIFY_GROUP")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")

    print(f"\n=== CS Non-Degeneracy Summary ===")
    print(f"  PASS:              {pass_count}")
    print(f"  RECLASSIFY_GLOBAL: {reclass_global}")
    print(f"  RECLASSIFY_GROUP:  {reclass_group}")
    print(f"  FAIL:              {fail_count}")
    print(f"  ERROR:             {error_count}")
    print(f"  SKIP:              {skip_count}")

    return 0 if fail_count == 0 and error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
