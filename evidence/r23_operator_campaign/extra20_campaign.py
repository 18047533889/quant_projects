"""Domain-valid bounded wrapper around the shared R23 operator campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import campaign as core
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _frame(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": a, "B": b})


def fixture_inputs(rows: int) -> dict[str, pd.DataFrame]:
    t = np.arange(rows, dtype=float)
    mid_a = 100.0 + 0.08 * t + 1.5 * np.sin(t / 7.0)
    mid_b = 120.0 + 0.06 * t + 1.2 * np.cos(t / 9.0)
    open_ = _frame(mid_a * (1.0 + 0.002 * np.sin(t)), mid_b * (1.0 + 0.002 * np.cos(t)))
    close = _frame(mid_a * (1.0 + 0.002 * np.cos(t)), mid_b * (1.0 + 0.002 * np.sin(t)))
    high = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()) + 1.0, columns=open_.columns)
    low = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()) - 1.0, columns=open_.columns)

    total_shares = _frame(1.0e9 + 2.0e6 * t, 1.2e9 + 1.5e6 * t)
    circulating_cap = total_shares * 0.8
    inputs = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "upper_limit": close * 1.10,
        "volume": _frame(1.0e6 + 1.0e5 * (1.0 + np.sin(t / 5.0)), 1.2e6 + 8.0e4 * (1.0 + np.cos(t / 6.0))),
        "pb": _frame(1.2 + 0.1 * (1.0 + np.sin(t / 8.0)), 1.5 + 0.1 * (1.0 + np.cos(t / 8.0))),
        "pe": _frame(12.0 + 0.5 * (1.0 + np.sin(t / 10.0)), 16.0 + 0.4 * (1.0 + np.cos(t / 10.0))),
        "total_capital": total_shares,
        "total_shares": total_shares,
        "float_shares": total_shares * 0.60,
        "free_float_shares": total_shares * 0.42,
        "circulating_cap": circulating_cap,
        "free_cap": circulating_cap * 0.55,
    }
    _assert_market_domain(inputs)
    return inputs


def _assert_market_domain(inputs: dict[str, pd.DataFrame]) -> None:
    for role, frame in inputs.items():
        values = frame.to_numpy(dtype=float)
        assert np.isfinite(values).all(), role
        assert (values > 0).all(), role
    assert (inputs["high"] >= inputs["open"]).all().all()
    assert (inputs["high"] >= inputs["close"]).all().all()
    assert (inputs["low"] <= inputs["open"]).all().all()
    assert (inputs["low"] <= inputs["close"]).all().all()
    assert (inputs["upper_limit"] > inputs["close"]).all().all()
    assert (inputs["float_shares"] <= inputs["total_shares"]).all().all()
    assert (inputs["free_float_shares"] <= inputs["total_shares"]).all().all()
    assert (inputs["free_cap"] <= inputs["circulating_cap"]).all().all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="evidence/r2/new-merge-operator-manifest.json")
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.rows <= 96 or not 1 <= args.limit <= 20:
        raise ValueError("extra20 guard requires rows<=96 and limit<=20")

    protocol = hashlib.sha256(
        Path(core.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    core.campaign_protocol_hash = lambda: protocol
    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["operators"]
    manifest = {str(row.get("canonical") or row.get("name")): row for row in manifest_data}
    recipes = json.loads(Path(args.recipes).read_text(encoding="utf-8"))["recipes"]
    recipe_map = {row["canonical"]: row for row in recipes}
    selected = sorted(recipe_map)[: args.limit]
    ledger = Path(args.ledger)
    inputs = fixture_inputs(args.rows)
    cut = max(1, args.rows - 11)
    changed_inputs = {role: frame.copy() for role, frame in inputs.items()}
    suffix = slice(cut, None)
    changed_inputs["open"].iloc[suffix, :] = inputs["open"].iloc[suffix, :] * 1.11
    changed_inputs["close"].iloc[suffix, :] = inputs["close"].iloc[suffix, :] * 0.98
    changed_inputs["high"].iloc[suffix, :] = np.maximum(
        changed_inputs["open"].iloc[suffix, :].to_numpy(),
        changed_inputs["close"].iloc[suffix, :].to_numpy(),
    ) + 1.25
    changed_inputs["low"].iloc[suffix, :] = np.minimum(
        changed_inputs["open"].iloc[suffix, :].to_numpy(),
        changed_inputs["close"].iloc[suffix, :].to_numpy(),
    ) - 1.10
    changed_inputs["upper_limit"].iloc[suffix, :] = (
        changed_inputs["close"].iloc[suffix, :] * 1.13
    )
    multipliers = {
        "volume": 1.17,
        "pb": 1.09,
        "pe": 0.93,
        "total_capital": 1.04,
        "total_shares": 1.08,
        "float_shares": 1.03,
        "free_float_shares": 0.97,
        "circulating_cap": 1.06,
        "free_cap": 0.94,
    }
    for role, multiplier in multipliers.items():
        changed_inputs[role].iloc[suffix, :] = (
            inputs[role].iloc[suffix, :] * multiplier
        )
    _assert_market_domain(changed_inputs)
    required_roles = {role for recipe in recipes for role in recipe["inputs"]}
    for role in required_roles:
        assert not np.array_equal(
            inputs[role].iloc[suffix, :].to_numpy(),
            changed_inputs[role].iloc[suffix, :].to_numpy(),
        ), role

    load_all()
    executed_slots = 0
    outputs_by_name = {}
    for canonical in selected:
        recipe = recipe_map[canonical]
        if canonical not in manifest:
            core.emit(ledger, {"canonical": canonical, "backend": "campaign", "status": "UNTESTED_NOT_IN_MANIFEST"})
            continue
        outputs = {}
        prepared = {}
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(canonical, backend, mode="any")
            if op is not None:
                prepared[backend] = (op, core.execution_fingerprint(canonical, backend, recipe, inputs, op))
        for backend in ("pandas_numpy", "polars"):
            op_and_fp = prepared.get(backend)
            if op_and_fp is None:
                core.emit(ledger, {"canonical": canonical, "backend": backend, "status": "UNTESTED_NO_SLOT", "reason": "manifest recommendation has no effective registry backend slot"})
                continue
            op, fingerprint = op_and_fp
            core.emit(ledger, {"canonical": canonical, "backend": backend, "status": "RUNNING", "rows": args.rows, "params": recipe["params"], "fingerprint": fingerprint, "fixture_domain": "positive_market_ohlc_shares_v1"})
            try:
                roles = recipe["inputs"]
                actual = core.array(op.calculate(*[core.as_backend(inputs[role], backend) for role in roles], **recipe["params"]))
                future = core.array(op.calculate(*[core.as_backend(changed_inputs[role], backend) for role in roles], **recipe["params"]))
                finite = int(np.isfinite(actual.astype(float)).sum())
                prefix = bool(actual.shape == future.shape and np.allclose(actual[:cut], future[:cut], equal_nan=True, rtol=1e-10, atol=1e-12))
                status, reason = core.execution_status(finite, prefix, recipe.get("allow_all_nan_reason"))
                outputs[backend] = actual
                core.emit(ledger, {"canonical": canonical, "backend": backend, "status": status, "reason": reason, "shape": list(actual.shape), "finite": finite, "future_prefix_invariant_all_inputs": prefix, "params": recipe["params"], "fingerprint": fingerprint, "fixture_domain": "positive_market_ohlc_shares_v1"})
            except Exception as exc:
                core.emit(ledger, {"canonical": canonical, "backend": backend, "status": "FAILED_EXECUTION_PENDING_TRIAGE", "reason": "review recipe or implementation; no automatic blame", "error_type": type(exc).__name__, "error": str(exc)[:500], "params": recipe["params"], "fingerprint": fingerprint, "fixture_domain": "positive_market_ohlc_shares_v1"})
            executed_slots += 1
            gc.collect()
        outputs_by_name[canonical] = outputs
        if set(outputs) == {"pandas_numpy", "polars"}:
            equal = bool(outputs["pandas_numpy"].shape == outputs["polars"].shape and np.allclose(outputs["pandas_numpy"], outputs["polars"], equal_nan=True, rtol=1e-10, atol=1e-12))
            core.emit(ledger, {"canonical": canonical, "status": "CANONICAL_PARITY" if equal else "CANONICAL_PARITY_FAILED", "pandas_polars_equal": equal, "backend_fingerprints": {backend: prepared[backend][1] for backend in ("pandas_numpy", "polars")}, "fixture_domain": "positive_market_ohlc_shares_v1"})

    print(json.dumps({"selected": selected, "ledger": str(ledger), "rows": args.rows, "executed_slots": executed_slots, "fixture_domain": "positive_market_ohlc_shares_v1"}, sort_keys=True))


if __name__ == "__main__":
    main()
