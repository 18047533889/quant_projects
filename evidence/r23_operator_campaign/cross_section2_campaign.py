"""Bounded parity resolution for cs_regression and cs_resid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import campaign as core
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


FIXTURE_DOMAIN = "cross_section_12_asset_nondegenerate_v1"


def fixture_inputs(rows: int, assets: int = 12) -> dict[str, pd.DataFrame]:
    if assets < 8:
        raise ValueError("cross-section fixture requires at least 8 assets")
    t = np.arange(rows, dtype=float)[:, None]
    a = np.linspace(-1.7, 2.1, assets, dtype=float)[None, :]
    x = a + 0.03 * t + 0.08 * np.sin(0.4 * a + t / 9.0)
    residual = 0.23 * np.sin(1.7 * a + t / 5.0) + 0.04 * a * a
    y = 0.65 + 1.8 * x + residual
    columns = [f"asset_{i:02d}" for i in range(assets)]
    inputs = {
        "x": pd.DataFrame(x, columns=columns),
        "y": pd.DataFrame(y, columns=columns),
    }
    assert_valid(inputs)
    return inputs


def assert_valid(inputs: dict[str, pd.DataFrame]) -> None:
    x = inputs["x"].to_numpy(dtype=float)
    y = inputs["y"].to_numpy(dtype=float)
    assert x.shape == y.shape and x.shape[1] >= 8
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (np.var(x, axis=1) > 1.0e-6).all()


def future_inputs(inputs: dict[str, pd.DataFrame], cut: int) -> dict[str, pd.DataFrame]:
    changed = {role: frame.copy() for role, frame in inputs.items()}
    assets = inputs["x"].shape[1]
    a = np.linspace(-1.0, 1.0, assets, dtype=float)[None, :]
    count = len(inputs["x"]) - cut
    time_wave = np.sin(np.arange(count, dtype=float)[:, None] / 3.0)
    changed["x"].iloc[cut:, :] = (
        inputs["x"].iloc[cut:, :].to_numpy() * (1.04 + 0.03 * a)
        + 0.12 * a * a
        + 0.02 * time_wave
    )
    changed["y"].iloc[cut:, :] = (
        inputs["y"].iloc[cut:, :].to_numpy() * (0.96 - 0.02 * a)
        + 0.19 * np.cos(2.2 * a + time_wave)
    )
    assert_valid(changed)
    for role in ("x", "y"):
        assert not np.allclose(
            inputs[role].iloc[cut:, :].to_numpy(),
            changed[role].iloc[cut:, :].to_numpy(),
        )
    return changed


def ols_residual_oracle(inputs: dict[str, pd.DataFrame]) -> np.ndarray:
    """Independent intercept OLS: first argument y on second argument x."""
    y = inputs["y"].to_numpy(dtype=float)
    x = inputs["x"].to_numpy(dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    for row in range(y.shape[0]):
        valid = np.isfinite(y[row]) & np.isfinite(x[row])
        if int(valid.sum()) < 3:
            continue
        design = np.column_stack([np.ones(int(valid.sum())), x[row, valid]])
        coefficients, _, _, _ = np.linalg.lstsq(design, y[row, valid], rcond=None)
        out[row, valid] = y[row, valid] - design @ coefficients
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--assets", type=int, default=12)
    args = parser.parse_args()
    if not 8 <= args.assets <= 32 or not 8 <= args.rows <= 96:
        raise ValueError("bounded fixture requires 8<=assets<=32 and 8<=rows<=96")

    protocol = hashlib.sha256(
        Path(core.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    core.campaign_protocol_hash = lambda: protocol
    recipes = json.loads(Path(args.recipes).read_text(encoding="utf-8"))["recipes"]
    if {recipe["canonical"] for recipe in recipes} != {"cs_regression", "cs_resid"}:
        raise ValueError("cross_section2 recipes must contain exactly the two unresolved canonicals")
    ledger = Path(args.ledger)
    inputs = fixture_inputs(args.rows, args.assets)
    cut = args.rows - 7
    changed = future_inputs(inputs, cut)
    oracle_actual = ols_residual_oracle(inputs)
    oracle_future = ols_residual_oracle(changed)

    load_all()
    summary = {}
    for recipe in sorted(recipes, key=lambda row: row["canonical"]):
        canonical = recipe["canonical"]
        outputs = {}
        fingerprints = {}
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(canonical, backend, mode="any")
            if op is None:
                core.emit(ledger, {"canonical": canonical, "backend": backend, "status": "UNTESTED_NO_SLOT", "fixture_domain": FIXTURE_DOMAIN})
                continue
            fingerprint = core.execution_fingerprint(canonical, backend, recipe, inputs, op)
            fingerprints[backend] = fingerprint
            core.emit(ledger, {"canonical": canonical, "backend": backend, "status": "RUNNING", "rows": args.rows, "assets": args.assets, "params": recipe["params"], "fingerprint": fingerprint, "fixture_domain": FIXTURE_DOMAIN, "historical_failure_resolution": "replaces invalid two-asset fixture evidence"})
            try:
                roles = recipe["inputs"]
                actual = core.array(op.calculate(*[core.as_backend(inputs[role], backend) for role in roles], **recipe["params"]))
                future = core.array(op.calculate(*[core.as_backend(changed[role], backend) for role in roles], **recipe["params"]))
                finite = int(np.isfinite(actual.astype(float)).sum())
                prefix = bool(actual.shape == future.shape and np.allclose(actual[:cut], future[:cut], equal_nan=True, rtol=1e-10, atol=1e-12))
                changed_suffix = bool(not np.allclose(actual[cut:], future[cut:], equal_nan=True, rtol=1e-10, atol=1e-12))
                oracle_equal = bool(
                    np.allclose(actual, oracle_actual, equal_nan=True, rtol=1e-10, atol=1e-12)
                    and np.allclose(future, oracle_future, equal_nan=True, rtol=1e-10, atol=1e-12)
                )
                status, reason = core.execution_status(finite, prefix, None)
                if status == "EXECUTED_FINITE" and not changed_suffix:
                    status = "FAILED_EXECUTION_PENDING_TRIAGE"
                    reason = "legal future perturbation did not change the output suffix"
                if status == "EXECUTED_FINITE" and not oracle_equal:
                    status = "FAILED_EXECUTION_PENDING_TRIAGE"
                    reason = "backend disagrees with independent intercept OLS oracle"
                outputs[backend] = actual
                core.emit(ledger, {"canonical": canonical, "backend": backend, "status": status, "reason": reason, "shape": list(actual.shape), "finite": finite, "future_prefix_invariant_all_inputs": prefix, "future_suffix_changed": changed_suffix, "independent_ols_oracle_equal": oracle_equal, "ols_dependent": "y_first_argument", "ols_independent": "x_second_argument", "ols_intercept": True, "params": recipe["params"], "fingerprint": fingerprint, "fixture_domain": FIXTURE_DOMAIN, "historical_failure_resolution": "fixture_issue_minimum_cross_section_is_3"})
            except Exception as exc:
                core.emit(ledger, {"canonical": canonical, "backend": backend, "status": "FAILED_EXECUTION_PENDING_TRIAGE", "reason": "review fixture or implementation; no automatic blame", "error_type": type(exc).__name__, "error": str(exc)[:500], "params": recipe["params"], "fingerprint": fingerprint, "fixture_domain": FIXTURE_DOMAIN})
        if set(outputs) == {"pandas_numpy", "polars"}:
            equal = bool(outputs["pandas_numpy"].shape == outputs["polars"].shape and np.allclose(outputs["pandas_numpy"], outputs["polars"], equal_nan=True, rtol=1e-10, atol=1e-12))
            core.emit(ledger, {"canonical": canonical, "status": "CANONICAL_PARITY" if equal else "CANONICAL_PARITY_FAILED", "pandas_polars_equal": equal, "backend_fingerprints": fingerprints, "fixture_domain": FIXTURE_DOMAIN, "historical_failure_resolution": "fixture_issue_minimum_cross_section_is_3"})
            summary[canonical] = equal
    print(json.dumps({"rows": args.rows, "assets": args.assets, "fixture_domain": FIXTURE_DOMAIN, "parity": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
