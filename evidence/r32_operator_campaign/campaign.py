"""Bounded, resumable real-execution campaign for recommended operators."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


TERMINAL = {
    "EXECUTED_FINITE",
    "EXECUTED_LEGITIMATE_WARMUP",
    "UNTESTED_NO_SLOT",
    "UNTESTED_NO_RECIPE",
    "FAILED_EXECUTION_PENDING_TRIAGE",
    "ABORTED_GUARD",
    "ABORTED_PROCESS",
}


def emit(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()


def successful_fingerprints(path: Path) -> dict[tuple[str, str], str]:
    latest_backend = {}
    latest_parity = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        canonical = row.get("canonical")
        backend = row.get("backend")
        if canonical and backend in {"pandas_numpy", "polars"}:
            latest_backend[(canonical, backend)] = row
        if canonical and row.get("status") in {"CANONICAL_PARITY", "CANONICAL_PARITY_FAILED"}:
            latest_parity[canonical] = row
    done = {}
    for canonical, parity in latest_parity.items():
        if parity.get("pandas_polars_equal") is not True:
            continue
        bound = parity.get("backend_fingerprints")
        if not isinstance(bound, dict):
            continue
        rows = {backend: latest_backend.get((canonical, backend))
                for backend in ("pandas_numpy", "polars")}
        if all(rows[backend]
               and rows[backend].get("status") in {"EXECUTED_FINITE", "EXECUTED_LEGITIMATE_WARMUP"}
               and rows[backend].get("future_prefix_invariant_all_inputs") is True
               and rows[backend].get("fingerprint") == bound.get(backend)
               for backend in rows):
            for backend, row in rows.items():
                done[(canonical, backend)] = row["fingerprint"]
    return done


def reconcile_interrupted(path: Path, watchdog_path: str | None) -> None:
    if not watchdog_path or not path.exists():
        return
    watchdog = json.loads(Path(watchdog_path).read_text(encoding="utf-8"))
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("backend"):
            latest[(row.get("canonical"), row.get("backend"))] = row
    reason = watchdog.get("test_guard_reason")
    terminal = "ABORTED_GUARD" if reason else "ABORTED_PROCESS"
    for (canonical, backend), row in latest.items():
        if row.get("status") == "RUNNING":
            emit(path, {"canonical": canonical, "backend": backend, "status": terminal,
                        "reason": reason or watchdog.get("signal_name") or "process ended before terminal row",
                        "watchdog": watchdog_path})


def fixture(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(2309)
    t = np.arange(rows, dtype=float)
    values = 0.01*t + np.sin(t/3.0) + 0.3*np.sin(t/11.0) + 0.08*rng.normal(size=rows)
    return pd.DataFrame({"A": values, "B": values[::-1] + 0.04*rng.normal(size=rows)})


def fixture_inputs(rows: int) -> dict[str, pd.DataFrame]:
    base = fixture(rows)
    return {
        "base": base,
        "other": base*0.37 + 0.11,
        # Independent nonlinear control used by three-input rolling formulas.
        "third": base.pow(2)*0.07 - base*0.13 + 0.05,
        "fourth": np.sin(base)*0.31 + base*0.04,
        "fifth": np.cos(base)*0.27 - base*0.03,
        "condition": base > base.median(),
        # Group operators consume a separate aligned panel of categorical labels,
        # not numeric values masquerading as groups.
        "group": pd.DataFrame(np.tile(["G_A", "G_B"], (rows, 1)), columns=base.columns),
    }


def as_backend(frame: pd.DataFrame, backend: str):
    return pl.from_pandas(frame.reset_index(drop=True)) if backend == "polars" else frame


def array(value):
    return value.to_numpy() if isinstance(value, pl.DataFrame) else value.to_numpy(dtype=float)


def execution_status(finite: int, prefix: bool, allow_all_nan_reason: str | None) -> tuple[str, str]:
    if not prefix:
        return "FAILED_EXECUTION_PENDING_TRIAGE", "future-input mutation changed the historical prefix"
    if finite:
        return "EXECUTED_FINITE", "bounded reviewed recipe produced finite values"
    if allow_all_nan_reason:
        return "EXECUTED_LEGITIMATE_WARMUP", allow_all_nan_reason
    return "FAILED_EXECUTION_PENDING_TRIAGE", "all-NaN output was not declared as a legitimate warmup"


def campaign_protocol_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def execution_fingerprint(canonical: str, backend: str, recipe: dict, inputs: dict, op) -> str:
    physical = getattr(op, "physical_spec", None)
    spec = physical() if callable(physical) else getattr(op, "_physical_spec", None)
    source_path = inspect.getsourcefile(type(op))
    source_hash = ""
    if source_path and Path(source_path).is_file():
        source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256()
    for role in sorted(recipe.get("inputs", ["base"])):
        fixture_hash.update(role.encode())
        fixture_hash.update(pd.util.hash_pandas_object(inputs[role], index=True).values.tobytes())
    payload = {
        "canonical": canonical,
        "backend": backend,
        "recipe": recipe,
        "fixture_seed": 2309,
        "fixture_hash": fixture_hash.hexdigest(),
        "implementation_class": f"{type(op).__module__}.{type(op).__qualname__}",
        "implementation_source_hash": getattr(spec, "implementation_source_hash", ""),
        "semantic_contract_hash": getattr(spec, "semantic_contract_hash", ""),
        "implementation_closure_hash": getattr(spec, "implementation_closure_hash", ""),
        "source_hash": source_hash,
        "campaign_protocol_hash": campaign_protocol_hash(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="evidence/r2/new-merge-operator-manifest.json")
    parser.add_argument("--recipes", default="evidence/r23_operator_campaign/recipes.json")
    parser.add_argument("--ledger", default="evidence/r23_operator_campaign/ledger.jsonl")
    parser.add_argument("--selection", choices=("recipes", "manifest"), default="recipes")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--reconcile-watchdog")
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        raise ValueError("limit must be between 1 and 30")
    if not 1 <= args.rows <= 128:
        raise ValueError("rows must be between 1 and 128")

    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["operators"]
    manifest = {str(row.get("canonical") or row.get("name")): row for row in manifest_data}
    recipes = json.loads(Path(args.recipes).read_text(encoding="utf-8"))["recipes"]
    recipe_map = {row["canonical"]: row for row in recipes}
    names = sorted(recipe_map) if args.selection == "recipes" else sorted(manifest)
    selected = names[args.offset : args.offset + args.limit]
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    reconcile_interrupted(ledger, args.reconcile_watchdog)
    done = successful_fingerprints(ledger)
    inputs = fixture_inputs(args.rows)
    frame = inputs["base"]
    cut = max(1, args.rows - 11)
    changed = frame.copy()
    changed.iloc[cut:, :] = changed.iloc[cut:, :]*-17.0 + 31.0

    load_all()
    executed_slots = 0
    cached_slots = 0
    for canonical in selected:
        recipe = recipe_map.get(canonical)
        if canonical not in manifest:
            emit(ledger, {"canonical": canonical, "backend": "campaign", "status": "UNTESTED_NOT_IN_MANIFEST"})
            continue
        outputs = {}
        prepared = {}
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(canonical, backend, mode="any")
            if op is not None and recipe is not None:
                prepared[backend] = (op, execution_fingerprint(
                    canonical, backend, recipe, inputs, op
                ))
        pair_cached = (
            set(prepared) == {"pandas_numpy", "polars"}
            and all(done.get((canonical, backend)) == prepared[backend][1]
                    for backend in prepared)
        )
        if pair_cached:
            cached_slots += 2
            continue
        for backend in ("pandas_numpy", "polars"):
            if recipe is None:
                emit(ledger, {"canonical": canonical, "backend": backend, "status": "UNTESTED_NO_RECIPE",
                              "reason": "no reviewed executable recipe in catalog"})
                continue
            op = OperatorRegistry.get(canonical, backend, mode="any")
            if op is None:
                emit(ledger, {"canonical": canonical, "backend": backend, "status": "UNTESTED_NO_SLOT",
                              "reason": "manifest recommendation has no effective registry backend slot"})
                continue
            fingerprint = prepared[backend][1]
            emit(ledger, {"canonical": canonical, "backend": backend, "status": "RUNNING",
                          "rows": args.rows, "params": recipe["params"], "fingerprint": fingerprint})
            try:
                roles = recipe.get("inputs", ["base"])
                call_inputs = [as_backend(inputs[role], backend) for role in roles]
                changed_inputs = dict(inputs)
                for role in roles:
                    altered = inputs[role].copy()
                    if role == "condition":
                        altered.iloc[cut:, :] = ~altered.iloc[cut:, :]
                    elif role == "group":
                        altered.iloc[cut:, :] = altered.iloc[cut:, :].replace(
                            {"G_A": "G_B", "G_B": "G_A"}
                        )
                    else:
                        altered.iloc[cut:, :] = altered.iloc[cut:, :]*-17.0 + 31.0
                    changed_inputs[role] = altered
                future_inputs = [as_backend(changed_inputs[role], backend) for role in roles]
                actual = array(op.calculate(*call_inputs, **recipe["params"]))
                future = array(op.calculate(*future_inputs, **recipe["params"]))
                finite = int(np.isfinite(actual.astype(float)).sum())
                prefix = bool(actual.shape == future.shape and np.allclose(
                    actual[:cut], future[:cut], equal_nan=True, rtol=1e-10, atol=1e-12
                ))
                status, reason = execution_status(
                    finite, prefix, recipe.get("allow_all_nan_reason")
                )
                outputs[backend] = actual
                emit(ledger, {"canonical": canonical, "backend": backend, "status": status,
                              "reason": reason, "shape": list(actual.shape), "finite": finite,
                              "future_prefix_invariant_all_inputs": prefix, "params": recipe["params"],
                              "fingerprint": fingerprint})
                executed_slots += 1
            except Exception as exc:
                emit(ledger, {"canonical": canonical, "backend": backend,
                              "status": "FAILED_EXECUTION_PENDING_TRIAGE",
                              "reason": "review recipe or implementation; no automatic blame",
                              "error_type": type(exc).__name__, "error": str(exc)[:500],
                              "params": recipe["params"], "fingerprint": fingerprint})
                executed_slots += 1
            gc.collect()
        if set(outputs) == {"pandas_numpy", "polars"}:
            left, right = outputs["pandas_numpy"], outputs["polars"]
            equal = bool(left.shape == right.shape and np.allclose(
                left, right, equal_nan=True, rtol=1e-10, atol=1e-12
            ))
            emit(ledger, {"canonical": canonical,
                          "status": "CANONICAL_PARITY" if equal else "CANONICAL_PARITY_FAILED",
                          "pandas_polars_equal": equal,
                          "backend_fingerprints": {
                              backend: prepared[backend][1]
                              for backend in ("pandas_numpy", "polars")
                          }})

    print(json.dumps({"selected": selected, "ledger": str(ledger), "rows": args.rows,
                      "manifest_size": len(manifest), "recipe_count": len(recipe_map),
                      "executed_slots": executed_slots, "cached_slots": cached_slots}, sort_keys=True))


if __name__ == "__main__":
    main()
