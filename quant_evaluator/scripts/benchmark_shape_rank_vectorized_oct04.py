"""Controlled, read-only benchmark for vectorized quantile-rank scoring.

The default CLI is a dry run. Pass ``--run`` explicitly to collect timings.
This harness never changes thread settings and is not a strict RSS limiter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_PATH = "quant_evaluator/metrics/quantile_shape.py"
_RANK_HELPER_PATH = "quant_evaluator/metrics/quantile_rank_numeric.py"
_ACTIVE_ROOT = _ROOT
_DEFAULT_CASES = tuple((q, f) for q in (5, 20, 512) for f in (48, 512, 4096))
_MEMORY_CAP_BYTES = 512 * 1024 * 1024
_legacy_default = None
_PINNED_BASELINE_SHA = None
_BASELINE_REF = "HEAD"


def _load_legacy():
    source = subprocess.check_output(
        ["git", "show", f"{_PINNED_BASELINE_SHA}:{_BASELINE_PATH}"], cwd=_ACTIVE_ROOT, text=True
    )
    tree = ast.parse(source, filename=f"{_PINNED_BASELINE_SHA}:{_BASELINE_PATH}")
    wanted = {"_as_matrix", "compute_quantile_rank_monotonicity"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("baseline AST is missing required rank functions")
    reference = next(node for node in nodes if node.name == "compute_quantile_rank_monotonicity")
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "spearmanr" for node in ast.walk(reference)):
        raise RuntimeError("baseline must contain the legacy independent SciPy implementation; select an earlier --baseline-ref")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, Any] = {"np": np}
    exec(compile(module, f"HEAD:{_BASELINE_PATH}", "exec"), namespace)
    return namespace["compute_quantile_rank_monotonicity"]


def _legacy(qr: np.ndarray) -> np.ndarray:
    global _legacy_default
    if _legacy_default is None:
        _legacy_default = _load_legacy()
    return _legacy_default(qr)


def _candidate(qr: np.ndarray) -> np.ndarray:
    return compute_quantile_rank_monotonicity(qr)


def _now() -> float:
    return time.perf_counter()


def _estimate_peak_memory_bytes(shape, *, tile_factors: int = 128) -> int:
    # Conservative working-set estimate for input, rank intermediates, outputs,
    # and temporary arrays. No F-by-F allocation is used by this benchmark.
    q, f = shape
    if type(q) is not int or type(f) is not int or q < 1 or f < 0:
        raise ValueError("shape must be a (Q, F) pair of non-negative builtin integers")
    if type(tile_factors) is not int or tile_factors < 1:
        raise ValueError("tile_factors must be a positive builtin integer")
    width = min(f, tile_factors)
    return int(q) * int(f) * 16 + int(q) * width * 24 + int(f) * 16 + int(q) * 8


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes(root=None) -> dict[str, str]:
    repo_root = Path(root).resolve() if root is not None else _ACTIVE_ROOT
    baseline = subprocess.check_output(
        ["git", "show", f"{_PINNED_BASELINE_SHA or 'HEAD'}:{_BASELINE_PATH}"], cwd=repo_root
    )
    candidate_path = repo_root / _BASELINE_PATH
    helper_path = repo_root / _RANK_HELPER_PATH
    benchmark_path = repo_root / "quant_evaluator/scripts/benchmark_shape_rank_vectorized_oct04.py"
    return {
        "baseline_HEAD": _hash_bytes(baseline),
        "candidate_worktree": _hash_bytes(candidate_path.read_bytes()),
        "rank_helper_worktree": _hash_bytes(helper_path.read_bytes()),
        "benchmark_worktree": _hash_bytes(benchmark_path.read_bytes()),
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
    }


def _validate_identity(expected: dict[str, Any], current: dict[str, Any]) -> bool:
    return expected == current


class _InputMutationError(RuntimeError):
    """Raised when an implementation changes the shared read-only panel."""


def _combined_source_hash(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return _hash_bytes(payload)


def _parse_case(case: Any, index: int) -> tuple[str, int, int]:
    if isinstance(case, str):
        import re
        match = re.fullmatch(r"q([0-9]+)_f([0-9]+)(?:_mixed)?", case)
        if not match:
            raise ValueError(f"invalid case name: {case!r}")
        q, f = map(int, match.groups())
        name = case
    elif isinstance(case, (tuple, list)) and len(case) == 2:
        q, f = case
        name = f"q{q}_f{f}_mixed"
    elif isinstance(case, dict) and set(case) == {"name", "Q", "F"}:
        name, q, f = case["name"], case["Q"], case["F"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"case {index} name must be non-empty text")
    else:
        raise ValueError(f"invalid case at index {index}: {case!r}")
    if type(q) is not int or type(f) is not int or q < 3 or f < 1:
        raise ValueError(f"case {index} dimensions must be builtin integers Q>=3, F>=1")
    return name, q, f


def _make_input(q: int, f: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # The first third is tie-rich integer data, the next third constant, and
    # the remainder continuous. A deterministic subset is partially masked.
    m = rng.integers(-3, 4, size=(q, f)).astype(np.float64)
    constant_end = (2 * f) // 3
    m[:, f // 3:constant_end] = 7.0
    if constant_end < f:
        m[:, constant_end:] = rng.standard_normal((q, f - constant_end))
    if q >= 4 and f:
        masked = np.arange(f) % 7 == 0
        m[1, masked] = np.nan
    return np.ascontiguousarray(m)


def _validate_output(value: Any, f: int, label: str) -> np.ndarray:
    out = np.asarray(value)
    if out.shape != (f,) or out.dtype != np.dtype(np.float64):
        raise RuntimeError(f"{label} returned shape/dtype {out.shape}/{out.dtype}; expected ({f},)/float64")
    return out


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint64), np.ascontiguousarray(b).view(np.uint64)
    )


def _validate_settings(rounds: int, seed: int, deadline_seconds: float) -> None:
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive builtin integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative builtin integer (bool is rejected)")
    if isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (int, float)):
        raise ValueError("deadline_seconds must be a positive finite number")
    if not np.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be a positive finite number")


def run_benchmark(*, cases=None, rounds=3, seed=20261004, deadline_seconds=120.0, root=None, baseline_ref="HEAD") -> dict:
    """Run pinned-baseline ABBA timing and return a qualification receipt."""
    global _ACTIVE_ROOT, _PINNED_BASELINE_SHA, _BASELINE_REF, _legacy_default
    _validate_settings(rounds, seed, deadline_seconds)
    selected = _DEFAULT_CASES if cases is None else cases
    if isinstance(selected, (str, dict)):
        selected = [selected]
    parsed = [_parse_case(c, i) for i, c in enumerate(selected)]
    if not parsed:
        raise ValueError("cases must not be empty")
    base_root = Path(root).resolve() if root is not None else _ROOT
    previous_root, previous_pin, previous_ref = _ACTIVE_ROOT, _PINNED_BASELINE_SHA, _BASELINE_REF
    previous_default = _legacy_default
    pinned_sha = subprocess.check_output(["git", "rev-parse", f"{baseline_ref}^{{commit}}"], cwd=base_root, text=True).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", pinned_sha):
        raise ValueError("baseline_ref did not resolve to a full commit SHA")
    _ACTIVE_ROOT = base_root
    _BASELINE_REF = baseline_ref
    _PINNED_BASELINE_SHA = pinned_sha
    _legacy_default = None
    started = _now()
    deadline = started + float(deadline_seconds)
    receipt: dict[str, Any] = {
        "status": "complete", "winner_eligible": True, "rounds": rounds,
        "seed": seed, "deadline_seconds": float(deadline_seconds),
        "memory_cap_bytes": _MEMORY_CAP_BYTES, "memory_cap_is_strict_rss": False,
        "schedule": {"warmup_calls": 1, "rounds": rounds, "blocks_per_round": 4,
                     "calls_per_block": 1, "order": "ABBA"},
        "cases": [], "errors": [], "source_hashes_before": None,
        "source_hashes_after": None, "runtime_identity": None,
        "baseline": {"revision": pinned_sha, "source_sha256": ""},
        "environment": {"frozen_head": pinned_sha},
        "candidate_source_sha256_before": "", "candidate_source_sha256_after": "",
        "runtime_before": None, "runtime_after": None,
        "results": {}, "budget": {"deadline_exceeded": False},
    }
    original_legacy = globals()["_legacy"]
    try:
        initial_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_root, text=True).strip()
        receipt["environment"]["frozen_head"] = initial_head
        initial_hashes = _source_hashes(base_root)
        initial_identity = _runtime_identity()
        receipt["source_hashes_before"] = initial_hashes
        receipt["runtime_identity"] = initial_identity
        receipt["runtime_before"] = initial_identity
        baseline_hash = initial_hashes.get("baseline_HEAD") or next(
            (v for k, v in initial_hashes.items() if "quantile_shape" in k),
            _hash_bytes(json.dumps(initial_hashes, sort_keys=True).encode()),
        )
        candidate_hash = _combined_source_hash({k: v for k, v in initial_hashes.items() if "baseline" not in k})
        receipt["baseline"]["source_sha256"] = baseline_hash
        receipt["candidate_source_sha256_before"] = candidate_hash
        for case_index, (name, q, f) in enumerate(parsed):
            if _now() >= deadline:
                receipt.update(status="partial", winner_eligible=False)
                receipt["budget"]["deadline_exceeded"] = True
                break
            estimate = _estimate_peak_memory_bytes((q, f), tile_factors=512)
            record: dict[str, Any] = {
                "name": name, "Q": q, "F": f, "estimated_peak_memory_bytes": estimate,
                "status": "pending", "legacy_seconds": [], "candidate_seconds": [],
            }
            receipt["cases"].append(record)
            result = {"warmup": {}, "samples": [], "bitwise_parity_each_call": True}
            receipt["results"][name] = result
            if estimate > _MEMORY_CAP_BYTES:
                record["status"] = "skipped_memory_cap"
                receipt.update(status="partial", winner_eligible=False)
                continue
            x = _make_input(q, f, seed + case_index)
            x.setflags(write=False)
            input_before = _hash_bytes(x.tobytes())
            record["input_sha256"] = input_before
            try:
                baseline_out = None
                for label, fn in (("legacy", _legacy), ("candidate", _candidate)):
                    if _now() >= deadline:
                        raise TimeoutError("deadline reached before warmup")
                    warmup_started = _now()
                    out = fn(x)
                    if _now() >= deadline:
                        raise TimeoutError("deadline reached after warmup")
                    if _now() - warmup_started > 10.0:
                        raise TimeoutError("single-call admission budget exceeded")
                    out = _validate_output(out, f, label)
                    if label == "legacy":
                        baseline_out = out.copy()
                        parity = _bitwise_equal(baseline_out, out)
                    else:
                        parity = _bitwise_equal(baseline_out, out)
                    result["warmup"][label] = {"parity": parity}
                    if not parity:
                        result["bitwise_parity_each_call"] = False
                        raise RuntimeError("warmup outputs are not bitwise identical")
                    if _hash_bytes(x.tobytes()) != input_before:
                        raise _InputMutationError("implementation mutated read-only benchmark input")
                schedule = ("legacy", "candidate", "candidate", "legacy")
                for round_index in range(rounds):
                    for mode in schedule:
                        if _now() >= deadline:
                            raise TimeoutError("deadline reached before timed call")
                        fn = _legacy if mode == "legacy" else _candidate
                        call_started = _now()
                        value = fn(x)
                        call_finished = _now()
                        elapsed = call_finished - call_started
                        if not np.isfinite(elapsed) or elapsed < 0:
                            raise RuntimeError("timer returned an invalid duration")
                        if elapsed > 10.0:
                            raise TimeoutError("single-call admission budget exceeded")
                        if call_finished >= deadline:
                            raise TimeoutError("deadline reached after timed call")
                        out = _validate_output(value, f, mode)
                        if not _bitwise_equal(baseline_out, out):
                            result["bitwise_parity_each_call"] = False
                            raise RuntimeError(f"{mode} output differs bitwise from legacy")
                        # Record only elapsed duration, never the returned output.
                        record[f"{mode}_seconds"].append(elapsed)
                        result["samples"].append({"mode": mode, "seconds": elapsed,
                            "bitwise_parity_each_call": True, "shape_ok": True, "dtype_ok": True})
                        if _hash_bytes(x.tobytes()) != input_before:
                            raise _InputMutationError("implementation mutated read-only benchmark input")
                input_after = _hash_bytes(x.tobytes())
                if input_before != input_after:
                    raise _InputMutationError("benchmark input changed during calls")
                record["status"] = "complete"
                record["input_hash_after"] = input_after
                if record["legacy_seconds"] and record["candidate_seconds"]:
                    result["candidate_over_legacy"] = float(np.mean(record["candidate_seconds"]) / np.mean(record["legacy_seconds"]))
            except TimeoutError as exc:
                record["status"] = "partial_deadline"
                receipt["errors"].append(f"{name}: {exc}")
                receipt.update(status="partial", winner_eligible=False)
                receipt["budget"]["deadline_exceeded"] = True
                break
            except _InputMutationError as exc:
                record["status"] = "partial_input_mutation"
                result["failure"] = str(exc)
                receipt["errors"].append(f"{name}: {exc}")
                receipt.update(status="partial", winner_eligible=False)
                break
            except Exception as exc:
                record["status"] = "failed"
                result["failure"] = f"{type(exc).__name__}: {exc}"
                receipt["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
                receipt.update(status="failed", winner_eligible=False)
                break
        final_hashes = _source_hashes(base_root)
        final_identity = _runtime_identity()
        receipt["source_hashes_after"] = final_hashes
        receipt["runtime_after"] = final_identity
        receipt["candidate_source_sha256_after"] = _combined_source_hash({k: v for k, v in final_hashes.items() if "baseline" not in k})
        final_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_root, text=True).strip()
        if final_hashes != initial_hashes:
            receipt["errors"].append("source hashes changed during benchmark")
            receipt.update(status="partial", winner_eligible=False)
        if final_head != initial_head:
            receipt["errors"].append("HEAD changed during benchmark")
            receipt.update(status="partial", winner_eligible=False)
        if not _validate_identity(initial_identity, final_identity):
            receipt["errors"].append("runtime identity changed during benchmark")
            receipt.update(status="partial", winner_eligible=False)
        return receipt
    except Exception as exc:
        receipt["errors"].append(f"{type(exc).__name__}: {exc}")
        receipt.update(status="failed", winner_eligible=False)
        return receipt
    finally:
        globals()["_legacy"] = original_legacy
        _ACTIVE_ROOT, _PINNED_BASELINE_SHA, _BASELINE_REF = previous_root, previous_pin, previous_ref
        _legacy_default = previous_default


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="perform timed benchmark; default is dry-run")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    parser.add_argument("--baseline-ref", default="HEAD", help="immutable legacy revision; required after the candidate is committed")
    parser.add_argument("--cases", nargs="+", help="explicit case names such as q20_f4096_mixed")
    parser.add_argument("--output", type=Path, help="write a new receipt file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"status": "dry_run", "timed": False, "winner_eligible": False,
                          "cases": args.cases or [list(x) for x in _DEFAULT_CASES]}, indent=2))
        return 0
    receipt = run_benchmark(cases=args.cases, rounds=args.rounds, seed=args.seed,
                            deadline_seconds=args.deadline_seconds, baseline_ref=args.baseline_ref)
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
