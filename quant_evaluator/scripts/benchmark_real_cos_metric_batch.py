#!/usr/bin/env python3
"""Bounded full-batch CPU/CUDA/auto A/B on verified DataAccess COS panels."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import resource
import statistics
import time
import traceback
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

DEFAULT_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
RANK_CHAIN = ("rank_ic", "rank_ic_series", "ic_std", "ic_ir")
PEARSON_CHAIN = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
QUANTILE_CHAIN = ("quantile_returns_full", "quantile_returns_daily",
                  "quantile_spread", "quantile_monotonicity",
                  "daily_quantile_monotonicity_rate")
RANK_SERIES_SINGLE = ("rank_ic_series",)
QUANTILE_FULL_SINGLE = ("quantile_returns_full",)
QUANTILE_DAILY_SINGLE = ("quantile_returns_daily",)
TURNOVER_SINGLE = ("factor_turnover_rate",)
POSITIVE_RATIO_SINGLE = ("rank_ic_positive_ratio",)
RANK_POSITIVE_PAIR = ("rank_ic", "rank_ic_positive_ratio")
METRICS = DEFAULT_METRICS
ALLOWED_BATCHES = (DEFAULT_METRICS, RANK_CHAIN, QUANTILE_CHAIN)
PEARSON_SINGLE_METRICS = tuple((metric,) for metric in PEARSON_CHAIN)
F32_SINGLE_METRICS = (RANK_SERIES_SINGLE, QUANTILE_FULL_SINGLE,
                      QUANTILE_DAILY_SINGLE, *PEARSON_SINGLE_METRICS)
MANIFEST_SHA256 = "b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864"
_BATCH = None
_LABELS = None


def _validate_f24_mixed_three_request(*, factors, metrics, run, output,
                                      verify_auto_reference=False):
    """Require a persisted run for the full-history F24 profile."""
    if factors != 24 or tuple(metrics) != DEFAULT_METRICS:
        return False
    if not run and not verify_auto_reference:
        raise ValueError("F24 mixed-three requires --run before full benchmark execution")
    if output is None:
        raise ValueError("F24 mixed-three requires --output to persist the receipt")
    return True


F24_SHAPE = (2586, 5461, 24)
F24_BACKEND_ORDER = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
F24_REFERENCE_DEFAULT = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / \
    "real_cos_f24_mixed_three_ab_20260929.json"


def _source_identity(source):
    """Return a public, stable binding for the full private loader identities."""
    if not isinstance(source, Mapping) or not isinstance(source.get("sources"), list):
        raise ValueError("source.sources must be a list")
    identities = []
    for item in source["sources"]:
        if not isinstance(item, Mapping):
            raise ValueError("each source identity must be an object")
        identity = {key: item.get(key) for key in (
            "factor_id", "uri", "sha256", "bytes", "etag", "manifest_sha256",
            "source_status")}
        factor_id, uri, digest = identity["factor_id"], identity["uri"], identity["sha256"]
        if (not isinstance(factor_id, str) or not factor_id
                or not isinstance(uri, str) or not uri.startswith("cos://")
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or digest not in uri or not uri.endswith(f"/{factor_id}.parquet")
                or type(identity["bytes"]) is not int or identity["bytes"] <= 0
                or not isinstance(identity["etag"], str) or not identity["etag"]
                or identity["manifest_sha256"] != MANIFEST_SHA256
                or not isinstance(identity["source_status"], str)):
            raise ValueError(f"invalid COS source identity for {factor_id!r}")
        identities.append(identity)
    factor_ids = [item["factor_id"] for item in identities]
    object_digests = [item["sha256"] for item in identities]
    if (len(identities) != 24 or len(set(factor_ids)) != 24
            or len(set(object_digests)) != 24):
        raise ValueError("F24 binding must contain 24 unique source identities and objects")
    identities.sort(key=lambda item: item["factor_id"])
    canonical = json.dumps(identities, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode()
    public_rows = sorted(({
        "source_identity_sha256": hashlib.sha256(json.dumps(
            item, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest(),
        "object_sha256": item["sha256"],
        "bytes": item["bytes"],
        "manifest_sha256": item["manifest_sha256"],
    } for item in identities), key=lambda item: item["source_identity_sha256"])
    public_canonical = json.dumps(public_rows, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode()
    return {
        "count": len(identities),
        "manifest_sha256": MANIFEST_SHA256,
        "total_bytes": sum(item["bytes"] for item in identities),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "summary_sha256": hashlib.sha256(public_canonical).hexdigest(),
        "sources": public_rows,
    }


def _validate_source_binding(binding):
    """Validate the sanitized F24 binding summary without exposing source locators."""
    if not isinstance(binding, Mapping):
        raise ValueError("source.source_binding is missing")
    rows = binding.get("sources")
    if (binding.get("count") != 24 or binding.get("manifest_sha256") != MANIFEST_SHA256
            or not isinstance(rows, list) or len(rows) != 24):
        raise ValueError("F24 source binding count or manifest mismatch")
    identities = []
    object_digests = []
    total_bytes = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source binding row must be an object")
        identity_digest = row.get("source_identity_sha256")
        object_digest = row.get("object_sha256")
        object_bytes = row.get("bytes")
        if (not isinstance(identity_digest, str) or len(identity_digest) != 64
                or any(char not in "0123456789abcdef" for char in identity_digest)
                or not isinstance(object_digest, str) or len(object_digest) != 64
                or any(char not in "0123456789abcdef" for char in object_digest)
                or type(object_bytes) is not int or object_bytes <= 0
                or row.get("manifest_sha256") != MANIFEST_SHA256):
            raise ValueError("invalid sanitized source binding row")
        identities.append(identity_digest)
        object_digests.append(object_digest)
        total_bytes += object_bytes
    if len(set(identities)) != 24 or len(set(object_digests)) != 24:
        raise ValueError("F24 source binding must contain 24 unique identities and objects")
    if binding.get("total_bytes") != total_bytes:
        raise ValueError("F24 source binding byte total mismatch")
    public_canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode()
    summary_digest = hashlib.sha256(public_canonical).hexdigest()
    if binding.get("summary_sha256") != summary_digest:
        raise ValueError("F24 source binding summary digest mismatch")
    canonical_digest = binding.get("canonical_sha256")
    if (not isinstance(canonical_digest, str) or len(canonical_digest) != 64
            or any(char not in "0123456789abcdef" for char in canonical_digest)):
        raise ValueError("F24 source binding canonical digest is malformed")
    return dict(binding)


def _public_source_report(source, *, include_binding=False):
    """Keep an allowlisted public summary; add strict bindings only for F24 mixed-three."""
    safe_fields = (
        "days", "assets", "date_span", "calendar_sessions",
        "missing_adj_vwap_partitions", "label", "factor_finite_ratio",
        "label_finite_ratio", "limitations",
    )
    public = {key: _plain(source[key]) for key in safe_fields if key in source}
    if include_binding:
        public["source_binding"] = _source_identity(source)
    return public

def _validate_auto_reference(reference):
    """Validate the archived F24 baseline before any COS data is loaded."""
    if not isinstance(reference, Mapping):
        raise ValueError("reference report must be a JSON object")
    request = reference.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("reference request is missing")
    expected_request = {
        "metrics": list(DEFAULT_METRICS), "backend_order": list(F24_BACKEND_ORDER),
        "factors": 24, "shape": list(F24_SHAPE), "dtype": "float64",
        "repeats_per_child": 2, "manifest_sha256": MANIFEST_SHA256,
        "compact_artifacts": True,
    }
    for key, expected in expected_request.items():
        if request.get(key) != expected:
            raise ValueError(f"reference request.{key} mismatch")
    if reference.get("parity_pass") is not True:
        raise ValueError("reference parity_pass must be true")
    runs = reference.get("runs")
    if not isinstance(runs, list) or len(runs) != 6:
        raise ValueError("reference must contain exactly six runs")
    for index, (run, backend) in enumerate(zip(runs, F24_BACKEND_ORDER), start=1):
        if (not isinstance(run, Mapping) or run.get("status") != "ok"
                or run.get("round") != index or run.get("backend_requested") != backend):
            raise ValueError(f"reference run {index} status/order mismatch")
        digest = run.get("artifact_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"reference run {index} lacks a compact artifact hash")
        expected_backend = "cuda" if backend == "cuda_strict" else "cpu"
        expected_metric_backends = {metric: expected_backend for metric in DEFAULT_METRICS}
        if (run.get("backend_used") != expected_backend
                or run.get("metric_backends") != expected_metric_backends):
            raise ValueError(f"reference run {index} backend result mismatch")
    configs = reference.get("config_hashes")
    run_configs = [run.get("config_hash") for run in runs]
    if (not isinstance(configs, list) or len(configs) != 1
            or not isinstance(configs[0], str) or len(configs[0]) != 64
            or any(value != configs[0] for value in run_configs)):
        raise ValueError("reference config hashes are inconsistent")
    if runs[1]["backend_used"] != "cuda" or runs[4]["backend_used"] != "cuda":
        raise ValueError("reference explicit CUDA rounds did not use CUDA")
    cuda_hashes = {runs[1]["artifact_sha256"], runs[4]["artifact_sha256"]}
    if len(cuda_hashes) != 1:
        raise ValueError("reference explicit CUDA artifact hashes disagree")
    for index in (2, 3):
        if (runs[index]["backend_used"] != "cpu"
                or runs[index].get("auto_backend_reason") != "metric_not_certified_for_profile"):
            raise ValueError("reference must identify the known CPU auto fallback")
    comparisons = reference.get("comparisons_to_first_cpu")
    if not isinstance(comparisons, Mapping):
        raise ValueError("reference parity comparisons are missing")
    required_metric_checks = (
        "descriptor", "values_rtol_1e-8_atol_1e-10", "finite_mask", "valid_mask",
        "counts", "provenance_observation_counts", "provenance", "metric_values",
    )
    for backend in ("cpu", "cuda_strict"):
        checks = comparisons.get(backend)
        if not isinstance(checks, list) or len(checks) != 2:
            raise ValueError(f"reference {backend} parity checks are incomplete")
        for check in checks:
            if not isinstance(check, Mapping):
                raise ValueError(f"reference {backend} parity checks are malformed")
            metric_checks = check.get("metrics")
            if (check.get("pass") is not True or check.get("config_hash") is not True
                    or not isinstance(metric_checks, Mapping)
                    or any(not isinstance(metric_checks.get(metric), Mapping)
                           or metric_checks[metric].get("pass") is not True
                           or any(metric_checks[metric].get(field) is not True
                                  for field in required_metric_checks)
                           for metric in DEFAULT_METRICS)):
                raise ValueError(f"reference {backend} parity checks failed or are incomplete")
    source = reference.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("reference source summary is missing")
    source_identity = _validate_source_binding(source.get("source_binding"))
    return {"config_hash": configs[0], "cuda_artifact_sha256": next(iter(cuda_hashes)),
            "source_identity": source_identity}


def _artifact_sha256(artifacts):
    return hashlib.sha256(json.dumps(
        _plain(artifacts), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _auto_cuda_route_matches(run):
    return (
        run.get("backend_used") == "cuda"
        and run.get("auto_backend_reason") == "certified_batch_real_cos_f24_mixed_three"
        and run.get("metric_backends") == {metric: "cuda" for metric in DEFAULT_METRICS}
    )


def _auto_reference_run_passes(run):
    return (
        _auto_cuda_route_matches(run)
        and run.get("artifact_hash_matches_explicit_cuda") is True
        and run.get("config_hash_matches_reference") is True
        and run.get("source_identity_matches_reference") is True
    )


def _plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return str(value)
    return value


def _metric_value(value):
    if value is None:
        return None
    return {name: _plain(getattr(value, name)) for name in (
        "metric_id", "value", "valid", "observation_count", "metric_version",
        "warnings", "sample_unit")}


def _worker(connection, backend, repeats):
    try:
        from quant_evaluator.runtime.evaluator import evaluate

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            bundle = evaluate(_BATCH, _LABELS, metrics=METRICS, backend=backend)
            timings.append(time.perf_counter() - start)

        artifacts = {}
        for metric in METRICS:
            artifact = bundle.artifacts[metric]
            values = np.asarray(artifact.values)
            finite_mask = np.isfinite(values)
            explicit_mask = getattr(artifact, "valid_mask", None)
            explicit_counts = getattr(artifact, "counts", None)
            observations = tuple(
                _metric_value(bundle.grouped_metrics[factor_id].get(metric))
                for factor_id in bundle.factor_ids
            )
            descriptor = {
                "type": type(artifact).__name__,
                "artifact_kind": _plain(getattr(artifact, "artifact_kind", None)),
            }
            if is_dataclass(artifact):
                descriptor.update({
                    field.name: _plain(getattr(artifact, field.name))
                    for field in fields(artifact)
                    if field.name not in {"values", "counts", "valid_mask", "provenance"}
                })
            else:
                raise TypeError(f"unsupported non-dataclass artifact: {type(artifact).__name__}")
            artifacts[metric] = {
                "descriptor": descriptor,
                "values": values.tolist(),
                "finite_mask": finite_mask.tolist(),
                "valid_mask": (finite_mask if explicit_mask is None else
                               np.asarray(explicit_mask)).tolist(),
                "counts": (None if explicit_counts is None else
                           np.asarray(explicit_counts).tolist()),
                "provenance_observation_counts": _plain(
                    artifact.provenance.get("observation_counts")),
                "provenance": _plain(artifact.provenance),
                "metric_values": observations,
            }
        meta = bundle.metadata
        connection.send({
            "status": "ok", "backend_requested": backend,
            "seconds": timings, "cold_s": timings[0],
            "warm_median_s": statistics.median(timings[1:]) if len(timings) > 1 else None,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "backend_used": meta.get("backend_used"),
            "auto_backend_reason": meta.get("auto_backend_reason"),
            "auto_backend_profile": meta.get("auto_backend_profile"),
            "metric_backends": _plain(meta.get("metric_backends")),
            "metric_observation_counts": {
                metric: [None if item is None else item.get("observation_count")
                        for item in artifacts[metric]["metric_values"]]
                for metric in METRICS},
            "peak_vram": meta.get("peak_vram"),
            "factor_tile_size": meta.get("factor_tile_size"),
            "factor_tiles_processed": meta.get("factor_tiles_processed"),
            "vram_budget_bytes": meta.get("vram_budget_bytes"),
            "config_hash": bundle.config_hash,
            "execution_receipt": _plain(meta.get("execution_receipt")),
            "artifacts": artifacts,
        })
    except BaseException as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=12)})
    finally:
        connection.close()


def _run_one(context, backend, repeats, timeout_s):
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, backend, repeats))
    process.start()
    sender.close()
    if receiver.poll(timeout_s):
        try:
            result = receiver.recv()
        except EOFError:
            result = {"status": "crash", "exitcode": process.exitcode}
    else:
        result = {"status": "timeout", "timeout_s": timeout_s}
        process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    receiver.close()
    if result.get("status") != "ok":
        raise RuntimeError(f"{backend}: {result}")
    return result


def _same(a, b):
    return json.dumps(_plain(a), sort_keys=True, separators=(",", ":"),
                      allow_nan=True) == json.dumps(_plain(b), sort_keys=True,
                                                    separators=(",", ":"), allow_nan=True)


def _close_values(a, b):
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return left.shape == right.shape and bool(np.allclose(
        left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


def _metric_values_same(a, b):
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if left is None or right is None:
            if left is not right:
                return False
            continue
        if not _same({k: v for k, v in left.items() if k != "value"},
                     {k: v for k, v in right.items() if k != "value"}):
            return False
        if left["value"] is None or right["value"] is None:
            if left["value"] is not right["value"]:
                return False
        elif not _close_values([left["value"]], [right["value"]]):
            return False
    return True


def _compare(reference, candidate):
    result = {"config_hash": reference["config_hash"] == candidate["config_hash"],
              "metrics": {}}
    for metric in METRICS:
        a, b = reference["artifacts"][metric], candidate["artifacts"][metric]
        result["metrics"][metric] = {
            "descriptor": _same(a["descriptor"], b["descriptor"]),
            "values_rtol_1e-8_atol_1e-10": _close_values(a["values"], b["values"]),
            "finite_mask": _same(a["finite_mask"], b["finite_mask"]),
            "valid_mask": _same(a["valid_mask"], b["valid_mask"]),
            "counts": _same(a["counts"], b["counts"]),
            "provenance_observation_counts": _same(
                a["provenance_observation_counts"], b["provenance_observation_counts"]),
            "provenance": _same(a["provenance"], b["provenance"]),
            "metric_values": _metric_values_same(a["metric_values"], b["metric_values"]),
        }
        result["metrics"][metric]["pass"] = all(
            result["metrics"][metric][field] for field in (
                "descriptor", "values_rtol_1e-8_atol_1e-10", "finite_mask",
                "valid_mask", "counts", "provenance_observation_counts",
                "provenance", "metric_values"))
    result["pass"] = result["config_hash"] and all(
        metric["pass"] for metric in result["metrics"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--factors", type=int, choices=(2, 8, 12, 13, 24, 32), default=8)
    parser.add_argument("--compact", action="store_true", help="write artifact hashes, not value arrays")
    parser.add_argument("--run", action="store_true",
                        help="run a preflight-approved F32 request or the exact F24 mixed-three request")
    parser.add_argument("--verify-auto-reference", action="store_true",
                        help="validate the archived F24 baseline, then run only 1-2 auto requests")
    parser.add_argument("--reference-report", type=Path, default=F24_REFERENCE_DEFAULT)
    parser.add_argument("--auto-runs", type=int, choices=(1, 2), default=1,
                        help="number of auto requests in reference verification")
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=1,
                        help="evaluations per auto request in reference verification")
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("timeout-s must be positive")

    global _BATCH, _LABELS, METRICS
    requested = tuple(args.metrics.split(","))
    if requested not in ALLOWED_BATCHES:
        if not ((args.factors == 2 and requested == RANK_SERIES_SINGLE)
                or (args.factors == 13 and requested in (
                    TURNOVER_SINGLE, POSITIVE_RATIO_SINGLE, RANK_POSITIVE_PAIR))
                or (args.factors == 32 and requested in (*F32_SINGLE_METRICS, PEARSON_CHAIN))):
            parser.error("metrics must be an exact default, rank-chain or quantile-chain set; "
                         "F2 permits rank_ic_series alone, F13 permits factor_turnover_rate "
                         "or rank_ic_positive_ratio alone, or their exact rank-IC/positive-ratio pair, "
                         "and F32 permits rank_ic_series, quantile_returns_full, "
                         "quantile_returns_daily or a Pearson metric alone, or the exact Pearson chain")
    if args.factors == 2 and requested not in (RANK_SERIES_SINGLE, RANK_CHAIN, QUANTILE_CHAIN):
        parser.error("F2 runs require rank_ic_series alone or an exact rank/quantile chain")
    if args.factors == 13 and requested not in (
            RANK_CHAIN, QUANTILE_CHAIN, TURNOVER_SINGLE, POSITIVE_RATIO_SINGLE, RANK_POSITIVE_PAIR):
        parser.error("F13 runs require the exact rank-chain, quantile-chain, "
                     "factor_turnover_rate, rank_ic_positive_ratio, or rank-IC/positive-ratio pair")
    if args.factors == 24 and requested not in (RANK_CHAIN, QUANTILE_CHAIN, DEFAULT_METRICS):
        parser.error("F24 runs require the exact rank-chain, quantile-chain, or mixed-three request")
    if args.factors == 32 and requested not in (*F32_SINGLE_METRICS, PEARSON_CHAIN):
        parser.error("F32 runs require a certified singleton (rank, quantile or Pearson) "
                     "or the exact Pearson chain")
    if args.verify_auto_reference:
        if args.run:
            parser.error("--verify-auto-reference cannot be combined with --run")
        if args.factors != 24 or requested != DEFAULT_METRICS:
            parser.error("--verify-auto-reference requires the exact F24 mixed-three request")
    try:
        f24_mixed_request = _validate_f24_mixed_three_request(
            factors=args.factors, metrics=requested, run=args.run, output=args.output,
            verify_auto_reference=args.verify_auto_reference)
    except ValueError as exc:
        parser.error(str(exc))
    preflight_request = requested in (PEARSON_CHAIN, QUANTILE_DAILY_SINGLE,
                                      *PEARSON_SINGLE_METRICS)
    if args.run and not ((args.factors == 32 and preflight_request) or f24_mixed_request):
        parser.error("--run applies only to a preflight-approved F32 request "
                     "or the exact F24 mixed-three request")
    if preflight_request:
        if args.factors != 32:
            parser.error("this request requires the exact F32 panel")
        from quant_evaluator.scripts.benchmark_real_cos_factor_batch import (
            preflight_factor_count_profile,
        )
        preflight = preflight_factor_count_profile(SimpleNamespace(
            manifest_sha256=MANIFEST_SHA256,
            profile_max_object_mib=128,
            profile_max_total_mib=2048,
            max_working_gib=50,
        ))
        print(json.dumps({"preflight": preflight}, ensure_ascii=False), flush=True)
        if preflight["status"] != "ready":
            raise SystemExit(2)
        if not args.run:
            print(json.dumps({"status": "preflight_only",
                              "note": "pass --run to start the F32 "
                                      + ("Pearson chain" if requested == PEARSON_CHAIN else
                                         "Pearson singleton" if requested in PEARSON_SINGLE_METRICS else
                                         "daily quantile")
                                      + " benchmark"}), flush=True)
            return
    reference = None
    reference_meta = None
    if args.verify_auto_reference:
        try:
            reference = json.loads(args.reference_report.read_text())
            reference_meta = _validate_auto_reference(reference)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"invalid auto reference report: {exc}")
    METRICS = requested
    load_start = time.perf_counter()
    _BATCH, _LABELS, source = load_real_batch(
        factors=args.factors, days=0, assets=5500,
        max_object_mib=128 if args.factors in (24, 32) else 64,
        max_total_mib=(2048 if args.factors in (24, 32) else
                       128 if args.factors in (2, 8) else 256),
        manifest_sha256=MANIFEST_SHA256)
    load_s = time.perf_counter() - load_start
    shape = (_BATCH.num_times, _BATCH.num_assets, _BATCH.num_factors)
    if shape != (2586, 5461, args.factors):
        raise RuntimeError(f"unexpected bound panel shape: {shape}")

    if args.verify_auto_reference:
        if shape != F24_SHAPE or _BATCH.values.dtype != np.dtype("float64"):
            raise RuntimeError("loaded panel does not match F24 reference shape/dtype")
        current_source_identity = _source_identity(source)
        source_match = current_source_identity == reference_meta["source_identity"]
        if not source_match:
            raise RuntimeError("loaded COS source identities do not match reference report")
        context = mp.get_context("fork")
        expected_cuda_hash = reference_meta["cuda_artifact_sha256"]
        runs = []
        for index in range(args.auto_runs):
            run = _run_one(context, "auto", repeats=args.repeats, timeout_s=args.timeout_s)
            artifact_hash = _artifact_sha256(run["artifacts"])
            run["round"] = index + 1
            run["artifact_sha256"] = artifact_hash
            run["artifact_hash_matches_explicit_cuda"] = artifact_hash == expected_cuda_hash
            run["config_hash_matches_reference"] = (
                run["config_hash"] == reference_meta["config_hash"])
            run["source_identity_matches_reference"] = source_match
            run["auto_cuda_route_matches_reference"] = _auto_cuda_route_matches(run)
            del run["artifacts"]
            runs.append(run)
            print(json.dumps({
                "verify_auto_round": index + 1, "backend_used": run["backend_used"],
                "auto_backend_reason": run["auto_backend_reason"],
                "auto_cuda_route_matches_reference": run["auto_cuda_route_matches_reference"],
                "metric_backends": run["metric_backends"],
                "metric_observation_counts": run["metric_observation_counts"],
                "peak_vram": run["peak_vram"],
                "artifact_hash_matches_explicit_cuda": run["artifact_hash_matches_explicit_cuda"],
                "config_hash_matches_reference": run["config_hash_matches_reference"]},
                ensure_ascii=False), flush=True)
        report = {
            "created_utc": datetime.now().astimezone().isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mode": "verify_auto_reference",
            "reference_report": str(args.reference_report),
            "reference_validation_pass": True,
            "request": {"metrics": METRICS, "backend": "auto", "auto_runs": args.auto_runs,
                        "repeats_per_run": args.repeats, "factors": 24,
                        "shape": shape, "dtype": str(_BATCH.values.dtype),
                        "manifest_sha256": MANIFEST_SHA256},
            "source": _public_source_report(source, include_binding=True), "source_identity_matches_reference": source_match,
            "reference_config_hash": reference_meta["config_hash"],
            "reference_explicit_cuda_artifact_sha256": expected_cuda_hash,
            "runs": runs,
            "parity_pass": bool(runs) and source_match and all(
                _auto_reference_run_passes(run) for run in runs),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                              allow_nan=False) + "\n")
        print(json.dumps({"parity_pass": report["parity_pass"],
                          "auto_routes": [run["backend_used"] for run in runs],
                          "output": str(args.output) if args.output else None},
                         ensure_ascii=False), flush=True)
        if not report["parity_pass"]:
            raise SystemExit(2)
        return

    if f24_mixed_request and _BATCH.values.dtype != np.dtype("float64"):
        raise RuntimeError(f"F24 mixed-three requires float64 input, got {_BATCH.values.dtype}")
    # Symmetric interleaving reduces order and cache bias across backends.
    order = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
    context = mp.get_context("fork")
    runs = []
    for index, backend in enumerate(order):
        run = _run_one(context, backend, repeats=2, timeout_s=args.timeout_s)
        run["round"] = index + 1
        runs.append(run)
        print(json.dumps({"round": index + 1, "backend": backend,
                          "cold_s": run["cold_s"], "warm_median_s": run["warm_median_s"],
                          "backend_used": run["backend_used"],
                          "auto_backend_reason": run["auto_backend_reason"],
                          "peak_vram": run["peak_vram"],
                          "peak_rss_kib": run["peak_rss_kib"]}), flush=True)

    by_backend = {backend: [run for run in runs if run["backend_requested"] == backend]
                  for backend in ("cpu", "cuda_strict", "auto")}
    reference = by_backend["cpu"][0]
    comparisons = {
        backend: [_compare(reference, run) for run in selected]
        for backend, selected in by_backend.items()
    }
    config_hashes = sorted({run["config_hash"] for run in runs})
    report_runs = runs if not args.compact else [
        {key: value for key, value in run.items() if key != "artifacts"} | {
            "artifact_sha256": _artifact_sha256(run["artifacts"])}
        for run in runs
    ]
    report = {
        "created_utc": datetime.now().astimezone().isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request": {"metrics": METRICS, "backend_order": order,
                    "factors": args.factors,
                    "repeats_per_child": 2, "timeout_s": args.timeout_s,
                    "shape": shape, "dtype": str(_BATCH.values.dtype),
                    "manifest_sha256": MANIFEST_SHA256, "compact_artifacts": args.compact},
        "source": _public_source_report(source, include_binding=f24_mixed_request), "load_s": load_s,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "config_hashes": config_hashes,
        "runs": report_runs, "comparisons_to_first_cpu": comparisons,
        "parity_pass": len(config_hashes) == 1 and all(
            comparison["pass"] for repeated in comparisons.values()
            for comparison in repeated),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                          allow_nan=False) + "\n")
    print(json.dumps({"parity_pass": report["parity_pass"],
                      "config_hashes": config_hashes,
                      "auto_routes": [run["backend_used"] for run in by_backend["auto"]],
                      "output": str(args.output) if args.output else None}), flush=True)
    if not report["parity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
