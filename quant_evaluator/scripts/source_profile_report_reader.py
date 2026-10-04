"""Load caller-trusted, completed real-COS ABBA receipts as typed DTOs.

Reports are not signed, and parsing is not proof that source/runtime identities
are current. The embedded qualification is only structurally self-consistent.
Before reusing records, a caller must re-capture the live context, re-qualify
against that context, and explicitly warm the selected runtimes in a cold
process. This module never calibrates or silently selects a route.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
import stat

from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
    SourceRouteProfileQualification,
    validate_source_route_profile_qualification,
)
from quant_evaluator.runtime.source_profile_report_schema import (
    REPORT_KIND, REPORT_KIND_V2, REPORT_RUN_BACKENDS, REPORT_RUN_ORDER,
    REPORT_SCHEMAS, REPORT_SHAPE, REPORT_STATUS, PEARSON_METRICS,
)

MAX_REPORT_BYTES = 1024 * 1024
REPORT_METRICS = PEARSON_METRICS
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_TOP_FIELDS = frozenset({
    "kind", "status", "run_started", "shape", "metric_ids", "manifest_sha256",
    "requested_tile_cap", "preflight", "request_shape", "oracle", "profile_records",
    "run_order", "oracle_reports", "qualification_winner", "auto_verification",
})
_PREFLIGHT_FIELDS = frozenset({
    "pass", "available_ram_bytes", "minimum_available_ram_bytes",
    "cos_cache_disk_free_bytes", "required_disk_bytes",
})
_ORACLE_REPORT_FIELDS = frozenset({"pass", "backend", "run_index", "metrics"})
_METRIC_ORACLE_FIELDS = frozenset({
    "pass", "expected_coverage", "observed_coverage", "shape_equal",
    "finite_mask_equal", "nan_mask_equal", "positive_infinity_mask_equal",
    "negative_infinity_mask_equal", "observation_counts_equal", "max_abs_error",
    "tolerance",
})
_AUTO_FIELDS = frozenset({
    "backend_used", "qualification_status", "qualification_applied",
    "qualification_winner", "output_matches_independent_oracle", "oracle_report",
    "values_sha256",
})
_DEFAULT_AUTO_FIELDS = _AUTO_FIELDS | {"cache_status"}


@dataclass(frozen=True)
class SourceProfileReport:
    """Strictly decoded report plus the validator's self-consistency result.

    `records` remain caller-trusted. Use a newly captured live context when
    applying them to runtime routing; do not treat this result as live evidence.
    """

    kind: str
    status: str
    manifest_sha256: str
    records: tuple[CounterbalancedRouteProfileRecord, CounterbalancedRouteProfileRecord]
    qualification: SourceRouteProfileQualification


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _mapping(value, expected, name):
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    keys = frozenset(value)
    if keys != expected:
        missing, extra = expected - keys, keys - expected
        raise ValueError(f"{name} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _list(value, name):
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _str(value, name):
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _int(value, name, *, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer")
    return value


def _float(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


def _hash(value, name):
    value = _str(value, name)
    if _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _tuple(values, name, parser):
    return tuple(parser(item, f"{name}[{index}]") for index, item in enumerate(
        _list(values, name)))


def _pair_str_int(value, name):
    pair = _list(value, name)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (_str(pair[0], name), _int(pair[1], name))


def _pair_str_float(value, name):
    pair = _list(value, name)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (_str(pair[0], name), _float(pair[1], name))


def _range(value, name):
    pair = _list(value, name)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (_int(pair[0], name), _int(pair[1], name))


def _dto(value, cls, parsers, name):
    raw = _mapping(value, frozenset(field.name for field in fields(cls)), name)
    kwargs = {}
    for field in fields(cls):
        parser = parsers[field.name]
        kwargs[field.name] = parser(raw[field.name], f"{name}.{field.name}")
    return cls(**kwargs)


def _context(value, name):
    parsers = {field.name: (lambda v, n: _hash(v, n)) for field in fields(SourceRouteProfileContext)}
    for field in ("request_shape",):
        parsers[field] = lambda v, n: _tuple(v, n, lambda x, p: _int(x, p, minimum=1))
    parsers["metric_ids"] = lambda v, n: _tuple(v, n, _str)
    parsers["metric_coverage"] = lambda v, n: _tuple(v, n, _pair_str_int)
    parsers["metric_error_tolerances"] = lambda v, n: _tuple(v, n, _pair_str_float)
    for field in ("expected_coverage_count", "requested_tile_size",
                  "live_source_admitted_max_tile_size", "gpu_admitted_max_tile_size"):
        parsers[field] = lambda v, n: _int(v, n)
    parsers["timing_scope"] = _str
    return _dto(value, SourceRouteProfileContext, parsers, name)


def _output(value, name):
    parsers = {
        "metric_id": _str, "values_sha256": _hash, "finite_mask_sha256": _hash,
        "observation_counts_sha256": _hash,
        "coverage_expected": _int, "coverage_observed": _int,
    }
    return _dto(value, MetricOutputReceipt, parsers, name)


def _comparison(value, name):
    parsers = {
        "metric_id": _str, "cpu_values_sha256": _hash, "cuda_values_sha256": _hash,
        "cpu_finite_mask_sha256": _hash, "cuda_finite_mask_sha256": _hash,
        "cpu_observation_counts_sha256": _hash, "cuda_observation_counts_sha256": _hash,
        "compared_value_count": _int, "finite_mask_equal": _bool,
        "observation_counts_equal": _bool, "max_abs_error": _float,
        "max_abs_error_tolerance": _float, "evidence_sha256": _hash,
    }
    return _dto(value, MetricComparisonReceipt, parsers, name)


def _measurement(value, name):
    parsers = {
        "backend": _str, "seconds": _float, "oom_count": _int,
        "source_tile_size": _int,
        "source_ranges": lambda v, n: _tuple(v, n, _range),
        "actual_tile_size": _int,
        "compute_ranges": lambda v, n: _tuple(v, n, _range),
        "execution_config_sha256": _hash, "execution_schedule_sha256": _hash,
        "execution_schedule_scope": _str, "correctness_validated": _bool,
        "coverage_expected": _int, "coverage_observed": _int,
        "outputs": lambda v, n: _tuple(v, n, _output),
    }
    return _dto(value, BackendRouteProfileMeasurement, parsers, name)


def _record(value, name):
    parsers = {
        "context": _context,
        "execution_order": lambda v, n: _tuple(v, n, _str),
        "cpu": _measurement, "cuda": _measurement,
        "comparison_evidence": lambda v, n: _tuple(v, n, _comparison),
    }
    return _dto(value, CounterbalancedRouteProfileRecord, parsers, name)


def _check_oracle_report(raw, context, expected_backend, expected_index, name):
    item = _mapping(raw, _ORACLE_REPORT_FIELDS, name)
    if (_bool(item["pass"], f"{name}.pass") is not True
            or _str(item["backend"], f"{name}.backend") != expected_backend
            or _int(item["run_index"], f"{name}.run_index") != expected_index):
        raise ValueError(f"{name} did not pass or is misbound")
    metric_map = _mapping(item["metrics"], frozenset(context.metric_ids), f"{name}.metrics")
    coverage = dict(context.metric_coverage)
    tolerances = dict(context.metric_error_tolerances)
    for metric in context.metric_ids:
        output = _mapping(metric_map[metric], _METRIC_ORACLE_FIELDS,
                          f"{name}.metrics.{metric}")
        for key in _METRIC_ORACLE_FIELDS - {
                "expected_coverage", "observed_coverage", "max_abs_error", "tolerance"}:
            if _bool(output[key], f"{metric}.{key}") is not True:
                raise ValueError(f"{name} failed {metric}.{key}")
        if (_int(output["expected_coverage"], metric) != coverage[metric]
                or _int(output["observed_coverage"], metric) != coverage[metric]):
            raise ValueError(f"{name} coverage does not match context for {metric}")
        error = _float(output["max_abs_error"], metric)
        tolerance = _float(output["tolerance"], metric)
        if error < 0.0:
            raise ValueError(f"{name} absolute error must be nonnegative for {metric}")
        if tolerance != tolerances[metric] or error > tolerance:
            raise ValueError(f"{name} tolerance failed for {metric}")


def _check_auto_receipt(raw, context, winner, run_index, name, *, default_auto=False,
                        expected_value_hashes=None):
    expected_fields = _DEFAULT_AUTO_FIELDS if default_auto else _AUTO_FIELDS
    auto = _mapping(raw, expected_fields, name)
    if (_str(auto["backend_used"], f"{name}.backend_used") != winner
            or _str(auto["qualification_status"], f"{name}.qualification_status")
            != "qualified_current_source"
            or _bool(auto["qualification_applied"], f"{name}.qualification_applied") is not True
            or _str(auto["qualification_winner"], f"{name}.qualification_winner") != winner
            or _bool(auto["output_matches_independent_oracle"],
                     f"{name}.output_matches_independent_oracle") is not True):
        raise ValueError(f"{name} did not verify the qualified winner and oracle output")
    if default_auto and _str(auto["cache_status"], f"{name}.cache_status") != "cache_hit":
        raise ValueError("sixth default-auto receipt must report cache_status=cache_hit")
    _check_oracle_report(auto["oracle_report"], context, winner, run_index,
                         f"{name}.oracle_report")
    value_hashes = _mapping(auto["values_sha256"], frozenset(context.metric_ids),
                            f"{name}.values_sha256")
    for metric, digest in value_hashes.items():
        _hash(digest, f"{name}.values_sha256.{metric}")
        if expected_value_hashes is not None and digest != expected_value_hashes.get(metric):
            raise ValueError(f"{name}.values_sha256.{metric} does not match "
                             "the selected backend profile")


def _check_report(payload):
    if type(payload) is not dict:
        raise ValueError("report must be a JSON object")
    kind = _str(payload.get("kind"), "kind")
    schema = REPORT_SCHEMAS.get(kind)
    if schema is None:
        raise ValueError("report kind is unsupported")
    expected_fields = (_TOP_FIELDS | {"default_auto_verification"}
                       if schema.default_auto_required else _TOP_FIELDS)
    is_v2 = schema.default_auto_required
    body = _mapping(payload, expected_fields, "report")
    if body["status"] != REPORT_STATUS:
        raise ValueError("report status is not complete")
    if _bool(body["run_started"], "run_started") is not True:
        raise ValueError("report does not attest that the measured run started")
    shape = _tuple(body["shape"], "shape", lambda x, n: _int(x, n, minimum=1))
    request_shape = _tuple(body["request_shape"], "request_shape",
                           lambda x, n: _int(x, n, minimum=1))
    if shape != schema.shape or request_shape != schema.shape:
        raise ValueError("report does not describe the fixed source-profile request")
    metrics = _tuple(body["metric_ids"], "metric_ids", _str)
    if metrics != schema.metric_ids:
        raise ValueError("report metric order differs from its fixed schema")
    manifest = _hash(body["manifest_sha256"], "manifest_sha256")
    if _int(body["requested_tile_cap"], "requested_tile_cap") != schema.requested_tile_cap:
        raise ValueError("report tile cap differs from its fixed schema")
    preflight = _mapping(body["preflight"], _PREFLIGHT_FIELDS, "preflight")
    if _bool(preflight["pass"], "preflight.pass") is not True:
        raise ValueError("report preflight did not pass")
    for key in _PREFLIGHT_FIELDS - {"pass"}:
        _int(preflight[key], f"preflight.{key}", minimum=0)
    if (preflight["available_ram_bytes"] < preflight["minimum_available_ram_bytes"]
            or preflight["cos_cache_disk_free_bytes"] < preflight["required_disk_bytes"]):
        raise ValueError("preflight pass flag conflicts with its resource measurements")
    if body["oracle"] != schema.oracle:
        raise ValueError("report does not name its fixed independent oracle")
    if _tuple(body["run_order"], "run_order", _str) != REPORT_RUN_ORDER:
        raise ValueError("report ABBA run order is invalid")

    raw_records = _list(body["profile_records"], "profile_records")
    if len(raw_records) != 2:
        raise ValueError("report must contain exactly two profile records")
    records = tuple(_record(item, f"profile_records[{index}]")
                    for index, item in enumerate(raw_records))
    context = records[0].context
    if context.request_shape != schema.shape or context.metric_ids != schema.metric_ids:
        raise ValueError("profile context differs from its fixed report schema")
    if (context.requested_tile_size != 16
            or records[0].execution_order != ("cpu", "cuda")
            or records[1].execution_order != ("cuda", "cpu")):
        raise ValueError("profile records do not follow the fixed cap-16 ABBA schedule")
    if records[1].context != context:
        raise ValueError("profile records do not share an identical context")

    reports = _list(body["oracle_reports"], "oracle_reports")
    if len(reports) != 4:
        raise ValueError("report must contain four per-run oracle receipts")
    for index, raw in enumerate(reports):
        backend = REPORT_RUN_BACKENDS[index]
        _check_oracle_report(raw, context, backend, index, f"oracle_reports[{index}]")

    winner = _str(body["qualification_winner"], "qualification_winner")
    if winner not in ("cpu", "cuda"):
        raise ValueError("qualification winner must be cpu or cuda")
    qualification = validate_source_route_profile_qualification(
        records, expected_context=context)
    expected_value_hashes = None
    if schema.auto_values_must_match_profile:
        profile = records[0].cuda if winner == "cuda" else records[0].cpu
        expected_value_hashes = {item.metric_id: item.values_sha256 for item in profile.outputs}
    _check_auto_receipt(body["auto_verification"], context, winner, 4, "auto_verification",
                        expected_value_hashes=expected_value_hashes)
    if is_v2:
        _check_auto_receipt(body["default_auto_verification"], context, winner, 5,
                            "default_auto_verification", default_auto=True,
                            expected_value_hashes=expected_value_hashes)

    if qualification.winning_backend != winner:
        raise ValueError("report winner differs from its validated profile records")
    return SourceProfileReport(kind, REPORT_STATUS, manifest, records, qualification)


def parse_source_profile_report(payload: str | bytes) -> SourceProfileReport:
    """Parse bounded UTF-8 JSON and reconstruct the exact supported report DTOs."""
    if type(payload) is str:
        try:
            raw = payload.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("report text is not valid UTF-8") from exc
    elif type(payload) is bytes:
        raw = payload
    else:
        raise TypeError("payload must be JSON text or bytes")
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds the 1 MiB size limit")
    try:
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_object,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("report must be valid strict UTF-8 JSON") from exc
    return _check_report(parsed)


def load_source_profile_report(path: str | Path) -> SourceProfileReport:
    """Read a bounded regular file without blocking on special files."""
    nonblocking = getattr(os, "O_NONBLOCK", None)
    if nonblocking is None:
        raise ValueError("nonblocking report open is unavailable")
    try:
        file_path = os.fspath(Path(path))
        fd = os.open(file_path, os.O_RDONLY | nonblocking
                     | getattr(os, "O_CLOEXEC", 0))
    except (OSError, TypeError, ValueError):
        raise ValueError("report file could not be opened") from None

    try:
        try:
            mode = os.fstat(fd).st_mode
        except OSError:
            raise ValueError("report file could not be inspected") from None
        if not stat.S_ISREG(mode):
            raise ValueError("report source must be a regular file")
        try:
            with os.fdopen(fd, "rb", closefd=True) as stream:
                fd = -1
                payload = stream.read(MAX_REPORT_BYTES + 1)
        except OSError:
            raise ValueError("report file could not be read") from None
    finally:
        if fd >= 0:
            os.close(fd)
    return parse_source_profile_report(payload)


__all__ = (
    "MAX_REPORT_BYTES", "SourceProfileReport", "load_source_profile_report",
    "parse_source_profile_report",
)
