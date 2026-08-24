#!/usr/bin/env python3
"""Read-only release gate for staged Status feature runs.

The pipeline creates artifacts; this module independently decides whether the
artifacts are safe to promote. It intentionally does not upload, rename, or
modify anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "schemas" / "feature_manifest.schema.json"
MANIFEST_NAME = "feature_manifest.json"

# The manifest schema cannot validate a Parquet payload, so keep a minimum
# release contract for columns that make publication and PIT checks possible.
REQUIRED_OUTPUT_COLUMNS = {
    "observation_date",
    "symbol",
    "publishable_flag",
    "pit_validity_flag",
    "availability_timestamp",
    "observation_time",
}

REQUIRED_FEATURE_FIELDS = (
    "canonical_feature_id",
    "field_name",
    "semantic_version",
    "market",
    "frequency",
    "source_version",
    "availability_rule",
    "formula_hash",
    "formula_version",
    "source_lineage",
)

# Terms such as "t+1" are valid for a close-known signal. Only explicit
# future-data/look-ahead language is a release blocker.
PIT_VIOLATION = re.compile(
    r"(?:\blook\s*[-_ ]?ahead\b|\blookahead\b|"
    r"\bfuture\s+(?:data|information|value|price|return|label)\b|"
    r"\bforward\s*[-_ ]?(?:looking|return|label|price)\b|"
    r"\bfuture_(?:data|information|value|price|return|label)\b|"
    r"未来(?:数据|信息|值|价格|收益|标签)?|前视|前瞻|泄漏|偷看未来)",
    re.IGNORECASE,
)


class PublishValidationError(ValueError):
    """Raised when one or more release gates fail."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_fingerprint(schema: Any) -> str:
    return hashlib.sha256(str(schema).encode("utf-8")).hexdigest()


def _error_path(error: jsonschema.ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return path or "$"


def _validate_manifest_schema(manifest: dict[str, Any], schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    return [
        f"manifest schema: {_error_path(error)}: {error.message}"
        for error in sorted(
            validator.iter_errors(manifest), key=lambda item: list(item.absolute_path)
        )
    ]


def _required_manifest_fields(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    top_level = (
        "run_id",
        "registry_version",
        "pipeline_version",
        "source_root",
        "output_root",
        "feature_set",
        "frequency",
        "date_range",
        "features",
        "output",
        "quality",
        "memory",
    )
    for name in top_level:
        if name not in manifest:
            errors.append(f"required manifest field missing: {name}")

    date_range = manifest.get("date_range")
    if isinstance(date_range, dict) and all(
        key in date_range for key in ("start", "end")
    ):
        try:
            if date.fromisoformat(date_range["start"]) > date.fromisoformat(
                date_range["end"]
            ):
                errors.append("date_range.start must not be after date_range.end")
        except (TypeError, ValueError):
            pass  # JSON Schema reports malformed dates.

    output = manifest.get("output")
    if isinstance(output, dict):
        for name in ("local_path", "sha256", "bytes", "rows", "schema_fingerprint"):
            if name not in output:
                errors.append(f"required output field missing: {name}")

    quality = manifest.get("quality")
    if isinstance(quality, dict):
        for name in ("rows", "coverage_ratio", "publishable_rows"):
            if name not in quality:
                errors.append(f"required quality field missing: {name}")

    memory = manifest.get("memory")
    if isinstance(memory, dict):
        for name in ("rss_bytes", "available_bytes", "total_bytes", "worker_count"):
            if name not in memory:
                errors.append(f"required memory field missing: {name}")

    features = manifest.get("features")
    if isinstance(features, list):
        for index, feature in enumerate(features):
            if not isinstance(feature, dict):
                errors.append(f"feature[{index}] must be an object")
                continue
            for name in REQUIRED_FEATURE_FIELDS:
                if name not in feature:
                    errors.append(f"required feature field missing: features[{index}].{name}")
            string_fields = (
                "canonical_feature_id",
                "field_name",
                "semantic_version",
                "frequency",
                "source_version",
                "availability_rule",
                "formula_version",
            )
            for name in string_fields:
                value = feature.get(name)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"feature[{index}].{name} must be a non-empty string")
            if feature.get("source_lineage") == []:
                errors.append(f"feature[{index}].source_lineage must not be empty")
    return errors


def _integrity_checks(
    manifest: dict[str, Any], workdir: Path
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    output = manifest.get("output", {})
    raw_path = output.get("local_path")
    if not isinstance(raw_path, str) or not raw_path:
        return ["output.local_path must identify a file"], {}
    output_path = Path(raw_path)
    if not output_path.is_absolute():
        output_path = workdir / output_path
    else:
        output_path = output_path.resolve()
        # Runs are often copied for validation/replay. Prefer the copied
        # artifact when the manifest contains an absolute path from its source
        # run, otherwise a replay could accidentally validate the original.
        try:
            output_path.relative_to(workdir)
        except ValueError:
            local_candidate = workdir / "outputs" / output_path.name
            if local_candidate.is_file():
                output_path = local_candidate
    output_path = output_path.resolve()
    if not output_path.is_file():
        return [f"output file not found: {output_path}"], {"path": str(output_path)}

    actual_sha256 = sha256_file(output_path)
    actual_bytes = output_path.stat().st_size
    try:
        parquet = pq.ParquetFile(output_path)
        actual_rows = parquet.metadata.num_rows
        actual_schema = parquet.schema_arrow
        actual_columns = set(actual_schema.names)
    except Exception as exc:  # Arrow exception types vary by version.
        return [f"output is not readable Parquet: {exc}"], {"path": str(output_path)}

    actual_fingerprint = schema_fingerprint(actual_schema)
    if output.get("sha256") != actual_sha256:
        errors.append(
            f"output.sha256 mismatch: manifest={output.get('sha256')} actual={actual_sha256}"
        )
    if output.get("bytes") != actual_bytes:
        errors.append(
            f"output.bytes mismatch: manifest={output.get('bytes')} actual={actual_bytes}"
        )
    if output.get("rows") != actual_rows:
        errors.append(
            f"output.rows mismatch: manifest={output.get('rows')} actual={actual_rows}"
        )
    if output.get("schema_fingerprint") != actual_fingerprint:
        errors.append("output.schema_fingerprint mismatch")
    missing_columns = sorted(REQUIRED_OUTPUT_COLUMNS - actual_columns)
    if missing_columns:
        errors.append(f"required output columns missing: {missing_columns}")

    return errors, {
        "path": str(output_path),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
        "rows": actual_rows,
        "schema_fingerprint": actual_fingerprint,
        "columns": sorted(actual_columns),
    }


def _quality_checks(
    manifest: dict[str, Any], output_info: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    quality = manifest.get("quality", {})
    rows = output_info.get("rows")
    if quality.get("rows") != rows:
        errors.append(f"quality.rows mismatch: quality={quality.get('rows')} output={rows}")
    publishable = quality.get("publishable_rows")
    if (
        not isinstance(publishable, int)
        or isinstance(publishable, bool)
        or publishable < 0
    ):
        errors.append("quality.publishable_rows must be a non-negative integer")
    elif rows is not None and publishable > rows:
        errors.append("quality.publishable_rows must not exceed output.rows")

    publishable_actual: int | None = None
    try:
        table = pq.read_table(
            output_info["path"],
            columns=[
                "publishable_flag",
                "pit_validity_flag",
                "availability_timestamp",
                "observation_time",
            ],
        )
        publishable_actual = sum(
            value == 1 or value is True
            for value in table["publishable_flag"].to_pylist()
        )
        if publishable != publishable_actual:
            errors.append(
                f"quality.publishable_rows mismatch: manifest={publishable} actual={publishable_actual}"
            )

        pit_values = table["pit_validity_flag"].to_pylist()
        if any(value != 1 for value in pit_values):
            errors.append("PIT violation: output.pit_validity_flag contains a value other than 1")

        observation_dates = table["observation_time"].to_pylist()
        availability = table["availability_timestamp"].to_pylist()
        earlier = sum(
            1
            for observed, available in zip(observation_dates, availability)
            if observed is not None
            and available is not None
            and available.date() < observed
        )
        if earlier:
            errors.append(
                f"PIT violation: availability_timestamp precedes observation_time in {earlier} rows"
            )
    except Exception as exc:  # Arrow exception types vary by version.
        errors.append(f"quality/PIT columns could not be read: {exc}")

    return errors, {"publishable_rows": publishable_actual}


def _memory_checks(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    memory = manifest.get("memory", {})
    for field in ("rss_bytes", "available_bytes", "total_bytes", "worker_count"):
        value = memory.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"memory.{field} must be a non-negative integer")
    available = memory.get("available_bytes")
    total = memory.get("total_bytes")
    if isinstance(available, int) and isinstance(total, int):
        if total <= 0:
            errors.append("memory.total_bytes must be positive")
        elif available < total * 0.20:
            errors.append(
                "memory gate failed: available_bytes must retain at least 20% of total_bytes"
            )
    return errors


def _pit_manifest_checks(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for index, feature in enumerate(manifest.get("features", [])):
        if not isinstance(feature, dict):
            continue
        for field in ("canonical_feature_id", "field_name", "availability_rule"):
            value = feature.get(field)
            if isinstance(value, str) and PIT_VIOLATION.search(value):
                errors.append(
                    f"PIT violation: features[{index}].{field} contains explicit future/look-ahead language"
                )
    return errors


def validate_publish(
    workdir: Path, schema_path: Path = DEFAULT_SCHEMA
) -> dict[str, Any]:
    """Validate one staged run and return a JSON-serializable report.

    A report with ``status == 'PASS'`` is the only release approval signal.
    """
    workdir = workdir.resolve()
    manifest_path = workdir / "manifests" / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PublishValidationError([f"manifest not found: {manifest_path}"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishValidationError([f"cannot read manifest: {exc}"]) from exc
    if not isinstance(manifest, dict):
        raise PublishValidationError(["feature manifest must be a JSON object"])

    errors: list[str] = []
    errors.extend(_validate_manifest_schema(manifest, schema_path))
    errors.extend(_required_manifest_fields(manifest))
    errors.extend(_pit_manifest_checks(manifest))
    integrity_errors, output_info = _integrity_checks(manifest, workdir)
    errors.extend(integrity_errors)
    quality_errors, quality_info = (
        _quality_checks(manifest, output_info) if output_info else ([], {})
    )
    errors.extend(quality_errors)
    errors.extend(_memory_checks(manifest))

    report = {
        "status": "PASS" if not errors else "BLOCKED",
        "run_id": manifest.get("run_id"),
        "manifest": str(manifest_path),
        "output": output_info,
        "quality": quality_info,
        "errors": errors,
    }
    if errors:
        raise PublishValidationError(errors)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a staged Status feature run before release."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="validate one local run directory"
    )
    validate.add_argument("--workdir", required=True)
    validate.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    args = parser.parse_args(argv)
    try:
        result = validate_publish(Path(args.workdir), Path(args.schema))
    except PublishValidationError as exc:
        result = {"status": "BLOCKED", "errors": exc.errors}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
