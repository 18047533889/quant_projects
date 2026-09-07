"""Verified, isolated factor artifacts using the existing batch Parquet writer.

No catalog registration, production publication or authoritative watermark.
Input result residency is owned by the caller; this function consumes its
separately admitted writer/DQ/conversion workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy
from factor_engine.runtime.dq_gates import assert_factor_dq
from factor_engine.storage.parquet_batch_writer import BatchParquetWriter


class ArtifactResourceRequirementError(MemoryError):
    reason_code = "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"

    def __init__(self, required, available):
        self.required_bytes, self.available_bytes = required, available
        super().__init__(f"writer requires {required} bytes; admitted {available} bytes")


def required_writer_workspace_bytes(value) -> int:
    if not isinstance(value, pd.Series):
        raise TypeError("v2 factor artifact requires a labelled pandas Series")
    # Conservative shape estimate, not a measured allocator guarantee. Includes
    # whole-factor DQ scratch plus one conversion/readback chunk and codec state.
    return max(1024 * 1024, int(value.memory_usage(index=True, deep=True)) * 4)


def _cell_fingerprints_equal(left, right):
    """Bound hash temporaries by rows; caller separately checks exact semantics."""
    if len(left) != len(right):
        return False
    for offset in range(0, len(left), 8192):
        before = pd.util.hash_pandas_object(left.iloc[offset:offset + 8192], index=True)
        after = pd.util.hash_pandas_object(right.iloc[offset:offset + 8192], index=True)
        if not before.equals(after):
            return False
        del before, after
    return True


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_digest(path, chunk_bytes):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_factor_artifact_receipt(receipt, root, run_id, ordinal, name, value, *, policy, budget_bytes):
    """Independently verify an external sink; booleans are never commit proof.

    Only the isolated v2 manifest format is accepted. The caller retains result
    residency and writer workspace throughout this readback. No files are
    mutated and no failed/unknown commit is retried here.
    """
    if not isinstance(policy, DefaultExecutionPolicy):
        raise TypeError("a validated DefaultExecutionPolicy is required")
    required = required_writer_workspace_bytes(value)
    if type(budget_bytes) is not int or budget_bytes < required:
        raise ArtifactResourceRequirementError(required, budget_bytes)
    if not isinstance(receipt, dict) or receipt.get("committed") is not True:
        raise ValueError("sink returned no committed artifact receipt")
    expected_root = (Path(root) / run_id / "values").resolve()
    path = Path(receipt.get("path", "")).resolve()
    if not path.is_relative_to(expected_root) or path.name != "manifest.json":
        raise ValueError("sink artifact is outside this run's isolated values scope")
    max_manifest_bytes = min(16 * 1024 * 1024, budget_bytes // 4)
    started = time.monotonic()
    with path.open("rb") as stream:
        payload = stream.read(max_manifest_bytes + 1)
    if len(payload) > max_manifest_bytes:
        raise ArtifactResourceRequirementError(len(payload) * 4, budget_bytes)
    if hashlib.sha256(payload).hexdigest() != receipt.get("sha256"):
        raise ValueError("sink manifest hash differs from receipt")

    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = item
        return result

    manifest = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(manifest, dict):
        raise ValueError("sink manifest must be an object")
    expected_identity = {
        "schema_version": "factor_engine.factor_artifact.v2", "run_id": run_id,
        "ordinal": ordinal, "factor_id": name, "policy_digest": policy.digest,
        "dtype": str(value.dtype), "rows": len(value),
    }
    if any(manifest.get(key) != item for key, item in expected_identity.items()):
        raise ValueError("sink manifest identity differs from requested output")
    if manifest.get("committed") is not True or manifest.get("production_published") is not False:
        raise ValueError("sink manifest is not an isolated committed artifact")
    if not manifest.get("generation") or manifest["generation"] != receipt.get("generation"):
        raise ValueError("sink artifact generation differs from receipt")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("sink manifest lacks chunk list")
    # Re-run DQ instead of trusting manifest reports supplied by the sink.
    assert_factor_dq(value, raise_on_fail=True)
    offset = total_bytes = 0
    seen = set()
    for chunk in chunks:
        if time.monotonic() - started >= policy.sink_flush_seconds:
            raise TimeoutError("sink receipt verification deadline exceeded")
        if not isinstance(chunk, dict):
            raise ValueError("invalid sink chunk record")
        filename = chunk.get("path")
        rows = chunk.get("rows")
        if (not isinstance(filename, str) or Path(filename).name != filename or
                filename in seen or type(rows) is not int or rows <= 0 or
                chunk.get("offset") != offset or offset + rows > len(value)):
            raise ValueError("sink chunks are not unique contiguous output coverage")
        seen.add(filename)
        chunk_path = (path.parent / filename).resolve()
        if chunk_path.parent != path.parent or not chunk_path.is_file():
            raise ValueError("sink chunk escapes artifact directory")
        size = chunk_path.stat().st_size
        if size != chunk.get("bytes") or size > policy.max_file_bytes:
            raise ValueError("sink chunk file size differs from manifest or exceeds policy")
        if _file_digest(chunk_path, min(1024 * 1024, budget_bytes // 16)) != chunk.get("sha256"):
            raise ValueError("sink chunk hash differs from manifest")
        parquet = pq.ParquetFile(chunk_path)
        metadata = parquet.metadata
        uncompressed = sum(metadata.row_group(group).column(col).total_uncompressed_size
                           for group in range(metadata.num_row_groups)
                           for col in range(metadata.num_columns))
        if metadata.num_rows != rows or uncompressed * 4 > budget_bytes:
            raise ValueError("sink chunk decoded shape exceeds admitted verification budget")
        restored = parquet.read().to_pandas()["factor_value"]
        expected = value.iloc[offset:offset + rows].rename("factor_value")
        pd.testing.assert_series_equal(restored, expected, check_exact=True,
                                       check_names=True, check_dtype=True,
                                       check_index_type=True, check_categorical=True)
        if not _cell_fingerprints_equal(restored, expected):
            raise ValueError("sink cell/axis fingerprint differs from requested output")
        offset += rows
        total_bytes += size
        del restored, expected
    if offset != len(value) or receipt.get("rows") != offset or receipt.get("bytes") != total_bytes:
        raise ValueError("sink receipt does not cover all requested rows/bytes")
    return {**receipt, "verified": True, "production_published": False}


def _artifact_directory(root, run_id, ordinal, generation):
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("invalid artifact run_id")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("invalid factor ordinal")
    if not isinstance(generation, str) or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
        raise ValueError("invalid artifact generation")
    return Path(root) / run_id / "values" / f"{ordinal:08d}-{generation}"


def reconcile_factor_artifact(root, run_id, ordinal, name, value, *, generation, policy, budget_bytes):
    """Read only one persisted intent's manifest after the writer has exited.

    The caller must supervise this verification with a finite deadline and hold
    workspace admission. This does not search, rewrite, or infer success from
    labels. Missing/partial/mismatched artifacts leave the commit unknown.
    """
    if not isinstance(policy, DefaultExecutionPolicy):
        raise TypeError("a validated DefaultExecutionPolicy is required")
    required = required_writer_workspace_bytes(value)
    if type(budget_bytes) is not int or budget_bytes < required:
        raise ArtifactResourceRequirementError(required, budget_bytes)
    directory = _artifact_directory(root, run_id, ordinal, generation)
    path = directory / "manifest.json"
    limit = min(16 * 1024 * 1024, budget_bytes // 4)
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ArtifactResourceRequirementError(len(payload) * 4, budget_bytes)
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("chunks"), list):
        raise ValueError("invalid reconciliation manifest")
    chunks = manifest["chunks"]
    if any(not isinstance(c, dict) or type(c.get("bytes")) is not int or c["bytes"] < 0
           for c in chunks):
        raise ValueError("invalid reconciliation chunk sizes")
    receipt = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
               "generation": generation, "committed": True,
               "rows": manifest.get("rows"), "bytes": sum(c["bytes"] for c in chunks)}
    # Includes duplicate-key rejection, identity, DQ, hashes, exact axes/values,
    # chunk bounds and full coverage. The synthetic receipt is not the oracle.
    verified = verify_factor_artifact_receipt(
        receipt, root, run_id, ordinal, name, value, policy=policy, budget_bytes=budget_bytes)
    # A writer may have exited between rename and directory fsync. Re-read
    # equality is not durability proof: flush only these verified existing
    # files and their directory chain, without changing or creating content.
    for filename in [c["path"] for c in chunks] + ["manifest.json"]:
        with (directory / filename).open("rb") as stream:
            os.fsync(stream.fileno())
    for parent in (directory, directory.parent, directory.parent.parent, Path(root)):
        _fsync_directory(parent)
    return {**verified, "reconciled": True}


def write_verified_factor_artifact(root, run_id, ordinal, name, value, *, policy, budget_bytes,
                                   generation=None):
    if not isinstance(policy, DefaultExecutionPolicy):
        raise TypeError("a validated DefaultExecutionPolicy is required")
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ArtifactResourceRequirementError(1, budget_bytes)
    required = required_writer_workspace_bytes(value)
    if required > budget_bytes:
        raise ArtifactResourceRequirementError(required, budget_bytes)
    if (not isinstance(value.index, pd.MultiIndex) or value.index.nlevels != 2 or
            not value.index.is_unique):
        raise ValueError("factor output requires a unique two-axis time/instrument index")
    if not pd.api.types.is_numeric_dtype(value.dtype):
        raise ValueError("factor output dtype must be numeric")
    # Internal paths are generated, never built from a factor-supplied name.
    if generation is None:
        generation = uuid.uuid4().hex
    directory = _artifact_directory(root, run_id, ordinal, generation)
    started = time.monotonic()
    dq = assert_factor_dq(value, raise_on_fail=True)
    directory.mkdir(parents=True, exist_ok=False)
    chunks = []
    chunk_target = min(policy.target_file_bytes, policy.max_file_bytes, budget_bytes // 8)
    estimated_row_bytes = max(1, int(value.memory_usage(index=True, deep=True)) // max(1, len(value)))
    rows_per_chunk = max(1, chunk_target // max(1, estimated_row_bytes * 2))
    rows_verified = 0
    write_seconds = verify_seconds = 0.0
    for offset in range(0, len(value), rows_per_chunk):
        if time.monotonic() - started >= policy.sink_flush_seconds:
            raise TimeoutError("factor artifact write deadline exceeded; staging retained")
        expected = value.iloc[offset:offset + rows_per_chunk]
        convert_started = time.monotonic()
        frame = expected.to_frame(name="factor_value")
        table = pa.Table.from_pandas(frame, preserve_index=True, safe=True)
        to_arrow_seconds = time.monotonic() - convert_started
        arrow_bytes = table.nbytes
        if table.nbytes * 4 > budget_bytes:
            raise ArtifactResourceRequirementError(table.nbytes * 4, budget_bytes)
        path = directory / f"chunk-{len(chunks):06d}.parquet"
        partial = directory / (path.name + ".partial")
        write_start = time.monotonic()
        result = BatchParquetWriter.write_pa_table(
            table, partial, compression="ZSTD-1",
            compress_row_group_bytes=max(1, min(table.nbytes or 1, chunk_target)),
        )
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, path)
        _fsync_directory(directory)
        write_seconds += time.monotonic() - write_start
        del table, frame
        verify_start = time.monotonic()
        readback_table = pq.ParquetFile(path).read()
        readback_bytes = readback_table.nbytes
        convert_started = time.monotonic()
        restored = readback_table.to_pandas()["factor_value"]
        to_pandas_seconds = time.monotonic() - convert_started
        pd.testing.assert_series_equal(restored, expected.rename("factor_value"),
                                       check_exact=True, check_names=True, check_dtype=True,
                                       check_index_type=True, check_categorical=True)
        # Equality alone treats signed zero as equal. Hash pandas cells with
        # axes as a second exact representation check, including zero sign.
        if not _cell_fingerprints_equal(expected, restored):
            raise ValueError("artifact cell/axis fingerprint differs after readback")
        digest = _file_digest(path, min(1024 * 1024, max(1, budget_bytes // 16)))
        rows_verified += len(restored)
        chunks.append({"path": path.name, "sha256": digest, "rows": len(restored),
                       "offset": offset, "bytes": path.stat().st_size,
                       "conversions": [
                           {"from": "pandas_series", "to": "arrow_table",
                            "rows": len(expected), "logical_arrow_bytes": arrow_bytes,
                            "seconds": to_arrow_seconds, "copied_bytes": None},
                           {"from": "arrow_table", "to": "pandas_series",
                            "rows": len(restored), "logical_arrow_bytes": readback_bytes,
                            "seconds": to_pandas_seconds, "copied_bytes": None},
                       ]})
        verify_seconds += time.monotonic() - verify_start
        del restored, readback_table
    if rows_verified != len(value):
        raise ValueError("artifact does not cover all requested result rows")
    manifest = {
        "schema_version": "factor_engine.factor_artifact.v2", "run_id": run_id,
        "ordinal": ordinal, "factor_id": name, "generation": generation,
        "policy_digest": policy.digest, "dtype": str(value.dtype),
        "rows": rows_verified, "chunks": chunks, "dq": dq.to_dict(),
        "committed": True, "verified": True, "production_published": False,
        "performance": {"write_seconds": write_seconds, "verify_seconds": verify_seconds,
                        "total_seconds": time.monotonic() - started,
                        "workspace_estimate_bytes": required, "workspace_budget_bytes": budget_bytes,
                        "estimate_basis": "STATIC_ESTIMATE", "codec": "zstd_level_1"},
    }
    manifest_path = directory / "manifest.json"
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with (directory / "manifest.partial").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(directory / "manifest.partial", manifest_path)
    _fsync_directory(directory)
    # Persist every newly created directory entry up to the pre-approved root.
    # Fsyncing only `values/` does not make a newly created run directory durable.
    for parent in (directory.parent, directory.parent.parent, Path(root)):
        _fsync_directory(parent)
    return {"path": str(manifest_path), "sha256": hashlib.sha256(payload).hexdigest(),
            "generation": generation, "committed": True, "verified": True,
            "rows": rows_verified, "bytes": sum(item["bytes"] for item in chunks),
            "production_published": False, "performance": manifest["performance"]}
