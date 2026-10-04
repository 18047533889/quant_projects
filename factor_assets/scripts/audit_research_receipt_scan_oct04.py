"""Paired research-only audit of scalar and batched receipt embedding scans."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np

from factor_assets.clustering.incremental_recall import (
    scan_exact_member_winner,
    scan_exact_member_winners,
)
from factor_assets.contracts.research_feature_receipt import ResearchFeatureReceipt
from factor_assets.similarity.unit_vectors import _unit_rows

_AUDIT = "research-receipt-scan-oct04-v1"
_MAX_RECEIPTS = 16
_MAX_BLOCKS = 10
_MAX_CHUNK_ROWS = 256
_MAX_CHUNK_BYTES = 4 * 1024**2
_MAX_BATCH_QUERIES = 16
_SOURCE_PATHS = (
    "factor_assets/scripts/audit_research_receipt_scan_oct04.py",
    "factor_assets/clustering/incremental_recall.py",
    "factor_assets/similarity/unit_vectors.py",
    "factor_assets/contracts/research_feature_receipt.py",
    "factor_assets/contracts/_canonical.py",
)


@dataclass(frozen=True)
class _ResearchMember:
    """Minimal adapter for the recall helper; not a production fingerprint."""

    factor_id: str
    embedding: tuple[float, ...]


def _source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return {
        path: hashlib.sha256((project_root / path).read_bytes()).hexdigest()
        for path in _SOURCE_PATHS
    }


def _receipt_identity(receipts: Sequence[ResearchFeatureReceipt]) -> tuple:
    rows = []
    for receipt in receipts:
        if type(receipt) is not ResearchFeatureReceipt:
            raise TypeError("receipts must be exact ResearchFeatureReceipt objects")
        try:
            validated = replace(receipt)
        except (TypeError, ValueError) as exc:
            raise ValueError("receipt failed fresh contract validation") from exc
        if validated != receipt:
            raise ValueError("receipt content identity is inconsistent with its fields")
        binding = receipt.source_binding
        rows.append((
            binding.factor_id, receipt.content_hash, binding.manifest_uri,
            binding.manifest_sha256, binding.source_uri, binding.source_sha256,
            binding.time_axis_hash, binding.asset_axis_hash, binding.values_hash,
            binding.validity_hash,
        ))
    return tuple(rows)


def _integrity_snapshot(receipts: Sequence[ResearchFeatureReceipt]) -> tuple:
    return tuple(sorted(_source_hashes().items())), _receipt_identity(receipts)


def _guarded_call(receipts, baseline, function):
    try:
        before = _integrity_snapshot(receipts)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("audit source or receipt identity drift detected before scan") from exc
    if before != baseline:
        raise RuntimeError("audit source or receipt identity drift detected before scan")
    try:
        result = function()
    finally:
        try:
            after = _integrity_snapshot(receipts)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("audit source or receipt identity drift detected during scan") from exc
        if after != before or after != baseline:
            raise RuntimeError("audit source or receipt identity drift detected during scan")
    return result


def _validate_receipts(receipts) -> tuple[tuple[ResearchFeatureReceipt, ...], list[str]]:
    # Subclasses may override length/iteration and defeat bounded admission.
    if type(receipts) not in (tuple, list):
        raise TypeError("receipts must be an exact tuple or list")
    if not 2 <= len(receipts) <= _MAX_RECEIPTS:
        raise ValueError("audit requires 2..16 receipts")
    receipts = tuple(receipts)
    _receipt_identity(receipts)
    ids = []
    first = receipts[0]
    if type(first) is not ResearchFeatureReceipt:
        raise TypeError("receipts must be exact ResearchFeatureReceipt objects")
    first_binding = first.source_binding
    common = (
        first_binding.manifest_uri, first_binding.manifest_sha256,
        first_binding.time_axis_hash, first_binding.asset_axis_hash,
        first.shape, first.time_axis_name, first.time_axis_dtype,
        first.asset_axis_name, first.asset_axis_dtype,
        first.embedding_spec, first.embedding_model_version,
    )
    if first.scope != "research_only" or first.source_binding_status != "caller_supplied":
        raise ValueError("audit accepts only caller-supplied research receipts")
    if len(first.shape) != 3 or first.shape[-1] != 1:
        raise ValueError("receipt shape must be T×N×1")
    width = len(first.embedding)
    if not 1 <= width <= 4096:
        raise ValueError("receipt embedding width must be between 1 and 4096")

    for receipt in receipts:
        if type(receipt) is not ResearchFeatureReceipt:
            raise TypeError("receipts must be exact ResearchFeatureReceipt objects")
        if receipt.scope != "research_only" or receipt.source_binding_status != "caller_supplied":
            raise ValueError("audit accepts only caller-supplied research receipts")
        binding = receipt.source_binding
        factor_id = binding.factor_id
        if not factor_id or factor_id != factor_id.strip() or factor_id in ids:
            raise ValueError("receipt factor IDs must be unique nonempty trimmed text")
        ids.append(factor_id)
        current = (
            binding.manifest_uri, binding.manifest_sha256,
            binding.time_axis_hash, binding.asset_axis_hash,
            receipt.shape, receipt.time_axis_name, receipt.time_axis_dtype,
            receipt.asset_axis_name, receipt.asset_axis_dtype,
            receipt.embedding_spec, receipt.embedding_model_version,
        )
        if current != common:
            raise ValueError("receipts must share manifest, axes, shape, and embedding spec/version")
        if len(receipt.embedding) != width:
            raise ValueError("receipt embedding widths must match")
        embedding = np.asarray(receipt.embedding, dtype=np.float64)
        if not np.isfinite(embedding).all():
            raise ValueError("receipt embeddings must be finite")
        if not np.any(embedding != 0.0):
            raise ValueError("receipt embeddings must be nonzero for cosine scanning")
    return receipts, ids


def _check_options(paired_blocks, seed, max_chunk_rows, max_chunk_bytes,
                   max_batch_queries, query_count) -> None:
    if type(paired_blocks) is not int or not 3 <= paired_blocks <= _MAX_BLOCKS:
        raise ValueError("paired_blocks must be an integer in 3..10")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 0..2**32-1")
    if type(max_chunk_rows) is not int or not 1 <= max_chunk_rows <= _MAX_CHUNK_ROWS:
        raise ValueError("max_chunk_rows must be an integer in 1..256")
    if type(max_chunk_bytes) is not int or not 1 <= max_chunk_bytes <= _MAX_CHUNK_BYTES:
        raise ValueError("max_chunk_bytes must be an integer in 1..4 MiB")
    if type(max_batch_queries) is not int or not 1 <= max_batch_queries <= _MAX_BATCH_QUERIES:
        raise ValueError("max_batch_queries must be an integer in 1..16")
    if query_count > max_batch_queries:
        raise ValueError("query count exceeds max_batch_queries")


def _score_scalar(receipt_map, member_ids, queries, *, max_chunk_rows, max_chunk_bytes):
    return [
        scan_exact_member_winner(
            receipt_map, member_ids, query,
            to_embedding=lambda row: np.asarray(row.embedding, dtype=np.float64),
            normalize_rows=_unit_rows,
            max_chunk_rows=max_chunk_rows,
            max_chunk_bytes=max_chunk_bytes,
        )
        for query in queries
    ]


def _score_batch(receipt_map, member_ids, queries, *, max_chunk_rows,
                 max_chunk_bytes, max_batch_queries):
    return scan_exact_member_winners(
        receipt_map, member_ids, queries,
        to_embedding=lambda row: np.asarray(row.embedding, dtype=np.float64),
        normalize_rows=_unit_rows,
        max_chunk_rows=max_chunk_rows,
        max_chunk_bytes=max_chunk_bytes,
        max_batch_queries=max_batch_queries,
    )


def _assert_same_winners(left, right) -> None:
    if len(left) != len(right):
        raise RuntimeError("scalar and batch scan result counts differ")
    for index, (scalar, batch) in enumerate(zip(left, right, strict=True)):
        if scalar is None or batch is None:
            if scalar is not batch:
                raise RuntimeError(f"scalar and batch winner presence differs at query {index}")
            continue
        if scalar[0] != batch[0] or float(scalar[1]) != float(batch[1]):
            raise RuntimeError(f"scalar and batch exact winner/score differs at query {index}")
        if not math.isfinite(float(scalar[1])) or not math.isfinite(float(batch[1])):
            raise RuntimeError(f"scan score is non-finite at query {index}")


def run_receipt_scan_audit(
    receipts,
    *,
    paired_blocks: int = 3,
    seed: int = 20261004,
    max_chunk_rows: int = 4,
    max_chunk_bytes: int = 1024 * 1024,
    max_batch_queries: int = 16,
) -> dict:
    """Compare scalar and batched exact-recall scans on supplied research vectors.

    This function neither loads factors nor invokes an embedding model. Receipts
    and their caller-supplied embeddings are inputs; results are research-only,
    not production fingerprints, admission decisions, or speed qualifications.
    """
    receipts, receipt_ids = _validate_receipts(receipts)
    member_count = (len(receipts) + 1) // 2
    members = receipts[:member_count]
    queries_receipts = receipts[member_count:]
    member_ids = [item.source_binding.factor_id for item in members]
    query_ids = [item.source_binding.factor_id for item in queries_receipts]
    _check_options(paired_blocks, seed, max_chunk_rows, max_chunk_bytes,
                   max_batch_queries, len(query_ids))

    raw_queries = np.asarray([item.embedding for item in queries_receipts], dtype=np.float64)
    unit_queries = np.ascontiguousarray(_unit_rows(raw_queries), dtype=np.float64)
    receipt_map = {
        item.source_binding.factor_id: _ResearchMember(
            item.source_binding.factor_id, item.embedding,
        )
        for item in members
    }
    baseline = _integrity_snapshot(receipts)

    def run_mode(mode):
        if mode == "scalar":
            return _guarded_call(receipts, baseline, lambda: _score_scalar(
                receipt_map, member_ids, unit_queries,
                max_chunk_rows=max_chunk_rows, max_chunk_bytes=max_chunk_bytes,
            ))
        return _guarded_call(receipts, baseline, lambda: _score_batch(
            receipt_map, member_ids, unit_queries,
            max_chunk_rows=max_chunk_rows, max_chunk_bytes=max_chunk_bytes,
            max_batch_queries=max_batch_queries,
        ))

    # Warm both code paths twice, outside reported timings.
    for _ in range(2):
        warm_scalar = run_mode("scalar")
        warm_batch = run_mode("batch")
        _assert_same_winners(warm_scalar, warm_batch)

    rng = random.Random(seed)
    paired = []
    expected = None
    for _ in range(paired_blocks):
        order = ["scalar", "batch"]
        rng.shuffle(order)
        seconds = {}
        block_results = {}
        for mode in order:
            started = time.perf_counter()
            block_results[mode] = run_mode(mode)
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("scan timing is invalid")
            seconds[mode] = elapsed
        _assert_same_winners(block_results["scalar"], block_results["batch"])
        if expected is None:
            expected = block_results["batch"]
        else:
            _assert_same_winners(expected, block_results["batch"])
        paired.append({"order": order, "seconds": seconds})

    assert expected is not None
    winner_ids = [None if result is None else result[0] for result in expected]
    scores = [None if result is None else float(result[1]) for result in expected]
    return {
        "audit": _AUDIT,
        "scope": "research_only",
        "source_binding_status": "caller_supplied",
        "timing_scope": (
            "per-call wall time includes scoped source/receipt integrity checks around "
            "the scanner; excludes source loading and embedding generation"
        ),
        "receipt_ids": receipt_ids,
        "receipt_content_hashes": [item.content_hash for item in receipts],
        "member_ids": member_ids,
        "query_ids": query_ids,
        "winner_ids": winner_ids,
        "scores": scores,
        "paired_blocks": paired,
        "scan_config": {
            "seed": seed,
            "paired_blocks_count": paired_blocks,
            "warmup_pairs": 2,
            "max_chunk_rows": max_chunk_rows,
            "max_chunk_bytes": max_chunk_bytes,
            "max_batch_queries": max_batch_queries,
            "embedding_width": len(receipts[0].embedding),
        },
        "shared_identity": {
            "manifest_uri": receipts[0].source_binding.manifest_uri,
            "manifest_sha256": receipts[0].source_binding.manifest_sha256,
            "time_axis_hash": receipts[0].source_binding.time_axis_hash,
            "asset_axis_hash": receipts[0].source_binding.asset_axis_hash,
            "shape": list(receipts[0].shape),
            "embedding_spec": receipts[0].embedding_spec,
            "embedding_model_version": receipts[0].embedding_model_version,
        },
        "memory_scope": "helper_chunk_arrays_only_not_total_rss",
        "source_hashes": _source_hashes(),
    }
