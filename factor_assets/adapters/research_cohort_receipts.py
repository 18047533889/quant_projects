"""Sequential, bounded bridge from a strict cohort to research receipts."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from itertools import islice
from numbers import Real
from typing import Any

import numpy as np

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_URI = re.compile(r"cos://[^\s]+/([0-9a-f]{64})/landing_manifest\.json\Z")
_DEFAULT_MAX_SLICE_BYTES = 128 * 1024**2


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty trimmed text")
    return value


def _embedding(value: Any, factor_id: str) -> tuple[float, ...]:
    try:
        values = tuple(islice(iter(value), 4097))
    except TypeError as exc:
        raise ValueError(f"embedding for {factor_id} must be a finite numeric sequence") from exc
    if not values or len(values) > 4096:
        raise ValueError(f"embedding for {factor_id} must contain 1..4096 values")
    result: list[float] = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (Real, np.floating)):
            raise ValueError(f"embedding for {factor_id} must contain real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"embedding for {factor_id} must be finite")
        if isinstance(item, np.floating) and np.asarray(item).dtype.itemsize > 8:
            if np.longdouble(number) != item:
                raise ValueError(f"embedding for {factor_id} loses precision as float64")
        result.append(number)
    return tuple(result)


def _validated_inputs(batch, provenance, embeddings, embedding_spec,
                      embedding_model_version, max_slice_bytes):
    from quant_evaluator.contracts.factor_batch import FactorBatch

    if type(batch) is not FactorBatch:
        raise TypeError("batch must be an exact QE FactorBatch")
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    factor_ids = batch.factor_ids
    if (not factor_ids or any(type(factor_id) is not str or not factor_id
                              or factor_id != factor_id.strip() for factor_id in factor_ids)
            or len(set(factor_ids)) != len(factor_ids)):
        raise ValueError("batch must contain unique nonempty trimmed factor IDs")
    if provenance.get("coverage_policy") != "strict":
        raise ValueError("cohort provenance must use strict coverage")
    if provenance.get("quarantined_factors") != []:
        raise ValueError("cohort provenance must have no quarantined factors")
    retained = provenance.get("retained_factor_ids")
    if not isinstance(retained, (list, tuple)) or tuple(retained) != tuple(factor_ids):
        raise ValueError("retained factor IDs must exactly match batch order")

    manifest_uri = _text("manifest_uri", provenance.get("manifest_uri"))
    match = _MANIFEST_URI.fullmatch(manifest_uri)
    if match is None:
        raise ValueError("manifest URI must be content-addressed COS landing_manifest.json")
    manifest_hash = match.group(1)
    rows = provenance.get("sources")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(factor_ids):
        raise ValueError("source rows must exactly match retained factors")
    checked_rows = []
    for factor_id, row in zip(factor_ids, rows):
        if not isinstance(row, Mapping) or row.get("factor") != factor_id:
            raise ValueError("source rows must match retained factor order")
        uri = _text("source URI", row.get("uri"))
        source_hash = row.get("sha256")
        if not isinstance(source_hash, str) or _SHA256.fullmatch(source_hash) is None:
            raise ValueError("source SHA-256 must be lowercase 64-character hex")
        if row.get("manifest_sha256") != manifest_hash:
            raise ValueError("source row manifest digest differs from content-addressed URI")
        downloaded_bytes = row.get("downloaded_bytes")
        if type(downloaded_bytes) is not int or downloaded_bytes <= 0:
            raise ValueError("downloaded_bytes must be a positive integer")
        checked_rows.append((factor_id, uri, source_hash))

    if not isinstance(embeddings, Mapping) or set(embeddings) != set(factor_ids):
        raise ValueError("embedding IDs must exactly match retained factor IDs")
    normalized_embeddings = {
        factor_id: _embedding(embeddings[factor_id], factor_id)
        for factor_id in factor_ids
    }
    embedding_spec = _text("embedding_spec", embedding_spec)
    embedding_model_version = _text("embedding_model_version", embedding_model_version)

    if type(max_slice_bytes) is not int or max_slice_bytes <= 0:
        raise ValueError("max_slice_bytes must be a positive integer")
    values = batch.values
    if values.dtype.kind not in {"i", "u", "f"} or values.dtype.fields is not None:
        raise TypeError("cohort values must have a real numeric dtype")
    if values.ndim != 3 or values.shape[2] != len(factor_ids) or not all(values.shape):
        raise ValueError("cohort values must be nonempty T×N×factor array")
    validity = batch.validity
    if validity is not None and validity.shape != values.shape:
        raise ValueError("validity shape must match cohort values")
    one_slice_bytes = values.shape[0] * values.shape[1] * values.dtype.itemsize
    if validity is not None:
        one_slice_bytes += validity.shape[0] * validity.shape[1] * validity.dtype.itemsize
    if one_slice_bytes > max_slice_bytes:
        raise ValueError(
            f"one-factor slice requires {one_slice_bytes} bytes, exceeding max_slice_bytes={max_slice_bytes}"
        )
    return manifest_uri, manifest_hash, checked_rows, normalized_embeddings, embedding_spec, embedding_model_version


def _build_one(batch, factor_index: int, factor_id: str, source_row,
               manifest_uri: str, manifest_hash: str, embedding,
               embedding_spec: str, embedding_model_version: str):
    """Construct and discard one child batch while returning metadata only."""
    from quant_evaluator.contracts.array_identity import authoritative_array_hash
    from quant_evaluator.contracts.factor_batch import FactorBatch
    from factor_assets.contracts.research_feature_receipt import ResearchSourceBinding
    from factor_assets.adapters.research_feature_receipt import build_research_feature_receipt

    _, source_uri, source_hash = source_row
    child = FactorBatch(
        (factor_id,), batch.time_axis, batch.asset_axis,
        batch.values[:, :, factor_index:factor_index + 1],
        validity=(None if batch.validity is None
                  else batch.validity[:, :, factor_index:factor_index + 1]),
    )
    binding = ResearchSourceBinding(
        factor_id=factor_id,
        manifest_uri=manifest_uri,
        manifest_sha256=manifest_hash,
        source_uri=source_uri,
        source_sha256=source_hash,
        time_axis_hash=authoritative_array_hash(child.time_axis.values),
        asset_axis_hash=authoritative_array_hash(child.asset_axis.values),
        values_hash=authoritative_array_hash(child.values),
        validity_hash=(None if child.validity is None
                       else authoritative_array_hash(child.validity)),
    )
    return build_research_feature_receipt(
        child,
        source_binding=binding,
        embedding=embedding,
        embedding_spec=embedding_spec,
        embedding_model_version=embedding_model_version,
    )


def build_cohort_research_feature_receipts(
    batch,
    provenance,
    *,
    embeddings,
    embedding_spec: str,
    embedding_model_version: str,
    max_slice_bytes: int = _DEFAULT_MAX_SLICE_BYTES,
) -> tuple:
    """Return one payload-free receipt per strictly retained factor, in order.

    Admission accounts for the values and optional validity bytes in one
    factor slice before any child FactorBatch is frozen. Coordinate/hash scratch
    is separate; no full cohort cube or array-bearing result is retained.
    """
    (manifest_uri, manifest_hash, rows, normalized_embeddings,
     embedding_spec, embedding_model_version) = _validated_inputs(
        batch, provenance, embeddings, embedding_spec,
        embedding_model_version, max_slice_bytes,
    )
    return tuple(
        _build_one(batch, index, factor_id, row, manifest_uri, manifest_hash,
                   normalized_embeddings[factor_id], embedding_spec,
                   embedding_model_version)
        for index, (factor_id, row) in enumerate(zip(batch.factor_ids, rows))
    )
