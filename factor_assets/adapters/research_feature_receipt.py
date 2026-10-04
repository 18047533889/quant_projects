"""Build research-only receipts from exact single-factor QE arrays.

This adapter performs no I/O. Source object and manifest identities are caller
declarations, not authenticated download receipts or governed snapshot refs.
"""
from __future__ import annotations

from factor_assets.contracts.research_feature_receipt import (
    ResearchFeatureReceipt,
    ResearchSourceBinding,
)


def build_research_feature_receipt(
    factor_batch,
    *,
    source_binding: ResearchSourceBinding,
    embedding,
    embedding_spec: str,
    embedding_model_version: str,
) -> ResearchFeatureReceipt:
    """Bind a supplied embedding and source declaration to one exact factor.

    No embedding model is run here. A matching array identity proves only that
    the declared in-memory slice matches; it does not authenticate source files.
    Missing validity remains absent, never an authoritative missing-reason mask.
    """
    from quant_evaluator.contracts.array_identity import authoritative_array_hash
    from quant_evaluator.contracts.factor_batch import FactorBatch

    if type(factor_batch) is not FactorBatch:
        raise TypeError("factor_batch must be an exact QE FactorBatch")
    if type(source_binding) is not ResearchSourceBinding:
        raise TypeError("source_binding must be a ResearchSourceBinding")
    if len(factor_batch.factor_ids) != 1:
        raise ValueError("research receipt requires exactly one factor")
    if factor_batch.factor_ids[0] != source_binding.factor_id:
        raise ValueError("source binding factor_id differs from factor_batch")
    values = factor_batch.values
    if values.dtype.kind not in {"i", "u", "f"} or values.dtype.fields is not None:
        raise TypeError("research factor values require a real numeric array")
    if values.ndim != 3 or values.shape[-1] != 1 or not all(values.shape):
        raise ValueError("research factor values require nonempty T x N x 1 shape")
    axes = (factor_batch.time_axis, factor_batch.asset_axis)
    coordinates = tuple(axis.values for axis in axes)
    if any(array is None for array in coordinates):
        raise ValueError("research receipt requires explicit time and asset coordinates")
    expected_shape = (len(coordinates[0]), len(coordinates[1]), 1)
    if values.shape != expected_shape:
        raise ValueError("research coordinates must align with factor values")
    actual_hashes = {
        "time_axis_hash": authoritative_array_hash(coordinates[0]),
        "asset_axis_hash": authoritative_array_hash(coordinates[1]),
        "values_hash": authoritative_array_hash(values),
        "validity_hash": (
            None if factor_batch.validity is None
            else authoritative_array_hash(factor_batch.validity)
        ),
    }
    for field, actual in actual_hashes.items():
        if actual != getattr(source_binding, field):
            raise ValueError(f"source binding {field} differs from factor_batch")
    return ResearchFeatureReceipt(
        source_binding=source_binding,
        values_dtype=str(values.dtype),
        validity_dtype=(
            None if factor_batch.validity is None
            else str(factor_batch.validity.dtype)
        ),
        shape=tuple(int(size) for size in values.shape),
        time_axis_name=axes[0].name,
        time_axis_dtype=str(coordinates[0].dtype),
        asset_axis_name=axes[1].name,
        asset_axis_dtype=str(coordinates[1].dtype),
        embedding=embedding,
        embedding_spec=embedding_spec,
        embedding_model_version=embedding_model_version,
    )
