"""Production bridge from preprocessed factor evidence to ANN fingerprints."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from factor_assets.contracts._canonical import canonical_digest
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact


class FingerprintEmbeddingModel(Protocol):
    """Embedding authority supplied by the caller; FA does not train a model."""

    version: str

    def embed(self, values: np.ndarray, validity_mask: np.ndarray) -> Sequence[float]: ...


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def build_similarity_fingerprint(
    feature_bundle: Any,
    profile: Any,
    *,
    embedding_model: FingerprintEmbeddingModel,
    embedding_spec: str,
    mask_policy: str,
    direction: str,
    aggregation_method: str,
    embedding_model_version: str,
    window: str,
) -> SimilarityFingerprintArtifact:
    """Build one production fingerprint from governed FP values and profile evidence.

    The supplied embedding model is the only vector producer. This adapter binds
    its result to the bundle/profile evidence and the complete SIM-01 spec.
    """
    required_bundle = ("bundle_id", "get_primary_values", "source_factor_ids", "config_hash",
                       "policy_id", "fitted_state_refs", "missing_reason_plane")
    missing = [name for name in required_bundle if not hasattr(feature_bundle, name)]
    if missing:
        raise TypeError(f"feature_bundle lacks required preprocessing evidence: {missing}")
    required_profile = ("factor_id", "content_hash", "snapshot_ref", "universe_ref")
    missing = [name for name in required_profile if not hasattr(profile, name)]
    if missing:
        raise TypeError(f"profile lacks required profile evidence: {missing}")
    factor_id = _required_text(profile.factor_id, "profile.factor_id")
    source_ids = tuple(feature_bundle.source_factor_ids)
    if source_ids != (factor_id,):
        raise ValueError("production fingerprint requires a single-factor bundle matching profile.factor_id")
    config_hash = _required_text(feature_bundle.config_hash, "feature_bundle.config_hash")
    bundle_id = _required_text(feature_bundle.bundle_id, "feature_bundle.bundle_id")
    profile_ref = _required_text(profile.content_hash, "profile.content_hash")
    snapshot = _required_text(profile.snapshot_ref, "profile.snapshot_ref")
    universe = _required_text(profile.universe_ref, "profile.universe_ref")
    for name, expected in (("snapshot_ref", snapshot), ("universe_ref", universe)):
        supplied = getattr(feature_bundle, name, None)
        if supplied not in (None, "") and supplied != expected:
            raise ValueError(f"feature_bundle.{name} must match profile.{name}")
    for value, name in (
        (embedding_spec, "embedding_spec"), (mask_policy, "mask_policy"),
        (direction, "direction"), (aggregation_method, "aggregation_method"),
        (embedding_model_version, "embedding_model_version"), (window, "window"),
    ):
        _required_text(value, name)
    if direction not in {"signed", "absolute"}:
        raise ValueError("direction must be 'signed' or 'absolute'")
    if getattr(embedding_model, "version", None) != embedding_model_version:
        raise ValueError("embedding_model.version must match embedding_model_version")

    values = np.asarray(feature_bundle.get_primary_values(), dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError("single-factor primary values must have shape T×N")
    plane = feature_bundle.missing_reason_plane
    if plane is None or not hasattr(plane, "usable"):
        raise ValueError("production fingerprint requires an authoritative usable mask")
    usable = np.asarray(plane.usable, dtype=bool)
    if usable.ndim == 3 and usable.shape[-1] == 1:
        usable = usable[..., 0]
    if usable.shape != values.shape:
        raise ValueError("authoritative usable mask must align with primary values")
    validity = usable & np.isfinite(values)
    if not validity.any():
        raise ValueError("fingerprint input has no finite usable observations")
    embedding = tuple(float(v) for v in embedding_model.embed(values.copy(), validity.copy()))
    if not embedding:
        raise ValueError("embedding model returned an empty vector")

    preprocessing_ref = canonical_digest(
        bundle_id, config_hash, feature_bundle.policy_id,
        tuple(feature_bundle.fitted_state_refs), mask_policy,
    )
    try:
        from factor_engine.runtime.factor_value_identity import compute_block_content_hash
    except ImportError as exc:  # pragma: no cover - composed production dependency
        raise RuntimeError("factor-engine value identity authority is required") from exc
    value_content_hash = compute_block_content_hash(
        values, factor_ids=source_ids, numeric_policy=str(values.dtype),
    )
    usable_content_hash = compute_block_content_hash(
        usable.astype(np.float32), numeric_policy="authoritative-usable-mask",
    )
    time_axis = tuple(np.asarray(feature_bundle.time_axis.axis_values).tolist())
    asset_axis = tuple(np.asarray(feature_bundle.asset_axis.axis_values).tolist())
    value_ref = canonical_digest(
        value_content_hash, usable_content_hash, time_axis, asset_axis,
        snapshot, universe, config_hash,
    )
    return SimilarityFingerprintArtifact(
        factor_id=factor_id,
        embedding=embedding,
        embedding_spec=embedding_spec,
        snapshot=snapshot,
        universe=universe,
        window=window,
        preprocessing_ref=preprocessing_ref,
        mask_policy=mask_policy,
        direction=direction,
        aggregation_method=aggregation_method,
        embedding_model_version=embedding_model_version,
        value_ref=value_ref,
        profile_ref=profile_ref,
    )


def assign_preprocessed_fingerprint(
    feature_bundle: Any,
    profile: Any,
    *,
    embedding_model: FingerprintEmbeddingModel,
    cluster_versions: Mapping[str, Any],
    fingerprints_by_id: Mapping[str, SimilarityFingerprintArtifact],
    embedding_spec: str,
    mask_policy: str,
    direction: str,
    aggregation_method: str,
    embedding_model_version: str,
    window: str,
    **assignment_options: Any,
) -> Any:
    """Create a governed fingerprint and feed it to incremental assignment."""
    fingerprint = build_similarity_fingerprint(
        feature_bundle, profile, embedding_model=embedding_model,
        embedding_spec=embedding_spec, mask_policy=mask_policy, direction=direction,
        aggregation_method=aggregation_method,
        embedding_model_version=embedding_model_version, window=window,
    )
    from factor_assets.clustering.incremental import incremental_assign

    fingerprint_map = dict(fingerprints_by_id)
    fingerprint_map[fingerprint.factor_id] = fingerprint
    return incremental_assign(
        [fingerprint], cluster_versions, fingerprint_map, **assignment_options,
    )


__all__ = [
    "FingerprintEmbeddingModel",
    "build_similarity_fingerprint",
    "assign_preprocessed_fingerprint",
]
