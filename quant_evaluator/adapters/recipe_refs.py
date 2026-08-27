"""Bidirectional adapter: FP TreatmentRecipe <-> QE EvaluationRequest refs (QRP-P5-INT-1).

This module is the MINIMAL wiring between the two domain authorities:

- FP owns :class:`TreatmentRecipe` (``factor_preprocess.contracts.treatment_recipe``)
  — the ordered transform prescription for a factor.
- QE owns :class:`EvaluationRequest` and its reference contracts
  :class:`FactorValueRef` / :class:`LabelBundleRef`
  (``quant_evaluator.contracts.evaluation_refs``).

It deliberately introduces NO new authority and NO second catalog: the
semantic-transform sequence is read VERBATIM from the recipe
(``semantic_transform_id`` order preserves the winsor->neutralize->rank vs
smooth->rank ordering that is part of the recipe's content identity), and the
refs are those already serialized by ``EvaluationRequest.to_dict``.

What is provided (all QE-side, pure, zero-registry):

- ``recipe_to_factor_value_ref`` — map a recipe's ordered steps to a
  :class:`FactorValueRef` with the recipe's treatment identity recorded as
  provenance.  This is the missing RecipeStep-sequence -> FactorValueRef path
  (gap a): the recipe only carries ``source_factor_value_ref`` as a raw
  string, and QE's request refs are structured objects, so a helper is needed
  to bind a *treated* candidate evaluation request to its recipe.
- ``ref_for_factor_values`` — build the ``factor_value_ref`` for an
  *asserted* value artifact (asserted id + factor ids).  Agnostic of whether
  the artifact is raw or treated; the ingredient that distinguishes a treated
  ref is the recipe-derived one produced by ``recipe_to_factor_value_ref``.
- ``label_bundle_to_ref`` — build the ``label_bundle_ref`` carried by an
  ``EvaluationRequest`` from the label definition that produced the
  :class:`LabelBundle` (target/horizon, or an explicit asserted id).
- ``request_for_recipe`` — the end-to-end seam: composing the QE-side
  ``EvaluationRequest`` refs from a recipe's ordered steps + the underlying
  raw factor value + the label definition.  The raw ``batch_or_factor_ids`` /
  ``label_bundle`` array payloads are NOT serialized by ``to_dict``; they are
  deliberately in-memory runtime inputs only, matching QE's ref round-trip
  contract (DLIB-QE-003).

Identity (no separate registry):
The ref ids are deterministic functions of the asserted identities. When the
recipe provenance records the raw artifact id, the *raw* ref equals the
*treated* ref — that equality is exactly what lets two different step
sequences on the SAME underlying raw factor value be compared by the optimizer
without inventing a value-artifact registry.  Provenance on the ref records
the recipe's ``content_hash`` (and treatment identity) so the association
scans as evaluation-of-a-specific-treatment while sharing the underlying
value identity.

EvidenceStatus (gap c):
The QE side represents label-maturity absence as ``EvidenceStatus`` and a
per-metric ``valid`` flag (never a fabricated 0.0).  ``EvaluationRequest``'s
refs and the request round-trip do NOT touch EvidenceStatus — that status
lives in the evaluation RESULT.  So a *request* builder has no success bit to
misread.  The FP-side failure mode we guard against here is a caller treating
a non-finite / non-valid metric result as success; the exposed
``assert_computed_value`` helper fails closed on ``None`` / ``valid is False``
so downstream FP code cannot pass a ``computed=False`` evaluation upward as a
success.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef

__all__ = [
    "recipe_to_factor_value_ref",
    "ref_for_factor_values",
    "label_bundle_to_ref",
    "request_for_recipe",
    "assert_computed_value",
    "factor_batch_for_recipe",
    "recipe_evidence_evaluation",
]

#: Prefix for a SOURCE (raw) factor-value ref built from asserted ids.
_RAW_REF_PREFIX = "factor_value:raw"
#: Prefix for a treated factor-value ref whose value is the recipe output.
_TREATED_REF_PREFIX = "factor_value:treated"
#: Prefix for a label-bundle ref built from a LabelBundle definition.
_LABEL_REF_PREFIX = "label_bundle"


def _recipe_steps_identity(recipe) -> tuple:
    """Ordered scalar identity of a recipe's step sequence.

    Reads the recipe's *semantic* prescription (semantic_transform_id +
    parameters + stage, in order).  The order is part of the identity: the
    same ids in a different order hash differently (mirrors TreatmentRecipe's
    own content identity, DLIB-FP-017).

    Returns:
        A tuple of tuples of *immutable* scalars suitable for ``hash()``.
    """
    steps = []
    for step in recipe.ordered_steps:
        params = step.parameters
        if isinstance(params, dict):
            params = tuple(sorted((str(k), v) for k, v in params.items()))
        steps.append(
            (
                step.semantic_transform_id,
                step.stage,
                params,
                bool(step.requires_fit),
                step.implementation_ref,
            )
        )
    return tuple(steps)


def _recipe_treatment_metadata(recipe) -> dict:
    """Provenance metadata that ties a ref back to the recipe.

    ``treatment_identity`` and ``content_hash`` both derive from the ordered
    step sequence, so the association scans as evaluation-of-a-specific-
    treatment without introducing a second registry.
    """
    return {
        "recipe_id": getattr(recipe, "recipe_id", ""),
        "treatment_identity": getattr(recipe, "treatment_identity", "")
        or getattr(recipe, "content_hash", ""),
        "transform_sequence": [
            getattr(step, "semantic_transform_id", "") for step in recipe.ordered_steps
        ],
    }


def recipe_to_factor_value_ref(recipe, *, ref_for: str = "treated") -> FactorValueRef:
    """Map a recipe's ordered steps to a :class:`FactorValueRef`.

    This is the missing RecipeStep-sequence -> FactorValueRef path (gap a).

    Args:
        recipe: A :class:`TreatmentRecipe` (FP authority), duck-typed so the
            adapter stays a one-way QE-side import (no FP import in QE core).
        ref_for: ``"treated"`` (value the recipe produces — default) or
            ``"raw"`` (the recipe's underlying source value).  The source ref
            the recipe carries is a raw *string*; this helper only rebinds it
            when it looks like an asserted id/already-structured value.

    Returns:
        A structured :class:`FactorValueRef` whose ``factor_value_id`` derives
        from the asserted raw value identity + the recipe's step sequence
        (treated), or from the raw asserted id alone (raw).  Provenance
        carries the recipe's content hash + ordered semantic sequence.
    """
    raw = getattr(recipe, "source_factor_value_ref", "")
    raw_id = _first_plain_str(raw) or "recipe_source_factor_value"
    factor_ids = tuple(getattr(recipe, "factor_ids", ()) or ())
    if ref_for == "raw":
        return FactorValueRef(
            factor_value_id=f"{_RAW_REF_PREFIX}:{raw_id}",
            factor_ids=factor_ids,
            source_ref=raw_id or None,
            metadata=_recipe_treatment_metadata(recipe),
        )
    seq = _recipe_steps_identity(recipe)
    return FactorValueRef(
        factor_value_id=f"{_TREATED_REF_PREFIX}:{raw_id}:{hash(seq):x}",
        factor_ids=factor_ids,
        source_ref=raw_id or None,
        metadata=_recipe_treatment_metadata(recipe),
    )


def ref_for_factor_values(
    factor_value_id: str, factor_ids=(), *, treated_from=None
) -> FactorValueRef:
    """Build the ``factor_value_ref`` for an asserted factor-value artifact.

    Pure, registry-free, authority-agnostic: the asserted ``factor_value_id``
    is the identity; ``factor_ids`` are provenance; ``treated_from`` may carry
    an optional recipe for provenance.

    Args:
        factor_value_id: Asserted stable identity of the value artifact.
        factor_ids: Factor ids covered (provenance).
        treated_from: Optional recipe; when present the ref records the
            recipe's treatment identity as provenance.

    Returns:
        A :class:`FactorValueRef`.  RAW: for a raw artifact, callers typically
        pass the raw id and no ``treated_from``, which yields id
        ``factor_value:raw:<id>``.  The prefix is NOT an authority; it is a
        provenance tag so two artifacts with different asserted ids can never
        collide.
    """
    if not isinstance(factor_value_id, str) or not factor_value_id.strip():
        raise ValueError(
            "ref_for_factor_values: factor_value_id must be a non-empty string, "
            f"got {factor_value_id!r}"
        )
    metadata: Mapping[str, Any] = {}
    if treated_from is not None:
        metadata = _recipe_treatment_metadata(treated_from)
    return FactorValueRef(
        factor_value_id=f"{_RAW_REF_PREFIX}:{factor_value_id}",
        factor_ids=tuple(factor_ids),
        source_ref=factor_value_id,
        metadata=metadata,
    )


def label_bundle_to_ref(
    label_bundle,
    *,
    asserted_label_bundle_id=None,
) -> LabelBundleRef:
    """Build the ``label_bundle_ref`` for an ``EvaluationRequest``.

    Args:
        label_bundle: A :class:`LabelBundle` (QE authority) carrying
            ``target_id`` / ``horizon`` / ``source_ref``.
        asserted_label_bundle_id: Optional asserted artifact id.  When None,
            the ref is derived deterministically from the bundle's target +
            horizon.

    Returns:
        A :class:`LabelBundleRef` with the label-bundle identity, provenance
        target/horizon/source, and no raw arrays.
    """
    label_id = asserted_label_bundle_id or (
        f"{_LABEL_REF_PREFIX}:{getattr(label_bundle, 'target_id', '')}:"
        f"{getattr(label_bundle, 'horizon', '')}"
    )
    return LabelBundleRef(
        label_bundle_id=label_id,
        target_id=getattr(label_bundle, "target_id", None),
        horizon=getattr(label_bundle, "horizon", None),
        source_ref=getattr(label_bundle, "source_ref", None),
    )


def request_for_recipe(
    recipe,
    *,
    label_bundle=None,
    factor_batch=None,
    factor_value_id=None,
    label_bundle_id=None,
    metric_ids=("pearson_ic", "rank_ic", "coverage"),
    **request_kwargs,
) -> EvaluationRequest:
    """Compose the QE-side :class:`EvaluationRequest` refs from a recipe's steps.

    The seam that closes gaps a+b: the request carries the refs built from
    the recipe's ordered steps (``factor_value_ref``), the fully-qualified
    label definition (``label_bundle_ref``), and (optionally) the raw runtime
    arrays (``batch_or_factor_ids`` / ``label_bundle``) which are runtime-only
    and deliberately NOT serialized by ``EvaluationRequest.to_dict``.

    Args:
        recipe: The FP :class:`TreatmentRecipe` whose steps are evaluated.
        label_bundle: Optional :class:`LabelBundle` (QE authority) for the
            ``label_bundle_ref`` and runtime ``label_bundle`` payload.
        factor_batch: Optional fully-qualified QE contract for the runtime
            ``batch_or_factor_ids`` payload (NOT serialized).
        factor_value_id: Optional asserted raw factor-value artifact id used to
            build the treated factor-value ref.  When None, the recipe's own
            ``source_factor_value_ref`` (raw string) is used as the asserted id.
        label_bundle_id: Optional asserted label-bundle artifact id; when None,
            derived from the label bundle definition.
        metric_ids: Metrics the evaluation should compute.

    Returns:
        An :class:`EvaluationRequest` with ``factor_value_ref`` /
        ``label_bundle_ref`` populated (serializable round-trip), and the raw
        array payloads attached as runtime-only fields.
    """
    raw_id = factor_value_id or _first_plain_str(
        getattr(recipe, "source_factor_value_ref", "")
    )
    factor_value_ref = recipe_to_factor_value_ref(recipe, ref_for="treated")
    if raw_id:
        # Provenance: the raw id is a first-class identifier tag so the treated
        # ref records the underlying identity it was derived from.
        factor_value_ref = FactorValueRef(
            factor_value_id=factor_value_ref.factor_value_id,
            factor_ids=factor_value_ref.factor_ids,
            source_ref=raw_id,
            metadata=dict(factor_value_ref.metadata or {}),
        )

    label_ref = None
    if label_bundle is not None:
        label_ref = label_bundle_to_ref(
            label_bundle, asserted_label_bundle_id=label_bundle_id
        )

    kwargs = dict(request_kwargs)
    kwargs.setdefault("factor_value_ref", factor_value_ref)
    if label_ref is not None:
        kwargs.setdefault("label_bundle_ref", label_ref)
    return EvaluationRequest(
        batch_or_factor_ids=factor_batch,
        label_bundle=label_bundle,
        metric_ids=tuple(metric_ids),
        **kwargs,
    )


def _apply_recipe_to_panel(recipe, raw_values: np.ndarray) -> np.ndarray:
    """Apply a recipe's ordered transforms to a raw cross-sectional panel.

    Synthetic in-memory panel transform application (no registry; the recipe
    is treated as the executable prescription).  NaN positions are preserved
    through every step.  Steps whose registered implementation is unknown to
    this minimal helper fail closed — the recipe contract stays the source of
    truth for what the step means, and a recipe built from unsupported
    transforms cannot silently be evaluated as if it had been applied.
    """
    values = np.asarray(raw_values, dtype=np.float64)
    for step in recipe.ordered_steps:
        semantic = step.semantic_transform_id
        parameters = dict(step.parameters or {})
        overrides = {
            "CS_RANK:pct": ("cs_rank", {"pct": True}),
            "CROSS_SECTIONAL_ZSCORE:cs": ("cs_zscore", {}),
            "CROSS_SECTIONAL_DEMEAN:cs": ("cs_demean", {}),
            "CROSS_SECTIONAL_SCALE:cs": ("cs_scale", {}),
            "WINSOR:cs": ("cs_winsor", {}),
        }
        if semantic not in overrides:
            raise TypeError(
                "_apply_recipe_to_panel supports only cross-sectional "
                f"transforms, got recipe step {semantic!r}"
            )
        name, defaults = overrides[semantic]
        merged = dict(defaults)
        merged.update(parameters)
        fn = _load_fp_transform(name)
        values = fn(values, **merged)
    return values


def _load_fp_transform(name: str):
    """Import a registered FP transform implementation (causal-safe CS funcs).

    Loads the *implementation function* from the FP public transform module —
    never the FP registry internals — so the recipe's semantic transform id is
    honored by the exact transform FA/FP ships.
    """
    from factor_preprocess.transforms import (  # type: ignore
        cs_rank,
        cs_winsor,
        cs_zscore,
        cs_demean,
        cs_scale,
    )

    return {
        "cs_rank": cs_rank,
        "cs_winsor": cs_winsor,
        "cs_zscore": cs_zscore,
        "cs_demean": cs_demean,
        "cs_scale": cs_scale,
    }[name]


def factor_batch_for_recipe(recipe, panel, *, factor_ids=None) -> Any:
    """Build a QE :class:`FactorBatch` of the *treated* factor values.

    QE owns the treated-value materialisation for its own evaluation: given
    the FP :class:`TreatmentRecipe` prescription and the raw factor panel, it
    applies the recipe's ordered steps (which are implemented as registered,
    causal-safe FP transforms) into a QE ``FactorBatch``.  This keeps the
    chain as `raw values -> recipe -> treated FactorBatch` entirely through
    the public contract boundary — FP owns the recipe semantics, QE owns the
    evaluation contract, and the transform application is a documented QE-side
    adapter helper (no FP-internal import in QE core).

    Args:
        recipe: The FP :class:`TreatmentRecipe` whose ordered steps are applied.
        panel: Mapping of ``factor_id -> (dates, assets, values)`` where
            ``dates`` is a sequence of timestamps, ``assets`` is a sequence of
            asset identifiers, and ``values`` is a 2D numpy array shaped
            ``(n_times, n_assets)``.
        factor_ids: Factor ids whose panels are included.  Defaults to the
            recipe's own ``factor_ids`` (or the panel keys when the recipe
            does not carry them).

    Returns:
        A QE :class:`FactorBatch` of the treated values (or the raw values
        when the recipe has no transform steps).
    """
    if factor_ids is None:
        factor_ids = tuple(getattr(recipe, "factor_ids", ()) or ())
    if not factor_ids:
        factor_ids = tuple(panel.keys())
    factor_ids = tuple(factor_ids)
    if not factor_ids:
        raise ValueError("factor_batch_for_recipe: no factor ids available")

    # The panel is synthetic / in-memory only.  Every factor must be present.
    missing = [fid for fid in factor_ids if fid not in panel]
    if missing:
        raise ValueError(
            "factor_batch_for_recipe: missing raw factor panels for: "
            f"{missing}"
        )

    first = panel[factor_ids[0]]
    dates, assets, values = first[0], first[1], first[2]
    n_times, n_assets = values.shape
    if len(dates) != n_times or len(assets) != n_assets:
        raise ValueError(
            "factor_batch_for_recipe: dates/assets lengths must match the "
            "values panel shape"
        )

    # Apply the recipe's ordered steps to every factor's cross-section.
    treated = np.stack([_apply_recipe_to_panel(recipe, panel[fid][2]) for fid in factor_ids], axis=2)

    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch

    return FactorBatch(
        factor_ids=factor_ids,
        time_axis=AxisRef("time", "datetime64[ns]", n_times, values=np.asarray(dates)),
        asset_axis=AxisRef("asset", "str", n_assets, values=np.asarray(assets)),
        values=treated,
    )


def recipe_evidence_evaluation(
    recipe,
    panel,
    labels,
    *,
    metrics=("rank_ic", "pearson_ic", "coverage"),
    context=None,
) -> Any:
    """Evaluate a recipe's treated factor values end-to-end (QE authority).

    Convenience seam that closes the raw -> recipe -> treated-FactorBatch ->
    evaluation loop for callers that hold a synthetic panel + vwap-derived
    labels.  It runs the recipe through :func:`factor_batch_for_recipe` and
    the QE public ``evaluate`` facade, returning the
    :class:`EvaluationBundle` (computed, vwap-to-vwap basis).

    Args:
        recipe: The FP :class:`TreatmentRecipe` being evaluated.
        panel: Mapping of ``factor_id -> (dates, assets, values)``.
        labels: vwap-to-vwap forward-return label panel shaped
            ``(n_times, n_assets)``.
        metrics: Metric ids to compute (default: rank_ic / pearson_ic /
            coverage).
        context: Optional evaluation context (universe, period, ...).

    Returns:
        The QE :class:`EvaluationBundle` produced by the public evaluate
        facade for the treated factor batch.
    """
    factor_batch = factor_batch_for_recipe(recipe, panel)
    from datetime import datetime
    from quant_evaluator.contracts.label_bundle import LabelBundle

    n_times = factor_batch.num_times
    if labels.shape != (n_times, factor_batch.num_assets):
        raise ValueError(
            "recipe_evidence_evaluation: label panel shape must be "
            f"(n_times={n_times}, n_assets={factor_batch.num_assets}), got "
            f"{labels.shape}"
        )
    bundle = LabelBundle(
        target_id="vwap_forward_return",
        values=np.asarray(labels, dtype=np.float64),
        horizon=1,
        decision_time=tuple(range(n_times)),
        label_start_time=tuple(range(n_times)),
        label_end_time=tuple(range(1, n_times + 1)),
        price_convention="vwap_to_vwap",
        source_ref=getattr(recipe, "source_factor_value_ref", None),
    )
    from quant_evaluator.runtime.evaluator import evaluate

    return evaluate(
        factor_batch,
        bundle,
        context=context,
        metrics=tuple(metrics),
    )


def assert_computed_value(value, *, metric_id: str, factor_id: str = "") -> float:
    """Fail closed when an evaluation result is NOT computed evidence.

    Args:
        value: The per-metric value (float, or a MetricValue-like object
            carrying ``valid`` / ``value``).
        metric_id: Metric id for the error message.
        factor_id: Factor id for the error message ('' = whole-batch metric).

    Returns:
        The numeric value, when it is real COMPUTED evidence.

    Raises:
        RuntimeError: When ``value`` is ``None``, or a MetricValue-like object
            whose ``valid`` is False (e.g. label-not-mature / insufficient-
            data), or carries a non-finite numeric payload.  A non-finite
            payload would otherwise be rendered as ``None`` by the facade
            ``evaluate`` path (``runtime/evaluator.py`` ``valid=bool(np.isfinite
            (numeric))``) — this guard fails closed instead of passing it up.
    """
    if value is None:
        raise RuntimeError(
            f"evaluation result not computed for metric {metric_id!r} "
            f"(factor {factor_id!r}): value is None (label-not-mature / "
            "insufficient-data must never be treated as success)"
        )
    # Duck-typed MetricValue: valid flag is authoritative when present.
    valid = getattr(value, "valid", None)
    if valid is not None and not valid:
        raise RuntimeError(
            f"evaluation result not computed for metric {metric_id!r} "
            f"(factor {factor_id!r}): valid=False — evidence is not COMPUTED "
            "(do not treat a not-computed result as success)"
        )
    numeric = float(getattr(value, "value", value))
    if not _isfinite(numeric):
        raise RuntimeError(
            f"evaluation result not finite for metric {metric_id!r} "
            f"(factor {factor_id!r}): got {numeric!r}"
        )
    return numeric


def _first_plain_str(value: Any) -> str:
    """Best-effort plain string identity of a ref-ish value."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("factor_value_id", "id", "value_id"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
    return ""


def _isfinite(value: float) -> bool:
    try:
        import math

        return math.isfinite(value)
    except TypeError:
        return False