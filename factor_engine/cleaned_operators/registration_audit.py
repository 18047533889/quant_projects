# -*- coding: utf-8 -*-
"""Bootstrap-time audit for operator replacements and backend signatures.

The historical loader deliberately layers audited implementations over imported
compatibility implementations.  Replacements are therefore allowed during
bootstrap, but they must not be silent: every canonical/backend replacement is
recorded with source, status and parameter-contract provenance.  The final
catalog exposes this history and validates high-risk fiscal signatures.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from cleaned_operators.registry import OperatorRegistry

_INSTALLED = False
_ORIGINAL_REGISTER = None
_HISTORY: list[dict[str, Any]] = []

_FISCAL_SIGNATURES: dict[str, tuple[str, ...]] = {
    "period_lag": ("x", "period_id", "periods", "revision_policy"),
    "period_change": (
        "x", "period_id", "periods", "mode", "require_consecutive", "revision_policy",
    ),
    "period_average": (
        "x", "period_id", "periods", "require_consecutive", "revision_policy",
    ),
    "period_cagr": (
        "x", "period_id", "periods", "periods_per_year", "sign_policy",
        "require_consecutive", "revision_policy",
    ),
    "quarter_from_cumulative": ("x", "period_id", "fiscal_quarter", "revision_policy"),
    "ttm_from_quarterly": (
        "x", "period_id", "periods", "require_consecutive", "revision_policy",
    ),
    "ttm_from_cumulative": ("x", "period_id", "fiscal_quarter", "revision_policy"),
    "yoy_by_period": (
        "x", "period_id", "periods", "denominator", "require_consecutive", "revision_policy",
    ),
}


def _params(operator: Any) -> tuple[str, ...]:
    metadata = getattr(operator, "metadata", None)
    return tuple(str(x) for x in (getattr(metadata, "param_names", None) or ()))


def _logical_names(logical: list[dict[str, Any]]) -> tuple[str, ...]:
    """Human-readable role/name pair for error messages."""
    return tuple(f"{entry['role']}:{entry['name']}" for entry in logical)


def _logical_signature(operator: Any) -> list[dict[str, Any]] | None:
    """Logical per-position signature: panel-vs-scalar role + required status.

    R7-232: arity alone is insufficient — ``(x, weight, window)`` and
    ``(x, window, weight)`` have equal length but bind a positional scalar to a
    panel slot.  This derives, per positional index, whether the param is a
    PANEL input (leading, required, positionally bound) or a SCALAR control
    (defaulted, may trail).  The panel prefix is the shape a positional call
    actually binds, so it must match across backends exactly.

    Role resolution order:
    1. ``metadata.panel_params`` / ``metadata.scalar_params`` (R7-224 declared);
    2. ``metadata.panel_arity`` (leading N are panel);
    The audit NEVER infers a panel/scalar role from the kernel signature or
    param defaults: a required positional scalar (``tick_tolerance``, ``q``,
    ``threshold``) has no default and would be mis-guessed as a panel input,
    exactly the drift R7-232 exists to catch.  An operator that does NOT declare
    its panel/scalar layout is not comparable at the logical level and returns
    ``None`` — the existing R4-100 arity gate still applies, but no role
    judgement is fabricated.
    """
    metadata = getattr(operator, "metadata", None)
    if metadata is None:
        return None
    names = list(getattr(metadata, "param_names", None) or ())
    if not names:
        return None
    declared_panel = list(getattr(metadata, "panel_params", None) or ())
    declared_scalar = list(getattr(metadata, "scalar_params", None) or ())
    panel_arity = getattr(metadata, "panel_arity", None)
    if not declared_panel and not declared_scalar and panel_arity is None:
        # No declared panel/scalar layout: NOT comparable (arity gate only).
        return None
    out: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        role: str | None = None
        if name in declared_panel:
            role = "panel"
        elif name in declared_scalar:
            role = "scalar"
        elif panel_arity is not None and index < int(panel_arity):
            role = "panel"
        elif panel_arity is not None:
            role = "scalar"
        if role is None:
            # Declared layout does not cover every param (partial declaration):
            # treat uncovered trailing params as scalar controls — a panel slot
            # must always be explicitly declared.
            role = "scalar"
        out.append({"name": name, "role": role, "position": index})
    return out


def install_registration_audit() -> None:
    """Wrap registry registration before any runtime modules are imported."""
    global _INSTALLED, _ORIGINAL_REGISTER
    if _INSTALLED:
        return
    _ORIGINAL_REGISTER = OperatorRegistry.register.__func__

    def audited_register(
        cls,
        operator: Any,
        *,
        canonical: str,
        backend: str = "pandas_numpy",
        aliases=None,
        source: str = "",
        status: str = "implemented",
        backend_explicit: bool = True,
        replace: bool = False,
        replacement_reason: str = "",
        expected_old_source: str = "",
        semantic_version: str = "1.0",
    ) -> None:
        canonical_name = canonical or operator.metadata.name
        previous_operator = cls._operators.get(canonical_name, {}).get(backend)
        previous_meta = dict(
            ((cls._catalog.get(canonical_name, {}).get("backend_meta") or {}).get(backend) or {})
        )
        if previous_operator is not None:
            _HISTORY.append(
                {
                    "canonical": canonical_name,
                    "backend": backend,
                    "previous_source": str(previous_meta.get("source", "")),
                    "new_source": str(source or ""),
                    "previous_params": list(_params(previous_operator)),
                    "new_params": list(_params(operator)),
                    "previous_type": type(previous_operator).__name__,
                    "new_type": type(operator).__name__,
                    "replacement_reason": str(replacement_reason or "bootstrap layer replacement"),
                    "semantic_version": str(semantic_version or "1.0"),
                }
            )
        _ORIGINAL_REGISTER(
            cls,
            operator,
            canonical=canonical_name,
            backend=backend,
            aliases=aliases,
            source=source,
            status=status,
            backend_explicit=backend_explicit,
            replace=replace,
            replacement_reason=replacement_reason,
            expected_old_source=expected_old_source,
            semantic_version=semantic_version,
        )

    OperatorRegistry.register = classmethod(audited_register)
    OperatorRegistry._bootstrap_replacement_history = _HISTORY
    _INSTALLED = True


def _surface_label(canonical: str) -> str | None:
    """The canonical's AuthoringTier surface (``daily``/``extended``/``research``/...).

    Deliberately imported lazily so this module has no import-time dependency on
    ``operator_surface`` (which itself imports the registry).
    """
    try:
        from cleaned_operators.operator_surface import classify_canonical

        return classify_canonical(canonical)
    except Exception:  # pragma: no cover - audit-only best effort
        return None


def _reconcile_surface_tags(operator: Any, canonical: str, surface: str | None) -> None:
    """Make the operator's metadata surface tags agree with the real surface.

    R11 P1-04: ``daily`` is a stale registration-time stamp when the canonical
    is actually ``research`` / ``extended`` / ``internal``.  Remove the
    misleading ``daily`` tag and add the honest surface tag so metadata never
    contradicts the single AuthoringTier authority.
    """
    if surface is None or surface in {"", "unclassified"}:
        return
    meta = getattr(operator, "metadata", None)
    if meta is None:
        return
    try:
        tags = list(getattr(meta, "tags", None) or [])
    except AttributeError:
        return
    changed = False
    if "daily" in tags and surface != "daily":
        tags = [t for t in tags if t != "daily"]
        changed = True
    if surface not in tags and surface in {"research", "extended", "internal", "legacy", "unsafe"}:
        tags.append(surface)
        changed = True
    if changed:
        try:
            meta.tags = tags
        except (AttributeError, TypeError):  # frozen metadata
            pass


def finalize_registration_audit() -> None:
    """Attach deterministic replacement/signature evidence to the final catalog."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _HISTORY:
        canonical = OperatorRegistry._aliases.get(row["canonical"], row["canonical"])
        item = dict(row)
        item["canonical"] = canonical
        grouped[canonical].append(item)

    for canonical, implementations in OperatorRegistry._operators.items():
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        backend_meta = dict(catalog.get("backend_meta") or {})
        signatures: dict[str, list[str]] = {}
        # R11 P1-04: reconcile the metadata ``daily`` tag against the SINGLE
        # surface authority (AuthoringTier / classify_canonical frozensets).
        # A helper like ``register_dual`` stamps ``daily`` at registration time,
        # before the module's union_extended / union_research runs — so a
        # research/extended operator can briefly carry a ``daily`` tag that
        # contradicts its real surface.  Metadata must not duplicate surface
        # authority; remove the stale ``daily`` tag on non-daily canonicals and
        # add the honest surface tag.
        _surface_tag = _surface_label(canonical)
        for backend, operator in implementations.items():
            params = list(_params(operator))
            signatures[backend] = params
            entry = dict(backend_meta.get(backend) or {})
            entry["param_names"] = params
            entry["implementation_type"] = type(operator).__name__
            backend_meta[backend] = entry
            _reconcile_surface_tags(operator, canonical, _surface_tag)
        catalog["backend_meta"] = backend_meta
        catalog["backend_signatures"] = signatures
        catalog["replacement_history"] = sorted(
            grouped.get(canonical, []),
            key=lambda row: (
                row["backend"], row["previous_source"], row["new_source"],
                row["previous_type"], row["new_type"],
            ),
        )
        unique_signatures = {tuple(value) for value in signatures.values() if value}
        catalog["backend_signature_consistent"] = len(unique_signatures) <= 1

    # R4-100: positional-arity gate for ALL canonicals.  The pandas_numpy
    # backend is the semantic reference; any other backend that accepts FEWER
    # positional parameters would silently mis-read a positional call that is
    # valid against the reference (e.g. KAMA(close, 10, 2, 30) once broke the
    # polars ``(close, window)`` backend).  Backends may EXTEND the contract
    # with defaulted trailing params, never shrink it.  Variable NAMES are not
    # compared — numerator/denominator vs float_shares/total_shares is fine when
    # arity and roles match (audit: "不要只比较参数变量名字").
    for canonical in sorted(OperatorRegistry._operators):
        signatures = OperatorRegistry._catalog.get(canonical, {}).get(
            "backend_signatures", {}
        )
        pandas_params = signatures.get("pandas_numpy")
        if not pandas_params:
            continue
        for backend, params in signatures.items():
            if backend == "pandas_numpy" or not params:
                continue
            if len(params) < len(pandas_params):
                raise RuntimeError(
                    f"backend signature arity mismatch for {canonical}/{backend}: "
                    f"{len(params)} positional params < pandas reference "
                    f"{len(pandas_params)} ({tuple(params)} vs {tuple(pandas_params)}). "
                    "A positional call valid against the reference would mis-read "
                    "or break this backend (R4-100)."
                )

    # R7-232: LOGICAL signature equivalence, not just param COUNT.  Two backends
    # with the same arity but swapped panel/scalar roles (pandas ``(x, weight,
    # window)`` vs polars ``(x, window, weight)``) are semantically DIFFERENT —
    # a positional call binds the scalar to a panel slot and vice versa.  This
    # compares, per positional index, whether both backends agree on
    # panel-vs-scalar role and required-vs-optional status.  A backend may
    # reorder or rename scalar params (trailing defaults), but the PANEL prefix
    # (positionally leading, required inputs) must match exactly — that is the
    # shape that a positional call actually binds.
    for canonical in sorted(OperatorRegistry._operators):
        implementations = OperatorRegistry._operators.get(canonical, {})
        pandas_op = implementations.get("pandas_numpy")
        if pandas_op is None:
            continue
        pandas_logical = _logical_signature(pandas_op)
        if pandas_logical is None:
            continue
        for backend, operator in implementations.items():
            if backend == "pandas_numpy":
                continue
            other_logical = _logical_signature(operator)
            if other_logical is None:
                continue
            # Panel prefix must match exactly (positional index + role).
            p_panels = [i for i, k in enumerate(pandas_logical) if k["role"] == "panel"]
            o_panels = [i for i, k in enumerate(other_logical) if k["role"] == "panel"]
            if p_panels != o_panels:
                raise RuntimeError(
                    f"R7-232 logical signature mismatch for {canonical}/{backend}: "
                    f"panel positional indices {p_panels} != pandas reference "
                    f"{o_panels} ({_logical_names(pandas_logical)} vs "
                    f"{_logical_names(other_logical)}).  Same arity but different "
                    "panel/scalar layout — a positional call binds a scalar to a "
                    "panel slot.  Fix the backend signature, not the gate."
                )

    for canonical, expected in _FISCAL_SIGNATURES.items():
        implementations = OperatorRegistry._operators.get(canonical, {})
        if not implementations:
            raise RuntimeError(f"missing strict fiscal canonical: {canonical}")
        for backend, operator in implementations.items():
            if backend == "sql":
                continue
            actual = _params(operator)
            if actual != expected:
                raise RuntimeError(
                    f"fiscal backend signature mismatch for {canonical}/{backend}: "
                    f"expected={expected}, actual={actual}"
                )
        catalog = OperatorRegistry._catalog[canonical]
        catalog["semantic_contract"] = "strict_fiscal_period_v2"
        catalog["certified_parameter_domain"] = {
            "revision_policy": sorted(["first_available", "latest_available"]),
            "require_consecutive": [False, True],
        }


def replacement_history() -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in _HISTORY)
