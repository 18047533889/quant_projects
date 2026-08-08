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
        )

    OperatorRegistry.register = classmethod(audited_register)
    OperatorRegistry._bootstrap_replacement_history = _HISTORY
    _INSTALLED = True


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
        for backend, operator in implementations.items():
            params = list(_params(operator))
            signatures[backend] = params
            entry = dict(backend_meta.get(backend) or {})
            entry["param_names"] = params
            entry["implementation_type"] = type(operator).__name__
            backend_meta[backend] = entry
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
