"""
10-domain metric catalog for quant_evaluator (QE-METRIC overhaul, section B).

QE-P0-01 / QE-P0-R-C2: this module is a **compatibility adapter / generated
read-only view** over the SINGLE metric authority in
``quant_evaluator.registry.metrics``.  It does NOT define its own
``MetricSpec`` / ``MetricRegistry`` / ``Domain`` — those are re-exported from
the registry so there is exactly ONE class of each in the package.

QE-P0-R-C2 (single populated authority): the 10-domain catalog specs are
registered into the ONE ``MetricRegistry`` instance in
``quant_evaluator.registry.metrics`` (``_REGISTRY``).  This module's
``_REGISTRY`` is that SAME instance — there is exactly ONE populated metric
catalog at runtime, never two.  The module-level ``CATALOG`` is an immutable
``MappingProxyType`` view over the 10-domain subset of that backing registry,
and the query helpers (``get_metric_spec``, ``get_metric_specs_by_domain``,
``list_all_metric_ids``, ``list_all_domains``) are read-only facades over it.

The catalog does NOT seal the shared registry at import (that would freeze the
functional API's BUILDING lifecycle).  ``CATALOG`` is always a read-only
``MappingProxyType``; ``seal_metric_registry()`` is available for callers that
explicitly want to freeze the shared registry.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Set, Tuple

# QE-P0-01: single authority — re-export the one MetricSpec / MetricRegistry /
# Domain from the registry.  No second definition lives here.
from quant_evaluator.registry.metrics import (
    Domain,
    MetricRegistry,
    MetricSpec,
    MetricStatus,
    MetricTier,
    _REGISTRY,
)

__all__ = [
    "Domain",
    "MetricSpec",
    "MetricRegistry",
    "CATALOG",
    "get_metric_spec",
    "get_metric_specs_by_domain",
    "list_all_metric_ids",
    "list_all_domains",
    "seal_metric_registry",
]


# ---------------------------------------------------------------------------
# Module-level registry + CATALOG view.
#
# QE-P0-R-C2: ``_REGISTRY`` is the SAME single populated MetricRegistry
# instance owned by ``quant_evaluator.registry.metrics``.  The 10-domain
# catalog specs were registered there (not here), so this module holds no
# second populated catalog.  The module-level ``CATALOG`` is a read-only
# ``MappingProxyType`` over the 10-domain subset (the catalog's historical
# contract), so existing callers doing ``CATALOG[metric_id]`` lookups and
# iteration keep working while any assignment raises ``TypeError``.
# ---------------------------------------------------------------------------
_REGISTRY: MetricRegistry = _REGISTRY


def _domain_catalog() -> Dict[str, MetricSpec]:
    """Return the 10-domain subset of the single registry as a plain dict."""
    return {
        metric_id: spec
        for metric_id, spec in _REGISTRY.to_dict().items()
        if spec.domain is not None
    }


CATALOG: Mapping[str, MetricSpec] = MappingProxyType(_domain_catalog())


def seal_metric_registry() -> None:
    """Seal the shared registry so it becomes immutable.

    Idempotent: calling it again after the registry is already sealed is a
    no-op.  Re-binds the module-level ``CATALOG`` to a fresh read-only view so
    ``CATALOG[...]`` reads keep working while assignment raises.
    """
    global CATALOG
    _REGISTRY.seal()
    CATALOG = MappingProxyType(_domain_catalog())


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------

def get_metric_spec(metric_id: str) -> MetricSpec:
    """
    Retrieve the MetricSpec for a given metric_id.

    Raises
    ------
    UnsupportedMetricError
        If metric_id is not in the catalog.
    """
    return _REGISTRY.get_metric_spec(metric_id)


def get_metric_specs_by_domain(domain: Domain) -> List[MetricSpec]:
    """
    Return all MetricSpec entries for a given Domain, ordered by metric_id.
    """
    return _REGISTRY.get_metric_specs_by_domain(domain)


def list_all_metric_ids() -> List[str]:
    """Return a sorted list of the 10-domain catalog metric IDs."""
    return sorted(_domain_catalog().keys())


def list_all_domains() -> List[Domain]:
    """Return a sorted list of all Domain enum values."""
    return _REGISTRY.list_all_domains()
