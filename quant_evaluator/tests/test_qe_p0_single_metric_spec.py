"""
QE-P0-01: exactly ONE MetricSpec / ONE MetricRegistry authority.

Regression guard for the dual-authority smell: ``metrics.catalog`` and
``registry.metrics`` each defined their own ``MetricSpec``.  The two must be
merged into a single class — ``metrics.catalog.MetricSpec`` must BE
``registry.metrics.MetricSpec`` (re-export), never a second definition.

The catalog layer may only be a generated read-only view / compatibility
adapter over the registry authority.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import quant_evaluator.metrics.catalog as catalog
import quant_evaluator.registry.metrics as registry_metrics


def test_single_metric_spec_class():
    """catalog.MetricSpec must be the SAME class as registry.MetricSpec."""
    assert catalog.MetricSpec is registry_metrics.MetricSpec, (
        "two distinct MetricSpec classes exist; must be merged into one"
    )


def test_catalog_metric_spec_is_not_a_second_definition():
    """The catalog must not define its own MetricSpec body."""
    # The catalog module must not have a MetricSpec class defined in its own
    # namespace (it must be imported/re-exported from the registry authority).
    assert catalog.MetricSpec.__module__ == registry_metrics.MetricSpec.__module__


def test_catalog_registry_uses_single_metric_spec():
    """The catalog's MetricRegistry must hold the single MetricSpec type."""
    spec = catalog.get_metric_spec("pearson_ic")
    assert isinstance(spec, registry_metrics.MetricSpec)
    assert type(spec) is registry_metrics.MetricSpec


def test_catalog_specs_carry_catalog_fields():
    """Catalog-style fields (domain, metric_id, implementation_hash) survive."""
    spec = catalog.get_metric_spec("pearson_ic")
    assert spec.domain is catalog.Domain.IC
    assert spec.metric_id == "pearson_ic"
    assert spec.implementation_hash  # auto-derived content hash
    assert spec.direction == "higher_is_better"


# ---------------------------------------------------------------------------
# QE-P0-R-C2: single POPULATED authority (no dual data).
# ---------------------------------------------------------------------------


def test_single_populated_registry_authority():
    """The two surfaces must share ONE backing MetricRegistry instance.

    ``metrics.catalog`` must be a read-only view over the SAME populated
    ``MetricRegistry`` that ``registry.metrics`` owns — never a second
    independently-populated catalog.
    """
    assert catalog._REGISTRY is registry_metrics._REGISTRY, (
        "metrics.catalog and registry.metrics hold TWO distinct populated "
        "MetricRegistry instances; there must be exactly ONE"
    )


def test_catalog_is_read_only_view_over_single_registry():
    """The catalog's CATALOG mapping must be a read-only view over the single
    registry's backing store (the 10-domain subset)."""
    from types import MappingProxyType

    assert isinstance(catalog.CATALOG, MappingProxyType)
    with pytest.raises(TypeError):
        catalog.CATALOG["__new__"] = None  # type: ignore[index]
    # Every catalog metric_id is served by the single registry.
    assert set(catalog.list_all_metric_ids()) <= set(registry_metrics.list_metrics())
    # The catalog exposes the 10-domain subset (its historical contract).
    assert len(catalog.list_all_metric_ids()) >= 1
    assert all(
        catalog.get_metric_spec(mid).domain is not None
        for mid in catalog.list_all_metric_ids()
    )


def test_catalog_metrics_are_registered_in_single_registry():
    """Every catalog metric_id resolves through the single registry."""
    for metric_id in catalog.list_all_metric_ids():
        spec = registry_metrics.get_metric(metric_id)
        assert spec.metric_id == metric_id
        assert spec is catalog.get_metric_spec(metric_id)
