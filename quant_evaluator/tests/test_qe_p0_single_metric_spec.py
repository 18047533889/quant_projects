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
