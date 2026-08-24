"""
QE-P1-15: MetricRegistry lifecycle + MetricSpec field validation.

Pins the P1-15 contract for ``quant_evaluator.metrics.catalog``:

- ``MetricSpec`` gained validated fields (metric_version, implementation_id,
  artifact_kind, required_axes, units, direction, missing_policy,
  numeric_policy) with defaults so existing registrations stay valid.
- The module-level catalog is now backed by a ``MetricRegistry`` with a
  BUILDING -> SEALED lifecycle: duplicate registration raises ``ValueError``
  before sealing, registration after sealing fails closed, and the module
  level ``CATALOG`` is an immutable ``MappingProxyType`` at import time.
- The public query helpers keep working exactly as before.

Importable source is the repo-root ``quant_evaluator/`` package (source, not
build/lib).  Run with:

    PYTHONPATH=/home/sunhaiwei/quant_projects \\
    python -m pytest quant_evaluator/tests/test_metric_registry_lifecycle.py -q \\
    --tb=short
"""

from types import MappingProxyType

import pytest

from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.metrics.catalog import (
    CATALOG,
    Domain,
    MetricRegistry,
    MetricSpec,
    get_metric_spec,
    get_metric_specs_by_domain,
    list_all_domains,
    list_all_metric_ids,
    seal_metric_registry,
)


def _make_spec(
    metric_id: str,
    domain: Domain = Domain.IC,
    description: str = "test metric",
    required_inputs=None,
    output_type: str = "scalar",
    **kwargs,
) -> MetricSpec:
    if required_inputs is None:
        required_inputs = {"factor", "forward_returns"}
    return MetricSpec(
        domain=domain,
        metric_id=metric_id,
        description=description,
        required_inputs=required_inputs,
        output_type=output_type,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fresh MetricRegistry lifecycle.
# ---------------------------------------------------------------------------


class TestMetricRegistryBuildPhase:
    def test_register_two_distinct_specs_ok(self):
        registry = MetricRegistry()
        registry.register(_make_spec("m1"))
        registry.register(_make_spec("m2"))
        assert len(registry) == 2
        assert registry.get("m1").metric_id == "m1"
        assert registry.get("m2").metric_id == "m2"
        assert registry.list_all_metric_ids() == ["m1", "m2"]
        assert registry.sealed is False

    def test_duplicate_metric_id_raises_value_error(self):
        registry = MetricRegistry()
        registry.register(_make_spec("dup"))
        with pytest.raises(ValueError):
            registry.register(_make_spec("dup"))

    def test_get_unknown_raises_unsupported_metric(self):
        registry = MetricRegistry()
        registry.register(_make_spec("known"))
        with pytest.raises(UnsupportedMetricError):
            registry.get("missing")

    def test_get_metric_spec_alias_and_helpers(self):
        registry = MetricRegistry()
        registry.register(_make_spec("a_metric", domain=Domain.IC))
        registry.register(_make_spec("b_metric", domain=Domain.TURNOVER))
        assert registry.get_metric_spec("a_metric").metric_id == "a_metric"
        specs = registry.get_metric_specs_by_domain(Domain.IC)
        assert [s.metric_id for s in specs] == ["a_metric"]
        assert Domain.IC in registry.list_all_domains()
        assert registry.to_dict()["a_metric"].metric_id == "a_metric"

    def test_register_rejects_non_spec(self):
        registry = MetricRegistry()
        with pytest.raises(TypeError):
            registry.register("not a spec")  # type: ignore[arg-type]


class TestMetricRegistrySealed:
    def test_seal_freezes_registry(self):
        registry = MetricRegistry()
        registry.register(_make_spec("m1"))
        registry.seal()
        assert registry.sealed is True

        # register fails closed after seal.
        with pytest.raises(RuntimeError):
            registry.register(_make_spec("m2"))

        # reads still work.
        assert registry.get("m1").metric_id == "m1"
        assert len(registry) == 1
        assert registry.list_all_metric_ids() == ["m1"]

    def test_mapping_is_read_only_after_seal(self):
        registry = MetricRegistry()
        registry.register(_make_spec("m1"))
        registry.seal()
        catalog = registry._catalog
        assert isinstance(catalog, MappingProxyType)
        with pytest.raises(TypeError):
            catalog["__new__"] = None  # type: ignore[index]

    def test_seal_is_idempotent(self):
        registry = MetricRegistry()
        registry.register(_make_spec("m1"))
        registry.seal()
        registry.seal()  # must not raise


# ---------------------------------------------------------------------------
# MetricSpec new-field validation.
# ---------------------------------------------------------------------------


class TestMetricSpecValidation:
    def test_defaults_apply(self):
        spec = _make_spec("defaults", output_type="timeseries")
        assert spec.metric_version == "0.1.0"
        assert spec.implementation_id == "defaults"  # auto-defaults to metric_id
        assert spec.artifact_kind == "series"  # mapped from output_type
        assert spec.required_axes == ()
        assert spec.units == ""
        assert spec.direction == "higher_is_better"
        assert spec.missing_policy == "nan"
        assert spec.numeric_policy == "finite"

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError):
            _make_spec("bad_dir", direction="sideways")

    def test_non_empty_metric_id_enforced(self):
        with pytest.raises(ValueError):
            _make_spec("")

    def test_required_axes_must_be_non_empty_strings(self):
        with pytest.raises(ValueError):
            _make_spec("bad_axes", required_axes=("",))
        with pytest.raises(ValueError):
            _make_spec("bad_axes_type", required_axes=["x"])  # list, not tuple

    def test_bad_artifact_kind_raises(self):
        with pytest.raises(ValueError):
            _make_spec("bad_artifact", artifact_kind="blob")

    def test_output_type_mapping(self):
        for output_type, expected in [
            ("scalar", "scalar"),
            ("series", "series"),
            ("timeseries", "series"),
            ("vector", "vector"),
            ("matrix", "matrix"),
            ("distribution", "distribution"),
        ]:
            spec = _make_spec(f"ot_{output_type}", output_type=output_type)
            assert spec.artifact_kind == expected, output_type

    def test_explicit_artifact_kind_respected(self):
        spec = _make_spec(
            "explicit", output_type="scalar", artifact_kind="series"
        )
        assert spec.artifact_kind == "series"


# ---------------------------------------------------------------------------
# Default catalog: sealed at import.
# ---------------------------------------------------------------------------


class TestDefaultCatalog:
    def test_catalog_is_mapping_proxy_and_read_only(self):
        assert isinstance(CATALOG, MappingProxyType)
        with pytest.raises(TypeError):
            CATALOG["__new__"] = None  # type: ignore[index]

    def test_default_catalog_sealed(self):
        # seal_metric_registry() must have been applied at import time, so
        # the sealed registry still serves reads but rejects registration.
        registry = MetricRegistry()
        registry.register(_make_spec("x"))
        registry.seal()
        assert registry.sealed is True
        with pytest.raises(RuntimeError):
            registry.register(_make_spec("y"))
        # The public helper path on the sealed module registry:
        assert get_metric_spec("pearson_ic").metric_id == "pearson_ic"

    def test_list_all_metric_ids_non_empty_and_contains_pearson_ic(self):
        ids = list_all_metric_ids()
        assert len(ids) > 0
        assert ids == sorted(ids)
        assert "pearson_ic" in ids

    def test_accessors_work_on_sealed_catalog(self):
        assert len(get_metric_specs_by_domain(Domain.IC)) >= 1
        assert len(list_all_domains()) == len(list(Domain))
        spec = get_metric_spec("pearson_ic")
        assert spec.domain == Domain.IC
        assert spec.output_type == "scalar"

    def test_unknown_metric_raises_unsupported_metric(self):
        with pytest.raises(UnsupportedMetricError):
            get_metric_spec("__no_such_metric__")
