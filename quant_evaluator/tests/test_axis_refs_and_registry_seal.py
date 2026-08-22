"""
QE-P0-05 + QE-P0-07: explicit axis refs on metric artifacts + sealed registry.

P0 red-team findings this test pins:

QE-P0-05 (explicit axis identity):
    A ``MetricArtifact`` whose payload's LAST dimension is F must carry its
    factor axis identity first-class (``factor_axis: FactorAxisRef``), never
    only in free-form ``provenance``.  Series artifacts add a ``TimeAxisRef``,
    quantile vectors a ``QuantileAxisRef``, matrices a ``MatrixAxisRefs``.
    The axis refs are typed, frozen, unique-validated, and survive
    ``to_dict``/``from_dict`` + pickle.  Backward compatible (``None``
    default), but when provided they must validate against the payload shape.

QE-P0-07 (sealed registry authority):
    The metric catalog must not be a public mutable dict.  ``register_metric``
    accepts specs during BUILDING, rejects duplicates with ``ValueError``, and
    after ``seal_metric_registry()`` the registry is SEALED: further
    registration raises ``RuntimeError`` and the catalog is an immutable
    read-only view.  The RankIC/Pearson alias DAG is verified: every alias
    resolves to exactly ONE spec, rank-family aliases declare ``spearman`` and
    pearson-family aliases declare ``pearson`` consistently, and
    ``ic.pearson.mean`` resolves to ``pearson_ic`` (its own spec, not the
    ``mean_ic`` alias — single-DAG).

Importable source is ``build/lib/quant_evaluator`` (the repo-root
``quant_evaluator/`` directory is a stub).  The bootstrap below forces the
build/lib package + subpackages onto ``sys.modules`` directly (same pattern
as ``test_metamorphic.py``) and prepends build/lib to ``sys.path`` for
intra-package imports.

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest quant_evaluator/tests/test_axis_refs_and_registry_seal.py -v --tb=short
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Importable-source bootstrap (identical pattern to test_metamorphic.py).
# ---------------------------------------------------------------------------
_BUILD_LIB = Path(__file__).resolve().parents[1] / "build" / "lib"
_PKG_ROOT = _BUILD_LIB / "quant_evaluator"
if str(_BUILD_LIB) not in sys.path:
    sys.path.insert(0, str(_BUILD_LIB))

_QE = sys.modules.get("quant_evaluator")
if _QE is None or not str(getattr(_QE, "__file__", "")).startswith(str(_PKG_ROOT)):
    import importlib.util as _ilu
    import os as _os

    for _pkg in ("", "contracts", "metrics", "kernels", "runtime", "planner",
                 "registry", "diagnosis", "api", "adapters", "backends",
                 "reporting"):
        _root = _PKG_ROOT if not _pkg else _PKG_ROOT / _pkg
        _init = _root / "__init__.py"
        if _init.exists():
            _name = "quant_evaluator" if not _pkg else f"quant_evaluator.{_pkg}"
            _spec = _ilu.spec_from_file_location(_name, str(_init))
            _mod = _ilu.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
    import quant_evaluator
    _qe_dir = _os.path.dirname(_os.path.abspath(quant_evaluator.__file__))
    assert _qe_dir.startswith(str(_PKG_ROOT)), f"bootstrap failed: {_qe_dir}"

from quant_evaluator.contracts.axis_refs import (
    FactorAxisRef,
    MatrixAxisRefs,
    QuantileAxisRef,
    TimeAxisRef,
    axis_ref_from_dict,
)
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.metric_artifacts import (
    DistributionMetricArtifact,
    MatrixMetricArtifact,
    MetricArtifact,
    ScalarMetricArtifact,
    SeriesMetricArtifact,
    VectorMetricArtifact,
)
from quant_evaluator.registry import metrics as registry_metrics
from quant_evaluator.registry.metrics import (
    MetricSpec,
    MetricStatus,
    MetricTier,
    CANONICAL_METRIC_ALIASES,
    catalog_snapshot,
    get_metric,
    list_metrics,
    register_metric,
    registry_state,
    resolve_alias,
    seal_metric_registry,
)

# ---------------------------------------------------------------------------
# QE-P0-05: axis ref construction.
# ---------------------------------------------------------------------------


class TestFactorAxisRef:
    def test_requires_unique_non_empty_factor_ids(self):
        ref = FactorAxisRef(factor_ids=("f1", "f2", "f3"))
        assert ref.num_factors == 3
        assert ref.factor_ids == ("f1", "f2", "f3")

        with pytest.raises(InvalidContractError):
            FactorAxisRef(factor_ids=())
        with pytest.raises(InvalidContractError):
            FactorAxisRef(factor_ids=("f1", "f1"))
        with pytest.raises(InvalidContractError):
            FactorAxisRef(factor_ids=("f1", ""))

    def test_order_hash_is_stable_and_canonical(self):
        a = FactorAxisRef(factor_ids=("f1", "f2", "f3"))
        b = FactorAxisRef(factor_ids=("f1", "f2", "f3"))
        assert a.factor_order_hash == b.factor_order_hash
        # Changing the ORDER changes the hash (order is part of the axis).
        c = FactorAxisRef(factor_ids=("f3", "f2", "f1"))
        assert a.factor_order_hash != c.factor_order_hash

    def test_order_hash_mismatch_rejected(self):
        with pytest.raises(InvalidContractError):
            FactorAxisRef(
                factor_ids=("f1", "f2"),
                factor_order_hash="0" * 64,
            )

    def test_factor_versions_aligned_and_optional(self):
        ref = FactorAxisRef(
            factor_ids=("f1", "f2"),
            factor_versions=("1.0.0", None),
        )
        assert ref.factor_versions == ("1.0.0", None)
        with pytest.raises(InvalidContractError):
            FactorAxisRef(
                factor_ids=("f1", "f2"),
                factor_versions=("1.0.0",),
            )
        assert FactorAxisRef(factor_ids=("f1",)).factor_versions is None

    def test_pickle_roundtrip(self):
        ref = FactorAxisRef(
            factor_ids=("f1", "f2"),
            factor_versions=("1.0.0", None),
        )
        assert pickle.loads(pickle.dumps(ref)) == ref

    def test_to_from_dict_roundtrip(self):
        ref = FactorAxisRef(
            factor_ids=("f1", "f2"),
            factor_versions=("1.0.0", None),
        )
        restored = axis_ref_from_dict(ref.to_dict())
        assert restored == ref


class TestTimeQuantileMatrixAxisRefs:
    def test_time_axis_ref(self):
        ref = TimeAxisRef(
            time_index=("2024-01-01", "2024-01-02"),
            time_zone="Asia/Shanghai",
        )
        assert ref.num_times == 2
        assert pickle.loads(pickle.dumps(ref)) == ref
        assert axis_ref_from_dict(ref.to_dict()) == ref
        # time_zone optional; time_index optional
        assert TimeAxisRef().num_times == 0
        assert TimeAxisRef(time_zone="UTC").time_zone == "UTC"

    def test_quantile_axis_ref_unique(self):
        ref = QuantileAxisRef(quantile_labels=("Q1", "Q2", "Q3"))
        assert ref.num_quantiles == 3
        with pytest.raises(InvalidContractError):
            QuantileAxisRef(quantile_labels=("Q1", "Q1"))
        with pytest.raises(InvalidContractError):
            QuantileAxisRef(quantile_labels=())
        assert pickle.loads(pickle.dumps(ref)) == ref
        assert axis_ref_from_dict(ref.to_dict()) == ref

    def test_matrix_axis_refs(self):
        ref = MatrixAxisRefs(
            row=QuantileAxisRef(("Q1", "Q2")),
            col=QuantileAxisRef(("LOW", "HIGH")),
        )
        assert ref.num_quantiles == 2
        assert pickle.loads(pickle.dumps(ref)) == ref
        assert axis_ref_from_dict(ref.to_dict()) == ref
        with pytest.raises(InvalidContractError):
            MatrixAxisRefs(
                row=QuantileAxisRef(("Q1", "Q2")),
                col=QuantileAxisRef(("LOW",)),
            )


# ---------------------------------------------------------------------------
# QE-P0-05: artifacts carry axis refs first-class.
# ---------------------------------------------------------------------------

_F3 = FactorAxisRef(factor_ids=("f1", "f2", "f3"))
_T2 = TimeAxisRef(time_index=("2024-01-01", "2024-01-02"), time_zone="Asia/Shanghai")
_Q2 = QuantileAxisRef(quantile_labels=("Q1", "Q2"))
_M2 = MatrixAxisRefs(
    row=QuantileAxisRef(("Q1", "Q2")),
    col=QuantileAxisRef(("LOW", "HIGH")),
)


@pytest.mark.parametrize("payload_kwargs,factor_axis", [
    ({"values": np.array([0.1, 0.2, 0.3])}, _F3),
    ({"values": np.ones((2, 3)), "time_index": ("a", "b")}, _F3),
    ({"values": np.ones((2, 3))}, _F3),
    ({"values": np.ones((2, 2, 3))}, _F3),
    ({"samples": np.ones((4, 3)), "stat_names": ("s1",)}, _F3),
], ids=["scalar", "series", "vector", "matrix", "distribution"])
def test_artifact_factor_axis_roundtrip(payload_kwargs, factor_axis):
    cls, kwargs = _artifact_cls_and_kwargs(payload_kwargs)
    artifact = cls(**kwargs, factor_axis=factor_axis)
    assert artifact.factor_axis == factor_axis

    # to_dict / from_dict preserves the axis ref.
    restored = MetricArtifact.from_dict(artifact.to_dict())
    assert restored == artifact
    assert restored.factor_axis == factor_axis

    # pickle preserves the axis ref.
    restored2 = pickle.loads(pickle.dumps(artifact))
    assert restored2 == artifact
    assert restored2.factor_axis == factor_axis


def _artifact_cls_and_kwargs(payload_kwargs):
    """Map a payload kwargs dict to (artifact class, full kwargs)."""
    if "samples" in payload_kwargs:
        return DistributionMetricArtifact, {
            "metric_id": "m", "domain": "d", "artifact_kind": "distribution",
            **payload_kwargs,
        }
    values = np.asarray(payload_kwargs["values"])
    if values.ndim == 1:
        return ScalarMetricArtifact, {
            "metric_id": "m", "domain": "d", "artifact_kind": "scalar",
            **payload_kwargs,
        }
    if values.ndim == 2:
        if "time_index" in payload_kwargs:
            return SeriesMetricArtifact, {
                "metric_id": "m", "domain": "d", "artifact_kind": "series",
                **payload_kwargs,
            }
        return VectorMetricArtifact, {
            "metric_id": "m", "domain": "d", "artifact_kind": "vector",
            **payload_kwargs,
        }
    return MatrixMetricArtifact, {
        "metric_id": "m", "domain": "d", "artifact_kind": "matrix",
        **payload_kwargs,
    }


def test_factor_axis_must_match_payload_f():
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(
            metric_id="m", domain="d", artifact_kind="scalar",
            values=np.array([0.1, 0.2]), factor_axis=_F3,
        )
    with pytest.raises(InvalidContractError):
        SeriesMetricArtifact(
            metric_id="m", domain="d", artifact_kind="series",
            values=np.ones((2, 3)), time_index=("a", "b"),
            factor_axis=FactorAxisRef(("f1", "f2")),
        )
    with pytest.raises(InvalidContractError):
        MatrixMetricArtifact(
            metric_id="m", domain="d", artifact_kind="matrix",
            values=np.ones((2, 2, 3)),
            factor_axis=FactorAxisRef(("f1",)),
        )
    with pytest.raises(InvalidContractError):
        DistributionMetricArtifact(
            metric_id="m", domain="d", artifact_kind="distribution",
            samples=np.ones((4, 3)), stat_names=("s1",),
            factor_axis=FactorAxisRef(("f1", "f2")),
        )


def test_series_time_axis_must_match_t():
    with pytest.raises(InvalidContractError):
        SeriesMetricArtifact(
            metric_id="m", domain="d", artifact_kind="series",
            values=np.ones((3, 3)), time_index=("a", "b", "c"),
            time_axis=TimeAxisRef(time_index=("x", "y")),
        )
    ok = SeriesMetricArtifact(
        metric_id="m", domain="d", artifact_kind="series",
        values=np.ones((2, 3)), time_index=("a", "b"),
        time_axis=_T2,
    )
    assert ok.time_axis == _T2


def test_vector_quantile_axis_must_match_k():
    with pytest.raises(InvalidContractError):
        VectorMetricArtifact(
            metric_id="m", domain="d", artifact_kind="vector",
            values=np.ones((3, 3)),
            quantile_axis=QuantileAxisRef(("Q1", "Q2")),
        )
    ok = VectorMetricArtifact(
        metric_id="m", domain="d", artifact_kind="vector",
        values=np.ones((2, 3)), quantile_axis=_Q2,
    )
    assert ok.quantile_axis == _Q2


def test_matrix_matrix_axis_must_match_k():
    with pytest.raises(InvalidContractError):
        MatrixMetricArtifact(
            metric_id="m", domain="d", artifact_kind="matrix",
            values=np.ones((3, 3, 3)),
            matrix_axis=MatrixAxisRefs(_Q2, QuantileAxisRef(("Q1", "Q2"))),
        )
    ok = MatrixMetricArtifact(
        metric_id="m", domain="d", artifact_kind="matrix",
        values=np.ones((2, 2, 3)), matrix_axis=_M2,
    )
    assert ok.matrix_axis == _M2


def test_axis_identity_is_first_class_not_only_provenance():
    """The axis ref must survive a round-trip that STRIPS provenance entirely."""
    artifact = SeriesMetricArtifact(
        metric_id="m", domain="d", artifact_kind="series",
        values=np.ones((2, 3)), time_index=("a", "b"),
        factor_axis=_F3, time_axis=_T2,
    )
    payload = artifact.to_dict()
    # Remove every free-form provenance key: axis identity must remain.
    payload["provenance"] = {}
    restored = MetricArtifact.from_dict(payload)
    assert restored.factor_axis == _F3
    assert restored.time_axis == _T2


def test_factor_order_hash_distinguishes_factor_order_in_artifact():
    a = ScalarMetricArtifact(
        metric_id="m", domain="d", artifact_kind="scalar",
        values=np.array([0.1, 0.2, 0.3]),
        factor_axis=FactorAxisRef(("f1", "f2", "f3")),
    )
    b = ScalarMetricArtifact(
        metric_id="m", domain="d", artifact_kind="scalar",
        values=np.array([0.1, 0.2, 0.3]),
        factor_axis=FactorAxisRef(("f3", "f2", "f1")),
    )
    assert a.factor_axis.factor_order_hash != b.factor_axis.factor_order_hash
    assert a != b  # axis identity is part of artifact equality


# ---------------------------------------------------------------------------
# QE-P0-07: sealed registry authority.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _registry_restore():
    """Record pre-test registry state; restore if the test sealed it.

    Tests that exercise the seal lifecycle intentionally mutate the shared
    registry module (BUILDING -> SEALED is irreversible by design).  We restore
    by re-running the module's registration block from a fresh import so
    later tests keep a BUILDING registry.
    """
    import importlib
    state_before = registry_state()
    yield
    if registry_state() == "sealed" and state_before == "building":
        importlib.reload(registry_metrics)
        assert registry_state() == "building", "registry reload must restore BUILDING state"


def test_catalog_is_read_only_view_even_before_seal():
    snap = catalog_snapshot()
    assert isinstance(snap, type(CANONICAL_METRIC_ALIASES))  # MappingProxyType
    with pytest.raises(TypeError):
        snap["__new__"] = None  # type: ignore[index]


def test_catalog_snapshot_and_seal_do_not_export_public_mutable_dict():
    import quant_evaluator.registry.metrics as rm
    # The backing dict is private.
    assert not hasattr(rm, "CATALOG")
    assert not hasattr(rm, "METRIC_CATALOG")
    snap = catalog_snapshot()
    with pytest.raises(TypeError):
        snap["__new__"] = "no"  # type: ignore[index]


def test_duplicate_registration_before_seal_fails():
    dup = MetricSpec(
        name="mean_ic", display_name="Duplicate", description="duplicate",
        status=MetricStatus.EXPERIMENTAL, tier=MetricTier.CORE,
        compute_fn=lambda *a, **k: None,
    )
    with pytest.raises(ValueError):
        register_metric(dup)


def test_register_seal_mutation_fails():
    fresh = MetricSpec(
        name="__qe_seal_probe__", display_name="Probe", description="probe",
        status=MetricStatus.EXPERIMENTAL, tier=MetricTier.CORE,
        compute_fn=lambda *a, **k: None,
    )
    register_metric(fresh)
    assert get_metric("__qe_seal_probe__").name == "__qe_seal_probe__"

    assert seal_metric_registry() == "sealed"
    assert registry_state() == "sealed"

    # After seal: catalog is an immutable view.
    snap = catalog_snapshot()
    assert isinstance(snap, type(CANONICAL_METRIC_ALIASES))
    with pytest.raises(TypeError):
        snap["__new__"] = None  # type: ignore[index]

    # After seal: further registration fails closed.
    with pytest.raises(RuntimeError):
        register_metric(
            MetricSpec(
                name="__qe_seal_probe_2__", display_name="Probe",
                description="probe", status=MetricStatus.EXPERIMENTAL,
                tier=MetricTier.CORE, compute_fn=lambda *a, **k: None,
            )
        )
    # The sealed catalog still serves reads.
    assert get_metric("mean_ic").name == "mean_ic"


def test_metric_spec_has_metric_version_default():
    for name in list_metrics():
        spec = get_metric(name)
        assert isinstance(spec.metric_version, str)
        assert spec.metric_version.strip()
    assert get_metric("mean_ic").metric_version == "1"


# ---------------------------------------------------------------------------
# QE-P0-07: RankIC/Pearson alias DAG verification.
# ---------------------------------------------------------------------------

RANK_ALIASES = [
    "ic.rank.daily", "ic.rank.mean", "ic.rank.median", "ic.rank.std",
    "ic.rank.ir", "ic.rank.hac_t", "ic.rank.hac_p",
]
PEARSON_ALIASES = [
    "ic.pearson.daily", "ic.pearson.mean", "ic.pearson.std", "ic.pearson.ir",
]


def test_every_alias_resolves_to_exactly_one_registered_spec():
    for alias in RANK_ALIASES + PEARSON_ALIASES:
        resolved = resolve_alias(alias)
        # resolve_alias maps unknown -> identity; the alias must be known.
        assert resolved != alias, f"{alias} is not a registered alias"
        spec = get_metric(resolved)  # raises KeyError if not registered
        assert spec.name == resolved
        # Exactly one registry entry holds that spec name.
        assert list_metrics().count(resolved) == 1


def test_rank_family_declares_spearman_pearson_family_declares_pearson():
    for alias in RANK_ALIASES:
        assert get_metric(resolve_alias(alias)).ic_method == "spearman", alias
    for alias in PEARSON_ALIASES:
        assert get_metric(resolve_alias(alias)).ic_method == "pearson", alias


def test_ic_pearson_mean_resolves_to_pearson_ic_not_mean_ic():
    """Single DAG: the pearson.mean alias is its OWN spec (pearson_ic)."""
    assert resolve_alias("ic.pearson.mean") == "pearson_ic"
    assert get_metric(resolve_alias("ic.pearson.mean")).compute_fn is not None
    # Both pearson_ic and mean_ic exist as distinct registry entries.
    assert "pearson_ic" in list_metrics()
    assert "mean_ic" in list_metrics()
    assert get_metric("pearson_ic").name != get_metric("mean_ic").name


def test_rank_mean_resolves_to_rank_ic_and_declares_spearman():
    assert resolve_alias("ic.rank.mean") == "rank_ic"
    assert get_metric("rank_ic").ic_method == "spearman"
