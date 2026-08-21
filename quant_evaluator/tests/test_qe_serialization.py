"""
QE-P0 — serialization / persistence of QE registry metrics and metric
artifacts (review QE-P0, "MappingProxyType not picklable").

Proves three things:

1. **MetricArtifact pickle round-trip.** Every :class:`MetricArtifact`
   subclass (scalar / series / vector / matrix / distribution) survives
   ``pickle.dumps``/``pickle.loads`` losslessly — identity, payload arrays,
   and provenance all round-trip.
2. **Registry MappingProxyType does not break pickle.** The registry's
   ``CANONICAL_METRIC_ALIASES`` is a ``types.MappingProxyType`` (read-only),
   which the stdlib pickle cannot handle by default. The product fix in
   ``registry/metrics.py`` registers a ``copyreg.pickle`` reducer for
   ``mappingproxy`` that converts to a plain dict on the way out and restores
   a read-only ``mappingproxy`` on the way back. This test pins that a
   mappingproxy survives pickle losslessly AND that the round-tripped object
   is still read-only, wherever it is embedded (alone, in a plain dict, or in
   a ``StreamingEvaluationResult``). The streaming result's own default
   ``provenance`` is also covered (its dataclass default used to be a fresh
   empty mappingproxy per instance, which broke pickle of an otherwise-empty
   result).
3. **to_dict / from_dict round-trip.** All four formal artifact types in
   ``contracts/artifact_types.py`` (ICSeriesArtifact, QuantileReturnArtifact,
   ProbePortfolioArtifact, ExposureArtifact) and every MetricArtifact subclass
   round-trip through ``to_dict``/``from_dict``, preserving identity (== and
   stable hash) and payload/provenance.

Importable source is ``build/lib/quant_evaluator`` (the repo-root
``quant_evaluator/`` directory is a stub). The bootstrap below forces the
build/lib package + subpackages onto ``sys.modules`` directly (same pattern
as ``test_metamorphic.py``) and prepends build/lib to ``sys.path`` for
intra-package imports.

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest quant_evaluator/tests/test_qe_serialization.py -v --tb=short
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
                 "reporting", "contracts"):
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

# ---------------------------------------------------------------------------
# Imports (all resolved against the pinned build/lib package).
# ---------------------------------------------------------------------------
from quant_evaluator.registry.metrics import CANONICAL_METRIC_ALIASES, get_metric
from quant_evaluator.runtime.streaming_evaluator import StreamingEvaluationResult

from quant_evaluator.contracts.artifact_types import (
    ExposureArtifact,
    ICSeriesArtifact,
    ProbePortfolioArtifact,
    QuantileReturnArtifact,
)
from quant_evaluator.contracts.metric_artifacts import (
    DistributionMetricArtifact,
    MatrixMetricArtifact,
    MetricArtifact,
    ScalarMetricArtifact,
    SeriesMetricArtifact,
    VectorMetricArtifact,
)

ARTIFACT_TYPE_CLASSES = (
    ICSeriesArtifact,
    QuantileReturnArtifact,
    ProbePortfolioArtifact,
    ExposureArtifact,
)

METRIC_ARTIFACT_CLASSES = (
    ScalarMetricArtifact,
    SeriesMetricArtifact,
    VectorMetricArtifact,
    MatrixMetricArtifact,
    DistributionMetricArtifact,
)

_NAN_AWARE_TOL = dict(equal_nan=True)


# ---------------------------------------------------------------------------
# Fixtures: one sample instance per artifact class.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifact_samples():
    return {
        "ICSeriesArtifact": ICSeriesArtifact(
            values=np.array([[0.5, -0.2], [np.nan, 0.3]]),
            time_index=("2024-01-02", "2024-01-03"),
            ic_method="pearson",
            factor_ids=("f1", "f2"),
            metric_id="ic.daily",
            provenance={"src": "qe-test", "config_hash": "abc123"},
        ),
        "QuantileReturnArtifact": QuantileReturnArtifact(
            values=np.array([[0.10, -0.05], [0.02, 0.01]]),
            n_quantiles=2,
            factor_ids=("f1", "f2"),
            metric_id="quantile_returns",
            provenance={"src": "qe-test", "n_quantiles": 2},
        ),
        "ProbePortfolioArtifact": ProbePortfolioArtifact(
            values=np.array([[1.5, 2.5], [1.6, 2.6]]),
            time_index=("2024-01-02", "2024-01-03"),
            factor_ids=("f1", "f2"),
            metric_id="probe_portfolio",
            provenance={"src": "qe-test", "weights": True},
        ),
        "ExposureArtifact": ExposureArtifact(
            values=np.array([[1.0, 0.0], [0.0, 1.0]]),
            exposure_type="loadings",
            factor_ids=("f1", "f2"),
            metric_id="exposure",
            provenance={"src": "qe-test", "method": "ols"},
        ),
        "ScalarMetricArtifact": ScalarMetricArtifact(
            metric_id="mean_ic",
            domain="ic",
            artifact_kind="scalar",
            values=np.array([0.1, 0.2, np.nan]),
            provenance={"src": "qe-test", "inputs": ["factor_batch", "label_bundle"]},
            created_from=("factor_batch", "label_bundle"),
        ),
        "SeriesMetricArtifact": SeriesMetricArtifact(
            metric_id="pearson_ic_series",
            domain="ic",
            artifact_kind="series",
            values=np.array([[0.1], [0.2], [np.nan]]),
            time_index=("2024-01-02", "2024-01-03", "2024-01-04"),
            provenance={"src": "qe-test"},
            created_from=("factor_batch", "label_bundle"),
        ),
        "VectorMetricArtifact": VectorMetricArtifact(
            metric_id="quantile_returns_full",
            domain="quantile",
            artifact_kind="vector",
            values=np.array([[0.1], [0.2], [0.05]]),
            provenance={"src": "qe-test"},
            created_from=("factor_batch", "label_bundle"),
        ),
        "MatrixMetricArtifact": MatrixMetricArtifact(
            metric_id="transition_matrix",
            domain="quantile",
            artifact_kind="matrix",
            values=np.array([[[0.8, 0.1], [0.1, 0.9]], [[0.9, 0.1], [0.1, 0.8]]]),
            provenance={"src": "qe-test"},
            created_from=("factor_batch", "label_bundle"),
        ),
        "DistributionMetricArtifact": DistributionMetricArtifact(
            metric_id="block_bootstrap_ci",
            domain="ic",
            artifact_kind="distribution",
            samples=np.array([[0.05, 0.15], [0.04, 0.16], [0.06, 0.14]]),
            stat_names=("mean_ic",),
            provenance={"src": "qe-test", "n_bootstrap": 100},
            created_from=("ICSeriesArtifact",),
        ),
    }


def _assert_same_value(left, right, label):
    assert type(left) is type(right), f"{label}: type {type(left).__name__} != {type(right).__name__}"
    assert left == right, f"{label}: value mismatch"
    assert hash(left) == hash(right), f"{label}: stable hash mismatch"
    # Identity of payload arrays: values equal elementwise (NaN-aware).
    if hasattr(left, "values"):
        assert np.array_equal(
            np.asarray(left.values), np.asarray(right.values), **(_NAN_AWARE_TOL)
        ), f"{label}: payload array mismatch"
    if hasattr(left, "samples"):
        assert np.array_equal(
            np.asarray(left.samples), np.asarray(right.samples), **(_NAN_AWARE_TOL)
        ), f"{label}: samples array mismatch"
    assert dict(left.provenance) == dict(right.provenance), f"{label}: provenance mismatch"


# ---------------------------------------------------------------------------
# 1. MetricArtifact pickle round-trip.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", METRIC_ARTIFACT_CLASSES, ids=lambda c: c.__name__)
def test_metric_artifact_pickle_roundtrip(cls, artifact_samples):
    obj = artifact_samples[cls.__name__]
    restored = pickle.loads(pickle.dumps(obj))
    _assert_same_value(obj, restored, f"MetricArtifact pickle ({cls.__name__})")
    # Metric identity / payload / provenance individually.
    assert restored.metric_id == obj.metric_id
    assert restored.domain == obj.domain
    assert restored.artifact_kind == obj.artifact_kind
    assert tuple(restored.created_from) == tuple(obj.created_from)
    assert dict(restored.provenance) == dict(obj.provenance)


@pytest.mark.parametrize("cls", METRIC_ARTIFACT_CLASSES, ids=lambda c: c.__name__)
def test_metric_artifact_pickle_protocols(cls, artifact_samples):
    obj = artifact_samples[cls.__name__]
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(obj, protocol=proto))
        _assert_same_value(obj, restored, f"MetricArtifact pickle p{proto} ({cls.__name__})")


# ---------------------------------------------------------------------------
# 2. Registry MappingProxyType does not break pickle.
# ---------------------------------------------------------------------------
def test_registry_aliases_mappingproxy_is_read_only():
    from types import MappingProxyType
    assert isinstance(CANONICAL_METRIC_ALIASES, MappingProxyType)
    with pytest.raises(TypeError):
        CANONICAL_METRIC_ALIASES["__new__"] = "no"  # type: ignore[index]
    with pytest.raises(TypeError):
        CANONICAL_METRIC_ALIASES["ic.rank.daily"] = "override"  # type: ignore[index]


def test_registry_aliases_mappingproxy_pickle_roundtrip():
    """The mappingproxy itself survives pickle and stays read-only."""
    restored = pickle.loads(pickle.dumps(CANONICAL_METRIC_ALIASES))
    assert isinstance(restored, type(CANONICAL_METRIC_ALIASES))
    assert dict(restored) == dict(CANONICAL_METRIC_ALIASES)
    with pytest.raises(TypeError):
        restored["__new__"] = "no"  # type: ignore[index]


@pytest.mark.parametrize("proto", range(5), ids=lambda p: f"p{p}")
def test_registry_aliases_mappingproxy_pickle_protocols(proto):
    restored = pickle.loads(pickle.dumps(CANONICAL_METRIC_ALIASES, protocol=proto))
    assert dict(restored) == dict(CANONICAL_METRIC_ALIASES)
    with pytest.raises(TypeError):
        restored["__new__"] = "no"  # type: ignore[index]


def test_registry_aliases_mappingproxy_embedded_in_dict_pickle():
    """A mappingproxy embedded in a bigger picklable object does not break it."""
    payload = {"aliases": CANONICAL_METRIC_ALIASES, "tag": "qe", "n": 3}
    restored = pickle.loads(pickle.dumps(payload))
    assert restored["tag"] == "qe"
    assert isinstance(restored["aliases"], type(CANONICAL_METRIC_ALIASES))
    assert dict(restored["aliases"]) == dict(CANONICAL_METRIC_ALIASES)
    with pytest.raises(TypeError):
        restored["aliases"]["__new__"] = "no"  # type: ignore[index]


def test_metric_spec_pickle_roundtrip():
    """MetricSpec pickles losslessly with its compute_fn intact."""
    spec = get_metric("mean_ic")
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
    assert restored.compute_fn is spec.compute_fn
    assert restored.requires == spec.requires


def test_streaming_result_pickle_default_provenance():
    """StreamingEvaluationResult pickles even with its (now plain-dict) default provenance."""
    result = StreamingEvaluationResult()
    assert isinstance(result.provenance, dict)
    restored = pickle.loads(pickle.dumps(result))
    assert isinstance(restored.provenance, dict)
    assert restored.provenance == {}
    assert restored.metrics == {}
    assert restored.get_metric("mean_ic") is None


def test_streaming_result_pickle_with_mappingproxy_provenance():
    """StreamingEvaluationResult survives pickle with an embedded mappingproxy provenance."""
    result = StreamingEvaluationResult()
    result.provenance = {"aliases": CANONICAL_METRIC_ALIASES, "parent": ("a", "b")}
    restored = pickle.loads(pickle.dumps(result))
    assert restored.provenance["parent"] == ("a", "b")
    assert dict(restored.provenance["aliases"]) == dict(CANONICAL_METRIC_ALIASES)


def test_streaming_result_pickle_with_full_provenance():
    """The evaluator's real stream provenance (MappingProxyType of tuples) survives pickle."""
    from quant_evaluator.runtime.streaming_evaluator import StreamingEvaluator
    evaluator = StreamingEvaluator()
    result = StreamingEvaluationResult()
    result.provenance = evaluator._stream_provenance(
        parent_identity=("id", 1),
        factor_ids=("f1", "f2"),
        asset_coords=(0, 1),
        timing_rule=(("0",), ("1",)),
        timing_rows=2,
        timing_first=(("a",), ("b",)),
        timing_last=(("a",), ("b",)),
    )
    restored = pickle.loads(pickle.dumps(result))
    assert dict(restored.provenance) == dict(result.provenance)
    assert restored.provenance["factor_ids"] == ("f1", "f2")


# ---------------------------------------------------------------------------
# 3. to_dict / from_dict round-trip for artifact types + MetricArtifact.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ARTIFACT_TYPE_CLASSES, ids=lambda c: c.__name__)
def test_artifact_type_to_from_dict_roundtrip(cls, artifact_samples):
    obj = artifact_samples[cls.__name__]
    restored = cls.from_dict(obj.to_dict())
    _assert_same_value(obj, restored, f"artifact_types to_dict/from_dict ({cls.__name__})")


@pytest.mark.parametrize("cls", METRIC_ARTIFACT_CLASSES, ids=lambda c: c.__name__)
def test_metric_artifact_to_from_dict_roundtrip(cls, artifact_samples):
    obj = artifact_samples[cls.__name__]
    restored = MetricArtifact.from_dict(obj.to_dict())
    _assert_same_value(obj, restored, f"MetricArtifact to_dict/from_dict ({cls.__name__})")
    assert isinstance(restored, cls)


@pytest.mark.parametrize("cls", METRIC_ARTIFACT_CLASSES, ids=lambda c: c.__name__)
def test_metric_artifact_to_dict_json_friendly(cls, artifact_samples):
    """to_dict output must be JSON-serializable (arrays become lists)."""
    import json
    obj = artifact_samples[cls.__name__]
    payload = obj.to_dict()
    json.dumps(payload)  # must not raise
    assert isinstance(payload, dict)
    assert "provenance" in payload
    assert isinstance(payload["provenance"], dict)
