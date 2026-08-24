"""
QE R21 sealed-split overlap guard (audit Q5 close).

Closes the quant_evaluator-side gap found by the LEAKAGE/PIT audit
(evidence/r2/R21-LEAKAGE-PIT-AUDIT.yaml Q5): FO has
``SearchSession.freeze_for_sealed_test`` rejecting overlapping sealed plans
(factor_optimizer/search/runner.py), but QE had NO sealed/snapshot/split
concept, so a QE evaluation could be pointed at a label split overlapping the
factor/label information boundary with nothing stopping it.

Contract under test:

- ``EvaluationRequest.split_ref`` (default ``None`` = no check, backward
  compatible) + ``EvaluationBundle.split_ref`` mirror.
- ``contracts/sealed_split.py``: ``SealedSplitRef`` (split id + the data
  window it covers: ``start_time``/``end_time`` inclusive, or ``as_of``) and
  the fail-closed ``check_sealed_split_overlap`` guard wired into the public
  ``runtime.evaluate`` facade.
- Overlap semantics (fail-closed): a sealed split window must be strictly
  after ALL factor/label decision times (``t >= start`` is an overlap,
  matching FO ``_sealed_test_disjoint``) and strictly after every label
  window's end (forward-label reach); unorderable/empty reference windows
  raise rather than silently disarming the gate.  The typed error is
  ``SealedSplitOverlapError`` (subclass of ``TimingContractError`` and
  ``ValueError``).

Importable source is ``build/lib/quant_evaluator`` (repo-root
``quant_evaluator/`` is a stub that shadows build/lib under pytest); same
bootstrap pattern as ``test_metamorphic.py``.

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest quant_evaluator/tests/test_sealed_split_overlap.py -v --tb=short
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Importable-source bootstrap (identical pattern to test_metamorphic.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   # quant_projects root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.api.requests import EvaluationBundle, EvaluationRequest
from quant_evaluator.contracts.errors import (
    SealedSplitOverlapError,
    TimingContractError,
)
from quant_evaluator.contracts.sealed_split import (
    SealedSplitRef,
    check_sealed_split_overlap,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

T, N, F = 8, 40, 1
SEED = 20260821
_DECISION = tuple(range(T))
_LABEL_START = tuple(range(T))
_LABEL_END = tuple(range(1, T + 1))


def _panel():
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    return values, labels


def _batch(values):
    return FactorBatch(
        factor_ids=("f0",),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
    )


def _bundle(labels):
    return LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=_DECISION,
        label_start_time=_LABEL_START,
        label_end_time=_LABEL_END,
    )


# ---------------------------------------------------------------------------
# 1. Backward compatibility: no split_ref -> evaluation works, no check.
# ---------------------------------------------------------------------------

def test_request_without_split_ref_evaluates():
    """A plain EvaluationRequest (split_ref default None) evaluates fine."""
    values, labels = _panel()
    request = EvaluationRequest(batch_or_factor_ids=_batch(values), label_bundle=_bundle(labels))
    assert request.split_ref is None
    bundle = evaluate(request)
    assert isinstance(bundle, EvaluationBundle)
    assert bundle.split_ref is None
    assert "rank_ic" in bundle.metric_values


def test_facade_without_split_ref_evaluates():
    """The positional facade (no request object at all) is unaffected."""
    values, labels = _panel()
    bundle = evaluate(_batch(values), _bundle(labels))
    assert bundle.split_ref is None
    assert "coverage" in bundle.metric_values


# ---------------------------------------------------------------------------
# 2. Non-overlapping split window -> works, split_ref round-trips to bundle.
# ---------------------------------------------------------------------------

def test_request_with_non_overlapping_split_ref_evaluates():
    """A sealed split strictly after the label window passes the guard."""
    values, labels = _panel()
    # decision_time 0..7, label_end 1..8: the sealed segment starts at 9,
    # strictly after every decision time and every label window end.
    split = SealedSplitRef(split_id="sealed_test_01", start_time=9, end_time=12)
    request = EvaluationRequest(
        batch_or_factor_ids=_batch(values),
        label_bundle=_bundle(labels),
        split_ref=split,
    )
    bundle = evaluate(request)
    assert isinstance(bundle, EvaluationBundle)
    assert bundle.split_ref is split
    assert "rank_ic" in bundle.metric_values


def test_facade_split_ref_kwarg_evaluates():
    """The facade accepts split_ref directly (request-object path is optional)."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="sealed_test_02", start_time=9, end_time=12)
    bundle = evaluate(
        _batch(values), _bundle(labels), split_ref=split,
        metrics=("coverage",),
    )
    assert bundle.split_ref is split
    assert "coverage" in bundle.metric_values


def test_as_of_boundary_is_not_an_overlap():
    """An as_of exactly at the last label window end is still disjoint."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="sealed_test_asof", as_of=8)
    check_sealed_split_overlap(
        split,
        decision_times=_DECISION,
        label_start_times=_LABEL_START,
        label_end_times=_LABEL_END,
    )  # must not raise


# ---------------------------------------------------------------------------
# 3. Overlapping split window -> typed ValueError, fail closed.
# ---------------------------------------------------------------------------

def test_split_window_overlapping_decision_times_raises():
    """A split whose start touches the factor/label window is rejected."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="leaky_split", start_time=4, end_time=6)
    request = EvaluationRequest(
        batch_or_factor_ids=_batch(values),
        label_bundle=_bundle(labels),
        split_ref=split,
    )
    with pytest.raises(SealedSplitOverlapError) as excinfo:
        evaluate(request)
    assert "leaky_split" in str(excinfo.value)


def test_split_window_containing_all_data_raises():
    """A split covering the entire history is rejected."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="full_history", start_time=0, end_time=10)
    with pytest.raises(SealedSplitOverlapError):
        evaluate(_batch(values), _bundle(labels), split_ref=split)


def test_label_forward_reach_into_split_raises():
    """A label window ending after the split end is forward-label leakage."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="short_split", start_time=9, end_time=10)
    # All decision times (0..7) and label starts are before the split start,
    # but the final label window ENDS at 11, past the split end 10: the
    # forward label reaches into the sealed segment.
    bundle = LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=_DECISION,
        label_start_time=_LABEL_START,
        label_end_time=tuple(range(1, T + 1))[:-1] + (11,),
    )
    with pytest.raises(SealedSplitOverlapError):
        evaluate(_batch(values), bundle, split_ref=split)


def test_factor_time_at_split_start_raises():
    """Explicit factor time coordinates at/after the split start are an overlap."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="factor_collision", start_time=6, end_time=10)
    batch = FactorBatch(
        factor_ids=("f0",),
        time_axis=AxisRef("t", "int", T, values=np.arange(T)),
        asset_axis=AxisRef("a", "str", N),
        values=values,
    )
    with pytest.raises(SealedSplitOverlapError):
        evaluate(batch, _bundle(labels), split_ref=split)


def test_error_is_typed_value_error():
    """The overlap error is both a TimingContractError and a ValueError."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="typed_check", start_time=3, end_time=5)
    try:
        evaluate(_batch(values), _bundle(labels), split_ref=split)
    except SealedSplitOverlapError as exc:
        assert isinstance(exc, TimingContractError)
        assert isinstance(exc, ValueError)
    else:  # pragma: no cover - test must fail if nothing raised
        pytest.fail("expected SealedSplitOverlapError")


def test_unorderable_reference_times_fail_closed():
    """A reference that cannot be proven disjoint must raise, not skip."""
    values, labels = _panel()
    split = SealedSplitRef(split_id="str_vs_int", start_time="a", end_time="z")
    with pytest.raises(SealedSplitOverlapError):
        evaluate(_batch(values), _bundle(labels), split_ref=split)


def test_inverted_split_window_rejected_at_construction():
    """start > end is malformed and never silently disarms the gate."""
    with pytest.raises(ValueError):
        SealedSplitRef(split_id="inverted", start_time=5, end_time=2)
    with pytest.raises(ValueError):
        SealedSplitRef(split_id="empty_ref")
    with pytest.raises(ValueError):
        SealedSplitRef(split_id="both_forms", start_time=1, end_time=2, as_of=1)


# ---------------------------------------------------------------------------
# 4. Serialization: to_dict/from_dict + pickle round-trips.
# ---------------------------------------------------------------------------

def test_sealed_split_ref_dict_roundtrip():
    ref = SealedSplitRef(
        split_id="sealed_test_dict", start_time=9, end_time=12,
        metadata={"search_session_id": "ss_1"},
    )
    restored = SealedSplitRef.from_dict(ref.to_dict())
    assert restored == ref
    assert restored.split_id == ref.split_id
    assert restored.window == ref.window
    assert dict(restored.metadata) == {"search_session_id": "ss_1"}


def test_sealed_split_ref_as_of_dict_roundtrip():
    ref = SealedSplitRef(split_id="asof_dict", as_of=8)
    restored = SealedSplitRef.from_dict(ref.to_dict())
    assert restored == ref
    assert restored.as_of == 8
    assert restored.window == (8, 8)


def test_sealed_split_ref_pickle_roundtrip():
    ref = SealedSplitRef(
        split_id="sealed_test_pickle", start_time=9, end_time=12,
        metadata={"search_session_id": "ss_2"},
    )
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(ref, protocol=proto))
        assert restored == ref
        assert restored.window == ref.window


def test_request_dict_roundtrip_preserves_split_ref():
    values, labels = _panel()
    split = SealedSplitRef(split_id="req_dict", start_time=9, end_time=12)
    request = EvaluationRequest(
        batch_or_factor_ids="dummy", label_bundle="dummy",
        split_ref=split,
        metric_ids=("rank_ic",),
        metadata={"k": "v"},
    )
    restored = EvaluationRequest.from_dict(request.to_dict())
    assert restored.split_ref == split
    assert restored.split_ref.window == (9, 12)
    assert restored.metric_ids == ("rank_ic",)
    assert restored.metadata == {"k": "v"}
    assert restored.batch_or_factor_ids is None  # not carried by to_dict


def test_request_without_split_ref_dict_roundtrip():
    request = EvaluationRequest(batch_or_factor_ids="dummy", label_bundle="dummy")
    restored = EvaluationRequest.from_dict(request.to_dict())
    assert restored.split_ref is None


def test_request_pickle_roundtrip_preserves_split_ref():
    split = SealedSplitRef(split_id="req_pickle", start_time=9, end_time=12)
    request = EvaluationRequest(
        batch_or_factor_ids="dummy", label_bundle="dummy", split_ref=split
    )
    restored = pickle.loads(pickle.dumps(request))
    assert restored.split_ref == split


def test_bundle_split_ref_roundtrip_via_pickle_and_evaluate():
    values, labels = _panel()
    split = SealedSplitRef(split_id="bundle_pickle", start_time=9, end_time=12)
    bundle = evaluate(
        _batch(values), _bundle(labels), split_ref=split, metrics=("coverage",)
    )
    assert bundle.split_ref is split
    restored = pickle.loads(pickle.dumps(bundle))
    assert restored.split_ref == split
    assert restored.request_id == bundle.request_id
    assert restored.metric_values == bundle.metric_values


def test_check_helper_rejects_overlap_directly():
    """The guard helper itself is fail-closed on a touching window."""
    with pytest.raises(SealedSplitOverlapError):
        check_sealed_split_overlap(
            SealedSplitRef(split_id="touch", start_time=7, end_time=9),
            decision_times=_DECISION,
            label_start_times=_LABEL_START,
            label_end_times=_LABEL_END,
        )


def test_check_helper_none_is_noop():
    check_sealed_split_overlap(
        None,
        decision_times=_DECISION,
        label_start_times=_LABEL_START,
        label_end_times=_LABEL_END,
    )  # must not raise
