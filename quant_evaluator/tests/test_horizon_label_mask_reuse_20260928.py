"""Frozen label-mask reuse without large panel copies."""

from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


def _label(values):
    array = getattr(values, "array", values)
    t, n = array.shape
    times = tuple(range(t))
    return LabelBundle(
        target_id="h1", values=values, horizon=1,
        decision_time=times, observation_time=times,
        label_start_time=tuple(i + 1 for i in times),
        label_end_time=tuple(i + 2 for i in times),
        asset_axis=AxisRef("asset", "str", n, np.asarray([f"S{i}" for i in range(n)])),
    )


def test_mask_clone_reuses_bytes_backed_values_and_matches_replace_hash():
    label = _label(np.arange(18, dtype=np.float64).reshape(6, 3))
    mask = np.ones(label.values.shape, dtype=bool)
    mask[1, 2] = False

    expected = replace(label, validity=mask)
    actual = label._with_validity_mask(mask)

    assert actual.values is label.values
    assert actual.content_hash == expected.content_hash
    assert actual.validity is not mask
    mask[0, 0] = False
    assert actual.validity[0, 0]
    with pytest.raises(ValueError):
        actual.values.flags.writeable = True
    with pytest.raises(ValueError):
        actual.validity.flags.writeable = True


def test_mask_clone_copies_values_without_immutable_bytes_owner():
    source = np.arange(18, dtype=np.float64).reshape(6, 3)
    label = _label(source)
    untrusted = object.__new__(LabelBundle)
    for name in label.__dataclass_fields__:
        if name != "content_hash":
            object.__setattr__(untrusted, name, getattr(label, name))
    object.__setattr__(untrusted, "values", source)
    mask = np.ones(label.values.shape, dtype=bool)

    expected = replace(untrusted, validity=mask)
    actual = untrusted._with_validity_mask(mask)

    assert actual.values is not source
    assert actual.content_hash == expected.content_hash
    assert not actual.values.flags.writeable
    source[0, 0] = -200.0
    assert actual.values[0, 0] == 0.0


def test_mask_clone_revalidates_timing_instead_of_trusting_source_state():
    label = _label(np.arange(18, dtype=np.float64).reshape(6, 3))
    untrusted = object.__new__(LabelBundle)
    for name in label.__dataclass_fields__:
        if name != "content_hash":
            object.__setattr__(untrusted, name, getattr(label, name))
    object.__setattr__(untrusted, "decision_time", (0, 2, 1, 3, 4, 5))
    mask = np.ones(label.values.shape, dtype=bool)

    with pytest.raises(ValueError, match="decision_time must be strictly increasing"):
        untrusted._with_validity_mask(mask)
