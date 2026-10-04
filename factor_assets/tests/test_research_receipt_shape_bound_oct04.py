"""Bounded shape admission for research receipt contracts."""

import hashlib

import pytest

from factor_assets.contracts.research_feature_receipt import (
    ResearchFeatureReceipt,
    ResearchSourceBinding,
)


def _binding():
    digest = hashlib.sha256(b"shape-bound-fixture").hexdigest()
    return ResearchSourceBinding(
        factor_id="factor-a",
        manifest_uri="cos://research/manifest.json",
        manifest_sha256=digest,
        source_uri="cos://research/factor.parquet",
        source_sha256=digest,
        time_axis_hash=digest,
        asset_axis_hash=digest,
        values_hash=digest,
    )


def _receipt(shape):
    return ResearchFeatureReceipt(
        source_binding=_binding(),
        values_dtype="float64",
        validity_dtype=None,
        shape=shape,
        time_axis_name="time",
        time_axis_dtype="int64",
        asset_axis_name="asset",
        asset_axis_dtype="object",
        embedding=(0.25, -0.5),
        embedding_spec="research-embedding-v1",
        embedding_model_version="fixture-model-1",
    )


def test_infinite_shape_iterator_is_rejected_after_at_most_four_items():
    class SentinelShape:
        def __init__(self):
            self.reads = 0

        def __iter__(self):
            while True:
                self.reads += 1
                if self.reads > 4:
                    raise AssertionError("shape iterator was consumed past bounded probe")
                yield 1

    shape = SentinelShape()
    with pytest.raises(ValueError, match="shape must be positive integer T×N×1"):
        _receipt(shape)
    assert shape.reads == 4


def test_finite_shape_generator_retains_three_axis_acceptance():
    receipt = _receipt(dim for dim in (3, 5, 1))
    assert receipt.shape == (3, 5, 1)


@pytest.mark.parametrize("shape", [(True, 5, 1), (3, 0, 1), (3, 5, 2)])
def test_finite_invalid_shape_dimensions_fail_closed(shape):
    with pytest.raises(ValueError, match="shape must be positive integer T×N×1"):
        _receipt(shape)
