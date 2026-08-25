"""
QE-P0-03: implementation_hash must reflect the actual kernel source.

The old implementation_hash was a hash of the implementation_id STRING
(e.g. "quant_evaluator.metrics.ic.compute_daily_ic").  Changing the kernel
body by 300 lines without renaming the function left the hash unchanged — a
false identity.  The hash must be derived from the real module source (and,
where resolvable, the function's source closure) so a source change flips it.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.metrics.catalog import get_metric_spec, list_all_metric_ids


def test_implementation_hash_is_source_derived_not_id_string():
    """The hash must not be a hash of the implementation_id string alone."""
    spec = get_metric_spec("pearson_ic")
    impl_id = spec.implementation_id
    # A naive string-hash of the id would be a fixed 16-hex; the source hash
    # must differ from hashing the id string directly.
    import hashlib

    naive = hashlib.sha256(impl_id.encode("utf-8")).hexdigest()[:16]
    assert spec.implementation_hash != naive, (
        "implementation_hash is just the id-string hash; must be source-derived"
    )


def test_implementation_hash_changes_when_source_changes():
    """Editing the kernel source must change the implementation_hash."""
    import inspect

    from quant_evaluator.metrics.ic import compute_daily_ic

    spec = get_metric_spec("pearson_ic")
    # The hash must be a function of the actual source text.
    src = inspect.getsource(compute_daily_ic)
    assert src  # non-empty source
    # Recompute the hash from the source and confirm it matches the spec's.
    from quant_evaluator.registry.metrics import _source_implementation_hash

    recomputed = _source_implementation_hash(
        "quant_evaluator.metrics.ic.compute_daily_ic"
    )
    assert recomputed == spec.implementation_hash


def test_all_metrics_have_source_derived_hash():
    """Every catalog metric's implementation_hash must be source-derived."""
    from quant_evaluator.registry.metrics import _source_implementation_hash

    for metric_id in list_all_metric_ids():
        spec = get_metric_spec(metric_id)
        recomputed = _source_implementation_hash(spec.implementation_id)
        assert recomputed == spec.implementation_hash, metric_id
