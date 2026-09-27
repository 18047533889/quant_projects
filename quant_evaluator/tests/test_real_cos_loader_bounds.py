"""Research COS factor loader bounds fail before remote reads."""

import pytest

from quant_evaluator.scripts.load_real_cos_factor_batch import _factors, SHA


@pytest.mark.parametrize("count,max_mib,total_mib", [
    (0, 64, 256),
    (33, 64, 256),
    (True, 64, 256),
    (2, 0, 256),
    (2, 129, 256),
    (2, True, 256),
    (2, 64, 0),
    (2, 64, 2049),
    (2, 64, True),
])
def test_research_loader_rejects_out_of_bounds_before_io(count, max_mib, total_mib):
    with pytest.raises(ValueError):
        _factors(count, max_mib, SHA, total_mib)


def test_research_loader_rejects_unbound_manifest_before_io():
    with pytest.raises(ValueError, match="manifest_sha256"):
        _factors(32, 128, "not-a-digest", 2048)
