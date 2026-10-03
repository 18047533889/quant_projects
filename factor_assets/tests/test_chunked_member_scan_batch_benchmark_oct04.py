"""Tiny real scanner checks and fail-closed batch benchmark contracts."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/benchmark_chunked_member_scan_batch_oct04.py"
SPEC = importlib.util.spec_from_file_location("fa_batch_benchmark_contract", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_tiny_real_workers_preserve_exact_outputs_and_reduce_preparation():
    kwargs = dict(members=8, dimensions=4, queries=3, seed=5,
                  max_input_bytes=1024**2, max_chunk_rows=2,
                  max_chunk_bytes=2*4*8*3)
    single = BENCH._worker(mode="single", **kwargs)
    batch = BENCH._worker(mode="batch", **kwargs)
    assert single["input_fingerprint"] == batch["input_fingerprint"]
    assert single["winners"] == batch["winners"]
    assert single["preparation_counts"] == {
        "member_embedding_conversion_calls": 24,
        "member_normalize_rows_calls": 12,
        "member_normalized_rows": 24,
    }
    assert batch["preparation_counts"] == {
        "member_embedding_conversion_calls": 8,
        "member_normalize_rows_calls": 4,
        "member_normalized_rows": 8,
    }


def test_parity_rejects_even_one_ulp_difference():
    with pytest.raises(RuntimeError, match="score"):
        BENCH._assert_parity([["M00000", 0.5]],
                             [["M00000", float(np.nextafter(0.5, 1.))]])


@pytest.mark.parametrize("changes", [
    {"members": True}, {"members": 0}, {"queries": 257},
    {"dimensions": 0}, {"max_input_bytes": True},
])
def test_invalid_envelope_rejected(changes):
    kwargs = dict(members=8, dimensions=4, queries=3, max_input_bytes=1024**2)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        BENCH._validate_envelope(**kwargs)
