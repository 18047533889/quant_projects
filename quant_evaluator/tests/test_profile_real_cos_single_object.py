import hashlib

import pandas as pd
import pyarrow as pa
import pytest

from quant_evaluator.scripts.profile_real_cos_single_object import (
    PhaseLedger,
    _TimedHasher,
    _TimedRead,
    _profile_axes,
    profile_f61,
)


def test_phase_ledger_counts_calls_and_time():
    ledger = PhaseLedger()
    ledger.add("head_s", 0.125)
    ledger.add("head_s", 0.375)
    assert ledger.seconds["head_s"] == pytest.approx(0.5)
    assert ledger.counts["head_s"] == 2


def test_timed_hash_and_read_preserve_results(tmp_path):
    ledger = PhaseLedger()
    payload = b"bounded-profile"
    timed_hash = _TimedHasher(hashlib.sha256(), ledger)
    timed_hash.update(payload)
    assert timed_hash.hexdigest() == hashlib.sha256(payload).hexdigest()
    path = tmp_path / "object.parquet"
    path.write_bytes(payload)
    with _TimedRead(path.open("rb"), ledger) as stream:
        assert stream.read(4) + stream.read() == payload
    assert ledger.counts["hash_compute_s"] == 2
    assert ledger.counts["hash_file_read_s"] == 2


def test_axis_profile_normalizes_and_hashes_axes():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02 09:30", "2024-01-01 09:30"]),
        "000001.SZ": [2.0, 1.0],
        "metadata": [9.0, 8.0],
    })
    axis, phases = _profile_axes(pa.Table.from_pandas(frame, preserve_index=False))
    assert (axis["rows"], axis["columns"]) == (2, 1)
    assert axis["first_date"] == "2024-01-01T00:00:00"
    assert axis["last_date"] == "2024-01-02T00:00:00"
    assert len(axis["axis_sha256"]) == 64
    assert set(phases) == {
        "arrow_to_pandas_s", "axis_normalize_filter_sort_s", "axis_hash_s"}


def test_profile_rejects_object_cap_above_hard_limit():
    with pytest.raises(ValueError, match="1..128"):
        profile_f61("0" * 64, max_object_mib=129)
