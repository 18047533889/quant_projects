"""Successful evidence must describe actual finite output, not only a label."""
import gzip
import hashlib
import json

import pytest

from reconcile_r15_combined import _load_execution


def _fixture(tmp_path, **overrides):
    item = dict(source_row=1, id="f1", status="EXECUTED",
                executed_formula="add(close, 1)", value_count=8,
                finite_count=7, result_hash="a" * 64)
    item.update(overrides)
    path = tmp_path / "execution.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(item) + "\n")
    summary = dict(complete=True, processed=1, requested_limit=1,
                   counts={"EXECUTED": 1})
    path.with_suffix(".summary.json").write_text(json.dumps(summary))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_accepts_pinned_finite_success(tmp_path):
    path, digest = _fixture(tmp_path)
    records, _ = _load_execution([path], [digest])
    assert records[(1, "f1")][0]["finite_count"] == 7


@pytest.mark.parametrize("overrides", [
    {"value_count": 0}, {"finite_count": 0}, {"finite_count": 9},
    {"finite_count": True}, {"value_count": "8"},
    {"result_hash": ""}, {"result_hash": "z" * 64},
    {"result_hash": None},
])
def test_rejects_incomplete_or_nonfinite_success(tmp_path, overrides):
    path, digest = _fixture(tmp_path, **overrides)
    with pytest.raises(ValueError):
        _load_execution([path], [digest])


def test_rejects_wrong_pin_and_pin_count(tmp_path):
    path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="pinned hash"):
        _load_execution([path], ["0" * 64])
    with pytest.raises(ValueError, match="pin count"):
        _load_execution([path], [])


def test_rejects_conflicting_summary(tmp_path):
    path, digest = _fixture(tmp_path)
    summary_path = path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["counts"] = {"EXECUTION_FAILED": 1}
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary status"):
        _load_execution([path], [digest])
