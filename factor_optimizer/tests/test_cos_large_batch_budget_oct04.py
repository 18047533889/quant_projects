"""Explicit large research budgets preserve default admission and ordering."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "large_cos_audit", Path(__file__).parents[1] / "examples/cos_batch_audit.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
MIB = 1024**2


def records(count, size):
    sha = "a" * 64
    return {f"f{i:02d}": dict(
        uri=f"{example.POOL}/{sha}/f{i:02d}.parquet", sha256=sha,
        bytes=size, verified=True, status="evaluated_optimization_pending")
        for i in reversed(range(count))}


def test_default_object_and_batch_limits_remain_restrictive():
    rows = [{"factors": records(3, 48*MIB)}]
    with pytest.raises(ValueError):
        example.select_manifest_records(rows, 1)
    with pytest.raises(ValueError):
        example.select_manifest_records(rows, 3, max_factor_bytes=64*MIB)


def test_explicit_large_budget_retains_lexicographic_selection():
    data = records(16, 96*MIB)
    data["f15"]["bytes"] = 1  # no implicit smallest-object prioritization
    selected = example.select_manifest_records(
        [{"factors": data}], 16, max_factor_bytes=128*MIB,
        max_batch_factor_bytes=2048*MIB)
    assert [name for name, _ in selected] == sorted(data)
    with pytest.raises(ValueError):
        example.select_manifest_records(
            [{"factors": data}], 16, max_factor_bytes=128*MIB,
            max_batch_factor_bytes=1024*MIB)


@pytest.mark.parametrize("cap", [True, 0, -1, 2048*MIB+1, 1.5])
def test_batch_budget_invalid_or_above_hard_ceiling_is_rejected(cap):
    with pytest.raises(ValueError):
        example.select_manifest_records(
            [{"factors": records(1, 1)}], 1,
            max_batch_factor_bytes=cap)


@pytest.mark.parametrize("cap", [1, 8*MIB-1, 128*MIB-1])
def test_loader_preserves_non_integral_mib_byte_limits_before_io(monkeypatch, cap):
    def stop_at_engine(*args, **kwargs):
        raise RuntimeError("validated budget reached engine boundary")

    monkeypatch.setattr(example, "DuckDBEngine", stop_at_engine)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
    with pytest.raises(RuntimeError, match="reached engine boundary"):
        example.load_cos_sample(max_factor_bytes=cap)
