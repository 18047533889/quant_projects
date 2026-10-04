"""Behavioral tests for FO research-batch memory admission."""
from pathlib import Path

import pytest


def _api():
    import importlib

    try:
        return importlib.import_module("factor_optimizer.research_memory_admission")
    except ModuleNotFoundError:
        pytest.fail("research memory admission API is not implemented")


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_estimate_covers_output_freeze_and_one_sequential_factor_workset():
    api = _api()
    assert api.estimate_incremental_peak_bytes((2, 3, 4)) == (
        18 * 2 * 3 * 4 + 128 * 2 * 3 + 256 * 1024**2 + 64 * 499
    )


def test_bootstrap_draw_estimate_and_validation():
    api = _api()
    base = api.estimate_incremental_peak_bytes((2, 3, 4))
    assert api.estimate_incremental_peak_bytes((2, 3, 4), bootstrap_draws=500) == base + 64
    for draws in (0, -1, True, 1.0):
        with pytest.raises((TypeError, ValueError)):
            api.estimate_incremental_peak_bytes((2, 3, 4), bootstrap_draws=draws)


def test_huge_bootstrap_draw_budget_rejects_before_reading_host_memory(monkeypatch):
    api = _api()
    def unexpected_read():
        pytest.fail("memory reader ran before budget rejection")
    monkeypatch.setattr(api, "available_memory_bytes", unexpected_read)
    with pytest.raises(MemoryError):
        api.require_memory_admission(
            (1, 1, 1), max_additional_peak_bytes=64 * 1024**3,
            reserve_bytes=0, bootstrap_draws=10**12,
        )


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, -1, 3), (2, 3), (True, 2, 3)])
def test_estimate_rejects_invalid_or_ambiguous_shape(shape):
    with pytest.raises((TypeError, ValueError)):
        _api().estimate_incremental_peak_bytes(shape)


def test_admission_uses_incremental_headroom_without_counting_input_twice():
    api = _api()
    shape = (2, 3, 4)
    estimate = api.estimate_incremental_peak_bytes(shape)
    assert api.require_memory_admission(
        shape, max_additional_peak_bytes=estimate,
        reserve_bytes=17, available_bytes=estimate + 17,
    ) == estimate
    with pytest.raises(MemoryError):
        api.require_memory_admission(
            shape, max_additional_peak_bytes=estimate - 1,
            reserve_bytes=0, available_bytes=estimate * 10,
        )
    with pytest.raises(MemoryError):
        api.require_memory_admission(
            shape, max_additional_peak_bytes=estimate,
            reserve_bytes=17, available_bytes=estimate + 16,
        )


def test_admission_fails_closed_when_available_memory_is_unknown():
    with pytest.raises(RuntimeError):
        _api().require_memory_admission(
            (1, 1, 1), max_additional_peak_bytes=1024**3,
            reserve_bytes=0, available_bytes=None,
        )


def test_cgroup_v2_headroom_uses_every_ancestor_limit(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/team/job/worker\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname, current, maximum, high in (
        ("", 100, "900", "800"),
        ("team", 250, "700", "600"),
        ("team/job", 300, "500", "450"),
        ("team/job/worker", 50, "max", "max"),
    ):
        directory = mount / dirname
        _write(directory / "memory.current", f"{current}\n")
        _write(directory / "memory.max", f"{maximum}\n")
        _write(directory / "memory.high", f"{high}\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) == 150


def test_cgroup_v2_missing_ancestor_limit_is_unknown_not_unlimited(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    _write(mount / "memory.current", "1\n")
    _write(mount / "memory.max", "max\n")
    _write(mount / "memory.high", "max\n")
    _write(mount / "team/memory.current", "1\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) is None


def test_true_hierarchy_root_may_omit_limit_files(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname, current, maximum, high in (
        ("team", 250, "700", "600"),
        ("team/job", 300, "500", "450"),
    ):
        directory = mount / dirname
        _write(directory / "memory.current", f"{current}\n")
        _write(directory / "memory.max", f"{maximum}\n")
        _write(directory / "memory.high", f"{high}\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) == 150


def test_true_root_limit_permission_error_fails_closed(tmp_path, monkeypatch):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname, current, maximum, high in (
        ("", 100, "900", "800"),
        ("team", 250, "700", "600"),
        ("team/job", 300, "500", "450"),
    ):
        directory = mount / dirname
        _write(directory / "memory.current", f"{current}\n")
        _write(directory / "memory.max", f"{maximum}\n")
        _write(directory / "memory.high", f"{high}\n")
    original_read_text = Path.read_text
    def deny_root_max(path, *args, **kwargs):
        if path == mount / "memory.max":
            raise PermissionError("test root limit is unreadable")
        return original_read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", deny_root_max)
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) is None


def test_all_unlimited_visible_v2_ancestors_leave_host_headroom(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 12345 kB\n")
    _write(proc / "self/cgroup", "0::/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname in ("team", "team/job"):
        directory = mount / dirname
        _write(directory / "memory.current", "1\n")
        _write(directory / "memory.max", "max\n")
        _write(directory / "memory.high", "max\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) == 12345 * 1024


def test_leaf_missing_memory_max_is_unknown_not_unlimited(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname in ("", "team", "team/job"):
        directory = mount / dirname
        _write(directory / "memory.current", "1\n")
        _write(directory / "memory.high", "max\n")
    _write(mount / "memory.max", "max\n")
    _write(mount / "team/memory.max", "max\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) is None


def test_unsupported_cgroup_v1_does_not_assume_host_ram_is_unlimited(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "5:memory:/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(sysroot / "fs/cgroup/memory")
        + " rw,nosuid - cgroup cgroup rw,memory\n"
    ))
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) is None


def test_namespace_mount_root_with_hidden_parent_limits_fails_closed(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/tenant/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 /tenant " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname in ("", "team", "team/job"):
        directory = mount / dirname
        _write(directory / "memory.current", "1\n")
        _write(directory / "memory.max", "max\n")
        _write(directory / "memory.high", "max\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) is None


def test_negative_cgroup_current_is_unknown(tmp_path):
    api = _api()
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    mount = sysroot / "fs/cgroup"
    _write(proc / "meminfo", "MemAvailable: 1000000 kB\n")
    _write(proc / "self/cgroup", "0::/team/job\n")
    _write(proc / "self/mountinfo", (
        "29 23 0:26 / " + str(mount) + " rw,nosuid - cgroup2 cgroup rw\n"
    ))
    for dirname in ("", "team"):
        _write(mount / dirname / "memory.current", "1\n")
        _write(mount / dirname / "memory.max", "max\n")
        _write(mount / dirname / "memory.high", "max\n")
    _write(mount / "team/job/memory.current", "-1\n")
    _write(mount / "team/job/memory.max", "100\n")
    _write(mount / "team/job/memory.high", "max\n")
    assert api.available_memory_bytes(proc_root=proc, sys_root=sysroot) is None
