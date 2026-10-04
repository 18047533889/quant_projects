"""Conservative memory admission for FO research batch optimization.

The estimate describes incremental working set above the caller's already
resident input.  It is a shape-based admission envelope, not an RSS bound or
an absolute guarantee against allocation failure.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Iterable

_UNSET = object()
_MIB = 1024**2
_GIB = 1024**3
_FIXED_CACHE_DIAGNOSTIC_BYTES = 256 * _MIB
_SEQUENTIAL_FACTOR_BYTES_PER_CELL = 128
_OUTPUT_AND_FREEZE_BYTES_PER_CELL = 18
_UNLIMITED_THRESHOLD = 1 << 60
_MOUNT_ESCAPE = re.compile(r"\\(040|011|012|134)")


def _mount_unescape(value: str) -> str:
    replacements = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    return _MOUNT_ESCAPE.sub(lambda match: replacements[match.group(1)], value)


def _positive_shape(shape: Iterable[int]) -> tuple[int, int, int]:
    if isinstance(shape, (str, bytes)):
        raise TypeError("shape must contain exactly three positive integers")
    try:
        dims = tuple(shape)
    except TypeError as exc:
        raise TypeError("shape must contain exactly three positive integers") from exc
    if len(dims) != 3 or any(type(value) is not int or value < 1 for value in dims):
        raise ValueError("shape must contain exactly three positive integers")
    return dims


def estimate_incremental_peak_bytes(shape: Iterable[int], *, bootstrap_draws: int = 499) -> int:
    """Estimate additional peak bytes for shape ``(T, N, F)``.

    The envelope counts the output and its immutable freeze buffers (18 bytes
    per output cell), one sequential factor's raw/frame/search/scratch working
    set (128 bytes per input cell), a fixed 256 MiB cache/diagnostic reserve,
    and 64 bytes per retained bootstrap draw.
    The caller's existing input batch and labels are already reflected in live
    available memory and are deliberately not counted a second time here.
    """
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be a positive integer")
    times, assets, factors = _positive_shape(shape)
    cells = times * assets * factors
    one_factor_cells = times * assets
    return (
        _OUTPUT_AND_FREEZE_BYTES_PER_CELL * cells
        + _SEQUENTIAL_FACTOR_BYTES_PER_CELL * one_factor_cells
        + _FIXED_CACHE_DIAGNOSTIC_BYTES
        + 64 * bootstrap_draws
    )


def _host_memavailable_bytes(proc_root: Path) -> int | None:
    try:
        lines = (proc_root / "meminfo").read_text(encoding="ascii").splitlines()
        for line in lines:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) < 3 or fields[2] != "kB":
                    return None
                value = int(fields[1]) * 1024
                return value if value >= 0 else None
    except (OSError, UnicodeError, ValueError, IndexError):
        return None
    return None


def _cgroup_v2_headroom(proc_root: Path, sys_root: Path) -> int | None:
    """Return the tightest v2 memory.max/high headroom for this process."""
    try:
        memberships = (proc_root / "self/cgroup").read_text(encoding="ascii").splitlines()
        membership = None
        for line in memberships:
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
                membership = PurePosixPath(fields[2])
                break
        if membership is None or not membership.is_absolute() or ".." in membership.parts:
            return None

        mountinfo = (proc_root / "self/mountinfo").read_text(encoding="ascii").splitlines()
        mount_root = None
        mount_point = None
        for line in mountinfo:
            fields = line.split()
            separator = fields.index("-")
            if len(fields) <= separator + 2 or fields[separator + 1] != "cgroup2":
                continue
            mount_root = PurePosixPath(_mount_unescape(fields[3]))
            mount_point = Path(_mount_unescape(fields[4]))
            break
        if mount_root is None or mount_point is None or not mount_root.is_absolute():
            return None
        if mount_root != PurePosixPath("/"):
            return None

        try:
            relative = membership.relative_to(mount_root)
        except ValueError:
            return None
        if mount_point.is_relative_to(Path("/sys")) and sys_root != Path("/sys"):
            mount_point = sys_root / mount_point.relative_to(Path("/sys"))
        leaf = mount_point.joinpath(*relative.parts)
        try:
            leaf.relative_to(mount_point)
        except ValueError:
            return None

        available = None
        current_dir = leaf
        while True:
            is_visible_true_root = (
                current_dir == mount_point and mount_root == PurePosixPath("/")
            )
            limit_values = []
            for limit_name in ("memory.max", "memory.high"):
                try:
                    text = (current_dir / limit_name).read_text(encoding="ascii").strip()
                except FileNotFoundError:
                    if is_visible_true_root:
                        continue
                    return None
                except OSError:
                    return None
                if text == "max":
                    continue
                limit = int(text)
                if limit < 0:
                    return None
                if limit >= _UNLIMITED_THRESHOLD:
                    continue
                limit_values.append(limit)
            if limit_values:
                current = int((current_dir / "memory.current").read_text(encoding="ascii").strip())
                if current < 0:
                    return None
                for limit in limit_values:
                    remaining = max(0, limit - current)
                    available = remaining if available is None else min(available, remaining)
            if current_dir == mount_point:
                break
            parent = current_dir.parent
            if parent == current_dir:
                return None
            current_dir = parent
        return _UNLIMITED_THRESHOLD if available is None else available
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def available_memory_bytes(*, proc_root: Path = Path("/proc"),
                           sys_root: Path = Path("/sys")) -> int | None:
    """Return min(host MemAvailable, current process cgroup-v2 headroom).

    Unknown or unsupported cgroup layouts fail closed.  In particular, a
    cgroup-v1-only process is not treated as unrestricted host memory.
    """
    proc_root = Path(proc_root)
    sys_root = Path(sys_root)
    host_available = _host_memavailable_bytes(proc_root)
    if host_available is None:
        return None
    cgroup_available = _cgroup_v2_headroom(proc_root, sys_root)
    if cgroup_available is None:
        return None
    return min(host_available, cgroup_available)


def require_memory_admission(
    shape: Iterable[int],
    *,
    max_additional_peak_bytes: int,
    reserve_bytes: int,
    bootstrap_draws: int = 499,
    available_bytes=_UNSET,
) -> int:
    """Raise before large allocations unless the incremental estimate fits.

    ``available_bytes`` is a testing seam.  Omission reads live host/cgroup
    headroom; explicit ``None`` means unknown and is rejected.
    """
    if type(max_additional_peak_bytes) is not int or max_additional_peak_bytes < 1:
        raise ValueError("max_additional_peak_bytes must be a positive integer")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    estimated = estimate_incremental_peak_bytes(shape, bootstrap_draws=bootstrap_draws)
    if estimated > max_additional_peak_bytes:
        raise MemoryError("estimated research batch peak exceeds configured memory budget")
    if available_bytes is _UNSET:
        available_bytes = available_memory_bytes()
    if type(available_bytes) is not int or available_bytes < 0:
        raise RuntimeError("available host/cgroup memory could not be determined")
    if available_bytes < estimated + reserve_bytes:
        raise MemoryError("insufficient host/cgroup headroom for research batch")
    return estimated


__all__ = (
    "available_memory_bytes", "estimate_incremental_peak_bytes",
    "require_memory_admission",
)
