# -*- coding: utf-8 -*-
"""R39 §1.2 / §26.4: unified ``PerformanceRunSummary``.

One object that captures the whole batch run's Time-to-Durable-Commit
(compile / planning / scan / compute / conversion / writer-wait / serialize /
fsync / catalog-commit) plus the byte-amplification fields and the §28 integer
KPIs, bound to the runtime environment so an evidence generator (another task)
can write reproducible ``r39_*.json`` artifacts.

Building blocks provided here:

- ``PerformanceRunSummary`` — the exact dataclass from spec §1.2.
- ``PerformanceRunSummary.from_counters(counts, timing)`` — maps a
  ``PerfCounters`` snapshot + timing/byte dict into the summary.
- ``capture_environment()`` — imports the versions it can, leaves the rest
  ``None``.
- ``render_r39_json(summary, path)`` — writes the R39 JSON evidence file.

The dict-based ``runtime_stats`` consumers are left untouched; this summary is
fed from the new structured counters (R39-P1-PERF-081).
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from runtime.perf_counters import PerfCounters

#: Timing / byte keys accepted by ``from_counters``.
_TIMING_KEYS: tuple[str, ...] = (
    "compile_ms",
    "planning_ms",
    "scan_ms",
    "compute_ms",
    "conversion_ms",
    "writer_wait_ms",
    "serialize_ms",
    "fsync_ms",
    "catalog_commit_ms",
    "total_ttdc_ms",
    "scan_bytes",
    "minimum_required_scan_bytes",
    "conversion_bytes",
    "output_bytes",
    "write_bytes",
    "rewrite_bytes",
    "metadata_scan_bytes",
)

#: Counter slot → PerformanceRunSummary field mapping (§1.2).
_COUNTER_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("future_count", "future_count"),
    ("factor_count", "factor_count"),
    ("physical_query_count", "physical_query_count"),
    ("parquet_file_open_count", "parquet_file_open_count"),
    ("parquet_file_write_count", "parquet_file_write_count"),
    ("fsync_count", "fsync_count"),
    ("sqlite_transaction_count", "sqlite_transaction_count"),
    ("full_factor_rescan_count", "full_factor_rescan_count"),
    ("watchdog_thread_created_count", "watchdog_thread_created_count"),
)


@dataclass
class PerformanceRunSummary:
    """Unified performance run summary (R39 §1.2 / §26.4)."""

    factor_count: int
    compile_ms: float
    planning_ms: float
    scan_ms: float
    compute_ms: float
    conversion_ms: float
    writer_wait_ms: float
    serialize_ms: float
    fsync_ms: float
    catalog_commit_ms: float
    total_ttdc_ms: float

    scan_bytes: int
    minimum_required_scan_bytes: int
    conversion_bytes: int
    output_bytes: int
    write_bytes: int
    rewrite_bytes: int
    metadata_scan_bytes: int

    future_count: int
    physical_query_count: int
    parquet_file_open_count: int
    parquet_file_write_count: int
    fsync_count: int
    sqlite_transaction_count: int
    full_factor_rescan_count: int
    watchdog_thread_created_count: int

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------
    @classmethod
    def from_counters(
        cls,
        counts: PerfCounters | None,
        timing: Mapping[str, Any] | None,
    ) -> "PerformanceRunSummary":
        """Build a summary from a ``PerfCounters`` snapshot + timing/byte dict.

        Counter slots map 1:1 to the §1.2 count fields; the ``timing`` dict
        carries the ms timings and byte-amplification fields.  Missing timing
        keys default to 0; missing counter slots default to 0.
        """
        snapshot = counts.snapshot() if counts is not None else {}
        timing = dict(timing or {})

        def _c(slot: str) -> int:
            try:
                return int(snapshot.get(slot, 0) or 0)
            except (TypeError, ValueError):
                return 0

        def _f(key: str) -> float:
            try:
                return float(timing.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def _b(key: str) -> int:
            try:
                return int(timing.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        kwargs: dict[str, Any] = {
            "factor_count": _c("factor_count"),
            "compile_ms": _f("compile_ms"),
            "planning_ms": _f("planning_ms"),
            "scan_ms": _f("scan_ms"),
            "compute_ms": _f("compute_ms"),
            "conversion_ms": _f("conversion_ms"),
            "writer_wait_ms": _f("writer_wait_ms"),
            "serialize_ms": _f("serialize_ms"),
            "fsync_ms": _f("fsync_ms"),
            "catalog_commit_ms": _f("catalog_commit_ms"),
            "total_ttdc_ms": _f("total_ttdc_ms"),
            "scan_bytes": _b("scan_bytes"),
            "minimum_required_scan_bytes": _b("minimum_required_scan_bytes"),
            "conversion_bytes": _b("conversion_bytes"),
            "output_bytes": _b("output_bytes"),
            "write_bytes": _b("write_bytes"),
            "rewrite_bytes": _b("rewrite_bytes"),
            "metadata_scan_bytes": _b("metadata_scan_bytes"),
        }
        for slot, field in _COUNTER_FIELD_MAP:
            kwargs[field] = _c(slot)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def amplification(self) -> dict[str, float]:
        """§1.1 amplification ratios (guard division by zero)."""
        out: dict[str, float] = {}

        def _ratio(n: int, d: int) -> float:
            return round(float(n) / float(d), 4) if d else 0.0

        out["scan_amplification"] = _ratio(
            self.scan_bytes, self.minimum_required_scan_bytes
        )
        out["conversion_amplification"] = _ratio(
            self.conversion_bytes, self.output_bytes
        )
        out["write_amplification"] = _ratio(self.write_bytes, self.output_bytes)
        out["rewrite_amplification"] = _ratio(
            self.rewrite_bytes, self.output_bytes
        )
        out["metadata_amplification"] = _ratio(
            self.metadata_scan_bytes, self.output_bytes
        )
        if self.factor_count:
            out["physical_queries_per_factor"] = round(
                self.physical_query_count / float(self.factor_count), 4
            )
        else:
            out["physical_queries_per_factor"] = 0.0
        return out

    def to_dict(
        self,
        environment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize the summary + optional environment binding.

        ``environment`` is the dict from ``capture_environment()``.  When
        omitted, the environment is captured lazily so a single ``render`` call
        is self-contained.
        """
        data: dict[str, Any] = asdict(self)
        data["amplification"] = self.amplification()
        env = dict(environment or capture_environment())
        data["environment"] = env
        data["schema"] = "r39_performance_run_summary/v1"
        return data


# ---------------------------------------------------------------------------
# environment capture
# ---------------------------------------------------------------------------

#: versions that must be present in ``to_dict()``'s ``environment``.
_VERSION_KEYS: tuple[str, ...] = (
    "python",
    "numpy",
    "pandas",
    "polars",
    "duckdb",
    "pyarrow",
)


def _module_version(name: str) -> str | None:
    if name == "python":
        return (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "") or None)
    except Exception:
        return None


def _git_sha() -> str | None:
    try:
        from runtime.lineage import resolve_git_commit_hash

        return resolve_git_commit_hash() or None
    except Exception:
        return None


def _build_identity() -> str | None:
    """FactorEngine build identity: ``factor_engine==<version>`` if known."""
    try:
        from importlib.metadata import version

        return f"factor_engine=={version('factor_engine')}"
    except Exception:
        pass
    # Fallback: parse the pyproject/egg-info next to this file.
    try:
        root = Path(__file__).resolve().parents[1]
        pkg = root / "factor_engine.egg-info" / "PKG-INFO"
        if pkg.is_file():
            for line in pkg.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Version:"):
                    return f"factor_engine=={line.split(':', 1)[1].strip()}"
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            for line in pyproject.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith("version"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return f"factor_engine=={val}"
    except Exception:
        pass
    return None


def _cpu() -> dict[str, Any]:
    out: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
    }
    return out


def _memory_cgroup() -> dict[str, Any]:
    """RAM limit from cgroup v2 (``memory.max``) or host ``sysconf``."""
    out: dict[str, Any] = {"ram_bytes": None, "memory_source": None}
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
            if text and text != "max":
                out["ram_bytes"] = int(float(text))
                out["memory_source"] = str(path)
                break
        except Exception:
            continue
    if out["ram_bytes"] is None:
        try:
            out["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            out["memory_source"] = "sysconf"
        except (ValueError, AttributeError):
            out["ram_bytes"] = None
    try:
        from runtime.resource_governor import effective_memory_limit_bytes

        out["effective_memory_limit_bytes"] = int(effective_memory_limit_bytes())
    except Exception:
        out["effective_memory_limit_bytes"] = None
    return out


def _storage_class() -> str | None:
    try:
        from runtime.resource_governor import spill_disk_speed_class

        return str(spill_disk_speed_class()) or None
    except Exception:
        return None


def capture_environment(
    *,
    source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture runtime environment binding fields.

    Reads the versions it can import; unknown versions stay ``None``.  ``source
    snapshot`` is an optional identity dict (data source snapshot / generation
    pointers) passed by the caller; default ``None``.
    """
    env: dict[str, Any] = {
        "git_sha": _git_sha(),
        "build_identity": _build_identity(),
        "source_snapshot": (
            dict(source_snapshot) if source_snapshot is not None else None
        ),
    }
    for key in _VERSION_KEYS:
        env[f"{key}_version"] = _module_version(key)
    env["cpu"] = _cpu()
    env["memory"] = _memory_cgroup()
    env["storage_class"] = _storage_class()
    env["platform"] = platform.system()
    return env


# ---------------------------------------------------------------------------
# R39 evidence writer
# ---------------------------------------------------------------------------
def render_r39_json(
    summary: PerformanceRunSummary,
    path: str | os.PathLike[str],
    *,
    environment: Mapping[str, Any] | None = None,
) -> str:
    """Write the R39 performance-run-summary JSON evidence artifact.

    Returns the resolved path string.  The payload is a parseable JSON object
    with the summary fields, amplification ratios, and the environment binding.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict(environment=environment)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(out)


__all__ = [
    "PerformanceRunSummary",
    "capture_environment",
    "render_r39_json",
]
