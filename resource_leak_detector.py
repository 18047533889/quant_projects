"""
Comprehensive resource leak detector for quant_projects.

Scans for:
- Memory leaks (growing allocations, unreleased buffers)
- File handle leaks (unclosed files)
- Database connection leaks (unclosed SQLite connections)
- Thread/process leaks (orphaned threads, zombie processes)
- Circular references preventing GC
- Temporary file accumulation
"""

import gc
import os
import sys
import tracemalloc
import psutil
import sqlite3
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import tempfile


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: datetime
    memory_rss_mb: float
    memory_vms_mb: float
    open_files: int
    threads: int
    processes: int
    temp_files: int
    gc_objects: int
    gc_garbage: int
    tracemalloc_top: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class LeakReport:
    """Report of detected resource leaks."""
    leak_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    evidence: List[str]
    suggested_fix: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None


class ResourceLeakDetector:
    """Detect resource leaks in quant_projects codebase."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.snapshots: List[ResourceSnapshot] = []
        self.reports: List[LeakReport] = []
        self.process = psutil.Process()
        self.initial_open_files = len(self.process.open_files())
        self.initial_threads = self.process.num_threads()

    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        mem_info = self.process.memory_info()

        # Get tracemalloc top allocations if enabled
        tracemalloc_top = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            tracemalloc_top = [
                (str(stat.traceback), stat.size)
                for stat in top_stats
            ]

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            memory_rss_mb=mem_info.rss / 1024 / 1024,
            memory_vms_mb=mem_info.vms / 1024 / 1024,
            open_files=len(self.process.open_files()),
            threads=self.process.num_threads(),
            processes=len(self.process.children(recursive=True)),
            temp_files=self._count_temp_files(),
            gc_objects=len(gc.get_objects()),
            gc_garbage=len(gc.garbage),
            tracemalloc_top=tracemalloc_top,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _count_temp_files(self) -> int:
        """Count temporary files in system temp directory."""
        temp_dir = Path(tempfile.gettempdir())
        count = 0
        try:
            for item in temp_dir.glob("*"):
                if item.name.startswith(("tmp", "factor_", "quant_")):
                    count += 1
        except (PermissionError, FileNotFoundError):
            pass
        return count

    def detect_memory_leaks(self) -> List[LeakReport]:
        """Detect memory leaks from snapshots."""
        reports = []

        if len(self.snapshots) < 2:
            return reports

        # Check for monotonic memory growth
        memory_trend = [s.memory_rss_mb for s in self.snapshots]
        if len(memory_trend) >= 3:
            growth = memory_trend[-1] - memory_trend[0]
            if growth > 100:  # 100 MB growth
                reports.append(LeakReport(
                    leak_type="memory_leak",
                    severity="high",
                    description=f"Memory grew by {growth:.1f} MB across {len(memory_trend)} snapshots",
                    evidence=[
                        f"Initial: {memory_trend[0]:.1f} MB",
                        f"Final: {memory_trend[-1]:.1f} MB",
                        f"Growth rate: {growth / len(memory_trend):.1f} MB per snapshot"
                    ],
                    suggested_fix="Check for unreleased numpy arrays, cached data not being cleared, or circular references"
                ))

        # Check GC garbage accumulation
        if self.snapshots[-1].gc_garbage > 10:
            reports.append(LeakReport(
                leak_type="circular_reference",
                severity="medium",
                description=f"{self.snapshots[-1].gc_garbage} objects in gc.garbage (circular references)",
                evidence=[f"gc.garbage contains {self.snapshots[-1].gc_garbage} uncollectable objects"],
                suggested_fix="Break circular references or use weakref"
            ))

        return reports

    def detect_file_leaks(self) -> List[LeakReport]:
        """Detect file handle leaks."""
        reports = []

        if len(self.snapshots) < 2:
            return reports

        current_files = self.snapshots[-1].open_files
        leaked_files = current_files - self.initial_open_files

        if leaked_files > 5:
            open_files = self.process.open_files()
            file_paths = [f.path for f in open_files if hasattr(f, 'path')]

            reports.append(LeakReport(
                leak_type="file_handle_leak",
                severity="critical",
                description=f"{leaked_files} file handles not closed",
                evidence=[
                    f"Initial open files: {self.initial_open_files}",
                    f"Current open files: {current_files}",
                    f"Leaked files: {leaked_files}",
                    f"Sample paths: {file_paths[:5]}"
                ],
                suggested_fix="Use context managers (with statement) for all file operations"
            ))

        return reports

    def detect_thread_leaks(self) -> List[LeakReport]:
        """Detect thread leaks."""
        reports = []

        current_threads = self.snapshots[-1].threads if self.snapshots else self.process.num_threads()
        leaked_threads = current_threads - self.initial_threads

        if leaked_threads > 3:
            reports.append(LeakReport(
                leak_type="thread_leak",
                severity="high",
                description=f"{leaked_threads} threads not cleaned up",
                evidence=[
                    f"Initial threads: {self.initial_threads}",
                    f"Current threads: {current_threads}",
                    f"Active threads: {[t.name for t in threading.enumerate()]}"
                ],
                suggested_fix="Ensure threads are joined or use thread pools with proper cleanup"
            ))

        return reports

    def detect_temp_file_accumulation(self) -> List[LeakReport]:
        """Detect temporary file accumulation."""
        reports = []

        if not self.snapshots:
            return reports

        temp_files = self.snapshots[-1].temp_files
        if temp_files > 20:
            reports.append(LeakReport(
                leak_type="temp_file_accumulation",
                severity="medium",
                description=f"{temp_files} temporary files detected in {tempfile.gettempdir()}",
                evidence=[f"Temp file count: {temp_files}"],
                suggested_fix="Use TemporaryDirectory context manager or explicit cleanup in finally blocks"
            ))

        return reports

    def scan_sqlite_connections(self) -> List[LeakReport]:
        """Scan for unclosed SQLite connections in code."""
        reports = []

        # Scan research_control for SQLite usage patterns
        research_control = self.project_root / "research_control"
        if research_control.exists():
            leaks = self._scan_directory_for_sqlite_leaks(research_control)
            reports.extend(leaks)

        return reports

    def _scan_directory_for_sqlite_leaks(self, directory: Path) -> List[LeakReport]:
        """Scan directory for SQLite connection leak patterns."""
        reports = []

        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".claude" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                lines = content.split('\n')

                for i, line in enumerate(lines, 1):
                    # Check for sqlite3.connect without context manager
                    if "sqlite3.connect(" in line and "with" not in line:
                        # Check if there's a context manager in next few lines
                        context_manager_found = False
                        for j in range(max(0, i-3), min(len(lines), i+3)):
                            if "with" in lines[j] and "connect" in lines[j]:
                                context_manager_found = True
                                break

                        if not context_manager_found:
                            reports.append(LeakReport(
                                leak_type="sqlite_connection_leak",
                                severity="high",
                                description="SQLite connection without context manager",
                                evidence=[f"Line {i}: {line.strip()}"],
                                suggested_fix="Use 'with sqlite3.connect(...) as conn:' or ensure conn.close() in finally block",
                                file_path=str(py_file),
                                line_number=i
                            ))
            except Exception as e:
                pass

        return reports

    def scan_multiprocessing_pools(self) -> List[LeakReport]:
        """Scan for multiprocessing pool leaks."""
        reports = []

        # Scan for Pool usage without context manager or close/join
        for module_dir in ["quant_evaluator", "factor_assets", "factor_preprocess"]:
            module_path = self.project_root / module_dir
            if module_path.exists():
                leaks = self._scan_directory_for_pool_leaks(module_path)
                reports.extend(leaks)

        return reports

    def _scan_directory_for_pool_leaks(self, directory: Path) -> List[LeakReport]:
        """Scan directory for multiprocessing Pool leak patterns."""
        reports = []

        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".claude" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                lines = content.split('\n')

                # Look for Pool creation
                for i, line in enumerate(lines, 1):
                    if "Pool(" in line:
                        # Check if using context manager
                        has_with = False
                        has_close = False
                        has_join = False

                        # Check current and nearby lines
                        for j in range(max(0, i-2), min(len(lines), i+20)):
                            if "with Pool" in lines[j]:
                                has_with = True
                                break
                            if ".close()" in lines[j]:
                                has_close = True
                            if ".join()" in lines[j]:
                                has_join = True

                        if not has_with and not (has_close and has_join):
                            reports.append(LeakReport(
                                leak_type="multiprocessing_pool_leak",
                                severity="high",
                                description="multiprocessing.Pool without proper cleanup",
                                evidence=[f"Line {i}: {line.strip()}"],
                                suggested_fix="Use 'with Pool(...) as pool:' or call pool.close() and pool.join()",
                                file_path=str(py_file),
                                line_number=i
                            ))

            except Exception as e:
                pass

        return reports

    def scan_numpy_array_accumulation(self) -> List[LeakReport]:
        """Scan for patterns that accumulate large numpy arrays."""
        reports = []

        # Check for common patterns: list.append in loops with arrays
        for module_dir in ["quant_evaluator", "factor_assets", "factor_preprocess"]:
            module_path = self.project_root / module_dir
            if module_path.exists():
                leaks = self._scan_directory_for_array_accumulation(module_path)
                reports.extend(leaks)

        return reports

    def _scan_directory_for_array_accumulation(self, directory: Path) -> List[LeakReport]:
        """Scan for patterns that accumulate numpy arrays without clearing."""
        reports = []

        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".claude" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                lines = content.split('\n')

                in_loop = False
                loop_start = 0
                accumulator_lists = set()

                for i, line in enumerate(lines, 1):
                    stripped = line.strip()

                    # Track loops
                    if stripped.startswith(("for ", "while ")):
                        in_loop = True
                        loop_start = i
                        accumulator_lists.clear()

                    if in_loop and stripped.startswith(("return", "break", "def ", "class ")):
                        in_loop = False

                    # Look for list.append in loops
                    if in_loop and ".append(" in stripped:
                        var_name = stripped.split(".append(")[0].strip()
                        if not var_name.startswith("self."):  # Local variable
                            accumulator_lists.add(var_name)

                # Report if accumulation pattern found without clearing
                if accumulator_lists:
                    reports.append(LeakReport(
                        leak_type="array_accumulation",
                        severity="medium",
                        description=f"Potential array accumulation in loops: {', '.join(accumulator_lists)}",
                        evidence=[f"Variables: {', '.join(accumulator_lists)}"],
                        suggested_fix="Clear accumulator lists after processing or use generators for streaming",
                        file_path=str(py_file),
                        line_number=loop_start
                    ))

            except Exception as e:
                pass

        return reports

    def run_full_scan(self) -> Dict[str, List[LeakReport]]:
        """Run full resource leak detection scan."""
        print("Running comprehensive resource leak detection...")

        # Start tracemalloc for memory profiling
        tracemalloc.start()

        # Take initial snapshot
        print("Taking initial resource snapshot...")
        self.take_snapshot()

        # Force garbage collection
        gc.collect()

        # Static code analysis
        print("Scanning for SQLite connection leaks...")
        sqlite_reports = self.scan_sqlite_connections()

        print("Scanning for multiprocessing Pool leaks...")
        pool_reports = self.scan_multiprocessing_pools()

        print("Scanning for numpy array accumulation...")
        array_reports = self.scan_numpy_array_accumulation()

        # Take final snapshot
        print("Taking final resource snapshot...")
        self.take_snapshot()

        # Runtime leak detection
        print("Analyzing memory leaks...")
        memory_reports = self.detect_memory_leaks()

        print("Analyzing file handle leaks...")
        file_reports = self.detect_file_leaks()

        print("Analyzing thread leaks...")
        thread_reports = self.detect_thread_leaks()

        print("Analyzing temporary file accumulation...")
        temp_reports = self.detect_temp_file_accumulation()

        # Stop tracemalloc
        tracemalloc.stop()

        return {
            "memory_leaks": memory_reports,
            "file_leaks": file_reports,
            "thread_leaks": thread_reports,
            "temp_file_leaks": temp_reports,
            "sqlite_leaks": sqlite_reports,
            "pool_leaks": pool_reports,
            "array_accumulation": array_reports,
        }

    def generate_report(self, results: Dict[str, List[LeakReport]]) -> str:
        """Generate formatted report."""
        lines = []
        lines.append("=" * 80)
        lines.append("RESOURCE LEAK DETECTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Project: {self.project_root}")
        lines.append("")

        # Summary
        total_issues = sum(len(reports) for reports in results.values())
        critical = sum(1 for reports in results.values() for r in reports if r.severity == "critical")
        high = sum(1 for reports in results.values() for r in reports if r.severity == "high")
        medium = sum(1 for reports in results.values() for r in reports if r.severity == "medium")
        low = sum(1 for reports in results.values() for r in reports if r.severity == "low")

        lines.append(f"SUMMARY: {total_issues} issues found")
        lines.append(f"  - Critical: {critical}")
        lines.append(f"  - High: {high}")
        lines.append(f"  - Medium: {medium}")
        lines.append(f"  - Low: {low}")
        lines.append("")

        # Detailed reports by category
        for category, reports in results.items():
            if not reports:
                continue

            lines.append(f"\n{'=' * 80}")
            lines.append(f"{category.upper().replace('_', ' ')}")
            lines.append(f"{'=' * 80}")

            for i, report in enumerate(reports, 1):
                lines.append(f"\n[{i}] {report.leak_type} ({report.severity})")
                lines.append(f"Description: {report.description}")

                if report.file_path:
                    lines.append(f"Location: {report.file_path}:{report.line_number}")

                lines.append("Evidence:")
                for evidence in report.evidence:
                    lines.append(f"  - {evidence}")

                lines.append(f"Suggested Fix: {report.suggested_fix}")

        # Resource snapshots
        if self.snapshots:
            lines.append(f"\n{'=' * 80}")
            lines.append("RESOURCE SNAPSHOTS")
            lines.append(f"{'=' * 80}")

            for i, snapshot in enumerate(self.snapshots):
                lines.append(f"\nSnapshot {i + 1}:")
                lines.append(f"  Memory RSS: {snapshot.memory_rss_mb:.1f} MB")
                lines.append(f"  Open Files: {snapshot.open_files}")
                lines.append(f"  Threads: {snapshot.threads}")
                lines.append(f"  Temp Files: {snapshot.temp_files}")
                lines.append(f"  GC Objects: {snapshot.gc_objects}")
                lines.append(f"  GC Garbage: {snapshot.gc_garbage}")

        lines.append("")
        return "\n".join(lines)


def main():
    """Run resource leak detection."""
    project_root = Path("/home/shw/quant_projects")

    detector = ResourceLeakDetector(project_root)
    results = detector.run_full_scan()
    report = detector.generate_report(results)

    # Print report
    print(report)

    # Save report
    report_path = project_root / "RESOURCE_LEAK_DETECTION_RESULTS.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
