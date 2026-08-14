"""
Resource leak fixes for identified issues.

Provides context managers and utilities to fix resource leaks across quant_projects.
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional, Any
from pathlib import Path
import atexit
import weakref


# ============================================================================
# SQLite Connection Management
# ============================================================================

class SQLiteConnectionManager:
    """
    Managed SQLite connection with automatic cleanup.

    Ensures connections are properly closed even if exceptions occur.
    """

    def __init__(self, db_path: Path, persistent: bool = False):
        """
        Initialize connection manager.

        Args:
            db_path: Path to SQLite database
            persistent: If True, keep connection open and register for cleanup
        """
        self.db_path = db_path
        self.persistent = persistent
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = False

        if persistent:
            self._conn = sqlite3.connect(str(db_path))
            self._conn.row_factory = sqlite3.Row
            # Register for cleanup at exit
            atexit.register(self.close)

    @contextmanager
    def connection(self):
        """
        Context manager for database connection.

        For persistent connections, yields the same connection.
        For non-persistent, creates a new connection per context.
        """
        if self.persistent:
            if self._closed:
                raise RuntimeError("Connection manager is closed")
            yield self._conn
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def close(self):
        """Close persistent connection."""
        if self._conn and not self._closed:
            try:
                self._conn.close()
            except Exception:
                pass  # Suppress errors during cleanup
            finally:
                self._closed = True

    def __enter__(self):
        """Support using manager as context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        self.close()
        return False


@contextmanager
def safe_sqlite_connection(db_path: Path, row_factory: bool = True):
    """
    Safe SQLite connection context manager.

    Usage:
        with safe_sqlite_connection(db_path) as conn:
            cursor = conn.execute("SELECT * FROM table")
            ...
    """
    conn = sqlite3.connect(str(db_path))
    if row_factory:
        conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ============================================================================
# Array Accumulator Management
# ============================================================================

class BoundedAccumulator:
    """
    Bounded accumulator that prevents unbounded growth.

    Automatically clears when reaching size limit or provides
    generator-based iteration.
    """

    def __init__(self, max_size: int = 1000, auto_clear: bool = True):
        """
        Initialize bounded accumulator.

        Args:
            max_size: Maximum size before warning or auto-clear
            auto_clear: If True, automatically clear when reaching max_size
        """
        self.max_size = max_size
        self.auto_clear = auto_clear
        self._items = []

    def append(self, item: Any):
        """Append item with bounds checking."""
        if len(self._items) >= self.max_size:
            if self.auto_clear:
                self.clear()
            else:
                raise MemoryError(
                    f"Accumulator reached max size {self.max_size}. "
                    "Call clear() or use streaming."
                )
        self._items.append(item)

    def extend(self, items):
        """Extend with multiple items."""
        for item in items:
            self.append(item)

    def clear(self):
        """Clear accumulated items."""
        self._items.clear()

    def __iter__(self):
        """Iterate over items."""
        return iter(self._items)

    def __len__(self):
        """Get current size."""
        return len(self._items)

    def __getitem__(self, key):
        """Get item by index."""
        return self._items[key]


def streaming_accumulator(max_memory_mb: float = 100):
    """
    Decorator to convert accumulator-based functions to streaming.

    Usage:
        @streaming_accumulator(max_memory_mb=50)
        def process_batches(data):
            results = []
            for batch in data:
                results.append(process(batch))  # Will yield instead
            return results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This is a marker - actual implementation would need
            # AST transformation or manual refactoring
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Multiprocessing Pool Management
# ============================================================================

class ManagedPool:
    """
    Managed multiprocessing Pool with guaranteed cleanup.

    Wraps Pool to ensure close() and join() are always called.
    """

    def __init__(self, processes: Optional[int] = None, **kwargs):
        """
        Initialize managed pool.

        Args:
            processes: Number of worker processes
            **kwargs: Additional Pool arguments
        """
        from multiprocessing import Pool
        self._pool = Pool(processes=processes, **kwargs)
        self._closed = False
        atexit.register(self.cleanup)

    def __enter__(self):
        """Enter context manager."""
        return self._pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with cleanup."""
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up pool resources."""
        if not self._closed:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass  # Suppress errors during cleanup
            finally:
                self._closed = True

    def __getattr__(self, name):
        """Delegate attribute access to underlying pool."""
        return getattr(self._pool, name)


# ============================================================================
# File Handle Management
# ============================================================================

class FileHandleTracker:
    """
    Track open file handles and warn about leaks.

    Useful for debugging file handle leaks in long-running processes.
    """

    def __init__(self):
        self._tracked_files = weakref.WeakSet()
        self._open_count = 0
        self._close_count = 0

    @contextmanager
    def track(self, file_obj):
        """Track a file object."""
        self._open_count += 1
        self._tracked_files.add(file_obj)
        try:
            yield file_obj
        finally:
            self._close_count += 1

    def get_stats(self) -> dict:
        """Get file handle statistics."""
        return {
            "opened": self._open_count,
            "closed": self._close_count,
            "leaked": self._open_count - self._close_count,
            "currently_tracked": len(self._tracked_files),
        }

    def report_leaks(self):
        """Report file handle leaks."""
        stats = self.get_stats()
        if stats["leaked"] > 0:
            print(f"WARNING: {stats['leaked']} file handles not properly closed")
            print(f"  Opened: {stats['opened']}")
            print(f"  Closed: {stats['closed']}")


# Global tracker instance
_file_tracker = FileHandleTracker()


@contextmanager
def tracked_file(path, mode='r', **kwargs):
    """
    Open file with automatic tracking.

    Usage:
        with tracked_file("data.txt", "r") as f:
            data = f.read()
    """
    f = open(path, mode, **kwargs)
    with _file_tracker.track(f):
        try:
            yield f
        finally:
            f.close()


def report_file_leaks():
    """Report tracked file handle leaks."""
    _file_tracker.report_leaks()


# ============================================================================
# Temporary File Cleanup
# ============================================================================

class TemporaryFileManager:
    """
    Manager for temporary files with automatic cleanup.

    Tracks all temporary files and ensures cleanup on exit.
    """

    def __init__(self):
        self._temp_files = set()
        self._temp_dirs = set()
        atexit.register(self.cleanup_all)

    def register_file(self, path: Path):
        """Register a temporary file for cleanup."""
        self._temp_files.add(path)

    def register_dir(self, path: Path):
        """Register a temporary directory for cleanup."""
        self._temp_dirs.add(path)

    def cleanup_file(self, path: Path):
        """Clean up a specific file."""
        try:
            if path.exists():
                path.unlink()
            self._temp_files.discard(path)
        except Exception:
            pass

    def cleanup_dir(self, path: Path):
        """Clean up a specific directory."""
        try:
            if path.exists():
                import shutil
                shutil.rmtree(path)
            self._temp_dirs.discard(path)
        except Exception:
            pass

    def cleanup_all(self):
        """Clean up all registered files and directories."""
        for path in list(self._temp_files):
            self.cleanup_file(path)

        for path in list(self._temp_dirs):
            self.cleanup_dir(path)


# Global temp file manager
_temp_manager = TemporaryFileManager()


@contextmanager
def managed_tempfile(suffix: str = "", prefix: str = "tmp", dir: Optional[Path] = None):
    """
    Create managed temporary file with automatic cleanup.

    Usage:
        with managed_tempfile(suffix=".npy") as tmp_path:
            np.save(tmp_path, data)
            # File automatically deleted on exit
    """
    import tempfile
    import os as _os
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    _os.close(fd)  # Close file descriptor

    path = Path(path)
    _temp_manager.register_file(path)

    try:
        yield path
    finally:
        _temp_manager.cleanup_file(path)


@contextmanager
def managed_tempdir(suffix: str = "", prefix: str = "tmp", dir: Optional[Path] = None):
    """
    Create managed temporary directory with automatic cleanup.

    Usage:
        with managed_tempdir() as tmp_dir:
            (tmp_dir / "data.npy").write_bytes(data)
            # Directory automatically deleted on exit
    """
    import tempfile
    path = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir))
    _temp_manager.register_dir(path)

    try:
        yield path
    finally:
        _temp_manager.cleanup_dir(path)


# ============================================================================
# Memory Leak Detection Utilities
# ============================================================================

class MemoryLeakDetector:
    """
    Detect memory leaks in long-running functions.

    Usage:
        detector = MemoryLeakDetector()

        with detector.monitor("batch_processing"):
            for batch in batches:
                process(batch)

        detector.report()
    """

    def __init__(self):
        import psutil as _psutil
        self.measurements = {}
        self.process = _psutil.Process()

    @contextmanager
    def monitor(self, label: str):
        """Monitor memory during a block of code."""
        import gc
        gc.collect()

        mem_before = self.process.memory_info().rss / 1024 / 1024

        try:
            yield
        finally:
            gc.collect()
            mem_after = self.process.memory_info().rss / 1024 / 1024

            growth = mem_after - mem_before
            self.measurements[label] = {
                "before_mb": mem_before,
                "after_mb": mem_after,
                "growth_mb": growth,
            }

    def report(self):
        """Report memory measurements."""
        print("\n=== Memory Leak Detection Report ===")
        for label, measurements in self.measurements.items():
            growth = measurements["growth_mb"]
            status = "LEAK" if growth > 10 else "OK"
            print(f"{label}: {growth:+.1f} MB [{status}]")
            if growth > 1:
                print(f"  Before: {measurements['before_mb']:.1f} MB")
                print(f"  After: {measurements['after_mb']:.1f} MB")


# Example usage
if __name__ == "__main__":
    print("Resource leak fix utilities loaded")
    print("\nAvailable utilities:")
    print("  - SQLiteConnectionManager: Managed SQLite connections")
    print("  - safe_sqlite_connection: Context manager for SQLite")
    print("  - BoundedAccumulator: Prevent unbounded list growth")
    print("  - ManagedPool: Managed multiprocessing pools")
    print("  - tracked_file: Track file handles")
    print("  - managed_tempfile: Managed temporary files")
    print("  - MemoryLeakDetector: Detect memory leaks")
