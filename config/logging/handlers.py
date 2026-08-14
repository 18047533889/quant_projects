"""
Custom log handlers with rotation, compression, and remote logging.
"""

import gzip
import logging
import logging.handlers
import os
import socket
from pathlib import Path
from typing import Optional


class RotatingFileHandlerWithCompression(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that compresses old log files.

    When a log file is rotated, it is compressed with gzip to save space.
    """

    def __init__(
        self,
        filename: Path,
        mode: str = 'a',
        maxBytes: int = 100 * 1024 * 1024,  # 100MB default
        backupCount: int = 10,
        encoding: Optional[str] = 'utf-8',
        delay: bool = False,
        compress_old: bool = True,
    ):
        """
        Initialize handler.

        Args:
            filename: Log file path
            mode: File open mode
            maxBytes: Maximum bytes before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first emit
            compress_old: Whether to compress rotated files
        """
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(filename),
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
        )
        self.compress_old = compress_old

    def doRollover(self):
        """
        Do a rollover and compress the old log file.
        """
        # Close current file
        if self.stream:
            self.stream.close()
            self.stream = None

        # Rotate files
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)

            # Move current to .1
            dfn = self.rotation_filename(f"{self.baseFilename}.1")
            if os.path.exists(dfn):
                os.remove(dfn)
            if os.path.exists(self.baseFilename):
                os.rename(self.baseFilename, dfn)

                # Compress the rotated file
                if self.compress_old:
                    self._compress_file(dfn)

        # Open new file
        if not self.delay:
            self.stream = self._open()

    def _compress_file(self, filepath: str):
        """
        Compress a log file with gzip.

        Args:
            filepath: Path to file to compress
        """
        try:
            compressed_path = f"{filepath}.gz"

            # Read and compress
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=9) as f_out:
                    f_out.writelines(f_in)

            # Remove original
            os.remove(filepath)

        except Exception as e:
            # Log compression failure but don't crash
            print(f"Warning: Failed to compress log file {filepath}: {e}")


class SyslogHandler(logging.handlers.SysLogHandler):
    """
    Enhanced syslog handler with better formatting.

    Supports both local Unix socket and remote TCP/UDP syslog.
    """

    def __init__(
        self,
        address: str = 'localhost:514',
        facility: int = logging.handlers.SysLogHandler.LOG_USER,
        socktype: int = socket.SOCK_DGRAM,
    ):
        """
        Initialize syslog handler.

        Args:
            address: Syslog server address ('host:port' or '/dev/log')
            facility: Syslog facility
            socktype: Socket type (SOCK_DGRAM for UDP, SOCK_STREAM for TCP)
        """
        # Parse address
        if ':' in address and not address.startswith('/'):
            host, port_str = address.rsplit(':', 1)
            address = (host, int(port_str))

        super().__init__(address=address, facility=facility, socktype=socktype)

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to syslog with error handling.
        """
        try:
            super().emit(record)
        except Exception as e:
            # Don't crash on syslog errors
            self.handleError(record)


class BufferedHandler(logging.Handler):
    """
    Buffered handler that flushes on threshold or error.

    Collects log records in memory and flushes them in batches
    for better performance with remote logging.
    """

    def __init__(
        self,
        target_handler: logging.Handler,
        capacity: int = 100,
        flush_on_error: bool = True,
    ):
        """
        Initialize buffered handler.

        Args:
            target_handler: Underlying handler to flush to
            capacity: Number of records to buffer before flushing
            flush_on_error: Whether to flush immediately on error/critical
        """
        super().__init__()
        self.target = target_handler
        self.capacity = capacity
        self.flush_on_error = flush_on_error
        self.buffer = []

    def emit(self, record: logging.LogRecord):
        """
        Buffer a log record.
        """
        self.buffer.append(record)

        # Flush if buffer is full
        if len(self.buffer) >= self.capacity:
            self.flush()

        # Flush immediately on error/critical
        elif self.flush_on_error and record.levelno >= logging.ERROR:
            self.flush()

    def flush(self):
        """
        Flush all buffered records to target handler.
        """
        self.acquire()
        try:
            for record in self.buffer:
                self.target.emit(record)
            self.buffer.clear()
            self.target.flush()
        finally:
            self.release()

    def close(self):
        """
        Flush and close handler.
        """
        try:
            self.flush()
        finally:
            self.target.close()
            super().close()


class MultiProcessSafeHandler(logging.Handler):
    """
    Handler wrapper that ensures thread and process safety.

    Uses file locking to ensure multiple processes can write to
    the same log file without corruption.
    """

    def __init__(self, target_handler: logging.Handler):
        """
        Initialize multi-process safe handler.

        Args:
            target_handler: Underlying handler to protect
        """
        super().__init__()
        self.target = target_handler
        self._lock_file = None

        # Create lock file if target is file handler
        if isinstance(target_handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            lock_path = str(target_handler.baseFilename) + '.lock'
            self._lock_file = Path(lock_path)

    def emit(self, record: logging.LogRecord):
        """
        Emit with file locking.
        """
        if self._lock_file:
            import fcntl

            lock_fd = None
            try:
                # Open lock file
                lock_fd = open(self._lock_file, 'a')

                # Acquire exclusive lock
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

                # Write log
                self.target.emit(record)

            except Exception:
                self.handleError(record)
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception:
                        pass
        else:
            # No locking needed
            self.target.emit(record)

    def flush(self):
        """Flush target handler."""
        self.target.flush()

    def close(self):
        """Close target handler."""
        self.target.close()
        super().close()
