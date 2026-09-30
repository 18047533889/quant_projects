"""Process-local, bounded QE source identity checks.

The default mode hashes every Python source file once, then uses filesystem
metadata guards to detect metadata-visible edits and re-hashes changed files.
This is a performance guard, not a proof of unchanged content: same-size
writes within one filesystem timestamp tick can retain every stat guard,
even without privileged metadata restoration. ``strict_full_content`` rehashes
every source file on every check when that stronger assurance is required.
Callers needing guaranteed content-based cache invalidation must keep strict
checks enabled.

Receipts describe source files on disk. They do not claim to identify Python
code already loaded into the interpreter; callers must keep the process-local
baseline associated with the same imported-code lifetime and avoid hot reloads
or runtime monkeypatching when using it to namespace a cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import stat
import threading
import weakref
from typing import Callable


class SourceIdentityError(RuntimeError):
    """Source identity could not be established within configured bounds."""


class SourceIdentityRaceError(SourceIdentityError):
    """Source metadata changed while content was being read."""


class UnsupportedSourceLayout(SourceIdentityError):
    """The source tree contains a symlink layout this implementation rejects."""


@dataclass(frozen=True)
class SourceIdentityReceipt:
    digest: str
    process_nonce: str
    process_id: int
    strategy: str
    stat_guard_verified: bool
    full_content_checked: bool
    drifted: bool
    changed_paths: tuple[str, ...]
    directory_changes: tuple[str, ...]
    stat_guard_fields: tuple[str, ...]
    source_size_bytes: int
    record_cap: int
    byte_cap: int
    drift_path_cap: int
    retry_cap: int
    files_hashed: int
    bytes_hashed: int
    stat_records: int
    attempts: int


@dataclass(frozen=True)
class _Guard:
    kind: str
    link_stat: tuple[int, ...]
    target_path: str | None = None
    target_stat: tuple[int, ...] | None = None
    link_text: str | None = None


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    """Fields that detect replacement, chmod, truncation, and ordinary edits."""
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _inside_root(root: str, path: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


# Register one callback, not a bound callback per instance: dead identity
# objects must not be retained by the process-wide at-fork registry.
_IDENTITIES = weakref.WeakSet()


def _reset_identity_locks_after_fork():
    for identity in list(_IDENTITIES):
        identity._lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_identity_locks_after_fork)


class ProcessSourceIdentity:
    """Bounded source-tree identity memo for one process.

    ``max_records`` bounds all directory entries visited during traversal;
    ``max_bytes`` bounds the total Python source size admitted for hashing;
    ``max_drift_paths`` bounds one update's changed path set; and
    ``max_retries`` bounds retries after a detected read race.

    Python source symlink files are accepted only when their resolved regular
    file target stays inside the root. Symlink directories are always rejected
    and never traversed. Other non-Python files are ignored after counting
    their directory entry toward ``max_records``.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        strict_full_content: bool = False,
        max_records: int = 10000,
        max_bytes: int = 256 * 1024 * 1024,
        max_drift_paths: int = 64,
        max_retries: int = 1,
        chunk_bytes: int = 1024 * 1024,
        _opener: Callable | None = None,
    ) -> None:
        self.root = os.path.realpath(os.fspath(root))
        if not os.path.isdir(self.root):
            raise SourceIdentityError(f"source root is not a directory: {self.root}")
        for name, value in (("max_records", max_records), ("max_bytes", max_bytes),
                            ("max_drift_paths", max_drift_paths), ("chunk_bytes", chunk_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be an integer in [0, 3]")
        if not isinstance(strict_full_content, bool):
            raise ValueError("strict_full_content must be a bool")
        self.strict_full_content = strict_full_content
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.max_drift_paths = max_drift_paths
        self.max_retries = max_retries
        self.chunk_bytes = chunk_bytes
        self._opener = _opener or open
        self._pid = os.getpid()
        self._nonce = secrets.token_hex(16)
        self._snapshot: dict[str, _Guard] | None = None
        self._file_digests: dict[str, str] = {}
        self._digest: str | None = None
        self._lock = threading.RLock()
        _IDENTITIES.add(self)

    @property
    def process_nonce(self) -> str:
        return self._nonce

    def _scan(self) -> tuple[dict[str, _Guard], int]:
        records: dict[str, _Guard] = {}
        visited = 0
        stack = [self.root]
        while stack:
            directory = stack.pop()
            try:
                entries = []
                with os.scandir(directory) as iterator:
                    for entry in iterator:
                        visited += 1
                        if visited > self.max_records:
                            raise SourceIdentityError(
                                f"source entry cap exceeded ({self.max_records})"
                            )
                        entries.append(entry)
                entries.sort(key=lambda entry: entry.name)
                dir_stat = os.stat(directory, follow_symlinks=False)
            except OSError as exc:
                raise SourceIdentityError(f"cannot inspect source directory {directory}: {exc}") from exc
            relative_directory = os.path.relpath(directory, self.root).replace(os.sep, "/")
            records[relative_directory] = _Guard("directory", _stat_signature(dir_stat))
            for entry in entries:
                try:
                    if entry.is_symlink() and entry.is_dir(follow_symlinks=True):
                        raise UnsupportedSourceLayout(
                            f"symlink directory is not traversed: {entry.path}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                        continue
                    if not entry.name.endswith(".py"):
                        continue
                    rel = os.path.relpath(entry.path, self.root).replace(os.sep, "/")
                    link_stat = entry.stat(follow_symlinks=False)
                    if entry.is_symlink():
                        target_path = os.path.realpath(entry.path)
                        if not _inside_root(self.root, target_path):
                            raise UnsupportedSourceLayout(
                                f"Python source symlink escapes root: {rel}"
                            )
                        target_stat_obj = os.stat(target_path)
                        if not stat.S_ISREG(target_stat_obj.st_mode):
                            raise UnsupportedSourceLayout(
                                f"Python source symlink target is not a regular file: {rel}"
                            )
                        guard = _Guard(
                            "symlink_file", _stat_signature(link_stat), target_path,
                            _stat_signature(target_stat_obj), os.readlink(entry.path),
                        )
                    else:
                        if not stat.S_ISREG(link_stat.st_mode):
                            raise UnsupportedSourceLayout(
                                f"Python source is not a regular file: {rel}"
                            )
                        guard = _Guard("file", _stat_signature(link_stat), entry.path,
                                       _stat_signature(link_stat))
                    records[rel] = guard
                except SourceIdentityError:
                    raise
                except OSError as exc:
                    raise SourceIdentityError(f"cannot inspect source entry {entry.path}: {exc}") from exc
        source_bytes = sum((guard.target_stat or (0, 0, 0, 0, 0, 0))[3]
                           for guard in records.values()
                           if guard.kind in ("file", "symlink_file"))
        if source_bytes > self.max_bytes:
            raise SourceIdentityError(
                f"source byte cap exceeded ({source_bytes} > {self.max_bytes})"
            )
        return records, visited

    def _hash_file(self, relative_path: str, guard: _Guard) -> tuple[str, int]:
        assert guard.target_path is not None and guard.target_stat is not None
        digest = hashlib.sha256()
        total = 0
        try:
            with self._opener(guard.target_path, "rb") as stream:
                before = os.fstat(stream.fileno())
                if _stat_signature(before) != guard.target_stat:
                    raise SourceIdentityRaceError(
                        f"source changed before content read: {relative_path}"
                    )
                while True:
                    block = stream.read(self.chunk_bytes)
                    if not block:
                        break
                    total += len(block)
                    if total > self.max_bytes:
                        raise SourceIdentityError("source byte cap exceeded while hashing")
                    digest.update(block)
                after = os.fstat(stream.fileno())
                if _stat_signature(after) != guard.target_stat:
                    raise SourceIdentityRaceError(
                        f"source changed during content read: {relative_path}"
                    )
        except SourceIdentityError:
            raise
        except OSError as exc:
            raise SourceIdentityError(f"cannot read source file {relative_path}: {exc}") from exc
        if total != guard.target_stat[3]:
            raise SourceIdentityRaceError(
                f"source size changed during content read: {relative_path}"
            )
        return digest.hexdigest(), total

    @staticmethod
    def _tree_digest(file_digests: dict[str, str]) -> str:
        digest = hashlib.sha256(b"QE-source-tree-content-v1\0")
        for relative_path in sorted(file_digests):
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_digests[relative_path].encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def identify(self, *, strict_full_content: bool | None = None) -> SourceIdentityReceipt:
        """Return an identity receipt or fail closed if the tree is unstable."""
        if strict_full_content is not None and type(strict_full_content) is not bool:
            raise ValueError("strict_full_content must be a bool or None")
        strict = self.strict_full_content if strict_full_content is None else strict_full_content
        with self._lock:
            if os.getpid() != self._pid:
                # A forked child must establish its own baseline and nonce.
                self._pid = os.getpid()
                self._nonce = secrets.token_hex(16)
                self._snapshot = None
                self._file_digests = {}
                self._digest = None

            read_files = 0
            read_bytes = 0
            last_race: SourceIdentityRaceError | None = None
            for attempt in range(1, self.max_retries + 2):
                before, visited = self._scan()
                prior = self._snapshot
                all_changed = sorted(
                    path for path in set(before) | (set(prior) if prior is not None else set())
                    if prior is None or before.get(path) != prior.get(path)
                )
                directory_changes = tuple(
                    path for path in all_changed
                    if any(guard is not None and guard.kind == "directory"
                           for guard in (before.get(path), prior.get(path) if prior is not None else None))
                ) if prior is not None else ()
                metadata_file_changes = tuple(
                    path for path in all_changed
                    if any(guard is not None and guard.kind in ("file", "symlink_file")
                           for guard in (before.get(path), prior.get(path) if prior is not None else None))
                ) if prior is not None else ()
                file_paths = {path for path, guard in before.items()
                              if guard.kind in ("file", "symlink_file")}
                if strict or prior is None:
                    to_hash = sorted(file_paths)
                else:
                    to_hash = sorted(
                        path for path in metadata_file_changes
                        if path in file_paths and before[path] != prior.get(path)
                    )
                digests = dict(self._file_digests)
                for deleted in set(digests) - file_paths:
                    del digests[deleted]
                try:
                    for path in to_hash:
                        file_hash, bytes_read = self._hash_file(path, before[path])
                        digests[path] = file_hash
                        read_files += 1
                        read_bytes += bytes_read
                    if strict or all_changed:
                        after, _ = self._scan()
                        if after != before:
                            raise SourceIdentityRaceError(
                                "source tree changed while hashing; retrying boundedly"
                            )
                except SourceIdentityRaceError as exc:
                    last_race = exc
                    if attempt <= self.max_retries:
                        continue
                    break

                content_changes = tuple(
                    path for path in file_paths
                    if prior is not None and self._file_digests.get(path) != digests.get(path)
                )
                source_changes = tuple(sorted(set(metadata_file_changes) | set(content_changes)))
                drifted = bool(source_changes)
                if drifted and len(source_changes) > self.max_drift_paths:
                    raise SourceIdentityError(
                        f"source drift cap exceeded ({len(source_changes)} > {self.max_drift_paths})"
                    )
                source_size = sum((guard.target_stat or (0, 0, 0, 0, 0, 0))[3]
                                  for guard in before.values()
                                  if guard.kind in ("file", "symlink_file"))
                self._snapshot = before
                self._file_digests = digests
                self._digest = self._tree_digest(digests)
                return SourceIdentityReceipt(
                    digest=self._digest,
                    process_nonce=self._nonce,
                    process_id=self._pid,
                    strategy=("strict_full_content" if strict
                              else "process_stat_guarded_changed_file_hash"),
                    stat_guard_verified=True,
                    full_content_checked=strict or prior is None,
                    drifted=drifted,
                    changed_paths=source_changes,
                    directory_changes=directory_changes,
                    stat_guard_fields=("device", "inode", "mode", "size", "mtime_ns", "ctime_ns"),
                    source_size_bytes=source_size,
                    record_cap=self.max_records,
                    byte_cap=self.max_bytes,
                    drift_path_cap=self.max_drift_paths,
                    retry_cap=self.max_retries,
                    files_hashed=read_files,
                    bytes_hashed=read_bytes,
                    stat_records=visited,
                    attempts=attempt,
                )
            raise SourceIdentityRaceError(
                f"source tree remained unstable after {self.max_retries + 1} attempts"
            ) from last_race
