"""Explicit, bounded lookup of caller-trusted source-profile report candidates.

Reports are parsed by the strict report reader, but are not authenticated or
qualified against the live runtime context here. Callers must re-qualify the
returned records before using them for routing. This module never scans for
reports, reads environment variables, benchmarks, or initializes CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Mapping

from quant_evaluator.scripts.source_profile_report_reader import (
    load_source_profile_report,
)

_MAX_ENTRIES = 32
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SourceQualificationCandidate:
    """Unverified typed receipt records from one strictly parsed report."""

    candidate_id: str
    records: tuple


@dataclass(frozen=True)
class SourceQualificationLookup:
    """Safe lookup outcome; reason codes never include paths or reader errors."""

    candidate: SourceQualificationCandidate | None
    reason_code: str


class FileSourceQualificationProvider:
    """Read reports only from an explicit fingerprint-to-path mapping.

    Configuration copies at most 32 entries and performs no filesystem reads.
    The strict reader bounds each report read to 1 MiB.
    """

    def __init__(self, report_paths: Mapping[str, str | os.PathLike[str]]):
        if type(report_paths) is not dict:
            raise TypeError("report_paths_must_be_dict")
        if len(report_paths) > _MAX_ENTRIES:
            raise ValueError("entry_limit")
        normalized = {}
        for fingerprint, path in report_paths.items():
            if type(fingerprint) is not str or _SHA256.fullmatch(fingerprint) is None:
                raise ValueError("fingerprint_invalid")
            if not isinstance(path, (str, os.PathLike)):
                raise ValueError("report_path_invalid")
            try:
                path_text = os.fspath(path)
                if not path_text:
                    raise ValueError("report_path_invalid")
                path = Path(path_text)
            except (TypeError, ValueError, OSError) as exc:
                raise ValueError("report_path_invalid") from exc
            normalized[fingerprint] = path
        self._report_paths = MappingProxyType(normalized)

    def lookup(self, request_fingerprint: str) -> SourceQualificationLookup:
        """Return a strict-reader candidate only for an exact request match."""
        if (type(request_fingerprint) is not str
                or _SHA256.fullmatch(request_fingerprint) is None):
            return SourceQualificationLookup(None, "request_fingerprint_invalid")
        path = self._report_paths.get(request_fingerprint)
        if path is None:
            return SourceQualificationLookup(None, "candidate_not_found")
        try:
            report = load_source_profile_report(path)
        except OSError:
            return SourceQualificationLookup(None, "report_unreadable")
        except Exception:
            # Do not expose parser messages, which can contain untrusted text.
            return SourceQualificationLookup(None, "report_invalid")
        try:
            records = report.records
            if type(records) is not tuple or len(records) != 2:
                return SourceQualificationLookup(None, "report_invalid")
            if any(record.context.request_content_sha256 != request_fingerprint
                   for record in records):
                return SourceQualificationLookup(None, "fingerprint_mismatch")
            manifest_sha256 = report.manifest_sha256
            if (type(manifest_sha256) is not str
                    or _SHA256.fullmatch(manifest_sha256) is None):
                return SourceQualificationLookup(None, "report_invalid")
        except Exception:
            return SourceQualificationLookup(None, "report_invalid")
        # An opaque report-binding label avoids disclosing the configured path.
        candidate_id = "abba:" + hashlib.sha256(
            (request_fingerprint + manifest_sha256).encode("ascii")).hexdigest()
        return SourceQualificationLookup(
            SourceQualificationCandidate(candidate_id, records), "candidate_found")


_lock = threading.RLock()
_configured_provider: FileSourceQualificationProvider | None = None


def _reset_after_fork() -> None:
    """Drop configured caller-trusted paths in a forked child."""
    global _lock, _configured_provider
    _lock = threading.RLock()
    _configured_provider = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def configure_source_qualification_provider(
    provider: FileSourceQualificationProvider,
) -> None:
    """Install an explicitly constructed provider without reading its files."""
    if type(provider) is not FileSourceQualificationProvider:
        raise TypeError("provider_invalid")
    global _configured_provider
    with _lock:
        _configured_provider = provider


def configure_source_qualification_provider_if_absent(provider: FileSourceQualificationProvider) -> bool:
    """Atomically claim an empty slot without replacing another caller."""
    if type(provider) is not FileSourceQualificationProvider:
        raise TypeError("provider_invalid")
    global _configured_provider
    with _lock:
        if _configured_provider is not None:
            return False
        _configured_provider = provider
        return True


def clear_source_qualification_provider_if_current(provider: FileSourceQualificationProvider) -> bool:
    """Atomically release only the slot owned by this exact provider."""
    if type(provider) is not FileSourceQualificationProvider:
        raise TypeError("provider_invalid")
    global _configured_provider
    with _lock:
        if _configured_provider is not provider:
            return False
        _configured_provider = None
        return True


def get_source_qualification_provider() -> FileSourceQualificationProvider | None:
    with _lock:
        return _configured_provider


def clear_source_qualification_provider() -> None:
    """Remove the process-level provider; primarily useful for lifecycle/tests."""
    global _configured_provider
    with _lock:
        _configured_provider = None


def lookup_source_qualification_candidate(
    request_fingerprint: str,
) -> SourceQualificationLookup:
    """Query the configured provider, returning a safe miss when absent."""
    configured = get_source_qualification_provider()
    if configured is None:
        return SourceQualificationLookup(None, "provider_not_configured")
    return configured.lookup(request_fingerprint)


__all__ = (
    "FileSourceQualificationProvider", "SourceQualificationCandidate",
    "SourceQualificationLookup", "clear_source_qualification_provider",
    "configure_source_qualification_provider", "get_source_qualification_provider",
    "configure_source_qualification_provider_if_absent",
    "clear_source_qualification_provider_if_current",
    "lookup_source_qualification_candidate",
)
