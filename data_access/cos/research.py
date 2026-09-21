"""Bounded exact-object research reads for declared, flat COS datasets.

No credential parsing, wildcard sync, persistent data copy or production/PIT
certification. The registered dataset still owns authorization and semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, unquote

from data_access.core.exceptions import ValidationError
from data_access.read.query_budget import QueryBudget, is_strict_semantics, validate_query_request
from data_access.runtime.prepared_read import DeadlineContext, VerifiedPhysicalScope


@dataclass(frozen=True)
class ResearchObjectRead:
    table: Any
    source_uri: str
    source_etag: str
    content_sha256: str
    downloaded_bytes: int
    dataset: str


def read_declared_cos_object(
    store, dataset: str, *, params: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None, time_range=None,
    instrument_filter=None, query_budget: QueryBudget | None = None,
    allow_research: bool = False,
) -> ResearchObjectRead:
    """Read one declared Parquet or small JSON object through the COS gateway.

    Resolve validated dataset parameters, authorize before network I/O, inspect
    exact-object metadata, admit up to 64 MiB (2 MiB for JSON), then fetch into a private ephemeral
    directory. Both metadata observations must agree; single-part MD5 ETags
    are verified against the downloaded content. DataAccess performs the
    actual column/time/instrument read with the original dataset contract.
    The temporary object is removed on success AND failure; the caller retains
    the Arrow table and remote provenance, not a persistent local mirror.

    This explicit research API rejects strict/production contexts. It does not
    bypass production generic storage/PIT gates or claim sealed provenance.
    """
    if allow_research is not True or is_strict_semantics():
        raise ValidationError("explicit research opt-in required; strict/production is unsupported")
    store.authorize_dataset(dataset)
    store._authorize_factor_params(dataset, params)
    ds = store._registry.get(dataset)
    json_metadata = ds.format in {'json', 'jsonl', 'ndjson'}
    scan_cap = (2 if json_metadata else 64) * 1024**2
    result_cap = (8 if json_metadata else 128) * 1024**2
    row_cap = 10_000 if json_metadata else 2_000_000
    budget = store._resolve_read_budget(ds, query_budget)
    budget = replace(
        budget, max_scan_bytes=min(budget.max_scan_bytes or scan_cap, scan_cap),
        max_rows=min(budget.max_rows or row_cap, row_cap),
        max_result_bytes=min(budget.max_result_bytes or result_cap, result_cap),
        max_elapsed_ms=min(budget.max_elapsed_ms or 60_000., 60_000.))
    validate_query_request(budget, columns=list(columns) if columns else None,
                           time_range=time_range)
    if budget.max_remote_requests is not None and budget.max_remote_requests < 3:
        raise ValidationError("remote request budget requires HEAD + GET + HEAD")
    from data_access.cos import remote
    if not remote.declares_cos_storage(ds) or not (json_metadata or ds.format in {'parquet', 'pq'}):
        raise ValidationError("research object must be a declared COS Parquet or JSON dataset")
    paths = remote.resolve_remote_paths(ds, time_range=time_range, params=params)
    if len(paths) != 1:
        raise ValidationError("research read requires exactly one object")
    parsed = urlsplit(paths[0])
    key = parsed.path.lstrip("/")
    if (not parsed.netloc or parsed.query or parsed.fragment or
            any(c in key for c in "*?[]{}") or
            ".." in PurePosixPath(unquote(key)).parts or
            not key.endswith(('.json', '.jsonl', '.ndjson') if json_metadata else ('.parquet',))):
        raise ValidationError("research read requires an exact safe object matching declared format")
    uri = "cos://" + parsed.netloc + "/" + key
    from data_access.cos.mirror import _cli_binary
    cli = _cli_binary()
    deadline = DeadlineContext.start(budget, source="research_cos_object")
    def metadata():
        result = remote.cos_cli_head(uri, cli=cli, timeout_s=deadline.remaining_secs())
        if (not result or result.get("key") != key or not result.get("etag") or
                type(result.get("size")) is not int or result["size"] <= 0):
            raise ValidationError("exact object metadata is missing or mismatched")
        return result
    before = metadata()
    # CLI metadata uses rounded human-readable sizes. A conservative upper
    # bound avoids treating a rounded-down size as an exact byte measurement.
    scan_upper = int(before["size"] * 1.01) + 1024
    if scan_upper > budget.max_scan_bytes:
        raise ValidationError("object exceeds download/scan budget")
    root = remote.ensure_cache_root_secure()
    with tempfile.TemporaryDirectory(prefix="research-object-", dir=root) as tmp:
        dest = Path(tmp) / ('object.json' if json_metadata else 'object.parquet')
        try:
            subprocess.run([cli, "cp", uri, str(dest)], check=True, capture_output=True,
                           text=True, timeout=deadline.remaining_secs())
        except (OSError, subprocess.SubprocessError) as exc:
            # Do not surface gateway stderr, which may contain provider details.
            raise ValidationError("COS exact-object transfer failed") from exc
        if dest.is_symlink() or not dest.is_file():
            raise ValidationError("object transfer did not produce a regular file")
        os.chmod(dest, 0o600)
        size = dest.stat().st_size
        if size <= 0 or size > scan_upper or size > budget.max_scan_bytes:
            raise ValidationError("downloaded object exceeds budget or is empty")
        after = metadata()
        if (before["etag"], before["size"]) != (after["etag"], after["size"]):
            raise ValidationError("source object changed during transfer")
        md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
        with dest.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                md5.update(chunk)
                sha.update(chunk)
        etag = str(before["etag"]).strip('"')
        if re.fullmatch(r"[0-9a-fA-F]{32}", etag) and md5.hexdigest() != etag.lower():
            raise ValidationError("object content digest does not match ETag")
        scope = VerifiedPhysicalScope(dataset_id=dataset, exact_objects=(str(dest),),
                                      contract_digest=store._contract_digest_for(dataset))
        table = store.read(dataset, columns=columns, time_range=time_range,
                           instrument_filter=instrument_filter,
                           query_budget=replace(budget, max_elapsed_ms=deadline.remaining_ms()),
                           physical_scope=scope, result="arrow", **dict(params or {})).to_arrow()
        deadline.check("research object read")
        return ResearchObjectRead(table, uri, etag, sha.hexdigest(), size, dataset)
