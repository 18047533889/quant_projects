"""Bounded benchmark-only phase timing for one verified F61 COS factor."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

MAX_OBJECT_MIB = 128
MIN_FREE_BYTES = 2 * MAX_OBJECT_MIB * 1024**2 + 64 * 1024**2

@dataclass
class PhaseLedger:
    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    def add(self, name, elapsed):
        self.seconds[name] = self.seconds.get(name, 0.0) + max(0.0, float(elapsed))
        self.counts[name] = self.counts.get(name, 0) + 1

class _TimedRead:
    def __init__(self, stream, ledger): self.stream, self.ledger = stream, ledger
    def read(self, *a, **k):
        started = time.perf_counter()
        try: return self.stream.read(*a, **k)
        finally: self.ledger.add("hash_file_read_s", time.perf_counter() - started)
    def __enter__(self): return self
    def __exit__(self, *exc): return self.stream.__exit__(*exc)
    def __getattr__(self, name): return getattr(self.stream, name)

class _TimedHasher:
    def __init__(self, hasher, ledger): self.hasher, self.ledger = hasher, ledger
    def update(self, value):
        started = time.perf_counter()
        try: return self.hasher.update(value)
        finally: self.ledger.add("hash_compute_s", time.perf_counter() - started)
    def hexdigest(self):
        started = time.perf_counter()
        try: return self.hasher.hexdigest()
        finally: self.ledger.add("hash_compute_s", time.perf_counter() - started)

@contextmanager
def instrument_existing_read(ledger, *, exact_uri, cli, factor_dataset):
    """Temporarily time calls in DataAccess; all original functions still run."""
    from unittest.mock import patch
    import data_access.cos.remote as remote
    import data_access.cos.research as research
    from data_access.read.read_handle import ReadHandle
    from data_access.store import DataAccessStore
    original_head = remote.cos_cli_head
    def head(uri, *a, **k):
        if uri != exact_uri: return original_head(uri, *a, **k)
        t = time.perf_counter()
        try: return original_head(uri, *a, **k)
        finally: ledger.add("head_s", time.perf_counter() - t)
    original_run = subprocess.run
    def run(argv, *a, **k):
        match = (isinstance(argv, (list, tuple)) and len(argv) >= 3 and
                 str(argv[0]) == cli and argv[1] == "cp" and argv[2] == exact_uri)
        if not match: return original_run(argv, *a, **k)
        t = time.perf_counter()
        try: return original_run(argv, *a, **k)
        finally: ledger.add("download_s", time.perf_counter() - t)
    original_open = Path.open
    def path_open(path, *a, **k):
        stream = original_open(path, *a, **k)
        if (path.name == "object.parquet" and path.parent.name.startswith("research-object-")
                and a and a[0] == "rb"):
            return _TimedRead(stream, ledger)
        return stream
    original_arrow = ReadHandle.to_arrow
    def to_arrow(handle):
        t = time.perf_counter()
        try: return original_arrow(handle)
        finally: ledger.add("parquet_arrow_materialize_s", time.perf_counter() - t)
    original_store_read = DataAccessStore.read
    def store_read(store, dataset, *a, **k):
        if dataset != factor_dataset: return original_store_read(store, dataset, *a, **k)
        t = time.perf_counter()
        try: return original_store_read(store, dataset, *a, **k)
        finally: ledger.add("parquet_store_read_s", time.perf_counter() - t)
    # Patch only research.py's module references, and restore on all exits.
    hmod = research.hashlib
    hashes = SimpleNamespace(
        md5=lambda *a, **k: _TimedHasher(hmod.md5(*a, **k), ledger),
        sha256=lambda *a, **k: _TimedHasher(hmod.sha256(*a, **k), ledger))
    smod = research.subprocess
    scoped_subprocess = SimpleNamespace(run=run, SubprocessError=smod.SubprocessError)
    with patch.object(remote, "cos_cli_head", head), \
         patch.object(research, "subprocess", scoped_subprocess), \
         patch.object(research, "hashlib", hashes), \
         patch.object(Path, "open", path_open), \
         patch.object(ReadHandle, "to_arrow", to_arrow), \
         patch.object(DataAccessStore, "read", store_read):
        yield

def _profile_axes(table):
    import pandas as pd
    from quant_evaluator.scripts.benchmark_real_cos_factor_tiles import axis_hash
    phases = {}
    t = time.perf_counter(); frame = table.to_pandas()
    phases["arrow_to_pandas_s"] = time.perf_counter() - t
    t = time.perf_counter()
    if "timestamp" not in frame: raise ValueError("timestamp column is missing")
    frame = frame.set_index("timestamp")
    frame.index = pd.to_datetime(frame.index).normalize()
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("factor object has duplicate axes")
    frame = frame.loc[:, [c for c in frame if c.endswith((".SZ", ".SH"))]].sort_index()
    phases["axis_normalize_filter_sort_s"] = time.perf_counter() - t
    t = time.perf_counter(); digest = axis_hash(frame)
    phases["axis_hash_s"] = time.perf_counter() - t
    return {"rows": int(len(frame)), "columns": int(len(frame.columns)),
            "axis_sha256": digest,
            "first_date": frame.index.min().isoformat() if len(frame) else None,
            "last_date": frame.index.max().isoformat() if len(frame) else None}, phases

def _read_once(record, manifest_sha, object_mib):
    from data_access.core.engine import DuckDBEngine
    from data_access.registry.loader import DatasetRegistry
    from data_access.store import DataAccessStore
    from factor_optimizer.research_manifest import read_bound_factor, read_bound_manifest
    from quant_evaluator.scripts.load_real_cos_factor_batch import BASE, _ds
    name, uri, digest, size = record
    if size > object_mib * 1024**2: raise ValueError("object exceeds strict byte bound")
    m_uri = f"{BASE}/metadata/{manifest_sha}/landing_manifest.json"
    md = _ds("source_manifest", m_uri.rsplit("/", 1)[0], "landing_manifest.json", "json")
    fd = _ds("factor_panel", uri.rsplit("/", 1)[0], name + ".parquet", "parquet")
    engine = DuckDBEngine(threads=1)
    try:
        store = DataAccessStore(DatasetRegistry({md.name: md, fd.name: fd}), engine)
        t = time.perf_counter(); snapshot = read_bound_manifest(store, md.name, allow_research=True)
        manifest_s = time.perf_counter() - t
        if snapshot.manifest_sha256 != manifest_sha: raise ValueError("manifest digest mismatch")
        ledger = PhaseLedger(); cli = os.environ["DATA_ACCESS_COS_CLI"]
        t = time.perf_counter()
        with instrument_existing_read(ledger, exact_uri=uri, cli=cli,
                                     factor_dataset=fd.name):
            bound = read_bound_factor(store, md.name, fd.name, name,
                allow_research=True, max_object_mib=object_mib, manifest_snapshot=snapshot)
        elapsed = time.perf_counter() - t; obj = bound.factor
        if (bound.manifest_sha256 != manifest_sha or obj.source_uri != uri or
                obj.content_sha256 != digest or obj.downloaded_bytes != size):
            raise ValueError("object identity differs from verified manifest")
        expected_counts = {"head_s": 2, "download_s": 1,
                           "parquet_store_read_s": 1,
                           "parquet_arrow_materialize_s": 1}
        if any(ledger.counts.get(key, 0) != value for key, value in expected_counts.items()):
            raise RuntimeError("phase hooks did not observe expected object stages")
        axes, conv = _profile_axes(obj.table)
        return {"factor_id": name, "uri": uri, "manifest_sha256": manifest_sha,
            "sha256": obj.content_sha256, "etag": obj.source_etag, "bytes": obj.downloaded_bytes,
            "manifest_read_s": manifest_s, "bound_factor_read_s": elapsed,
            "phases_s": {**ledger.seconds, **conv, "unattributed_bound_read_s":
                max(0.0, elapsed - sum(ledger.seconds.values()))},
            "phase_counts": ledger.counts, "axis": axes}
    finally: engine.close()

def profile_f61(manifest_sha, *, max_object_mib=128):
    if type(max_object_mib) is not int or not 1 <= max_object_mib <= MAX_OBJECT_MIB:
        raise ValueError("max_object_mib must be in 1..128")
    root = Path(os.environ.get("DATA_ACCESS_COS_CACHE_ROOT", ""))
    if not root.is_absolute() or not root.exists(): raise ValueError("cache root must exist")
    if shutil.disk_usage(root).free < MIN_FREE_BYTES: raise OSError("insufficient temp space")
    cli = os.environ.get("DATA_ACCESS_COS_CLI", "")
    if cli != "/usr/local/bin/admin-cos" or not Path(cli).is_file():
        raise ValueError("approved DATA_ACCESS_COS_CLI is not configured")
    os.environ.setdefault("ASHARE_PARQUET_ROOT", "/home/sunhaiwei/cos_data")
    from data_access.core.engine import DuckDBEngine
    from data_access.registry.loader import DatasetRegistry
    from data_access.store import DataAccessStore
    from factor_optimizer.research_manifest import read_bound_manifest
    from quant_evaluator.scripts.benchmark_real_cos_factor_tiles import select_records
    from quant_evaluator.scripts.load_real_cos_factor_batch import BASE, _ds
    uri = f"{BASE}/metadata/{manifest_sha}/landing_manifest.json"
    md = _ds("source_manifest", uri.rsplit("/", 1)[0], "landing_manifest.json", "json")
    engine = DuckDBEngine(threads=1)
    try:
        store = DataAccessStore(DatasetRegistry({md.name: md}), engine)
        snap = read_bound_manifest(store, md.name, allow_research=True)
        if snap.manifest_sha256 != manifest_sha: raise ValueError("F61 manifest digest mismatch")
        records = select_records(snap.factors, 61, max_object_mib, 6144)
    finally: engine.close()
    record = records[len(records)//2]
    first = _read_once(record, manifest_sha, max_object_mib)
    second = _read_once(record, manifest_sha, max_object_mib)
    if (first["sha256"], first["bytes"], first["axis"]) != (second["sha256"], second["bytes"], second["axis"]):
        raise ValueError("sequential object identity or axis changed")
    return {"schema": "qe_cos_single_object_profile.v1", "manifest_sha256": manifest_sha,
        "f61_selected_count": len(records), "selection": "middle record by (bytes,factor_id) in F61 sample",
        "object_max_bytes": max_object_mib*1024**2,
        "temporary_object_policy": "private DataAccess temp, removed after each read",
        "repeat_definition": "second sequential remote read; cache state uncontrolled; no persistent local cache",
        "measurements": [{"pass": "first_read", **first}, {"pass": "sequential_repeat", **second}]}

def main(argv=None):
    from quant_evaluator.scripts.benchmark_real_cos_metric_batch import MANIFEST_SHA256
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    p.add_argument("--max-object-mib", type=int, default=128)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    result = profile_f61(a.manifest_sha256, max_object_mib=a.max_object_mib)
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if a.output: a.output.write_text(data, encoding="utf-8")
    else: sys.stdout.write(data)
    return 0

if __name__ == "__main__": raise SystemExit(main())
