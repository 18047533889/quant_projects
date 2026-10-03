"""Bounded public incremental-assignment A/B with the same stable norm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factor_assets.clustering import incremental as current
from factor_assets.contracts.cluster_governance import ClusterScale, ClusterVersionArtifact
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact

BASELINE = "0925681e59259fb54ac2f635cc283580ce583cac"


def _source_hashes():
    paths = [Path(current.__file__),
             Path(__file__).resolve(),
             ROOT / "factor_assets/similarity/unit_vectors.py"]
    helper = ROOT / "factor_assets/clustering/incremental_recall.py"
    if helper.exists():
        paths.append(helper)
    return {path.relative_to(ROOT).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _fingerprint(identifier, vector):
    return SimilarityFingerprintArtifact(
        factor_id=identifier, embedding=tuple(map(float, vector)),
        embedding_spec="incremental-batch-ab-v1", snapshot="synthetic-20261003",
        universe="synthetic-embedding-library", window="fixed-ab-window",
        preprocessing_ref="none", mask_policy="finite-nonzero",
        direction="signed", aggregation_method="embedding-cosine",
        embedding_model_version="seeded-normal-v1", value_ref=f"synthetic:{identifier}",
        profile_ref=f"synthetic-profile:{identifier}")


def _load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:factor_assets/clustering/incremental.py"],
        cwd=ROOT, text=True)
    module = types.ModuleType("_incremental_batch_before_cache")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASELINE}:incremental.py", "exec"), module.__dict__)
    # Normalize identically: isolate batch reuse/domain-check costs, rather
    # than conflating corrected arithmetic with a cache-performance claim.
    module._normalize_rows = current._normalize_rows
    return module, hashlib.sha256(source.encode()).hexdigest()


def _payload(result):
    payload = result.to_dict()
    payload.pop("created_at", None)
    return payload


def _oracle(result, queries, mapping, clusters):
    if len(result.candidates) != len(queries) * len(clusters):
        raise AssertionError("public candidate coverage is not exact")
    for index, (query, assignment) in enumerate(zip(queries, result.assignments, strict=True)):
        q = np.asarray(query.embedding, dtype=np.longdouble)
        q /= np.sqrt(np.sum(q * q))
        # Public candidate tuples are ordered query-major then cluster-major.
        start = index * len(clusters)
        candidates = result.candidates[start:start + len(clusters)]
        if [candidate.cluster_id for candidate in candidates] != list(clusters):
            raise AssertionError("public cluster coverage/order changed")
        if len(candidates) != len(clusters) or assignment.factor_id != query.factor_id:
            raise AssertionError("public query/candidate coverage changed")
        for candidate in candidates:
            cluster = clusters[candidate.cluster_id]
            scored = []
            for identifier in cluster.member_factor_ids:
                v = np.asarray(mapping[identifier].embedding, dtype=np.longdouble)
                v /= np.sqrt(np.sum(v * v))
                scored.append((identifier, float(np.sum(q * v))))
            top = max(score for _, score in scored)
            expected = next(identifier for identifier, score in scored if score == top)
            if candidate.factor_id != expected or abs(candidate.similarity - top) > 2e-14:
                raise AssertionError("public candidate differs from independent longdouble cosine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, default=2048)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not (128 <= args.members <= 4096 and 16 <= args.queries <= 256
            and 4 <= args.dimensions <= 128 and 1 <= args.clusters <= 32
            and args.clusters <= args.members and 2 <= args.rounds <= 7):
        parser.error("sizes exceed the bounded benchmark envelope")
    if args.report.exists():
        parser.error("refusing to overwrite an existing evidence report")
    if any(os.environ.get(name) != "1" for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        parser.error("set OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1")
    if not hasattr(np, "longdouble") or np.finfo(np.longdouble).nmant <= 52:
        parser.error("independent oracle requires extended-precision longdouble")
    available = next(int(line.split()[1]) * 1024 for line in
                     Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 4 * 1024**3 or shutil.disk_usage(ROOT).free < 1024**3:
        parser.error("insufficient available RAM or disk for bounded benchmark")
    if os.getloadavg()[0] > (os.cpu_count() or 1):
        parser.error("host load exceeds CPU count; defer controlled timing")
    source_hashes_before = _source_hashes()
    rng = np.random.default_rng(20261003)
    vectors = rng.normal(size=(args.members + args.queries, args.dimensions))
    members = [_fingerprint(f"M{i:05d}", v) for i, v in enumerate(vectors[:args.members])]
    queries = [_fingerprint(f"Q{i:05d}", v) for i, v in enumerate(vectors[args.members:])]
    mapping = {fp.factor_id: fp for fp in members + queries}
    clusters = {}
    for index, part in enumerate(np.array_split(np.arange(args.members), args.clusters)):
        identifiers = tuple(members[int(i)].factor_id for i in part)
        name = f"C{index:03d}"
        clusters[name] = ClusterVersionArtifact(
            logical_cluster_id=name, cluster_set_version_ref="synthetic-batch-ab",
            member_factor_ids=identifiers, representative_factor_id=identifiers[0],
            scale=ClusterScale.MICRO_CLUSTER)
    baseline, baseline_hash = _load_baseline()
    def call(module):
        return module.incremental_assign(queries, clusters, mapping, batch_id="synthetic-fixed-ab")
    expected = _payload(call(baseline))
    new_result = call(current)
    if _payload(new_result) != expected:
        raise AssertionError("public baseline/new artifacts differ")
    _oracle(new_result, queries, mapping, clusters)
    samples = {"baseline": [], "current": []}
    load_before = os.getloadavg()
    started = time.time()
    for _ in range(args.rounds):
        for name, module in (("baseline", baseline), ("current", current),
                             ("current", current), ("baseline", baseline)):
            start = time.perf_counter()
            result = call(module)
            elapsed = time.perf_counter() - start
            if _payload(result) != expected:
                raise AssertionError("timed public artifacts differ")
            samples[name].append(elapsed)
    source_hashes_after = _source_hashes()
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("declared recall sources changed during timing")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report = {"kind": "incremental_batch_recall_ab.v1", "status": "complete",
        "members": args.members, "queries": args.queries, "dimensions": args.dimensions,
        "clusters": args.clusters, "abba_rounds": args.rounds, "samples_seconds": samples,
        "medians_seconds": medians, "speedup": medians["baseline"] / medians["current"],
        "baseline_revision": BASELINE, "baseline_source_sha256": baseline_hash,
        "source_hashes": source_hashes_after, "public_artifact_exact_parity": True,
        "independent_longdouble_oracle": True, "unix_started": started,
        "unix_finished": time.time(), "host_load_before": load_before,
        "host_load_after": os.getloadavg(), "python": platform.python_version(),
        "numpy": np.__version__, "limitations": ["Synthetic embeddings, not real factor-value scoring.",
        "Baseline uses current stable normalization; timing isolates batch reuse and validation.",
        "Source digests cover declared recall files only, not transitive or in-memory closure.",
        "Shared business host load is not fully controlled; does not certify all sizes/backends."]}
    payload = json.dumps(report, indent=2, allow_nan=False)
    if len(payload.encode()) > 1024**2:
        raise RuntimeError("report exceeds the bounded evidence size")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(payload + "\n")
    print(json.dumps({"status": "complete", "medians_seconds": medians,
                      "speedup": report["speedup"], "parity": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
