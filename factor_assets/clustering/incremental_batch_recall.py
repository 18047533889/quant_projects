"""Bounded preparation policy for oversized exact incremental batch recall."""
from __future__ import annotations

import numpy as np

from factor_assets.clustering.incremental_recall import scan_exact_member_winners

MAX_BATCH_PREPARATION_BYTES = 32 * 1024**2
QUERY_BLOCK_ROWS = 256


def prepare_oversized_batch_winners(
    fingerprints, cluster_versions, fingerprints_by_id, *, cache_bytes,
    to_embedding, normalize_rows, unit_queries_out=None,
):
    """Reuse member chunks across query blocks without changing row reductions.

    Return an empty mapping when no bounded batch preparation is admitted; the
    caller then uses the existing single-query path. This estimates additional
    query buffers and winner bookkeeping, not inputs, output audits or RSS.
    """
    count = len(fingerprints)
    if count < 2:
        return {}
    width = len(fingerprints[0].embedding)
    if any(len(fp.embedding) != width for fp in fingerprints):
        return {}
    oversized = [(cid, cv) for cid, cv in cluster_versions.items()
                 if cv.member_factor_ids and
                 len(cv.member_factor_ids) * (width * 8 + 64) + 256 > cache_bytes
                 and any(fingerprints_by_id.get(fid) is not None
                         for fid in cv.member_factor_ids)]
    estimate = count * (width * 8 * 3 + len(oversized) * 128)
    if not oversized or estimate > MAX_BATCH_PREPARATION_BYTES:
        return {}
    # Normalize separately exactly as in the existing query-major path.
    queries = np.empty((count, width), dtype=np.float64)
    for index, fp in enumerate(fingerprints):
        query = to_embedding(fp)
        queries[index] = normalize_rows(query.reshape(1, -1))[0]
    winners = {}
    for cid, cv in oversized:
        results = []
        for start in range(0, count, QUERY_BLOCK_ROWS):
            results.extend(scan_exact_member_winners(
                fingerprints_by_id, cv.member_factor_ids,
                queries[start:start + QUERY_BLOCK_ROWS],
                to_embedding=to_embedding, normalize_rows=normalize_rows,
            ))
        winners[cid] = results
    if unit_queries_out is not None:
        unit_queries_out.append(queries)
    return winners
