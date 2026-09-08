"""Default durable batch call after an administrator approves the data profile.

Set FACTOR_ENGINE_V2_PROFILE to that existing profile. It must explicitly bind
the date range, universe, canonical HFQ source and isolated output directory.
It must also contain ``expected_source_content_digests``: a mapping from every
required dataset (including ashare_stock_daily_adj) to its approved frozen
DataAccess source-snapshot content digest (32 or 64 lowercase hex characters).
This is the exact prepared object-set identity, not a manifest path or approval
label. Obtain it through the administrator's snapshot approval workflow; do not
automatically pin whatever snapshot happens to be current when execution starts.
Optional manifest tokens are additional change detectors, not substitutes.
Different dependency/warmup ranges may select different object sets: the current
one-digest-per-dataset profile rejects mismatches instead of changing scope.
This example neither invents that authorization nor publishes factors.

Usage from an existing factor-definition program::

    if __name__ == "__main__":
        receipt = materialize_all(all_factors)

No backend, concurrency, batch-size or memory tuning is required from callers.
The default policy uses auto regional backend selection and 80% of measured
effective remaining memory as the shared admission pool. This is not a hard
bound on all native allocator or process-family RSS allocations.
The default worker writes each completed factor through verified durable
artifacts and returns descriptors. The receipt includes per-factor terminal
states; inspect failures rather than treating a completed call as all-success.
The v2 implementation remains subject to its current evidence/admission gates.
The script main guard is required by spawn; it is not a performance option.
"""

from factor_engine import get_engine


def materialize_all(factors):
    with get_engine() as engine:
        return engine.run_many(factors)
