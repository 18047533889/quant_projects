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
The default execution purpose is research_compute: uncertified but otherwise
valid implementations can be evaluated without changing strict input/PIT gates.
Research artifacts remain UNVERIFIED and are not production publications.
An explicitly approved production_compute profile retains certification gates.
Define research formulas on the complete authoring surface, for example::

    from factor_engine.api.factor import Factor
    from factor_engine.api.dsl_parser import parse_recommended_expr
    factors = [Factor(name="robust_residual",
                      expr=parse_recommended_expr("cs_huber_resid(close, open)"))]
    # Inside the main guard: receipt = materialize_all(factors)

The same managed entry is available as ``POST /factor-engine/default/compute``
through the existing authenticated asynchronous job queue, with a JSON body::

    {"factors": [{"name": "robust_residual", "formula": "cs_huber_resid(close, open)"}]}

The server must have its approved profile configured. This endpoint accepts no
backend, mode, source, worker or memory overrides. Its job summary retains the
durable receipt, including partial failures. Legacy compute endpoints are kept.
Invalid request structure is rejected before submission. Malformed formulas and
unknown operator calls become per-factor REJECTED manifest rows; valid peers
continue through the same bounded pipeline. Real HTTP/DataAccess acceptance is
still required; synthetic contract tests do not authorize real input or output.

The CLI accepts the same factors-only JSON object using the approved profile::

    python -m factor_engine.run_pipeline default-compute factors.json

It prints the durable receipt and exits nonzero for a partial/failed batch.
No backend, resource, data-source or output override is accepted by this command.

The default policy uses auto regional backend selection and 80% of measured
effective remaining memory as the shared admission pool. This is not a hard
bound on all native allocator or process-family RSS allocations.
The default worker writes each completed factor through verified durable
artifacts and returns descriptors. The receipt includes per-factor terminal
states; inspect failures rather than treating a completed call as all-success.
The v2 implementation remains subject to its current evidence/admission gates.
The script main guard is required by spawn; it is not a performance option.

Revalidate a completed run without resubmitting its factor definitions::

    if __name__ == "__main__":
        receipt = revalidate_completed_run(previous_run_id)

This retains the original run identity and checks previously committed bytes.
It does not yet resume unfinished work: pending/uncertain commits are rejected
until persisted worker ownership and exit can be proven. Keep the same approved
profile and original artifacts; changing their identity is not a resume.
"""

from factor_engine import get_engine


def materialize_all(factors):
    with get_engine() as engine:
        return engine.run_many(factors)


def revalidate_completed_run(run_id):
    with get_engine() as engine:
        return engine.run_many((), resume_run_id=run_id)
