# RAW summary cache ABBA result (2026-10-04)

## Outcome

The stored four-run ABBA comparison shows identical optimizer outputs and candidate ledgers in cached and uncached modes. The cache reduced RAW-summary computations from 1,777 to 33 per measured run, with 1,744 hits and zero failures. Timing was mixed: one pair favored cached by 9.7799%, the other favored uncached by 3.7316%. The median paired time reduction is 3.0241%, descriptive only; it does not establish a stable speedup.

## Scope and method

- Executed in server-c terminal session `session46462terminal0` on the strict F16 cohort: 500 dates × 256 assets × 16 factors. It is not a full-market or multiyear evaluation. The sealed TEST partition was not evaluated.
- Order was uncached → cached → cached → uncached. Warmup was a separate cached run (220.58 s) and is excluded from both timing pairs.
- Both pairs had 1,777 RAW-summary requests. Each uncached run recomputed 1,777 summaries; each cached run computed 33 and hit 1,744. All four runs had zero RAW-summary failures.
- Pair 1: uncached 199.4152 s; cached 179.9126 s; cached time reduction 9.7799%.
- Pair 2: cached 186.0443 s; uncached 179.3517 s; cached time reduction −3.7316%.
- The stored equivalence record says all outputs and ledgers were identical across five runs (the separate warmup plus four ABBA runs). The result fingerprint, cohort fingerprint, optimized-values hash, optimized-validity hash, per-factor plan identities, and candidate-ledger hashes agree. All 16 factors were `raw_retained` / `NO_OP_RAW`; this is the optimizer's selected result under the recorded search and gates, not evidence of a method implementation defect.

## Integrity and qualification note

The JSON records matching before/after selected-source hashes, runtime objects, and runtime fingerprints. Its measured benchmark-script SHA-256 is `70fe29acf1eb334375d4d914d629dea223859d5185a40034235d1d7e9b366ec5`; the recorded `research_batch.py` and `research_summary_cache.py` hashes are respectively `b78ceee015de3dca93670b7a958503202f7d6923b6c4ba1696d25ec6615fa323` and `882a8bae78e885b88218e593b30ebb1d22c32a51df07f05f2f4130a9a1ed9d2b`.

A subsequent qualification-gate edit was manually checked against the five stored run receipts and passed 5/5. That check validates the gate against the existing receipts; it is not an ABBA rerun under the new script. The current benchmark script hash is `632752854b7791ff7bebf76c6d4e037b34560f78e2ccd62fb7ee23528af60deb`, and that post-edit hash was not backfilled into the measurement JSON. Do not attribute the recorded timings or outputs to the post-edit script.

## Interpretation

This is a cache-semantics/equivalence result on the stated bounded cohort, plus noisy paired timings. It supports neither a “fastest” claim nor a claim that the optimizer has no bugs. The speed comparison is not a statistical-significance test or a performance guarantee.

Source receipt: [`real_raw_summary_cache_oct04.json`](real_raw_summary_cache_oct04.json).
