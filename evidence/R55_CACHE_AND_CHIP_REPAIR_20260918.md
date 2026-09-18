# R55 cache safety and chip-coordinate repairs

## Runtime cache

A caller-provided cache survived two run_many calls and reused an old ts_mean result after source generation changed. The reproducer's calculation count stayed at 1 despite values increasing by 1000; see R55_CACHE_BEFORE_EXCERPT.txt (the full original before-log remains on server).

run_many now binds a cache view to the current source snapshot plus the caller namespace. Views share the backing cache; they do not duplicate its contents, clear the caller cache, or mutate caller scope. Same-snapshot reuse remains tested. Changed snapshots/datasets/semantic versions cannot reuse the old result. Unknown snapshot proof disables reuse.

Streaming pins the exact initial scope and rejects a changed snapshot before the next wave emits. Inner waves do not rebind/nest the namespace. Snapshot and permission errors propagate before sink writes with either an existing cache or the default absent-cache path. A mock-backed real DataAccessSource manifest-token change is covered.

Final normal root integration, evidence/r55-cache-final-root.log: **150 passed**, no skipped cases. Covers the 12 new boundary cases, existing shared-DAG/cross-wave streaming, DataAccess source/manifest identity, default stream controls, and zero-headroom resource contracts.
Watchdog: returncode 0; elapsed 72.483215931 s including setup; sampled family peak 469377024 bytes.

## Turnover/CPT correctness

- A legal turnover sample requires sufficiently observed chip mass: exp(-sum(turnover_history)) <= .10. The test uses decimal turnover .2 and window 20; the model guard is not relaxed.
- Twelve turnover outputs and CPT now exclude all standard identity/time metadata from calculation and preserve it. Previously timestamp was overwritten with numeric estimates. Bridge-injected __fe_time__ is explicitly preserved.
- Misaligned price/turnover coordinates and multi-instrument long input fail explicitly rather than mixing assets.
- The turnover support floor is five observations. Declaring window>=2 allowed 2..4 to produce NaN forever; the formal and native domain now require window>=5. Five is explicitly tested as a valid minimal window.
- Turnover semantic versions increase to 3, CPT to 2; cache-identity tests cover the changes.

Before evidence: r55-turnover-domain-root.log (11 timestamp-coordinate failures); r55-turnover-fixed-root.log (13 remaining reserved-axis failures after the first fix). Final normal root family suite, evidence/r55-chip-final-root.log: **259 passed**, no skipped cases. Includes full legacy turnover/prospect/tail tests, 85 new chip/domain/coordinate cases, 65 identity cases, and three independently valid report/revision fixtures.
Watchdog: returncode 0; elapsed 58.750013430 s including setup; sampled family peak 468680704 bytes.

Both runs used 1536 MiB sampled RSS / 240 s test guards without a trigger. Sampling is not a hard memory cap. These timings include startup and are not kernel-throughput benchmarks. Warnings about unsupported Polars implementations were not suppressed or promoted to certified/GPU status. Chip/CPT remains a shared NumPy CPU kernel per Polars column; no pandas-panel copy or GPU claim was introduced.

## Still open

The strict full-canonical campaign is not complete. Pytest preloads 17 modules and collects 1826 canonicals while ordinary load_all currently has 1789; production bootstrap/policy is being audited rather than pretending test-only visibility is production availability. Generic sample failures continue to be resolved without all-NaN exemptions.
These regressions are not full multi-backend certification, 110k real-factor execution, updated CSV delivery, production publication, or a claim of optimal performance.
