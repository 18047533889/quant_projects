# Synthetic multi-horizon label-staging A/B (2026-09-28)

## Scope

These in-memory synthetic L20 runs compare per-tile label upload with the explicit
`GPUExecutionPolicy(prefer_resident_labels=True)` preference. They are bounded
transfer/correctness checks, not market-data benchmarks, automatic-route evidence,
or proof that resident staging is faster. Labels are staged resident only when
admission succeeds; otherwise the runtime uses per-tile staging, and runtime OOM
releases the resident pool and restarts the result from tile zero per tile.

## F24/H4 repeated A/B

Input: 2,586 dates × 5,461 assets × 24 factors, four labels, five rank-chain
metrics, tile size 4 (six factor tiles). The four runs alternated per-tile,
resident, per-tile, resident. Full artifact parity was true for both resident
comparisons. Per-tile H2D was 5,422,904,064 bytes/run; resident H2D was
3,163,360,704 bytes/run, saving 2,259,543,360 bytes (41.66%).

| Run | Mode | Seconds | H2D bytes |
|---:|---|---:|---:|
| 1 | per-tile | 105.334 | 5,422,904,064 |
| 2 | resident | 104.130 | 3,163,360,704 |
| 3 | per-tile | 105.299 | 5,422,904,064 |
| 4 | resident | 104.250 | 3,163,360,704 |

The mean was 105.3165 s per-tile versus 104.190 s resident (about 1.07% lower
for resident). System load varied during the sequence (1-minute load 2.51–5.18),
so this small difference is inconclusive and is not a speed claim. GPU free
memory remained 43.65 GiB; host MemAvailable remained 70.62–70.90 GiB after
runs. The preflight required at least 30 GiB host and 20 GiB GPU free, set the
VRAM policy fraction to 0.30, and estimated 2.95 GiB resident payload. No safety
threshold was approached.

## Earlier F12/H3 repeated A/B

Input: 2,586 dates × 5,461 assets × 12 factors, three labels, four rank-chain
metrics, tile size 4 (three tiles), alternated twice per mode. Artifacts,
configuration hashes, counts, masks, and provenance matched exactly. Per-tile
versus resident H2D was 2,372,520,528 versus 1,694,657,520 bytes (resident saved
677,857,008 bytes). Times were per-tile 38.445/38.347 s and resident
37.768/38.546 s. A CPU pytest process was also running, making these timings
inconclusive.

## Decision and replay

Keep per-tile label staging as the default. The measured and verified benefit is
reduced H2D traffic; these noisy small wall-time differences do not establish a
repeatable latency gain. Callers may opt in with
`GPUExecutionPolicy(prefer_resident_labels=True)` and must inspect
`label_staging_preference` (requested mode) and `label_staging_mode` (actual
mode) in result metadata. Resident admission or runtime OOM may safely fall back
to per-tile upload.

The F24/H4 run used the synthetic ABBA harness run in session 85387; F12/H3 is
recorded in `PUBLIC_RUNTIME_CAPABILITIES.md`. Inputs were generated in memory,
with no input artifact saved. COS-bound real-data access was unavailable for this
experiment.
