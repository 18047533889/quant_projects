# Bounded source-axis materialization (2026-10-03)

## What changes in normal calls

QE uses `adapters/source_axis_materializer.py` when assembling a factor tile.
The helper copies exact axes without constructing indexers or reindexing.
For other axes, it validates membership and uses row/column slices when the
positions form contiguous unit-stride runs. It retains indexed selection
for reordered or noncontiguous axes. The caller does not need a new option
to enable these paths.

This change preserves float64 conversion and writes into strided output
views. Shared-axis validation runs before final-axis validation. Duplicate
source axes still fail closed. In memory-bounded mode, output sharing
storage with the source raises before writing, because snapshot-safe
overlap handling may exceed the logical chunk budget. Unbounded mode uses
the prior reindex path for overlap.

`chunk_bytes` bounds the logical row-selection plus column-selection payload.
It does not cap process RSS, pandas object overhead, Arrow buffers, source
prefetch, or GPU memory. QE keeps the default at 8 MiB. You must budget those
other allocations when choosing source and GPU policies.

## Matched in-memory measurements

The [final receipt](benchmarks/source_axis_materializer_ab_20261003_final.json)
records fresh-process A/B/B/A runs for each width. Each process prepares and
touches its inputs and output before timing. A uses the previous fancy-index
row and column algorithm; B uses the current helper. Both include axis-indexer
work in their timed path.

The source has 2,588 dates and 5,461 assets. The target takes the first 2,586
dates and all assets. This matches the contiguous sub-axis shape of the
F48 source, but the benchmark repeats one synthetic panel into two or four
strided output slices. It does not exercise independent COS factors.

| Output width | Previous median | Current median |
| --- | ---: | ---: |
| 2 | 125.538 ms | 62.635 ms |
| 4 | 253.621 ms | 153.844 ms |

The runner compared values and NaN masks against pandas reindex.
The receipt contains identical output hashes for both methods.
Each median contains two measurements. These measurements support the
local assembly improvement on this workload; they do not establish
COS-to-GPU speedup or an optimum for other axis layouts.

The reported RSS deltas are process high-water-mark increments after
preparation. Earlier allocation peaks can hide later growth, so do not use
these deltas as total or guaranteed maximum memory.

To reproduce without reading COS:

```bash
.venv/bin/python -m quant_evaluator.scripts.benchmark_source_axis_materializer \
  --repeats 1 \
  --output quant_evaluator/docs/benchmarks/source_axis_materializer_UNIQUE.json
```

Use a fresh output name. The runner limits repetitions to 1..8 and gives
each measurement process a 120-second timeout. The receipt binds the
materializer and benchmark script to their source hashes.

## Default auto and explicit choices

Use `evaluate_factor_source_batch(source, labels, metrics=..., backend="auto")`
for bounded source evaluation. Omitted `backend` means `auto`.
You can request `cpu` or `cuda_strict`; `max_tile_size` can reduce the
source's declared batch width, and `gpu_policy` controls GPU resource policy.

The source selector still requires a measured evidence envelope.
For the F48 float64 mixed-three profile, a declared/requested cap of 16
selects the certified execution width 2 when resources permit.
Memory admission and measured optimal width are different checks.
Read [F48 admission and usage](benchmarks/F48_AUTO_ADMISSION_20261002.md)
for exact dimensions, metrics, hardware and budget requirements.

The materialized `evaluate` API has its own admission policy. Its F48
mixed-three request currently falls outside that policy and selects CPU.
The source-API F48 receipt does not certify materialized F48 execution.
This materialization improvement changes neither registry and does not
establish per-metric global fastest-backend selection.

## Verification scope

Regression coverage checks exact axes, reordered/partial axes, empty input,
duplicate and missing axes, mixed numeric dtype conversion, nonnumeric
failure, strided output, and overlap rejection/snapshot behavior.
The source-tile tests also check shared-axis membership and logical chunk
budgets. An independent real-source rerun and the full QE suite are separate
gates; retain their actual result rather than treating this microbenchmark
as evidence for all metrics.

## Real COS route verification and suite results

The [ordinary F48 receipt](benchmarks/f48_cap16_axis_materializer_ordinary_20261003.json)
records 48 real factors at 2,586 dates and 5,461 assets, with RankIC,
quantile spread and factor turnover rate (144 scalar results).
Default auto selected CUDA without selector injection and took 58.984 seconds;
strict CUDA took 57.096 seconds. Both used 24 width-2 tiles, reported
2,586,991,616 peak VRAM bytes and zero OOM retries. The route, factor coverage,
request identity, historical reference and direct auto/CUDA checks passed,
including finite masks and observation counts.

The declared QE source hash stayed unchanged during the run:
`4a554388f5b36f7f1324e1232b41b2955834fe1c41d4a2ffb3a8ac382621d3db`.
This attestation does not cover the DataAccess/optimizer dependency closure.
The run uses the documented authorized COS environment settings in the F48
admission document. It does not compare the previous and current materializer
in interleaved real-source runs, nor does it rerun CPU. Do not derive an
end-to-end speedup from the prior day's timing.

The full QE suite passed 5,020 tests with 26 skips and 50 warnings in
208.25 seconds. The skips include metrics requiring specialized evidence;
they do not establish coverage for those inputs. The registered metric/formula
reference check passed. The four additional contiguous/noncontiguous alignment
checks passed in a separate run after the full suite began collection.
FE extreme EWM correctness remains an independent unresolved task; this
QE release makes no claim that that smoothing implementation is fixed.
