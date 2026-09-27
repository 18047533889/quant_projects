# Horizon label mask reuse A/B (2026-09-28)

## Scope and safety

This is a bounded contract-construction benchmark, not a full evaluation
latency claim or an auto-backend certification. It ran on server-c with two
2586-date × 5461-asset float64 LabelBundle objects and one common boolean
mask. Original labels were fully constructed before the measurement. The
comparison was dataclasses.replace(label, validity=mask) versus
label._with_validity_mask(mask) in the same Python process, in that order.
RSS was read from psutil.Process().memory_info().rss; time used
time.perf_counter(). The replace objects were dropped and garbage-collected
before the reuse measurement. A concurrent non-QE batch job made wall time
unsuitable for a throughput claim.

| Operation, two labels | Elapsed | Incremental RSS |
|---|---:|---:|
| Baseline dataclasses.replace | 1.751 s | 255.6 MiB |
| Immutable-value reuse | 1.621 s | 13.5 MiB |

The pre-replace RSS was 391.4 MiB. After dropping replace results and garbage
collection it was 418.0 MiB, so incremental RSS values use the preceding
baseline for each operation. Both versions produced identical content_hash
values; the reuse path held the exact original immutable values objects. The
observed speed difference is too small and too exposed to concurrent CPU load
to claim a stable latency gain. The useful verified effect is avoiding two
additional full-panel value buffers. Editable or untrusted value buffers still
take the copying path.

Focused contract/horizon tests passed (15), and the full QuantEvaluator suite
passed (2944 passed, 26 skipped) after the change. Neither test total proves
every metric has real-data numerical correctness.
