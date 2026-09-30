# Selective GPU Pearson repair A/B

The bounded benchmark calls the complete batched_pearson_ic public API from
baseline 6984c311f and the working tree. The baseline module is loaded from
git show into memory; no source copy is written. Calls alternate order after
warmup. Wall time includes host synchronization; CUDA events measure device
work. Each case uses six measurements of a 128 × 8 × 5000 panel on an NVIDIA
L20. The adjacent JSON records exact IC, NaN-mask and count parity.

| Case | Baseline wall ms | New wall ms | Change | Parity |
|---|---:|---:|---:|---|
| Safe float64 | 0.915 | 0.869 | 5.0% faster | Exact |
| 50% unsafe float64 | 2.729 | 1.290 | 52.7% faster | Exact |
| All unsafe float64 | 4.545 | 1.630 | 64.1% faster | Exact |
| float32 bounded path | 0.846 | 0.846 | Within measurement noise | Exact |

Run from the repository root with
.venv/bin/python quant_evaluator/scripts/benchmark_gpu_pearson_repair_public_ab.py --repeats 6.
This synthetic primitive benchmark does not measure F61 end to end.
