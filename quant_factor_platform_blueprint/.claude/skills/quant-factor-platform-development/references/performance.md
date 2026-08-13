# Performance Rules

Optimize measured hot paths only. QE/FP are batch/matrix-first. Reuse ranks/masks/exposure matrices in-request. Avoid Python loops over factors and repeated groupby. Keep float64 accumulators for statistical reliability. Never change semantics to hit a benchmark.
