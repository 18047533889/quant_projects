# Operator production hardening plan

This branch applies the next operator governance batch:

1. Freeze an explicit daily canonical allowlist and fail closed for new runtimes.
2. Move `causal_bfill` to unsafe, global diagnostics/model fitting to research, and `constant` to internal.
3. Canonicalize primitive production evidence after registry deduplication.
4. Reclassify recursive indicators as stateful rather than Polars native.
5. Add operator-specific NaN/Inf evidence requirements and fail-closed capability routing.
6. Add regression tests against GTJA185 and repository factor manifests.
