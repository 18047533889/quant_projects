# Operator production hardening

The public daily DSL is fail-closed. Registering a runtime does not make it
available to factor authors. Every canonical must be assigned explicitly to one
of: daily, research, unsafe, legacy, or internal. Unknown registrations are
`unclassified` and fail CI.

Key production rules:

- `bfill`, `causal_bfill`, lead/next, random operators and full-sample norms are removed from the runtime.
- `causal_linear_extrapolate` is research-only and unavailable to daily formulas.
- Global diagnostics and model-fitting utilities are research-only.
- `constant` is an internal IR/runtime helper, not a public operator.
- Recursive indicators are stateful, not native Polars expressions.
- Primitive evidence is stored under the final post-dedup canonical name.
- High-risk statistical operators cannot be marked Polars production-safe until
  their required DuckDB NaN and Inf edge evidence exists.
