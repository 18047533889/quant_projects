# Backend coverage (generated counts summary)

> MACHINE-GENERATED — do not edit by hand. Counts derive ONLY from
> `docs/PHYSICAL_IMPLEMENTATION_MATRIX.md` (same run). Manual counts elsewhere are not authoritative.
> Generated (UTC): 2026-08-20T12:29:33.930390+00:00 | HEAD: `f798b527004511d2fc9611ca1d09a49cc8783232`.

- total canonicals: **1624**
- physical backend rows: **4033**

| physical_backend | implemented | selectable(prod) | spec_complete | impl_id_bound | oracle_passed | oracle_NOT_RUN | edge_passed | parity_passed | prod_evidence | prod_admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clickhouse_sql | 394 | 385 | 394 | 394 | 296 | 98 | 0 | 0 | 0 | 0 |
| duckdb_sql | 394 | 385 | 394 | 394 | 296 | 98 | 0 | 0 | 0 | 0 |
| pandas_numpy | 1622 | 1603 | 2 | 2 | 1297 | 325 | 0 | 1622 | 0 | 0 |
| polars | 1623 | 1605 | 8 | 8 | 1299 | 324 | 0 | 0 | 0 | 0 |

## Direct-use split (per canonical)

- total canonicals: **1624**
- mining_visible: **1565**
- composition_usable: **1565**
- terminal_usable: **1392**
- production_admitted: **0**
- directly_usable: **0**

## Honest caveats

- DuckDB parity/production-safe sets fail closed to `{column, literal}` — `evidence/primitive_verified.json` provenance is stale (regeneration running separately). `parity_passed=no` below that set is the fail-closed state, not a measured failure.
- pandas edge / production certification: NOT_RUN (no artifact consulted by this generator).
- production certification gates: NOT_RUN for this slice.
