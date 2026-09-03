# Backend coverage (generated counts summary)

> MACHINE-GENERATED — do not edit by hand. Counts derive ONLY from
> `factor_engine/docs/PHYSICAL_IMPLEMENTATION_MATRIX.md` (same run). Manual counts elsewhere are not authoritative.
> Generated (UTC): 2026-09-03T09:40:31.721023+00:00 | HEAD: `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026`.

- total canonicals: **1740**
- physical backend rows: **4321**

| physical_backend | implemented | selectable(prod) | spec_complete | impl_id_bound | oracle_passed | oracle_NOT_RUN | edge_passed | parity_passed | prod_evidence | prod_admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clickhouse_sql | 422 | 413 | 422 | 422 | 321 | 101 | 0 | 0 | 0 | 0 |
| duckdb_sql | 422 | 413 | 422 | 422 | 321 | 101 | 0 | 0 | 0 | 0 |
| pandas_numpy | 1738 | 1719 | 2 | 2 | 1297 | 441 | 0 | 1738 | 0 | 0 |
| polars | 1739 | 1721 | 8 | 8 | 1299 | 440 | 0 | 0 | 0 | 0 |

## Direct-use split (per canonical)

- total canonicals: **1740**
- mining_visible: **1674**
- composition_usable: **1673**
- terminal_usable: **1500**
- production_admitted: **86**
- directly_usable: **86**

## Honest caveats

- DuckDB parity/production-safe sets fail closed to `{column, literal}` — `evidence/primitive_verified.json` provenance is stale (regeneration running separately). `parity_passed=no` below that set is the fail-closed state, not a measured failure.
- pandas edge / production certification: NOT_RUN (no artifact consulted by this generator).
- production certification gates: NOT_RUN for this slice.
