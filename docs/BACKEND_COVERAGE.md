# Backend coverage (generated counts summary)

> MACHINE-GENERATED — do not edit by hand. Counts derive ONLY from
> `docs/PHYSICAL_IMPLEMENTATION_MATRIX.md` (same run). Manual counts elsewhere are not authoritative.
> Generated (UTC): 2026-08-20T01:41:44.852227+00:00 | HEAD: `724cf286c567512efeefb2e24622eb0bb665680d`.

- total canonicals: **1612**
- physical backend rows: **4009**

| physical_backend | implemented | selectable(prod) | spec_complete | impl_id_bound | oracle_passed | oracle_NOT_RUN | edge_passed | parity_passed | prod_evidence | prod_admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clickhouse_sql | 394 | 385 | 394 | 394 | 296 | 98 | 0 | 0 | 0 | 0 |
| duckdb_sql | 394 | 385 | 394 | 394 | 296 | 98 | 79 | 79 | 79 | 0 |
| pandas_numpy | 1610 | 1591 | 2 | 2 | 1297 | 313 | 0 | 1610 | 0 | 0 |
| polars | 1611 | 1593 | 8 | 8 | 1299 | 312 | 77 | 77 | 77 | 0 |

## Direct-use split (per canonical)

- total canonicals: **1612**
- mining_visible: **1553**
- composition_usable: **1553**
- terminal_usable: **1380**
- production_admitted: **0**
- directly_usable: **0**

## Honest caveats

- DuckDB parity/production-safe sets fail closed to `{column, literal}` — `evidence/primitive_verified.json` provenance is stale (regeneration running separately). `parity_passed=no` below that set is the fail-closed state, not a measured failure.
- pandas edge / production certification: NOT_RUN (no artifact consulted by this generator).
- production certification gates: NOT_RUN for this slice.
