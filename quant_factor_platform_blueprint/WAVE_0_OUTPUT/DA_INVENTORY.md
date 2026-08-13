# DataAccess (DA) inventory

## version & public exports

- Distribution is `data-access`; import name is `data_access`; Python `>=3.10`; license is `Proprietary`.
- Declared version is `0.10.2`. Runtime `data_access.__version__` calls `full_version(importlib.metadata.version("data-access"))`, normally adding `+build.<sha>`; source-tree fallback is `0.8.0+local`. `__build_sha__` is also exported as a module attribute.
- Console entrypoints: `data-access-server = data_access.service.__main__:main`; `data-access-quality = data_access.quality.cli:main`.
- Root `__all__`: `get_store`, `reset_store`, `get_shared_engine`, `reset_shared_engine`, `DataAccessStore`, `DuckDBEngine`, `DataAccessError`, `ValidationError`, `AmbiguousSemanticFieldError`, `DataError`, `EngineError`, `DeadlineExceeded`, `AggregationSpec`, `AggregationItem`, `aggregate_minute_to_daily`, `aggregate_minute_bundle`, `MarketCalendar`, `MarketSession`, `get_market_calendar`, `get_market_session`, `PITEventIndex`, `PITEventRecord`, `PlanNode`, `build_physical_plan`, `ContractIR`, `build_contract_ir`, `QueryBudget`, `KeyPolicy`, `DataSnapshot`, `ReadResult`, `SqlReadResult`, `ScanHandle`, `ReadHandle`, `RelationHandle`, `DataRequest`, `ReadPlan`, `SemanticField`, `SemanticFieldCatalog`, `get_semantic_catalog`, `COSDatasetContract`, `COS_DATASET_CONTRACTS`, `get_cos_contract`, `require_cos_contract`, `validate_panel_request`, `resolve_event_clock`, `normalize_return_values`, and `semantic_contract_fingerprint`.
- `get_store()` is the primary in-process facade. It installs COS runtime semantics and patches `data_access.store.get_store` so callers cannot bypass those contracts by import path.
- There is no root `DataAccess` class. The canonical facade is `DataAccessStore` obtained from `get_store()`; remote callers use `data_access.service.DataAccessClient`.
- Setuptools maps the physical repository directory to `data_access.*` with explicit package entries. Packaged Python subpackages are `core`, `registry`, `read`, `write`, `cos`, `service`, `clickhouse`, `quality`, `contract`, `security`, `runtime`, `snapshot`, and `r30`.

## subpackage inventory

| subpackage | purpose | public entrypoints |
|---|---|---|
| root/store | Governed read/write facade, registry ownership, PIT/universe joins, factor-lake reads, SQL, cache and snapshot integration. | `get_store`, `DataAccessStore`, root exports |
| read | Logical/physical read planning, PIT, handles, predicates, calendars, semantic fields, budgets, schema epochs and waves. | Read results/handles, `DataRequest`/`ReadPlan`, `QueryBudget`, predicates, semantic catalog, formats |
| write | Governed publish/upsert, generation atomicity, authorization and metadata security. | Package facade intentionally exposes only `mutation_lock`; publish/upsert are advanced modules |
| core | DuckDB engine, storage resolution/authorization, retry, namespaces, audit, exceptions and build metadata. | `DuckDBEngine`, shared-engine factories, capabilities, storage helpers and exceptions |
| snapshot | Immutable local/remote object identities and pre-execution verification. | `ResolvedObject`, `ResolvedSourceSnapshot`, resolver and verifier APIs |
| registry | Dataset YAML loading, parameterized/static datasets, parameter/schema/layout validation, path authorization. | `Dataset`, `DatasetRegistry`, `ParametricDataset`, `StaticDataset`, `load_registry`, validation helpers |
| runtime | Prepared-read pipeline, resource admission, cache management, deadlines and startup gates. | `GlobalResourceGovernor`, `ResourceReservation`, `CacheManager`, `PreparedRead`, `ReadPipeline` |
| service | FastAPI server and typed HTTP client boundary. | `DataAccessClient`, `ReadRequest`, `ReadResponseMeta` |
| security | Principal/policy authorization, credentials, run modes, provenance frames, redaction and execution contexts. | `CredentialProvider`, `DataPrincipal`, authorizers, `GovernedFrame`, context helpers |
| contract | Compiled runtime dataset contract: physical layout, selectors, required filters and temporal axes. | `RuntimeDatasetContract`, `ContractCompiler`, `PhysicalPartitionSpec`, `FileSelector`, `TemporalAxisSpec` |
| cos | COS/S3 remote reads, secure credentials, mirrors and hybrid routing. | Mirror constants, `ensure_local_mirror_for_dataset`, `cos_cache_root`, `cos_remote_backend`, `prepare_cos_remote_paths`, `should_read_cos_remote` |
| clickhouse | Optional ClickHouse panel reads and factor-series writes. | `ClickHouseConfig`, `execute_query`, `execute_select`, `insert_factor_series`, `verify_factor_write` |
| quality | Dataset/table quality validation and CLI reporting. | `QualityReport`, `validate_table`; `data-access-quality` |
| r30 | Additive maturity/governance: calendar/universe/experiment snapshots, lineage, coverage, data-change impact, leases, policies, training and multi-asset contracts. | Lazy module exports; import symbols from individual modules |
| config | Registry and semantic catalog YAML package data. | No Python API |
| deploy | systemd/Kubernetes deployment assets. | None |
| ops | Operational maintenance scripts, including dataset-stat refresh. | Script-only; not packaged |
| benchmarks | Read/COS/factor performance harnesses. | Script-only; not packaged |
| scripts/docs/constraints | Verification/bootstrap tooling, architecture/runbooks and environment constraints. | None as library API |

## read/ detailed capabilities (PIT, calendar, universe, snapshot, COS, DuckDB, query budget, schema epoch)

`data_access.read.__init__` is the public read facade. It exports `DataSnapshot`, `ReadResult`, `SqlReadResult`, `QueryBudget`, `ScanHandle`, `ReadHandle`, `RelationHandle`, `DataRequest`, `ReadPlan`, aggregation and temporal-join specs, semantic catalog, formats/adapters, and the typed predicate AST (`Eq`, `Between`, `And`, etc.) with DuckDB/Polars compilers.

Not exported by that facade and therefore advanced/internal unless explicitly promoted: `schema_epoch`, `wave_planner`, `read_session`, `managed_reader`, `read_auto_router`, `query_cache`, `partition_planner`, `metadata_plane`, `manifest`, `coverage`, `stats`, `telemetry`, `visible_catalog`, `dependency_extractor`, `source_binding`, `sql_ast_validator`, `sql_escape`, `streaming_integrity`, backend counters, PIT index builders/loaders, and most physical planning helpers.

- **PIT:** `DataAccessStore.read_asof(...)` is the direct as-of entrypoint; `read_joined(...)` performs governed temporal joins; COS runtime installs `read_cos_events` and `read_cos_events_asof`. `PITEventIndex`/`PITEventRecord` are root-public types, while build/load/prune functions remain module-level internals.
- **Calendar:** root-public `MarketSession`, `MarketCalendar`, `get_market_session(market)`, and `get_market_calendar(market, ...)`; the store also has `get_calendar(market)`. These encode sessions, breaks, holidays, early closes and availability timing.
- **Universe:** there is no public `get_universe()` or `read_universe()`. Use `read_joined(..., universe=..., time_varying_universe=True)` or `read_factors(..., universe=...)`. The store's `_resolve_universe_instruments` is private. `r30.universe_snapshot.UniverseSnapshot.from_store` is an advanced immutable adapter.
- **Snapshot:** canonical APIs are in `data_access.snapshot`: `resolve_source_snapshot(...)`, `ResolvedSourceSnapshot`, and `verify_snapshot_before_execute(...)`. `DataSnapshot` is the read-result data identity. No symbol literally named `SourceSnapshot` exists.
- **DuckDB:** root-public `DuckDBEngine`, `get_shared_engine()` and `reset_shared_engine()` own governed connections, deadlines and execution. `RelationHandle` provides governed SQL over declared DA relations. `cos.s3_duckdb` configures DuckDB httpfs but is internal.
- **S3/COS:** `data_access.cos` exports mirror constants/helpers and remote path preparation. Lower-level `S3Credentials`, URI authorization, credential resolution, hybrid path construction and httpfs setup are implementation APIs.
- **Query budget:** `QueryBudget` and `resolve_query_budget` are public. Enforcement covers result rows, scan files/objects/bytes, remote requests, Arrow/memory, Polars collection and absolute deadlines.
- **Schema epoch:** `SchemaEpoch`, `SchemaMigration` and `SchemaEpochGate` exist in `read/schema_epoch.py` but are not exported from `read` or root; they are planner/runtime internals.
- **Predicate AST:** public through `data_access.read`: `Filter`, column predicates, logical nodes, `parse_filters`, `compile_filter_duckdb`, `compile_filter_polars`. Canonical hash and constraint-analysis helpers exist but are not facade exports.
- **Wave planner:** `BatchDataRequest`, `ReadWave`, `PhysicalFactorDAG`, `ReadWavePlanner` and `LegacyFactorBatchPlan` exist in `read/wave_planner.py`, but are not facade exports.
- **Semantic catalog:** public `SemanticField`, `SemanticFieldCatalog`, `get_semantic_catalog`, `parse_semantic_field`, `normalize_table_units` and reset helper. Root promotes the first three.
- **Contract IR:** root-public `ContractIR` and `build_contract_ir(...)`; compiled runtime contracts are public through `data_access.contract`.

## canonical public API (the classes/functions new packages should call)

Signatures below are source signatures. `Any` is intentional in the current DA annotations.

```python
get_store() -> DataAccessStore

DataAccessStore(
    registry: DatasetRegistry, engine: DuckDBEngine, *,
    principal: Any = None, authorizer: Any = None,
    credential_provider: Any = None,
) -> None

DataAccessStore.read(
    dataset: str, *, columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None, filters: Any = None,
    limit: int | None = None, engine: str = "auto", result: str = "auto",
    prefer_polars: bool = False, batch_size: int = 100000,
    query_budget: QueryBudget | None = None, normalize_units: bool = False,
    mode: str = "auto", allow_sparse: bool = False,
    allow_effective_time: bool = False,
    physical_scope: str | Sequence[str] | None = None, **params: Any,
) -> ReadHandle

DataAccessStore.read_arrow(
    dataset: str, *, columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None, filters: Any = None,
    limit: int | None = None, query_budget: QueryBudget | None = None,
    mode: str = "auto", allow_sparse: bool = False,
    allow_effective_time: bool = False, **params: Any,
) -> pyarrow.Table

DataAccessStore.read_asof(
    dataset: str, *, as_of: Any, columns: Sequence[str] | None = None,
    instrument_filter: Sequence[str] | None = None, limit: int | None = None,
    query_budget: QueryBudget | None = None, **params: Any,
) -> ReadResult

DataAccessStore.scan(...same core controls...) -> ScanHandle
DataAccessStore.read_frame(...same core controls...)
DataAccessStore.read_uri(uri: str, *, ..., query_budget: QueryBudget | None = None, ...) -> ReadHandle

DataAccessStore.read_joined(
    anchor: str, fields: Any, *, joins: Mapping[str, Any] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    universe: str | None = None, time_varying_universe: bool = True,
    query_budget: QueryBudget | None = None, ...,
) -> ReadHandle

DataAccessStore.read_factors(
    factor_ids: Sequence[str], *, time_range: tuple[Any, Any] | None = None,
    universe: str | None = None, frequency: str | None = None,
    layout: str = "long", columns: Sequence[str] | None = None,
    limit: int | None = None, engine: str = "auto", result: str = "auto",
    prefer_polars: bool = False, batch_size: int = 100000,
    query_budget: QueryBudget | None = None,
    versions: Mapping[str, str] | None = None,
    require_same_data_snapshot: bool = False,
    require_same_universe: bool = False,
    route_matrix_threshold: int = 100, **params: Any,
) -> ReadHandle

DataAccessStore.get_calendar(market: str | None) -> Any | None
DataAccessStore.get_factor_catalog(dataset: str = "factor_lake", *, discover: bool = True)

DataAccessStore.sql(
    query: str, *, read_datasets: Sequence[str],
    read_params: Mapping[str, Mapping[str, Any]] | None = None,
    read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
    view_columns: Mapping[str, Sequence[str]] | None = None,
    params: Sequence[Any] | None = None,
    query_budget: QueryBudget | None = None,
) -> pyarrow.Table
```

```python
ReadHandle.to_arrow() -> pyarrow.Table
ReadHandle.to_pandas()
ReadHandle.to_polars()
ScanHandle.collect() -> ReadResult
RelationHandle.collect() -> Any
RelationHandle.sql(select_sql: str, *, params: Sequence[Any] | None = None) -> RelationHandle

get_market_session(market: str | None) -> MarketSession | None
get_market_calendar(
    market: str, *, store: Any = None,
    trading_days: Sequence[date] | None = None,
    holidays: set[date] | None = None, force_reload: bool = False,
) -> MarketCalendar

build_contract_ir(
    registry: Any, contracts: Mapping[str, Any] | None = None,
    catalog: Any = None, *, external_contract_datasets: Sequence[str] = (),
) -> ContractIR

resolve_source_snapshot(
    dataset: str, *, policy: str = "latest",
    pin_snapshot_id: str | None = None,
    paths: Sequence[str] | None = None, files: Sequence[Any] | None = None,
    list_objects_fn: Callable | None = None, head_object_fn: Callable | None = None,
    source_manifest_fn: Callable | None = None, fallback_fn: Callable | None = None,
    strict: bool | None = None,
) -> ResolvedSourceSnapshot
verify_snapshot_before_execute(
    snapshot: ResolvedSourceSnapshot, *, remote_meta_fn: Any = None,
    local_stat_fn: Any = None, strict: bool | None = None,
) -> None

class CredentialProvider(Protocol):
    def resolve(self) -> CredentialMaterial: ...

GlobalResourceGovernor(
    *, max_active_queries: int = 64,
    max_total_reserved_memory: int | None = None,
    max_total_scan_bytes_inflight: int | None = None,
    max_remote_concurrency: int = 16, max_duckdb_concurrency: int = 8,
    per_principal_active: int | None = None,
) -> None
```

Advanced, implemented but not root-public:

```python
CalendarSnapshot.build(market: str, calendar: Any, source_version: str | None = None) -> CalendarSnapshot
CalendarSnapshot.from_store(store: Any, market: str) -> CalendarSnapshot | None
UniverseSnapshot.build(universe_id: str, market: str | None, members: Iterable[Any] | None,
                       membership_policy_version: str, tradability_policy_version: str,
                       source_snapshot: Any = None) -> UniverseSnapshot
UniverseSnapshot.from_store(store: Any, universe_id: str, members: Iterable[Any] | None = None,
                            *, market: str | None = None,
                            membership_policy_version: str = "1",
                            tradability_policy_version: str = "1", ...) -> UniverseSnapshot
DataAccessClient(base_url: str, *, api_key: str | None = None,
                 timeout: float = 300.0, client: httpx.Client | None = None)
DataAccessClient.read_arrow(dataset: str, *, ..., format: Literal[... ] = "arrow_ipc",
                            max_rows: int | None = None) -> tuple[pyarrow.Table, ReadResponseMeta]
```

`GovernedFrame` is public from `data_access.security`, but is a provenance boundary type, not a general read facade. `SourceRef`, `SourceSnapshot`, `CredentialsProvider`, and `ResourceGovernor` do not exist as DA symbols; new packages must not invent compatibility imports for them.

## confirmed-gap (what new packages may need to add via adapters)

- **Canonical factor-definition identity:** DA stores `factor_definition` in `r30.artifact_meta.FactorArtifactMetadata` and has generic stable digests, but no canonical parser/normalizer/hash contract for factor definitions. FP/FA should own or adapt this identity while using DA for source/data identity.
- **Complexity profile:** no public factor-expression complexity model (operator count/depth/statefulness/cost class) was found. DA's scan/join costs, mining profiles and resource budgets describe data access, not factor-definition complexity.
- **Evaluation metrics:** no canonical IC/RankIC, turnover, decay, coverage, robustness, portfolio or optimizer-objective metrics API is present. These belong in QuantEvaluator/FactorOptimizer and should consume DA-governed frames/snapshots.
- **FactorAssets adapter:** DA already reads computed factor values by `factor_id` with `read_factors(...)` or generic `read("factor_lake", factor_id=...)`, and has `FactorCatalog` metadata. The gap is a clean cross-package FactorAssets repository protocol and canonical artifact-definition/version semantics, not raw factor-id I/O.
- **Universe facade:** DA owns universe semantics and applies them in joined/factor reads, but lacks public `get_universe()`/`read_universe()`. An adapter may wrap registered universe datasets or `UniverseSnapshot.from_store`; it must not duplicate PIT/tradability logic.
- **Promoted wave/schema APIs:** `ReadWavePlanner` and schema epoch logic exist but are not facade-public. New packages should request explicit DA adapters/promotions rather than reproduce grouping, source binding, schema checks or query budgets.
- **Stable source reference DTO:** no `SourceRef` symbol exists. DA has dataset names/params, `DataRequest`, `SourceBinding`, runtime contracts and resolved snapshots, but no small root-public DTO matching blueprint terminology.

## external dependencies

| group | dependencies |
|---|---|
| required | `PyYAML>=6.0`, `pandas>=2.0`, `pyarrow>=14.0`, `duckdb>=0.10`, `numpy>=1.24`, `sqlglot>=25.0` |
| polars | `polars>=0.20` |
| client | `httpx>=0.27`, `pydantic>=2.0` |
| service | `fastapi>=0.110`, `uvicorn[standard]>=0.27` |
| clickhouse | `clickhouse-connect>=0.8` |
| dev | `pytest>=8.0`, `build>=1.2`, `twine>=5.0` |

No GPL/AGPL/LGPL package is directly declared in `pyproject.toml`; DA itself is proprietary and the declared libraries are normally MIT/BSD/Apache-family. A definitive transitive conclusion requires scanning a locked environment because dependencies are lower-bounded and extras resolve dynamically. No GPL/AGPL/LGPL dependency is evident from the declared set. `botocore.config.Config` is lazily imported by `read/read_contract.py` but `botocore` is not declared, so remote-path packaging should be checked separately.

## sys.path/leakage findings

- No `sys.path` mutation appears in normal package `__init__.py` files or normal library runtime modules.
- Script bootstrap hacks: `scripts/benchmark_workloads.py`, `scripts/compact_dataset.py`, and `ops/refresh_dataset_stats.py` insert repository paths into `sys.path`. Tests and generated closure-test subprocess snippets do the same. Do not copy these into new packages.
- Standalone verifier scripts use physical imports such as `from r30...`, `from security...`, and `from service...`; these are not installed-library APIs.
- Fallback imports in `r30/execution_lease.py` and `read/query_cache.py` include `from runtime...` rather than `data_access.runtime...`, creating cross-layout/runtime leakage if the optional modules are absent.
- Production DA does not import FactorEngine; normal references are comments/docs or compatibility descriptions. FactorEngine imports found are in integration tests.
- Hard-coded local defaults exist in `cos/mirror.py` under `/home/shw/quant_projects/data/...` and in benchmark scripts. These are environment leakage, though not `sys.path` hacks; deployments should override through DA config.
- Some tests use `dataaccess.snapshot...`, while the packaged name is `data_access.snapshot...`; no `dataaccess` alias is declared and new packages must use `data_access`.
- Never import physical top-level directories (`read`, `runtime`, `security`) or append `/home/shw/quant_projects/dataaccess`; use the setuptools-mapped `data_access.*` API.
