# FactorEngine (FE) inventory

## version & public exports

- Package root: `/home/shw/quant_projects/factor_engine`. There is no root `__init__.py`; FE is a namespace-style source tree. Public modules are imported as top-level packages (`api`, `expr`, `ir`, `backend`, etc.), not as `factor_engine.*`.
- Distribution: `factor-engine`, version **0.3.1**, Python `>=3.10`. Console entry point: `factor-engine-serve = service.app:main`.
- Main `api` exports: `Factor`, `col`, `field`, `rank`, `ts_mean`, `ts_std`, `ts_std_dev`, `zscore`, `delay`, `make_cleaned_call_factory`. Other allow-listed operators lazy-load through `api.__getattr__`.
- `expr` exports: `Expr`, `ensure_expr`, `ColumnRef`, `FieldRef`, `Literal`, `CleanedCall`, `expression_payload`, `canonical_expression`.
- `ir` exports: `IRNode`, `Analyzer`, `AnalysisResult`.
- Canonical runtime entry point: `runtime.engine.FactorEngine`; construct backends with `backend.factory.build_backend`.

## subpackage inventory

| Subpackage | Purpose |
|---|---|
| `api` | User-facing factor DSL, `Factor`, parser, columns/fields, operator allowlists and compatibility APIs. |
| `audit` | Admission/audit matrices and operator governance checks. |
| `backend` | Pandas/Polars/SQL/Hybrid execution, semantics, routing, kernels, certification and cost models. |
| `benchmarks` | Backend/operator benchmarks and coverage baselines. |
| `cache` | Column/plan caching and cache policy. |
| `cleaned_operators` | Canonical runtime operator implementations, grouped by domain/family; runtime source of truth. |
| `data` | Data support and contracts. |
| `docs` | FE documentation and evidence/reference assets. |
| `evidence` | Evidence manifests and certification artifacts. |
| `examples` | Runnable DSL, backend, batch and materialization examples. |
| `execution` | Lower-level execution buffers and support. |
| `expr` | Expression AST nodes and canonical expression serialization. |
| `fields` | Field catalog, typed descriptors and market/data metadata. |
| `ir` | Typed IR, analyzer, schema, semantic/history contracts. |
| `market` | Market, calendar, universe and cross-market semantics. |
| `mining` | Factor mining/admission integrations and production gates. |
| `modeling` | Predictive/modeling operators and lifecycle contracts. |
| `operator_contracts` | Generated/static operator metadata snapshots (JSON/CSV/TXT). |
| `planner` | Logical/physical planning, optimization, plan hashes and lowering. |
| `planning` | Backend-region and physical-plan contract types. |
| `research_operators` | Research-only operator surface and experiments. |
| `research_tools` | Research/mining helpers and registries. |
| `runtime` | `FactorEngine`, contexts, identity, scheduling, resources, batch/incremental services. |
| `scripts` | Operational and audit scripts. |
| `security` | Factor ID, access and production security policy. |
| `semantic` | Semantic identity/continuity and validation support. |
| `service` | Optional HTTP/service layer and job/security APIs. |
| `storage` | Data-source abstractions, catalogs, parquet/materializers and persistence. |
| `tests` | Unit, integration, parity, audit and performance tests. |
| `tools` | Developer/audit/maintenance tooling. |
| `util` | Shared utility helpers. |

`planning` and `planner` are separate packages. `operator_contracts` contains data artifacts, not Python modules. `research_operators` exists but is absent from the `pyproject.toml` package include list (which does include `research_tools`).

## DSL/AST/IR core (define, parse, canonicalize, hash)

- Define with `api.Factor(name=..., expr=..., freq=..., universe=..., description=...)`; `expr` is an `Expr` tree. Arithmetic on `Expr` creates nodes. `api.col`/`field` create references and operator factories create `CleanedCall` nodes.
- Parse with `api.dsl_parser.parse_expr(text, surface="daily", dialect="native", dialect_version=..., budget=...)`, or `parse_factor(...)`, which preserves `source_expr`, surface and dialect metadata. Parsing is restricted Python-expression syntax with explicit complexity/size budgets.
- `expr.canonical.canonical_expression(node)` and `expression_payload(node)` provide deterministic expression serialization.
- `Analyzer.lower(expr)` produces `AnalysisResult` containing typed `IRNode`, dependencies and history/semantic analysis. `ir.schema` and `ir.types` own value, semantic, temporal, history and axis-effect contracts.
- Plan hashes are in `planner.plan_hash`: `source_expression_hash`, `typed_ir_structural_hash`, `typed_ir_semantic_hash`, `logical_plan_hash`, `optimized_plan_hash`, `physical_plan_hash`. `storage.catalog.compute_ir_hash` is the persistence/catalog hash API.
- There is no class named `FactorDefinition`; the canonical definition object is `api.factor.Factor`.
- Serialization is split: canonical AST serialization is `expression_payload`/`canonical_expression`; formula definitions retain source DSL plus Factor metadata. No single public `Factor.to_json()` API was found.

## operator registry/contracts (semantic_type, domain, lookback, stateful, complexity)

- Authoring lookup uses `api.operator_registry.build_dsl_allowlist(surface=..., dialect=..., dialect_version=...)`. `build_authoring_allowlist`, `build_research_mining_allowlist` and `build_production_mining_allowlist` separate authoring/research/production surfaces. `api.__getattr__` resolves allow-listed factories lazily.
- Runtime resolution bridges through `backend.cleaned_bridge`; implementations live in `cleaned_operators`.
- `operator_contracts/operator_contracts.json` is a generated snapshot. `operator_summary.txt` reports 1,266 operators: 1,179 daily, 73 extended, 7 unsafe, 3 internal, 3 research, 1 legacy; 1,162 have evidence and 104 do not. JSON records canonical name, surface, evidence, backend flags, certification (`level`, `pit_safe`, `checkpointable`) and parameter count.
- Semantic contracts are centralized in `ir.types`: `SemanticType`, `SemanticTypeBundle`, `HistoryContract`, `AxisEffectContract`, `history_contract_for`, `check_history_contract_declared`, `axis_effect_contract_for` and registration APIs.
- Lookback/history is computed by `ir.analyzer` from operator requirements. Cost metadata is in `backend.operator_cost`: `CostSpec`, `OperatorCost`, `cost_spec_for`, `get_operator_cost`, `estimate_backend_cost`, `estimate_plan_cost`.
- Stateful/checkpointability metadata appears in operator certification/contracts and the stateful operator/runtime families.

Cleaned-operator families (20 directories including docs/cache): `ashare`, `closure`, `common`, `cross_section`, `fundamental`, `index_listing`, `intraday`, `microstructure`, `overhaul`, `polars_native`, `price_volume`, `relation`, `search`, `shareholder`, `stateful`, `technical`, `ts_model`, `valuation`, plus `docs` and `__pycache__`. Principal semantic families are cross-sectional (`cs_*`), time-series/model (`ts_*`), technical/decay/window, price-volume, fundamental/valuation, intraday/microstructure, relation/shareholder/index-listing, A-share and stateful.

## compute/materialize APIs

- Construct `FactorEngine(backend, data_source, cache=None, run_mode=..., production_fallback_policy=...)`.
- `engine.compile(factor, pit_enforce=..., pit_forbid_forward_fill=...)` performs Expr -> IR -> logical plan -> optimization and returns `(optimized_plan, analysis)`.
- `engine.run(factor, ...)` computes through the injected data source/backend and returns a result dictionary; examples use `out["result"]` and `out["analysis"]`.
- `engine.materialize(...)` computes and writes output. Variants include `materialize_from_config`, `materialize_incremental`, `materialize_incremental_from_config`, `materialize_incremental_from_event`, `materialize_matrix` and ClickHouse methods.
- `storage` provides `ParquetMaterializer` and other materializers. Data access is injected through `storage.datasource.DataSource`; FE already owns compute/PIT/materialization integration.

## batch/scheduler APIs

- `FactorEngine.compile_many`, `analyze_batch`, `run_many`, `run_many_iter`, `run_many_parallel` compute multiple factors and share compilation/data/CSE paths.
- Config/materialization APIs include `run_many_from_config`, `run_many_from_config_parallel`, `materialize_many_from_config`, `materialize_many_from_config_parallel`, `materialize_many_fast`, `materialize_sharded` and incremental-many methods.
- `runtime.adaptive_batch_scheduler.AdaptiveBatchScheduler` supplies resource-aware batch `run` and `materialize`; `runtime.batch_service.materialize_shared_nodes_parallel` and pipeline helpers support shared-node parallel execution.
- `plan_many_fast` and `materialize_matrix` cover fast planning and matrix output. CSE is performed in multi-factor compilation.
- Resource/runtime controls include the adaptive scheduler, resource monitor/governance modules, execution contexts created by `FactorEngine._make_context`, and multibackend region schedulers. These are FE-owned and should be invoked through engine/scheduler boundaries.

## backends (pd/polars/duckdb)

- Use `backend.factory.build_backend(name)`. Names include `debug`, `pandas`, `pandas_modin`, `polars`, `polars_lazy`, `polars_long`/`polars_native`/`long_polars`, `auto_long`/`hybrid_long`, DuckDB/SQL aliases and `auto`/`hybrid`.
- Implementations include `PandasBackend`, `PolarsBackend`, `PolarsLongBackend`, DuckDB/SQL pushdown, Hybrid backends and `DebugBackend`.
- `backend.backend_router.BackendRouter.select` routes based on plan/capability/evidence; capability/registration/certification modules hold backend support metadata.
- Inject the backend into `FactorEngine`; external packages should not call cleaned kernels directly.

## canonical hash for factor identity (USE_EXISTING / NEED_ADAPTER / TRUE_GAP)

**USE_EXISTING, with a thin adapter recommended for raw `Factor` callers.** FE exposes deterministic canonical expression serialization and plan/IR hashes, notably `source_expression_hash`, `typed_ir_structural_hash`, `typed_ir_semantic_hash`, plus `runtime.factor_identity.compute_factor_identity` and `FactorSemanticIdentity`. `storage.catalog.compute_ir_hash` persists IR identity. Most semantic hash APIs accept lowered/typed plans rather than a bare `Factor`, so FA should compile/canonicalize through FE and select the semantic identity hash; it must not implement a second hash scheme.

## complexity profile (USE_EXISTING / NEED_ADAPTER / TRUE_GAP)

**USE_EXISTING, with a thin adapter recommended.** `api.dsl_parser.ComplexityBudget` bounds AST nodes/depth/arity/literals. `backend.operator_cost` exposes `CostSpec`, `OperatorCost`, `get_operator_cost`, `cost_spec_for`, `estimate_backend_cost` and `estimate_plan_cost`; analyzer output carries history requirements. No class literally named `CostProfile` was found. FA may adapt these FE results into its public profile shape but must not duplicate complexity analysis.

## example factor definition end-to-end

From `/home/shw/quant_projects/factor_engine/examples/pandas_factor.py`:

```python
import pandas as pd
from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource

class InMemorySeriesSource(DataSource):
    def __init__(self, data: dict[str, pd.Series]) -> None:
        self.data = data
    def load_column(self, name: str):
        return self.data[name]

# close is a timestamp/instrument MultiIndex Series in the full example.
factor = Factor(name="mom_2_rank", expr=rank(ts_mean(col("close"), 2)))
engine = FactorEngine(
    backend=PandasBackend(),
    data_source=InMemorySeriesSource({"close": close}),
)
out = engine.run(factor)
print(out["result"])
```

`examples/simple_factor.py` similarly defines `rank(ts_mean(col("close"), 20) - ts_mean(col("close"), 5))` and calls `FactorEngine.run`. `examples/batch_materialize.py` demonstrates run plus `ParquetMaterializer.materialize`.

## dependencies

Base: `PyYAML>=6.0`, `numpy>=1.24`, `pandas>=2.0`, `pyarrow>=14.0`, `scipy>=1.10`.

Optional extras: `pandas`; `accel` (bottleneck/numba); `talib`; `polars`; `modin`; `backtest`; `data` (`data-access`); `performance` (`psutil`, `threadpoolctl`); `full` (`data-access`, `polars`, `duckdb`, `psutil`); `service` (FastAPI/Uvicorn/Pydantic); `dev` (pytest/service/httpx).

No `license` field is declared in `pyproject.toml`, and no root `LICENSE*`/`COPYING*` file was found. License is **undeclared/unknown** from FE package metadata.

## sys.path/leakage findings

- Explicit path injection exists in scripts including `stress_test_quick.py`, `validate_production_scenarios.py`, `check_backend_coverage.py`, and in `examples/batch_materialize.py` (`sys.path.insert(0, project_root)`). Tests/operational scripts contain further inserts.
- These are not cross-package contracts; consumers should install/use FE's public package modules.
- No root `__init__.py` injection exists because there is no root `__init__.py`. `api/__init__.py` performs lazy operator resolution, not path injection.
- Top-level package discovery (`where=["."]`) and `service.app:main` reinforce source-layout leakage. New packages should call the public FE modules and never copy path hacks.

## 5-line summary

1. FE is version 0.3.1 and exposes the `api` DSL plus `runtime.engine.FactorEngine`.
2. Expr/IR, operator contracts, backend routing, CSE, materialization and scheduling are already FE-owned.
3. Canonical expression, typed IR/plan hashes and runtime factor identity exist; adapt them at the Factor boundary.
4. Complexity exists via parser budgets, analyzer history and operator/plan cost APIs, without one `CostProfile` class.
5. FE has no declared license metadata and contains script/example `sys.path` injections that new packages must not copy.
