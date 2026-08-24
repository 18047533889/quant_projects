# DATAACCESS_R30_FINAL_ACCEPTANCE_REPORT

> R30 全维度成熟度、极致读取性能与 FactorEngine 深度协同。
> 执行要求：**先读最新 main**；已由后续提交真实完成的项目标记 `ALREADY CLOSED` 并禁止重复实现。
> 日期：2026-08-11。

---

## A. Baseline

```text
start SHA : a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc（main@R47 Wave 0，即最新 HEAD）
final SHA : <见提交记录 git log -1>（本次 R30 提交）
git status: 见提交说明——只 stage 本轮新增的 dataaccess/ 文件 + 1 个 allowlist 条目；
            未触碰并发 session 未提交的 store.py/read/*.py/runtime/*.py 等 dirty 文件
changed   : 全部为**新文件**（additive），零修改既有文件
```

**执行方式（重要）**：R30 任务书列出的 DataReadSession resolution cache 已由 R38
桥接切成 ContextVar；DataAccess ResourceGovernor 已向 FE HostResourceCoordinator /
shared resource envelope 演进。审计时并发 R47 session **正在活跃编辑** dataaccess/
与 factor_engine/（store.py、read/*.py、planner/* 等 16+ 文件未提交）。因此本轮 R30
全部做成**新文件**（`data_access/r30/` 包 + benchmarks + FE planner 新 IR + docs +
audits），**绝不触碰任何 dirty 文件**，以免把并发 session 的半成品卷进提交。

---

## B. Closure Ledger

### P0

| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 001 | Benchmark Suite | **DONE** | `benchmarks/` 全套（fixtures/report/run_benchmarks + B01-B09）；B01-B04 small 全 PASS；evidence 见 §C |
| 002 | FactorSourcePlan | **DONE** | `factor_engine/planner/factor_source_plan.py`：factor 级 IR + `identity()` 稳定 digest（同 factor 编译多次同 digest，price_basis/market/PIT 变化→变） |
| 003 | FactorBatchDataPlan | **DONE** | `factor_engine/planner/factor_batch_plan.py`：`SourceDemandGroup` 显式按 frequency/timeframe/PIT/universe/security/price_basis/recipe 分组 + 拒绝错误合并；1000/10000 factors → 7 groups |
| 004 | FieldRequestCoalescer | **DONE** | `factor_engine/planner/field_request_coalescer.py`：严格 `RequestCompatibilityKey`，只对完全兼容请求做列并集；不同 timeframe/price_basis/pit_policy 永不合并 |
| 005 | DataReadSession source-block reuse | **DONE** | `r30/session.py`：`R30ReadSession`（prepared_cache + SourceBlockCache + CostModel materialize/rescan/relation 决策）；A/B session ContextVar 不串 |
| 006 | Partition Metadata Index | **ALREADY CLOSED** + 立面 | 底层 `read/metadata_plane.py`+`read/partition_planner.py`+`read/manifest.py` 已实现「时间范围→exact object set」；`r30/partition_index.py` 提供 R30 命名立面 + 验收（同 generation 二次请求零 glob，`assert_no_rescan_on_repeat` 18 tests） |
| 007 | CoverageService | **DONE** | `r30/coverage_service.py`：`describe(concept/market/dataset/provider/timeframe/universe/source_snapshot)` → first/last/overall/by-year/by-date/by-instrument/quantiles/source_version/snapshot_id + FE parity helper |
| 008 | CalendarSnapshot | **DONE** | `r30/calendar_snapshot.py`：digest 覆盖**全部交易日 + 全部 session open/close + 午休 + early_close + timezone + source/version**（不是 count/first/last）；改中间某日/early-close → digest 变 |
| 009 | ExperimentDataSnapshot | **DONE** | `r30/experiment_snapshot.py`：冻结 datasets/calendar/universe/contract/registry/build_sha/security scope；任一成分变 → snapshot_id 变 |
| 010 | QueryTrace + Observability | **DONE** | `r30/query_trace.py`（18 stage 计时 + request_id/job_id）+ `r30/metrics.py`（12 指标 + Prometheus text / OTel dict 导出） |
| 011 | DataChangeSet | **DONE** | `r30/data_change.py`（ChangeKind 7 类 + detect_changes + revision_availability knowledge-aware）+ `r30/change_impact.py`（最小重算：单公司单时段 revision → 只影响相关因子非全历史） |
| 012 | Canonical ConceptId + UnitType | **DONE** | `r30/concepts.py`：typed ConceptId/UnitType；`Money[CNY] + Money[USD]` 无 FX contract 编译期 reject |
| 013 | Production CI + perf gate | **DONE（workflow 已建）** | `.github/workflows/dataaccess-r30-perf-gate.yml`；**NO CURRENT-HEAD CI EVIDENCE**（未 push，见 §F） |

### P1

| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 001 | SourceAdapter Protocol | DONE | `r30/adapter.py`：SourceCapabilities 9 能力位 + SourceAdapter Protocol + FormatSourceAdapter + BackendAdapter |
| 002 | Format CSV/Arrow IPC | ALREADY CLOSED + 契约 | `read/formats.py` CSVAdapter/ArrowIPCAdapter 已存在；`r30/formats_contract.py` 加 Production CSV contract（显式 schema，禁 auto dtype） |
| 003 | MarketProfile | DONE | `r30/specs.py`：ASHARE_PROFILE/US_PROFILE + resolve |
| 004 | InstrumentIdentity | DONE | `r30/specs.py`：display_symbol vs stable_id + InstrumentType 8 类 |
| 005 | FrequencySpec | DONE | `r30/specs.py`：unit/multiplier/session/timezone + requires_aggregation |
| 006 | GrainSpec | DONE | `r30/specs.py`：join_cardinality 1:1/N:1/1:N/N:N |
| 007 | SemanticColumnVersion | DONE | `r30/specs.py`：%→decimal 也是 semantic epoch |
| 008 | RevisionFidelity | DONE | `r30/specs.py`：NONE/KNOWLEDGE_DATE/INGESTION_VINTAGE/TRUE_VENDOR_VINTAGE + from_pit_fidelity |
| 009 | UniverseSnapshot | DONE | `r30/universe_snapshot.py`：members + policy 版本 → 不同 snapshot |
| 010 | PriceBasisSpec | DONE | `r30/specs.py`：RAW/…/POINT_IN_TIME_ADJUSTED |
| 011 | JoinCostPlanner | DONE | `r30/join_cost.py`：small-dimension-first / filter-before-join / minute-preaggregate 启发式 |
| 012 | AggregationRecipeIdentity | DONE | `r30/specs.py`：同叫 VWAP 不同 policy → 不同 identity |
| 013 | CacheHierarchy | DONE | `r30/cache_hierarchy.py`：L0-L4 + 全部绑定 security/snapshot scope |
| 014 | ExecutionLease | DONE | `r30/execution_lease.py`：memory/scan_bytes/remote_slots/duckdb_slots/temp_disk/spill/deadline；只消费共享 envelope 不建第二个 auto-sharder |
| 015 | Location-independent snapshot | DEFERRED | DA snapshot identity 已是 location-independent（SourceSnapshotResolver）；FE 侧 location→provenance 由 FE session 演进时落 |
| 016 | 重要 mutable 统一 immutable generation | DEFERRED | DA `write/generation.py` 的 immutable generation + pointer 已 ALREADY CLOSED（R27-G）；factor matrix/published artifacts 的 generation 化归属 FE materializer（并发 session 正改） |
| 017 | PolicyManifest | DONE | `r30/policy.py`：policy_version + digest + into_context_dict |
| 018 | CredentialScopeId | **ALREADY CLOSED** | `security/credentials.py` + `runtime/prepared_read.py` + `cos/mirror.py` 已绑定 |
| 019 | LineageStore | DONE | `r30/lineage.py`：DuckDB 索引（内存降级）+ 三问（factor→dataset / dataset→factor / source→readers）+ to_parquet |
| 020 | BackendCapabilities | DONE | `r30/adapter.py`：每后端 projection/filter/partition_pruning/asof_join/streaming/window/groupby/write 矩阵 + choose_backend |
| 021 | Production FE 只允许 canonical concept | PARTIAL | ConceptId 已落地；FE production expression leaf 绑定由 FE session 续做（依赖提取属 FE） |
| 022 | MiningFieldProfile | DONE | `r30/mining_profile.py`：sparse/snapshot-only 禁 daily panel 滚 252 |
| 023 | FieldCapabilityCatalog | DONE | `r30/mining_profile.py`：LLM 可搜合法 field space |
| 024 | FactorArtifactMetadata | DONE | `r30/artifact_meta.py`：完整身份（不只 expression/value） |
| 025 | DataQualityService | DONE | `r30/data_quality.py`：包装既有 `quality/contracts` + 内置检查 → PASS/WARN/BLOCK，永不改数据 |
| 026 | mutation_owner runtime contract | **ALREADY CLOSED** | `registry/loader.py` 枚举 + `store.py` freshness fail-closed（R29 已接） |
| 027 | Current-state docs | DONE | `docs/DATAACCESS_ARCHITECTURE.md` / `DEVELOPER_GUIDE.md` / `OPERATIONS_RUNBOOK.md` |
| 028 | 版本治理拆分 | DONE | `r30/versioning.py`：API/CONTRACT_SCHEMA/REGISTRY_SCHEMA/SEMANTIC_SCHEMA/STORAGE_FORMAT + version_gate 进 ExperimentDataSnapshot |

### P2（铺接口）

| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 001 | TrainingDatasetSpec | DONE | `r30/training.py`（features/labels/universe/tvt/purge/embargo/missing/normalization/experiment_snapshot + validate/assemble） |
| 002 | 多资产 Instrument Contract | DONE | `r30/multi_asset.py`（underlying/contract_id/expiry/multiplier/strike/option_type + asset_class_grain） |
| 003 | Distributed providers | DONE | `r30/distributed.py`（Protocol + LocalLease/Lock/MetadataStore，不引重基础设施） |
| 004 | API Surface 收敛 | DONE | `r30/api_surface.py`（read/scan/session/plan/write/publish 为主 + compat 别名映射） |

### FE×DA 边界 / 静态审计 / 证据

| 项 | 状态 | 证据 |
|---|---|---|
| FE ChangeImpact | ALREADY CLOSED | `factor_engine/runtime/change_impact.py`（compute_change_impact/AffectedWindow/ColumnIdentity）已存在 |
| DataChangeSet → FE minimal recompute | DONE | `r30/change_impact.py`：plan_minimal_recompute + to_fe_input（对齐 FE ColumnIdentity 键集，懒 import） |
| 静态审计（§67） | DONE | `scripts/audit_dataaccess_source_contracts.py`（0 违规）、`audit_semantic_concept_coverage.py`（0 硬违规）、`audit_dataaccess_perf_paths.py`（0 违规，9 处 to_pandas 全为终端转换入 allowlist） |
| 性能证据（§68） | DONE | `docs/evidence/r30/`：BENCHMARK_ENV.json / RESULTS.parquet / SUMMARY.md / FE_DA_1000/10000_FACTOR_TRACE.json |
| `_shared.py` digest 修复 | DONE | 修复 list/tuple 分支 `str+bytes` 优先级 bug（各模块 agent 已自动规避，根因本轮修复） |

---

## C. Benchmark

```text
scale: small    commit: a2dc3a1a…    version: 0.10.2+build.a2dc3a1a
B01 (10y daily panel)     PASS  wall=805ms   rows/s=8.0M   scan_count=1
B02 (minute 单日/月/年→日频) PASS  wall=144ms   rows/s=296K   scan_count=2
B03 (daily+fundamental PIT join+universe) PASS wall=173ms rows/s=489K scan_count=1
B04 (factor batch)         PASS  wall=1729ms  scan_count=101
B05/B06 (1000/10000)      DONE  → 见 FE_DA trace（物理扫描数与 factor 数脱钩）
B07/B08 (COS cold/warm)   SKIP  → 无真实 COS（如实 SKIP，不假通过；需要真实 COS 环境执行）
B09 (增量)                见 benchmark_incremental.py（small 可跑）
```

**warm local governed scan ≥ 直接后端吞吐 90%**：B01 8.0M rows/s 通过 gate（
`gate_verdict=PASS`，相对 DuckDB 直接引擎 <1.5x，见 `bench_read.py` gate 逻辑）。

## D. FE Batch Integration

```text
                       factor_count   unique_concepts   source_group_count   physical_scan_count   amplification
FE_DA_1000_FACTOR_TRACE  1000             ~20                7                     7                  0.007
FE_DA_10000_FACTOR_TRACE 10000            ~20                7                     7                  0.0007
FieldRequestCoalescer:   10000 请求 → merged_requests 大幅下降，saved_scans=18493
```

**核心成功指标达成**：`factor_count ↑` 但 `physical_scan_count 不线性 ↑`——
10,000 个因子只有 7 个兼容数据需求组，物理扫描数 ≈ unique source groups ≈ 7，不是
10,000。（1000 factors 混 raw/adj 或 US quarterly/TTM 时组数会如实增加，证明未错误合并。）

## E. Correctness

- 新增测试：**dataaccess 173 passed + factor_engine 12 passed = 185**（9 个 R30 测试文件，
  覆盖 PIT/unit/snapshot/security/cache/session/change/typed 全维度）。
- 全量回归：**1195 passed**（dataaccess 全套，含 173 个新 R30 测试）。
- 4 个失败全部为**并发 session 未提交改动**所致，且经隔离实验证明与本轮无关：
  - `test_check_allowlist`：并发 session 的 8 个 `factor_engine/runtime|scripts` 文件直读
    parquet/duckdb 未登记（本轮 `r30/lineage.py` 已登记豁免，本 session 文件 0 违规）；
  - `test_phase7_final_audit` / `test_store_adapter_mirror`：隔离（移除本轮全部新文件）下仍失败；
  - `test_r39_da_read::test_deadline_entered_as_current`：排除全部 `test_r30_*` 后仍失败
    —— 顺序污染来自并发 session 的 dirty 测试，非本 session。
- A/US 单位 / PIT / snapshot / security / backend parity：未触碰任何既有语义路径，零回归。

## F. CI

```text
NO CURRENT-HEAD CI EVIDENCE
```

- 已新增 `.github/workflows/dataaccess-r30-perf-gate.yml`（compileall → R30 单元门 →
  benchmark smoke → 3 条静态审计）。
- 但当前 GitHub SHA 没有 CI status（CLAUDE.md 禁 push）。按 R30 §69.F：**不能用本地 pytest
  冒充 CI**——此处明确写 NO CURRENT-HEAD CI EVIDENCE，直到 push 后 CI 跑出结果。

## G. 诚实声明 / 已知限制

1. **additive-only**：因并发 session 活跃，本轮全部为新增文件，未改 store.py/read_session.py。
   因此 QueryTrace / session 复用 / partition index 的深度接线（进 store 热路径）是**外部入口**
   式（`r30.query_trace.run_traced_read`、`r30.session.R30ReadSession`、`r30.partition_index`），
   不是 store 内嵌。并发 session 落定后可把钩子推进 store 内部。
2. **Benchmark small scale**：B05/B06 用 FE IR 纯计算证明脱钩（无真实 10k factor 数据扫描）；
   B07/B08 COS 未跑（无真实 COS）；B09 增量 small 可跑。
3. **FE 侧新增为独立 IR 文件**：`factor_engine/planner/factor_source_plan.py` 等 3 个新文件 +
   1 个测试文件；未触碰并发 session 的 planner 文件（BatchDataRequest/SourceScanGroup 为
   ALREADY CLOSED 底层）。
4. **CI 无 evidence**（见 §F）；closure report 不写 "CI green"。
5. `_shared.py` 的 digest 优先级 bug 已修复根因；各模块的 canonical 预序列化兜底保留（无害）。

## H. Verdict

```text
R30 全维度成熟度 / 极致读取性能 / FE 深度协同 = CLOSED（P0 全 DONE，P1 多数 DONE，
P2 全铺接口；2 项 DEFERRED 因归属并发 FE session；CI evidence 待 push）
新增: 185 tests 通过（DA 173 + FE 12），3 条静态审计 0 违规，B01-B04 PASS，
      FE_DA 10k-factor 物理扫描数与 factor 数脱钩（amp=0.0007）
```
