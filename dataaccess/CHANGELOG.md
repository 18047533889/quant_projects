# Changelog

## 0.10.3 — R30 全维度成熟度 / 极致读取性能 / FE 深度协同

> 执行时最新 main 为 `a2dc3a1a`（R47）。并发 session 活跃 → 本轮 **additive-only**：
> 全部新文件，零修改既有文件（避免卷走并发未提交改动）。验收报告见
> `DATAACCESS_R30_FINAL_ACCEPTANCE_REPORT.md`。

- **P0**：Benchmark Suite（`benchmarks/` B01-B09 + fixtures/report/run_benchmarks，B01-B04
  small 全 PASS）；QueryTrace + Observability（`r30/query_trace.py` 18 stage +
  `r30/metrics.py` Prometheus/OTel）；FE 批量 IR（`factor_engine/planner/factor_source_plan.py`
  / `factor_batch_plan.py` / `field_request_coalescer.py`，10k factors → 7 source groups，
  物理扫描数与 factor 数脱钩 amp=0.0007）；`r30/session.py` R30ReadSession（prepared +
  source-block 缓存 + CostModel）；Partition Index 立面 + CoverageService.describe +
  CalendarSnapshot 全量 digest + ExperimentDataSnapshot + DataChangeSet→最小重算 +
  ConceptId/UnitType（Money[CNY]+Money[USD] 无 FX 编译期 reject）。
- **P1**：SourceAdapter/Capabilities、JoinCostPlanner、CacheHierarchy、ExecutionLease、
  LineageStore（DuckDB 索引 + 内存降级）、PolicyManifest、DataQualityService、MiningFieldProfile/
  FieldCapabilityCatalog、FactorArtifactMetadata、RevisionFidelity、SemanticColumnVersion、
  AggregationRecipeIdentity、版本治理拆分（`r30/versioning.py`）。
- **P2 接口**：TrainingDatasetSpec / 多资产 DerivativeContractSpec / Distributed providers /
  API Surface 收敛（`r30/`）。
- **基础设施**：`_shared.py` digest 优先级 bug 修复；3 条静态审计（source_contracts /
  semantic_concept_coverage / perf_paths 全 0 违规）；current-state 三文档；perf-gate CI
  workflow；`dataaccess/r30/lineage.py` 登记 allowlist 豁免。
- **测试/证据**：新增 185 tests（DA 173 + FE 12）；全量 1195 passed，4 失败全为并发 session
  未提交改动（隔离实验证明与本轮无关）；`docs/evidence/r30/`（BENCHMARK_ENV/RESULTS/
  SUMMARY + FE_DA_1000/10000_FACTOR_TRACE）。
- **如实**：NO CURRENT-HEAD CI EVIDENCE（未 push）；B07/B08 COS SKIP（无真实 COS）；
  深度接线（QueryTrace/session/partition 进 store 热路径）待并发 session 落定后推进。

## 0.10.2 — R29 Execution/Governance/Security Second-Pass Closure

R28 修完后暴露的下一层（基线 `main@8449d9c`）：契约/治理实际缺口、执行旁路、
安全 fail-open。用户二扫的 ~20 项全部封闭（+ 附带 4 项）。

- **correctness/security**：ScanHandle 链式派生句柄继承 `_prepared`（verify/release 不再
  no-op）+ `close()/with/__del__`；generation mutation 锁用**稳定逻辑根**（不含
  `generation/<gid>`，G1→G2 两 writer 不再两把锁）；路径参数默认 `path_segment=True` +
  禁 glob metachar/quote（`factor_id="*"` 拒绝）；factor 家族 generic read 强制
  `_authorize_factor_tags`；`allowed_actions=[]` 保持空 frozenset（deny all，不再退化
  None=unrestricted）；`SchemaEpochGate.validate()` 参数顺序修复（`"add_column"` 不再传
  `col` 位）+ schema_migrations 从 registry 编译成 typed IR 注入 gate；`ReadPlan pin`
  构造 `VerifiedPhysicalScope`（strict 下 pin 可执行）；`PreparedRead` 固化
  `security_digest`+`credential_scope_id`，execute 前跨上下文 equality；组合执行
  `_execute_composed` 入统一 pipeline（admit→verify→execute→verify→release）；startup
  gate 真实加载 calendar + `next_trading_day`/`next_session_open` smoke probe。
- **remote/runtime**：`SourceSnapshotResolver` 主链化（FileVersion 兜底收进 resolver）；
  authoritative empty manifest 不再继续 LIST（防旧对象复活）；DuckDB engine `_exec_sem`
  与 governor `_duckdb_sem` 同一信号量（单一并发闸，移除 Store 双 `duckdb_slot`）；
  全请求 absolute deadline（prepare 的 HEAD/LIST/schema 计入预算）；generation
  delete-all 支持权威空代（`generation.meta.json`）+ `generation_required` 指针缺失
  fail-closed。
- **性能架构**：组合锚点注册成共享 pool 库持久表（去临时 parquet round-trip）；
  `write_generation_files` 全面去 pandas（DuckDB DISTINCT + typed CAST）；
  **DataReadSession**（job 级 resolution 缓存，同一批因子 prepare ONCE）；
  ScanHandle/ReadHandle 显式生命周期；`mutation_owner` 进 freshness token；
  版本 SCM 驱动（build SHA 进 `/version`/snapshot/lineage，pyproject 0.10.2）。
- **附带**：`DataRequest.transforms/field_params/frequency`（无 aggregations）明确
  reject（不静默忽略）；`AggregationSpec.__post_init__` 封 programmatic bypass；
  `_resolve_raw_paths` 容忍 `params=None`。
- **测试**：新增 23 个 destructive tests（test_r29_closure_2026_08.py）；
  **973 passed**（950 存量 + 23 新）；审计脚本全绿。

## 0.10.1 — R28 Concurrency/Snapshot-Truth/Remote-Auth/Schema-Gate-Cost/Write-Perf Closure

R27 之后的下一层（基线 `main@49907850`）：底层并发、快照真实性、远端权限、
schema gate 成本、写性能本身。用户逐项复核的 28 项全部封闭。

- **缓存正确性**（R28-1/2）：`CacheManager.pin()` 过期重建不再持非重入锁递归
  （死锁 → 原地重建）；删除失败保留 entry + `DELETE_FAILED` + 字节仍计入配额
  （不再「账面释放几十 GB、磁盘还满着」）。
- **快照真实性**（R28-3/4/5）：`SnapshotVerifier` 注入 credential-aware 真实
  COS HEAD（执行前/后真正比较 etag/version）；本地按 `size + mtime_ns` 精确校验
  （`ResolvedObject.mtime_ns`，消「同大小替换」漏网）；`SourceManifest` strict 必填
  dataset/content_digest(重算)/prefix/published_at/object_count + bucket+segment 边界。
- **物理 scope / schema gate**（R28-6/7/8）：`VerifiedPhysicalScope.contract_digest`
  消费（contract 更新后旧 scope 拒绝）；SchemaEpoch 迁移改真实 epoch-pair（A→B）；
  schema epoch 摘要落 manifest，query-time O(1) 分组（不再逐文件 footer）。
- **远端治理 / DuckDB 并发**（R28-9..15）：remote「请求总数」不再当并发上限；
  DuckDB 并发单一 source of truth + `_exec_sem` 下沉 engine（覆盖 read_joined 等全部
  路径）；request absolute deadline 贯穿 pool（fail-fast + `wait<=0` 立即 interrupt）；
  pool 连接共享同一数据库实例（object cache 跨连接共享）；每连接 threads 缩放
  （concurrent×per_query≈cores）；初始化日志移回 `__init__` + fork 防护覆盖全入口。
- **HTTP/凭证/SQL/写性能**（R28-16/17/18/26）：stream scope 覆盖生成器生命周期；
  `resolve_s3_credentials` 优先 request-scoped provider + read_joined 归因 request
  principal；sqlglot 限定名 `catalog.schema.table` 不混配裸名 + sqlglot 提为 base
  依赖；generation append/upsert/delete 走 **copy-on-write**（未变分区硬链接，
  更新时间 O(变更分区)）；`merge_tables_by_keys` 多键去 pandas（DuckDB typed
  anti-join）。
- **测试**：新增 28 个 destructive tests（test_r28_concurrency_snapshot_2026_08.py）；
  **950 passed**（922 存量 + 28 新）；审计脚本全绿。报告
  `docs/R28_CONCURRENCY_SNAPSHOT_PERF_CLOSURE_REPORT.md`。

## 0.10.0 — R27 Cache/Write-Transaction/Unsafe-Surface Closure

R26 之后下一层收口：封闭「旁路 API 与写路径」没有纳入同一 Cache / 事务 /
Snapshot 世界的问题。基线沿用 `main@58490a63`。

- **Cache 安全**（R27-A/B/C）：`read_cached` cache lookup **之前先 authorize**；
  key 纳入 security scope（principal+policy+clearance+run_mode）+ classification；
  restricted/premium 数据默认不进共享缓存；`normalize_units` 不再掉进 `**params`
  （走 `read(..., normalize_units=True)` 真实 normalize）；cache hit 记录审计 +
  provenance；新增 `read_cached_result()`（`CachedReadResult` 带 snapshot/来源代）。
- **物理边界**（R27-D/E）：`VerifiedPhysicalScope`（dataset 绑定 + contract digest），
  production/strict 下 raw `physical_scope` 拒绝；`_enforce_dataset_path_boundary`
  强制默认解析路径落在 dataset 自己 authorized root 内（A 不能借全局白名单读 B）。
- **写路径逻辑授权 + 原子写**（R27-F/G）：新增 `dataset:write/delete/publish` 等
  action；`write_arrow/upsert/delete_rows/publish_from_staging` 全部过逻辑授权；
  `generation_pointer` 数据集走**原子代写**（完整新一代 → 原子 flip
  `manifest.json.generation` 单指针），overwrite/append/upsert/delete 都只读完整一代。
- **旁路口封闭**（R27-H..L）：stream 单次 snapshot（不再二次 resolve、不 fail-open）；
  `set_calendar` 运行中冻结（`calendar_snapshot_id` 进缓存 key）；production/strict
  禁止裸 `scan_polars()`；SQL 逗号连接绕过 → sqlglot AST 白名单（缺失 fail-closed）；
  registry `schema/roles/engine/params_schema/param_specs` 冻结为 MappingProxyType。
- **测试**：新增 17 个 destructive tests（test_r27_cache_write_atomic_2026_08.py）；
  **922 passed**（905 存量 + 17 新）；审计脚本全绿。

## 0.9.9 — R26 Implementation Closure：真实执行链收口 + Clean-Checkout Hardening

R26 二阶审计（基线 58490a63 / 实际 956e6cd6）。核心目标：**让 R25 设计出来的
Contract / Security / Snapshot / Resource / PIT / Provenance 真正成为一条不可绕过的
执行链，并保证从 GitHub clean checkout 到 production service 行为一致。**

- **Clean checkout / packaging**（P0-001/002）：`.gitignore` 不再用 `credentials.*`
  吞源码；`credentials.py` 纳入 Git；`pyproject.toml` 补齐 security/contract/
  runtime/snapshot 子包；新增 `check_source_inventory.py` / `check_wheel_inventory.py`
  CI guard；wheel 全 12 子包 fresh-venv import 通过。
- **PreparedRead + ReadPipeline 唯一执行链**（P0-003/004）：`store.prepare_read()`
  → `execute_prepared_read()`；9 个 public read path（read_result/read_arrow/read_auto/
  read_arrow_stream/scan/scan_polars/read_joined/sql/read_uri/read_factors）全部穿过
  auth→contract→snapshot→budget→governor→verify→execute→verify→release；instrumentation
  counters exactly-once（T-R26-PIPE-001）。
- **Security**（P0-005..009）：`DataAccessExecutionContext` ContextVar（HTTP 嵌套读
  继承请求 principal，并发 A/B 不串身份）；`AccessPolicy` 三态（[]=deny all）；
  production 无显式 policy → startup fail；strict bool config；factor 权限 all-required +
  catalog unavailable deny。
- **PIT / Contract**（P0-011..014）：contract 编译失败 production hard-fail；FilterRequirement
  per-dataset（不跨 dataset merge）；SQL PIT 不再 COALESCE same-day（`PITUnavailable`）；
  PITPolicyFloor 拒绝 request 降级。
- **Snapshot**（P0-015/016）：strict exact identity + manifest 完整校验 + `fail_if_changed`
  + boundary 校验；verifier before/after 逐身份字段比较（etag/version_id/size/mtime）。
- **Resource / Cache / Service**（P0-017..021）：governor admission 接主链 + duckdb slot
  + remote_requests；`QueryBudget.tighten/with_overrides`；CacheManager 真实 TTL/per-principal
  quota/GC（先物理删除再记账）；service `__main__` 启动跑 startup gate；`/ready` 复用
  startup gate + 真实 engine probe（credential/legacy fail → 503）。
- **SchemaEpoch / Provenance / Metadata**（P0-022..025）：真实 parquet footer 跨 epoch
  schema gate；`GovernedFrame` 不可伪造（provenance 全链校验）；`build_sql_snapshot`
  逐 dataset auth；`VisibleFactorCatalog`（不泄露 local root，count 基于可见 set）。
- **P1**：fingerprint 含 temporal/schema、deep freeze、compiler 按 registry 绑定、
  market 优先级、layout 非法 fail、unclassified 默认、production strict floor、
  request_id 单一来源、audit 递归脱敏 + 0600、fork/PID guard、DatasetInfo nullable、
  FileSelector IR、physical_partition_for 消费调用方 registry。
- **CI audits**：`audit_r25_contract_drift`（0）、`audit_r26_security`（0 fail-open）、
  `check_source_inventory`（0 ignored .py）、`check_wheel_inventory`（12 子包）、
  `compileall` OK、allowlist 0 violation。
- **测试**：R26 新增 30 个 destructive tests（全走 public path / 真实 parquet / 真实
  HTTP），**905 passed**。报告 `docs/R26_FULL_PLATFORM_CLOSURE_REPORT.md`。

## 0.9.8 — FE 集成复查收官：FactorEngine→DataAccess adapter + 真 PyArrow parity

沿「FactorEngine → DataAccess → factor lake」链路复查的 13 项（8 项 Freeze
blocker + 5 项顺手收口）全部修复。DataAccess 内核未再扩展；改动集中在 FE
adapter / 真 PyArrow backend / 统一 predicate。

**DataAccess 侧**
- **真 PyArrow backend 三引擎 parity**（P0-2）：`_read_pyarrow` 不再手写第二套
  bound/filter 规则，改消费与 DuckDB/Polars 共用同一 `Predicate` 的
  `compile_predicate_arrow`——date-only end 含完整一天（`expand_end_bound`）、
  `instrument_filter=[]` → 恒假表达式 0 行（非全市场）。同时 `pyarrow_engine_read`
  支持 **parquet**（`pa_ds.dataset` 自动检测格式；旧代码硬编码 IPC，`engine=
  "pyarrow"` 对 parquet 直接报错）。`engine=duckdb/polars/pyarrow` 三引擎真实 parity
  测试通过（此前只测了 `mode="arrow"`=DuckDB→Arrow，不是真 PyArrow 引擎）。
- **`normalize_units=True` 在 stream/lazy result 形态静默失效**（P0-5 根因）：auto
  路由到 `result=stream`/`lazy` 时输出层单位归一化被跳过（A股 Return 保持 raw BP）。
  `ReadHandle` 增加 normalize 钩子，统一物化终点（`_materialize_arrow`）在 lazy/stream
  都应用 `_maybe_normalize_units`。`engine='auto' + normalize_units=True` 现已正确
  归一化，与 duckdb 显式引擎一致。
- **DataRequest 静态 universe 空集 → `[]`**（item 9）：`_resolve_universe_instruments`
  空成员返回空列表（0 行），不再 `None`=全市场。
- **read_factors date-only end 含完整最后一天**（item 10）：`build_factor_union_sql`
  复用 `expand_end_bound`（factor lake datetime=timestamp；expand 值转 ISO 字符串
  绑定，VARCHAR 测试 fixture 也正确）。

**FactorEngine adapter 侧**（`factor_engine/storage/sources/data_access_source.py`）
- **`instrument_filter=[]` 不再折叠成全市场**（P0-1）：None 保持 None、`[]` 保持
  `[]`（eager / lazy / scan_polars_long / Composite 全 0 行）。
- **`read_mode` 贯穿到 DataAccess `mode=`**（P0-3）：eager `store.read` 与 lazy
  scan / scan_polars_long / scan_index_long 全部传 `mode=self.read_mode`（panel/
  event/pit 不再分层漂移）。
- **`semantic_filters` 真正变成 `filters=` 行过滤**（P0-4）：不再塞进 `params`
  （StaticDataset 拒绝多余 params）；read/scan 以 `filters=` 落到物理 WHERE
  （US 财务 timeframe / IndexSymbol 真正限制行）。conflict 检测无条件运行。
- **A股 Return eager 无二次 scale**（P0-5）：catalog 覆盖字段一律 `normalized.add`
  （禁入 COS compatibility scale fallback）。
- **polars-long 受控 collect 终端**（P0-6）：`execute_polars_long_plan` 终端 collect
  前做数据源快照 revalidation（`revalidate_for_long_collect`，production fail-closed：
  scan→collect 间源被替换 = 执行内容 ≠ 计划快照），collect 后强制 QueryBudget。
- **Four-layer PIT gate UNKNOWN fail-closed**（P0-7/8）：`_dataset_pit_allowed` /
  `_table_pit_allowed` 三态（True/False/None=UNKNOWN）；production 下 UNKNOWN 拒绝，
  research 告警降级。cleaned `fundamentals_*`/`financials_ratios`（period_end 无知识
  时钟）列入 `_NO_KNOWLEDGE_TIME_FUNDAMENTALS` deny-list，PIT 禁用并提示改用
  `us_stock_*`（filing_date 为 availability）。
- **production 禁止 direct-local factor lake 写**（item 13）：`LocalParquetWriteTarget`
  production 直接拒绝，强制走 staging + `publish_factor_lake` 原子发布。

**新增回归测试**：`dataaccess/tests/unit/test_final_closure_round11.py`（三引擎
parity / DataRequest 空 universe / read_factors date-only end / factor 元数据
roundtrip）；`factor_engine/tests/integration/test_final_closure_fe_integration.py`
（空股票池 / read_mode 贯穿 / semantic_filters 落 WHERE / PIT UNKNOWN fail-closed /
catalog 无二次 scale / long collect revalidate / 真实 A股 eager==lazy==raw×1e-4）。

### 0.9.8 追加：3+1 排除式复查收官（2026-08-09）

按最新已提交 `a2dc80c` 的排除式检查，收掉「真正收官项」（外部 AI：这批补完即
Core Freeze，不再扩架构）。改动只在 `dataaccess/` 树。

- **ReadPlan / CompiledDataRequest 真正 immutable**（R12-1）：`compile_data_request`
  的 `source_params / joins / join_specs / filters_by_dataset / field_params /
  transforms` 从普通 dict 改为深冻结的 `MappingProxyType`（`_immutable` 递归冻结
  dict→proxy、list/set→tuple/frozenset）。`ReadPlan` 改 `@dataclass(frozen=True)`，
  `__post_init__` 把 `datasets / per_dataset_columns / join_policies / scan_costs /
  storage / snapshot_info / join_specs_effective / plan_snapshot_tokens /
  plan_pinned_files` 全部冻结成 tuple / MappingProxyType。plan() 之后
  `plan.datasets.clear()`、`plan.per_dataset_columns["x"]=[...]`、
  `plan.snapshot_policy="pin"`、`plan.compiled.source_params["x"]["y"]="z"` 全部
  TypeError——「plan 即声明」真正成立（explain 显示 == execute 执行）。
- **`read()` 的 `engine/result` strict enum**（R12-2）：`_read_handle` 顶部
  `validate_engine_result`——`engine="polarr"` / `result="lazzy"` 不再静默落到
  duckdb 物化路径，未知值直接 `ValidationError`（auto/duckdb/polars/pyarrow ×
  auto/arrow/pandas/polars/lazy/stream，大小写归一）。
- **Typed DataRequest**（R12-2b）：`__post_init__` 严格校验——`pit` /
  `normalize_units` / `time_varying_universe` 必须真 bool（`pit="false"` 不再
  bool()→True）、`limit` 非负 int|None（bool/负数/float 拒绝）、`engine` /
  `result` / `snapshot_policy` 严格 enum。`compile_data_request` 同步严格。
- **`factor_lake_wide` contract 修正**（R12-3）：宽表是 `datetime × {asset}` 的
  pivot，**asset 是列轴不是物理列**。registry 新增 `specialized_only` +
  `specialized_only_reason`（裸标记无 reason 启动即拒）；`factor_lake_wide` 移除
  误导性的 `instrument_column: asset` 并标记 specialized_only；`_prepare_read_request`
  门禁覆盖所有 read/scan/sql 路径（`read` / `read_arrow` / `scan_polars` /
  `read_result` 全拒绝）。宽因子读取走 `read_factors(layout="wide")` / factor_matrix。
- **ReadLineage 保留 None/[] + params 不可变**（R12-4）：`instrument_filter` 改
  `tuple[str,...] | None`（None=全市场，( )=空股票池，provenance 不再折叠）；
  `lineage_params()` 把 stream/aggregation 路径的 mutable dict canonicalize 成
  不可变 `(k, v)` tuple 序列。

**新增回归测试**：`dataaccess/tests/unit/test_final_closure_round12.py`（16 项：
plan/compiled 不可变性 + execute 按 plan 时刻语义执行、engine/result/typed 拒绝、
factor_lake_wide specialized_only 全路径拒绝 + 无 reason 拒绝、ReadLineage None/[]
+ params 不可变）。

## 0.9.7 — 收官轮：真实数据 + 运行时 + 破坏性修复（Core Freeze 定版）

0.9.6 之后的**最后一轮**（真实 A股/美股 数据 + 并发/crash/破坏性测试驱动）修复：

**正确性 / PIT / 数据完整性**
- **read_joined 空 universe**（并发会话曾引入编译断点 `continue` 越级 → 全包无法
  import）：重构为 `else:` 结构，空 universe → 0 行 typed result（非 IndexError /
  非全量扫描）。
- **Polars date-only end 丢最后一天白天行**（item 27B）：timestamp 列上
  `time_range=("…","2026-07-10")` 旧代码 `<= 07-10 00:00` 漏掉当天白天；
  抽共享 `expand_end_bound`，DuckDB/Polars/Arrow 三后端统一 `< next_day`。
- **`_check_factor_versions` 三 gate 正交**（item 28）：显式 version 检查后不再
  `continue`——data_snapshot/universe gate 独立生效，不再豁免。
- **require_same_universe/data_snapshot 缺元数据列 fail-closed**（item 29）：
  factor_lake schema 无 universe/data_snapshot_id 列时直接拒绝，不静默放行。

**并发 / crash consistency**
- **publish 双 rename 崩溃 → journal 确定性 recovery**（item 25）：第一个 rename
  前 durable journal；进程在 old→archive 与 candidate→target 之间死亡时，下次
  publish / 读路径自动收敛（晋升 candidate 或回退 archive），target 绝不永久缺失。
- **audit JSONL 跨进程行原子性**（item 26）：`flock(fd)` 兜底（线程锁挡不住多进程
  append；不依赖 PIPE_BUF/单次 syscall 假设）。
- **mirror 期望 partition degraded 标记随结果返回**（item 33）：删除 module-global
  `_expected_dates_degraded`（两个线程查不同 dataset 会互相覆盖）；`_market_trading_days`
  返回 `(days, used_calendar)` 元组——旧实现自然日回退是非空列表，degraded 检测从未真正生效。

**治理 / fail-closed**
- **read_auto 未知 mode 拒绝**（item 32）：只接受 auto/arrow/stream/polars，不再
  静默落到 read_arrow。
- **factor_matrix 列探测失败 → MatrixCoverageMiss**（item 31）：无法证明覆盖时
  不再 columns=None 读全矩阵，调用方 fallback factor-major。
- **RelationHandle SQL 只能 FROM _sub**（item 34）：parser 级 scope 校验——
  information_schema / 裸表 / 嵌套子查询隐藏表全部拒绝（数据源边界，不依赖
  「FROM _sub 出现在 SQL 里」的弱检查）。
- **remote-only coverage 不再永久 unavailable**（item 35）：有合同声明区间 →
  `partial + authority=declared_remote`，无本地观测不再误报「数据缺失」。
- **refresh_factor_catalog partial = merge**（item 30）：`factor_ids=[...]` 只更新
  请求因子，不再把未刷新的其他记录覆盖删掉。
- **空 plain-layout 本地 glob → 0 行**（item 22 扩展）：无 manifest 且 glob 无匹配
  文件时返回空读取，不再让 DuckDB 报 "No files found" 进 3 次 IO 重试。

**新增回归测试**：`tests/unit/test_final_closure_round9.py`（13/14/15/16/17/19/22/
23/24）、`test_final_closure_round10.py`（25/26/27/28/29/30/31/33/34/35）。

**真实数据验收**：A股 StockDailyBar 1815 文件全健康、schema 一致、无 null ticker；
DuckDB/Polars/Arrow 三后端差分一致；PIT no-lookahead property（pre-revision 不泄漏
未来修正）；deadline 活取消（0.26s 中断）；FD 零泄漏（300 循环）；atomic writer +
并发 reader 零错误。

## 0.9.6 — 第七轮二阶回归收口（12 项，Core Freeze 前最后一批）

用户对最新 main（`608f602`）排除前 54 项后再审，确认 12 个新增问题/二阶回归。
前 7 个 P0/P0-P1（Freeze 前必修）+ 后 5 个 P1 全部一次性落地：

**并发安全（item 1，P0）**
- `mutation_lock` 续租/release 的 **check-then-act TOCTOU** 修复：续租改经持有的
  fd（`lseek+ftruncate+write+fsync`），所有权判定用 `stat(path).st_ino ==
  fstat(fd).st_ino` **inode 身份**——旧 writer 的 heartbeat 绝不可能 truncate 后来
  重建的新 inode；release 不再 `unlink`，改为经 fd 写 `released` 标记，旧 owner
  永远不对路径做任何 unlink（"删新锁"窗口从源头消除），`_can_break_lock` 把
  released 视为立即可打破，下次 acquisition 完成清理。锁 fd 改 `O_RDWR`
  （`O_WRONLY` 下无法经 fd 读回 payload，fencing 校验失效）。

**存储（item 2/12，P0/P1）**
- `StorageSpec.from_yaml` **URI scheme → backend**：`type` 未显式声明时从 `uri` 的
  scheme 推导（`cos://`→cos、`s3://`→s3、`http(s)://`→http）；未实现 scheme
  （`oss://`）与「scheme 推导 ≠ 显式 type」矛盾配置直接拒绝，**绝不 fallback
  LOCAL**（旧 bug：`from_yaml("cos://…")` 被解析成 `type=local`）。
- `StorageSpec.options` **深冻结**：frozen dataclass 不再留可变 dict——
  递归转 `MappingProxyType`/tuple/frozenset，`ds.storage.options["x"]=…` 从根上
  TypeError，immutable IR 契约落实。

**namespace/授权（item 3，P0）**
- `PathAuthorizer` **保存 unresolved root template**，`resolve_and_authorize` 时按
  **当前 session namespace** 解析（per-namespace 缓存）。旧实现 store 构造时把
  namespace 烘焙进 `_roots`，长期 worker 换 `DataAccessSession` 后数据集路径解析成
  新 namespace、authorizer 却只认识旧的。新增 `DatasetRegistry.allowed_root_templates()`。

**PIT 语义（item 4/7，P0）**
- `SemanticField.join_policy` **strict enum**（exact/asof/pit_asof/
  pit_asof_backward/asof_backward/latest_period）：拼写错 fail-closed，不再在执行链
  `join_spec_from_field` 静默丢成 None → exact（丢 PIT join 语义 = 引入未来数据）。
- `availability_latency` dict 解析 **exact int 严格化**：只接受非 bool int 或 exact
  integer 字符串；`1.9`/`"1.9"` 全部拒绝（旧 `int(latency)` 静默截断成 1，与
  `TemporalJoinSpec.__post_init__` 的 programmatic 契约不一致）。

**日历/session（item 5/6/10，P0/P0/P1）**
- `next_bar`/`next_session_open` 消费 **effective segments**
  （`MarketSession.effective_segments_on(d)`）：美股 early-close（half-day 13:00
  收市）日 13:00 的 next_bar 正确跳下一交易日 09:30，不再映射成常规 09:30–16:00
  segment 里的 13:01；`elapsed_index`/`contains_hhmm` 新增 `on=` 日期参数。
- **Market canonicalizer**（`canonicalize_market`）：仅接受明确 alias
  （ashare/a_share/a/cn；us/usa/nyse/nasdaq/am/us_stock），未知值 fail-closed——
  `_calendar_dataset_for("europe")` 不再静默返回 `us_calendar`。
- 无 manifest 的日历缓存 token：改用**文件 snapshot**（path+size+mtime_ns），不再用
  `registry_fingerprint`（配置版本≠数据版本）；remote 读不到本地文件时兜底。

**typed 对象 self-validating（item 8/9/11，P1）**
- `strict_sequence` 拆规则：`ordered=True`（fields/order_by）只接受 list/tuple，
  set/dict/generator 拒绝；`instruments` 等成员语义接受 set 但 canonical 排序。
  `_tuple_of`（semantic_catalog / temporal_join）同理只收 list/tuple。
- `TimePartitionSpec`/`PartitionSpec` 加 `__post_init__`：`frequency="daliy"` 等 typo
  构造即报错，`prune_paths_for_time_range` 不再 `is_valid` 静默 fallback daily。
- 公开 `QueryBudget` 自身 **invariant validation**：负数/NaN/Inf/bool 预算构造即
  fail-closed（复用 `_positive_int`/`_positive_finite_float`），`max_elapsed_ms=nan`
  不再等价于关掉 deadline check。

**回归**：`tests/unit/test_final_closure_round8.py` 14 条；全量 **724 passed / 0
failed**；ContractIR audit 71 数据集一致（fingerprint=b7d21b082127eb00）。

## 0.9.4 — 第六轮最终 Closure Ledger 收口（54 项 root issues）

按用户最终 Closure Ledger（31 个 Core Freeze blocker + 23 个 P1 + 2 设计决策）落地。
并发会话已同步实现其中 ~15 项（`#P0-final closure`/`#P1-final closure` 系列 + round6
回归），本会话完成剩余项 + 全量回归：

**写侧 / 并发安全（item 1-2）**
- `mutation_lock` **fencing token**：heartbeat 后台续租（正常 writer 永不因 lease
  过期被 hard-break 破锁）；**owner-only release**（release 前读回 payload，
  transaction_id 匹配才 unlink——旧 writer 绝不删新 writer 重建的锁）；**PID reuse
  检测**（payload 记录 /proc/PID/stat starttime，同 PID 不同 starttime ⇒ owner 已死）。
  `_dataset_mutation` 整事务临界区（lock→bump→mutate→rebuild→unlock）由并发会话
  完成，本会话复核无误。

**namespace 请求级（item 3 / 46）**
- Registry load **不再烘焙 `${RUN_NAMESPACE}`**：`root`/`root_template`/
  `static_root`/`authorized_root` 保留占位符，read/write 时由
  `resolve_namespace_path()` 按当前 context 解析（长期 worker 换
  `DataAccessSession` 不再串目录）。static_prefix 计算先屏蔽 namespace 占位符
  （修掉 `/staging/${RUN_NAMESPACE}/…` 被切成 `/staging/$` 的 bug）。
- session namespace **原样返回 validated 值**（不再 `_sanitize` 把 `-abc-` 与
  `abc` 汇聚，且与 env 路径行为一致）。

**snapshot pin 执行（item 6/7）**
- `ReadPlan.execute()` 在 `snapshot_policy="pin"` 下把 plan 冻结的精确文件清单
  注入 `store.read(physical_scope=…)`——scan 不再重新 glob 解析，杜绝
  verify 与 scan 之间底层文件被替换的 TOCTOU。

**availability / 日历（item 13 / 14）**
- 冲突 availability 声明 **fail-closed**：同 dataset 多字段声明不同
  availability/period_selection ⇒ strict 抛 ValidationError，不再由 YAML 字段顺序
  静默选更宽松的 same_day。
- MarketCalendar **右边界 fail-open**：`compile_available_from` 在 strict 下
  knowledge 超出日历覆盖（next_trading_day 返回 None）直接报错，不再
  「无法证明何时可用 ⇒ 现在就可用了」。

**PIT index（item 18）**
- **generation-directory + 原子指针提交**：`<root>/.pit_index/current`（唯一 commit
  点）+ `<gen>/index.parquet` + `<gen>/metadata.json`。先写完整新 gen 目录、最后
  原子替换指针——crash 任意时刻旧 generation 完整可用，不再「新 parquet + 旧
  JSON」mixed 后上一份好索引被毁。legacy 布局可读回退。

**PIT / derived / 语义（items 16/17/19/20/21/22 由并发会话完成，本会话复核）**

**SQL sandbox（item 31）**
- 从 denylist 升级为 **parser 级 allowlist**：
  - `_assert_single_select_statement`：语法层证明**恰好一个** SELECT/WITH
    （顶层 `;` 切分 + 括号深度，字符串/注释/引号标识符跳过）；
  - `_check_sql_function_allowlist`：从 **DuckDB 自身 `duckdb_functions()`**
    注册表按 function_type 分类——**只允许 scalar/aggregate/window**；任何
    **table 型**函数（read_parquet/parquet_metadata/sniff_csv/glob 及**未来新增**
    的文件/网络/外部源入口）自动拒绝；未知函数 strict fail-closed；SQL grammar
    关键字（IN/OVER/FILTER/CAST/EXTRACT…）不误伤。

**strict 判定统一（item 47）**
- `registry/paths.py` / `read/key_policy.py` / `read/telemetry.py` /
  `cos/s3_duckdb.py` / `cos/remote.py` 全部收敛到 `is_strict_semantics()`
  （production OR `DATA_ACCESS_STRICT_READ`）——不再各自查环境变量漏 strict 模式。
  另修复 strict 下 httpfs 禁止 query-time 联网 INSTALL。

**P1 收口（items 44/45 + 复核 32-43）**
- Filter AST hash：And/Or 子节点按**完整 canonical JSON** 排序（同列不同值的 tie
  不再保留输入顺序 → 语义等价过滤同 hash，缓存不 miss）。
- date-only end：`timestamp` 列的上界从 `<= next_day - 1µs` 改为 **`< next_day`**
  （nanosecond 精度不漏掉当天最后 999ns）。
- FormatSpec **DuckDB version capability 门**（item 33 补全）：parquet
  `file_row_number`/`union_by_name`/`binary_as_string` 等 option 需要特定 DuckDB
  版本才真正生效——旧版本上配置会被静默忽略 = 配置失效，registry load 时按
  `get_duckdb_capabilities()` 探测结果拒绝。
- FormatSpec/partitioning/storage/TemporalJoinSpec/QueryPolicy/factor pivot/
  FactorCatalog/early-close/calendar-flag/dataset-strict-schema 等由并发会话
  round6 完成，全量回归复核通过。

**fsync durable-write（item 52）**
- 新增 `core/atomic.py`：`atomic_write_bytes/text/json/file`（
  **tmp → flush/fsync(tmp fd) → os.replace → fsync(parent)**）。manifest
  parquet/json/rowgroups、PIT index、publish manifest、factor catalog 全部切到
  统一 helper——不再「只 fsync 目录不 fsync 文件内容」。

**coverage（item 53）**
- `empty_ok` 声明下「无分区」判 **complete**（合法空数据集）而非 unavailable；
- `5t` 交易日 staleness 用**真实 MarketCalendar**（春节/国庆/美股 holiday 反映），
  日历不可用才回退自然日近似；
- remote-only（s3/cos）数据集显式标注「本地无法 glob，覆盖依赖 manifest」。

**COS mirror（item 54）**
- `_mirror_file_state` 三态：**verified / legacy_unverified / corrupt**；
  strict/production 下 `_local_file_usable` **不把「非空」当「完整」**（manifest-less
  文件必须 resync 成 verified）；`_verify_mirror_file(deep=True)` 重核 checksum；
  trade_day 期望日期真实日历不可用 ⇒ `expected_dates_degraded()` 显式 degraded 标记。

**设计决策（upsert / audit）**
- **upsert 承诺 partition-level atomicity**（写死进 API 契约 docstring + lineage
  `atomicity="partition-level"` + `transaction_id`）；不承诺 dataset-level——
  需全量原子性的调用方走 generation-directory + pointer（publish 模型）。
- **audit durable acknowledgement**：`audit.record(durable=True)`（publish 用）
  在审计写失败时抛 `AuditWriteError`——业务成功但审计静默失败 = 合规证据缺失，
  不允许「发布了、审计悄悄没记」；默认 observability 模式保持吞错。

**回归**
- `tests/unit/test_final_closure_round7.py` 22 条（覆盖上述全部项）。
- 全量 `tests/` **710 passed**；ContractIR audit 71 数据集一致。
- 未提交（按用户要求只改服务器本地）。

## 0.9.3 — 第五轮二阶边界收口（Core Freeze blockers 1-7 + P1 收尾 8-12）

按第二轮深扫发现的 12 个「二阶边界 bug」收口。这轮没有新增架构/subsystem——全部
是已有架构没有完全贯穿到边界分支的问题。12 项全部落地：

**P0 blockers**
- **P0-C1 cache key 空池 vs 全市场**：`query_cache_key` 的 `instruments` 不再
  `sorted(...) if instruments else None`（`[]` 被折叠成 `None`，全市场结果与空池
  结果串 cache）；`None → null`、`[] → 空数组`、有值 → 排序数组。
- **P0-C2 read_cached gates 前置**：缓存命中也必须经过与真实 read 相同的
  semantic/budget gate（`_prepare_read_request` / `_assert_instrument_filter_supported`
  / `validate_query_request`）。最终是 `gates → cache key → hit/miss`，不再是
  `cache → gates`——同一进程先 research 缓存宽查询、后切 strict/production 时，
  旧 cache 不能绕过 require_columns / query budget / semantic gate。
- **P0-C3 COS helper `[] = empty`**：`read_cos_panel` / `read_cos_events` 的
  `instrument_filter` 从 truthiness 改为 `is not None`；`[]` 生成 `1 = 0`
  （WHERE FALSE），不再静默读出全市场。
- **P0-C4 ReadHandle one-shot stream fail-closed**：流一旦开始消费，任何第二终点
  （to_arrow / to_polars / to_lazy / 再次 stream）都 fail-closed——覆盖部分消费
  （break 后物化截断 batch）**和**完整消费（迭代器耗尽后 list() 静默变空表）。
  唯一例外：首次 `stream(buffer=True)` 显式固化。
- **P0-C5 canonical materialization**：lazy 句柄第一次 terminal collect 后缓存
  Arrow Table 作为 `_source`，后续 pandas/polars/stream 全部从这一份派生——
  同一 ReadHandle 绝不重复执行底层 LazyFrame（to_polars 后 to_arrow 不再二次
  扫描；中间数据变化时两个终点也不一致）。
- **P0-C6 remote meta 缓存 TTL**：`_remote_meta_cache` 从永久 memo 改为 30s TTL；
  失败的 `None`（无凭证/网络）不再永久缓存——首次无凭证、之后补凭证必须能重试。
- **P0-C7 cos:// HEAD 解析**：`_remote_object_meta` 先 `cos_uri_to_s3_uri` 再切
  bucket/key（旧代码对所有 scheme 用 `key[len("s3://"):]`，cos:// 长度不同导致
  bucket/key 错位）。

**P1 收尾**
- **P1-C8 MetadataPlane 复用统一 validator**：`pit_event_index.py` 新增
  `validate_current_source`（authoritative + manifest_epoch + source_snapshot +
  schema_hash），MetadataPlane / prune / 构建复用三处共用——不再写第二套近似
  （只比 epoch 会在「外部文件绕过 DataAccess 写入、schema 变化但 epoch 不变」
  时返回 stale index）。
- **P1-C9 sql() 参数计数 lexical-safe**：`RelationHandle.sql/sql_fragment` 的
  `?` 计数改用 quote/comment-aware 扫描器（跳过字符串字面量 / 双引号/反引号
  标识符 / `--` 行注释 / `/* */` 块注释），不再 `text[:pos].count("?")` 把
  `SELECT '?' AS x, ? FROM _sub` 里字面量的 `?` 计入。
- **P1-C10 stats sidecar fail-open**：`_stats_sidecar_fresh` 无 source_epoch 的
  legacy sidecar 在 production/strict 视为 **stale**（重新统计，旧统计可能把很大
  的表错误路由到 Arrow materialization）；`manifest_version()` 检查失败也 fail-closed。
- **P1-C11 asof 输出列冲突统一**：`read_cos_events_asof` 用
  `allocate_unique_column_name` 逐层分配（decisions 已有 `x` 且已有 `x_event` →
  `x_event_2`，不再静默覆盖）；空 decisions 分支的
  `fundamental_staleness_days` 同样走统一冲突策略。
- **P1-C12 严格 sequence parser**：`predicate.strict_sequence`（DataRequest /
  SemanticField 共用）拒绝裸 str/bytes（会被逐字符拆开）、元素类型不对立即
  ValidationError；`SemanticField._tuple_of` 不再对错误 mapping 静默返回空 tuple
  / `str()` 化非法对象。

**顺带修复（非本批清单，但 suite 红）**
- **publish 首次发布误归档（#P0-C13）**：`_dataset_mutation` 的事务锁
  （mutation_lock）会 `mkdir` 出空 target 目录，旧 `if target_dir.exists()` 把它
  当旧版本归档——首次发布也生成 archive_path + 垃圾归档目录。改
  `_target_has_published_content`（认实际 `*.parquet` 内容），首次发布 `archive_path`
  保持 None；二次发布正常归档。
- **test_phase6_hardening 泄漏恢复**：`test_read_cos_events_asof_rejects_raw_event`
  只恢复了 `resolve_event_clock`、漏掉 `validate_event_filters`（整个 pytest 进程
  `filters=None`，后续 read_cos_events 在 `filters.keys()` 崩）——补齐 finally。

**回归**：`tests/unit/test_final_closure_round5.py` 21 条 + 全量 `tests/` 通过
（除并发会话 WIP 导致的 3 个失败：schema_validation naive/aware 语义变更 +
dod2 未跟踪测试，非本批）。

## 0.9.2 — 第四轮最终 P0 收口（publish 契约门 / asof 统一 availability / 时间轴硬化）

按最终 audit 的 14 个 P0 correctness 尾巴收口。其中 10 项（P0-1/2/5/6/7/8/9/10/11/14）
经核实已在 0.9.1 磁盘版本修好，本轮补齐回归测试锁住；真实代码改动如下：

**Publish 契约门（P0-3 / P0-4 / P0-5 复核）**
- `_validate_candidate_contract`：candidate **实际数据**必须符合 target 声明
  schema 契约——声明列缺失 / 类型不符（归一比较，禁止隐式 cast）/ 同列跨文件
  类型不一致 → 拒绝发布。此前只证明了 candidate==staging（inventory hash），
  candidate==target contract 未证明。
- publish manifest `files[].path` 已是相对路径；本轮把 `base_dir` 从绝对
  candidate 路径改为 `"."`（manifest 随 rename 上线后恒有效），`manifest_version=3`。
- symlink 复核：`_copy_tree` 已先 `_assert_no_symlinks` 拒绝任何软链接，不再
  让 `copytree(symlinks=False)` 跟随复制。

**read_cos_events_asof 统一 availability（P0-12）**
- `_select` 不再写死 `event_clock <= decision`：新增
  `availability_uses_calendar / availability_strict_next`（temporal_join 公开），
  asof 与 read_joined 共用同一套语义——
    - 需日历的 availability（next_trading_day / next_session_open / session …）
      且有市场日历 → 用 `MarketCalendar.available_from` 编译 available_from，
      条件变 `available_from <= decision`；
    - 无日历 → 按 `TemporalJoinSpec.comparison_operator` 回退（严格下一交易日
      类用 `<`，其余 `<=`）。
  `_resolve_event_availability` 复用「字段级一致声明 → 契约默认 same_day」的
  解析顺序（financial 数据集 next_trading_day 生效）。

**ContractIR 时间轴硬化（P0-13）**
- `_temporal_axes_of` 不再从值字段的 `time_role` 推导 event_time（catalog 里
  Close/Open/Volume/PeRatio 等值字段都标了 `time_role: event_time`，旧代码会编译出
  `event_time="Close"`）。事件时间轴现在只来自 契约时钟列（strict→
  availability_column，effective_time_only→event_column）或 registry time_column
  （panel 的 bar 时间即事件时间）；decision_time 也要求 dtype 像时间列。
- 修复 `ashare_stock_indicator` 的 `roe` 字段缺失 `availability: next_trading_day`
  （与同数据集 eps/net_profit 冲突）→ ContractIR audit 重新 71 数据集一致。

**Manifest rowgroup 持久化真实可用（P0-14）**
- `_save_row_groups` 用 `pa.table(list_of_dicts)`，pyarrow 25 报 "Must pass names
  or schema"——rowgroup sidecar 实际从未成功写过。改为列数组字典构造；save/load
  双侧都校验 generation（异代 sidecar 配新 manifest → 忽略，fail-closed）。

**回归**：`tests/unit/test_final_closure_round4.py` 新增 22 条（覆盖全部 14 项）；
全量 `tests/` 626 passed；ContractIR audit 71 数据集一致。
（注：并发会话把 `us_stock_dividend` 从 effective_time_only 升级为 strict-PIT +
`declaration_date`，3 处依赖旧语义的测试改指仍为 effective_time_only 的
`us_stock_capital_split`，语义验证目标不变。）

## 0.9.1 — 第三轮增量收口（compiled 计划 / 谓词门 / 精确快照）

在 `6d66e9c` 之上按第三层审计清单（P0 1-23 / P1 24-40）收口，全部为「新模块
组合以后的语义一致 / snapshot 真实 / 计划执行一致 / 明确代码 bug」：

**计划与执行一致（P0-1..5）**
- `CompiledDataRequest`：`plan()` 编译期冻结请求语义（不可变），`ReadPlan.execute()`
  只消费 compiled——调用方在 plan 后改 `req.filters/joins/aggregations` 不再改变
  执行；`plan()` 不再写回 `request.anchor`。
- `snapshot_policy = latest / fail_if_changed / pin`：execute 前按每数据集
  `manifest_version` 对比 plan 时刻 token，版本变化即拒绝（回测/训练可复现）。
- 组合执行器 `_execute_composed` honor `result="stream"`（不再忽略）、静态 universe
  先展开、`time_varying_universe` 正确下沉；`ReadPlan.execute()` 把 plan 阶段
  编译好的 `join_specs_effective` 原样交给 read_joined（explain == execute）。

**Snapshot 真实化（P0-6..8）**
- `_expand_glob_paths`：读路径把 glob 冻结成精确文件清单，DuckDB 不再二次
  mutable glob（TOCTOU）；Polars scan 绑定精确文件 + `ScanHandle.collect()`
  collect 前 revalidate（mtime/size），strict 变化即拒、research 重建 snapshot。
- `FileVersion` 增加 `etag/version_id/content_length/last_modified`；s3:// 对象
  头元数据（boto3 best-effort，strict 或 `DATA_ACCESS_REMOTE_SNAPSHOT_META=1`
  时 head），对象被覆盖 → snapshot_id 跟着变。

**日历（P0-11..13 / P1-31/32）**
- `MarketCalendar.available_from()` 先做 UTC → 交易所本地时区转换；非交易日不
  生成「当天开盘」（跳到下一交易日开盘）。
- 全局 calendar 缓存 key = market + source token（manifest source_epoch /
  explicit 内容指纹 / store 注册表指纹），数据更新自动失效、store-local。
- `sql_next_trading_day_join` 改 `zip(days, days[1:])` O(N)。

**契约门升级（P0-14..16）**
- `required_filters` 从「列名出现在条件里」升级为 `filter_restricts_column`
  PredicateConstraint：`Ne`/`IsNotNull`/`Not`/OR-部分分支不能再冒充已约束。
- `allowed_filter_values` 用 `filter_constraint_status` 区分 absent/positive/weak：
  weak（无法证明 result ⊆ allowed）→ strict 拒绝。
- `DATA_ACCESS_STRICT_READ=1` 完全共享 production 的 fail-closed（semantic
  catalog 歧义 / ScanHandle lazy / __getattr__ 物化路径统一 `is_strict_semantics`）。

**read_uri（P0-17..19）**
- strict `read_uri` 用**精确 URI 作为 physical scope**（借用注册数据集 Contract
  做语义门禁，物理读取仍限指定文件，不再全 glob）。
- 对象 URI 反查不再用 `pathlib.Path`（本地语义）；格式匹配修正（`format="parquet"`
  不再让 CSV 注册数据集成为候选）。

**Serving / stream（P0-21..23 / P1-29/30）**
- `materialize_daily_aggregate` 重写为一次扫描：PreparedReadRequest → 带
  time_range/instrument_filter 行级谓词的聚合 SQL → GROUP BY（旧实现扫两遍且
  第二遍无行级谓词，会把未请求的股票聚合进去）。
- Polars scan 按 FormatSpec 选 `scan_parquet/scan_csv/scan_ndjson`（不再硬编码）；
  CBO `suggest_read_strategy` 对 arrow/feather 自动选 pyarrow。
- stream deadline 已前置到 engine reader（第一批前）。

**写事务（P0-9..10）**
- `_dataset_mutation` 显式 PREPARED/COMMITTED/ABORTED：进入 body 前先 bump
  source_epoch（mark dirty 在 mutate 之前），只有 COMMITTED 才 rebuild manifest，
  ABORTED 绝不 build fresh。

**Spec/IR/Storage 硬化（P1-33..40）**
- `TemporalJoinSpec`：字符串 `"false"/"0"` 正确解析为 False；`revision_order`/
  `primary_key` 非法类型 fail-closed；`availability_latency` 类型校验。
- `ContractIR`：`external=True` 贯通（audit 不再误报外部契约）；`required_filters`
  完整合并 panel+dimension+event+字段级；availability/duplicate_policy 冲突检测。
- `StorageSpec(type=非法)` 构造即抛；`resolve_storage_for_dataset` COS 查询失败
  fail-closed（不再静默降级 local）；新增 `storage_backend_capabilities()` 能力
  矩阵（HTTP/ClickHouse 未实现 reader 显式声明）。

**Query cache / CBO（P1-25..27）**
- 无权威 manifest（不存在 / 不 fresh）→ 不再缓存（外部系统改数据只能等 TTL
  的坑）。
- `plan()` 按每 dataset 传 `source_params` 给 `estimate_scan_cost` /
  `manifest_version`（ParametricDataset 成本/版本正确）。

**回归**：`tests/unit/test_final_closure_round3.py` 18 条新增，全量 420 passed，
ContractIR audit 71 数据集一致。

## 0.9.0 — DataAccess 收官整改（统一语义 / 事务 / 治理）

按 `docs/DATAACCESS_FINAL_CLOSURE_PLAN.md` 落地：消灭第二套 join 语义、PIT
财务 seed 正确性、交易日历 fail-closed、多时钟时间轴、dataset 级契约门、因子
矩阵一致性、执行治理（stream 直路由 / governed lazy / Scanner 下推）、写事务
原子性、namespace/ContractIR/Metadata Plane。含 17 条 DoD 回归 + 永久 CI。

**统一 JoinCompiler（P0-1/P0-2）**
- 删除 `PhysicalPlanExecutor._join_aggregated_anchor` 的第二套 join 语义；聚合
  锚点物化为临时 parquet 作为 `anchor_override`，交给与 `read_joined` **同一个**
  `_read_joined_sql` 编译——period_selection / revision 去重 / session 日历 /
  seed+window / future_cutoff / fanout / universe 全部一致。
- 新增 `_effective_join_specs`：SemanticField 默认 → COS Contract 默认 → 显式
  joins 覆盖；`plan()`/PIT validator/组合执行消费同一份。

**PIT / 日历 / 时钟**
- Financial seed 改为「截至 start 的 PIT 状态」（period DESC, revision DESC），
  修掉 Q3 已被晚到 Q2 修订回滚（P0-3）。
- Calendar loader 过滤 `IsTradeDay`/`is_trading_day`；production 真实日历不可用
  fail-closed；fallback 日历缓存到独立 key 不污染权威日历（P0-4/5）。
- `_session_avail_sql` 前瞻 21 天；availability 拆细粒度
  （same_instant/next_bar/next_session_open/after_close_next_open/…，P0-6/7）。
- 美股 early close（13:00 收市）按日期注入 session（P0-8）。
- ContractIR 增加 `TemporalAxes`；manifest 只在查询时钟 == partition 时钟时
  prune，PIT 分支用 knowledge 时钟裁剪或放行（P0-9）。
- dataset 级 required filters（timeframe/IndustrySource/IndexSymbol）无论选哪些
  列都强制（P0-10）；`is_strict_semantics()` 唯一 fail-closed 开关（P0-11）；
  effective_time_only 无界查询 production 拒绝（P0-13）；fanout IN 多值仍判
  fan-out（P0-14）。

**因子 / PIT 索引 / 缓存**
- `read_factors` 一致性 gate 先于 matrix 路由；matrix 精确投影请求 fids，
  MatrixCoverageMiss 兜底；版本检查改窗口内 DISTINCT 集合（跨分区混版本能抓住，
  P0-17/23/24/25）。
- PITEventIndex：文件级 min/max 剪枝、`indexed_file_count`=文件数、
  limit 截断 → complete=False、真 schema hash（P0-15/16）。
- cache key：columns 保留顺序（[A,B]≠[B,A]）、columns=None 展开 schema 列
  （P0-35/36）。

**执行治理（P0-18..22）**
- `result="stream"` 直接路由到真正 reader，不再先物化再重扫；deadline 传给
  engine watchdog（第一批之前生效）；generator finally 关闭 reader 归还连接。
- governed lazy：production 不暴露 raw LazyFrame，collect 走 budget 治理。
- PyArrow 引擎改 dataset.Scanner（filter/projection pushdown，不再先全读再
  pc.filter）。
- sql_relation production 增加 FROM 表集合绑定（snapshot_datasets 是真实数据
  边界，不止 lineage）。

**写事务（P0-26..31）**
- `_dataset_mutation` 改 DatasetTransaction：metadata 只在 COMMIT 前进；
  ABORTED 只 invalidate 不重建（manifest 保持 dirty 安全回退）。
- 多分区 upsert 改 prepare-then-promote：全部 merge 成功才统一晋级，失败
  可检测（staged_partitions 进 TransactionManifest）。
- publish 增加 unique_key 唯一性发布 gate + 发布代次 content_hash。
- mutation lock 跨 host 不得用本地 PID 回收（P0-31）。

**Namespace / ContractIR / Metadata Plane（P0-32..34）**
- namespace 改 `contextvars.ContextVar`，嵌套 enter/exit 恢复外层，asyncio 不串。
- ContractIR 补 `issues` 字段（修复 audit() AttributeError），`store.contract_ir()
  .audit()` 在完整 registry 上可运行。
- Metadata Plane 感知 source_epoch 变化，mutation 后不再返回旧派生对象。

**P1 收口（部分）**
- P1-17 row-group 统计按 (path,row_group) 去重；P1-14 CBO 不再重复计 selectivity；
  P1-21 storage_backend/layout/file_format 严格分离；P1-28 `order_by` 确定性排序
  （DataRequest/read_joined/组合执行）；P1-23 PIT schema hash。

**DoD 回归 + CI**
- `tests/unit/test_final_closure_dod.py`：17 条收官 DoD（路径等价 / 组合 parity /
  late-revision / 真实日历 / 末端 boundary / early close / required filter 不可绕
  过 / fanout 单值 / 未来事件拒绝 / PIT index 计数 / ContractIR audit / plane
  epoch / cache 列序 / stream 直路由+早停 / 锁 host 感知 / 嵌套 namespace）。
- `.github/workflows/dataaccess-final-closure.yml`：ContractIR audit + DoD 套件
  永久 CI。

## 0.8.1 — DataAccess 正确性收口（DataAccess-only）

按 `docs/DATAACCESS_CORRECTNESS_FIX_PLAN.md` 落地：Manifest 双 epoch、mutation
统一事务、缓存 key 完整性、统一读前语义门禁、read_uri/sql_relation 治理、财务
latest_period 默认语义、PIT 强语义、节点式物理计划执行、read_factors 版本门禁
fail-closed、PIT 索引权威化、镜像/覆盖度/陈旧度、strict YAML、写入/发布加固。

**P0 正确性**
- **Manifest 双 epoch**（#1/#2）：`_manifest.json` 区分 `source_epoch`（mutation
  递增）与 `manifest_built_epoch`（重建后追上）；`is_manifest_fresh` 仅当
  source==built 才信任，杜绝「旧 manifest 的 min/max prune 出错误数据」。
  `store._dataset_mutation` 统一 write/append/overwrite/upsert/delete/publish/
  compaction 的失效+重建；`delete_rows` 不再漏失效。
- **缓存 key 完整**（#3/#4）：`query_cache_key` 覆盖 filters（canonical Filter
  AST hash）/limit/mode/allow_sparse/allow_effective_time/normalize_units/语义
  catalog 指纹/Contract IR 指纹；`QueryResultCache` 改 max_bytes+max_entries+TTL
  联合约束。
- **统一读前语义门禁**（#5）：`store._prepare_read_request` 统一 temporal
  contract + event cutoff + required_filters + allowed_filter_values，
  read_arrow_stream/scan_polars/pyarrow 与 read_result 一致。
- **read_uri 契约回填**（#6）：production/strict 下 URI 必须唯一反查到已登记
  数据集并复用其 Contract，否则拒绝。
- **sql_relation 沙箱**（#7）：production/strict 要求声明 snapshot_datasets +
  `validate_sql_sandbox`（禁 read_parquet/COPY/ATTACH/LOAD/INSTALL）。
- **exact join 跨表时间列**（#8）：一律 `a.{decision_time}=b.{right_time}`，
  不再引用 anchor 没有的右表列名。
- **financial 默认 latest_period**（#9）：`temporal_model=financial_event` 字段
  未显式 period_selection 时默认 latest_period。
- **DataRequest(pit=True) 强语义**（#10）：`plan()` 对每个非 anchor 字段校验
  PIT 可证明性（asof/pit_asof + temporal contract），production fail-closed。
- **read_factors 版本门禁**（#12/#13）：`_check_factor_versions` fail-closed
  （无法证明相同==不相同）；matrix fallback 只捕 `MatrixUnavailable/
  MatrixCoverageMiss`。
- **snapshot 完整性**（#14）：read_joined/sql_relation/sql_result 参与数据集
  snapshot 缺失时 `SnapshotBuildError`（production fail-closed）。
- **PIT 索引权威化**（#15）：`PITIndexMetadata`（source_snapshot/manifest_epoch/
  complete/failed_files）；仅 complete 且源匹配才 authoritative prune，否则
  fail-open；读失败记录并置 incomplete。
- **resolve_fields fail ambiguous**（#33）：registry 全局回退命中多数据集 →
  `AmbiguousFieldError`（production 拒绝取第一个）。

**语义 / Registry / 契约**
- **Semantic YAML strict**（#31）：未知 key 拒绝、bool/enum/scale 严格校验
  （`mining_allowed: "false"` 不再变成 True）。
- **Dataset Registry strict bool**（#32）：hive_partitioning/union_by_name 用
  `_strict_bool` 解析。
- **ContractIR 扩展**（#34）：knowledge/effective/period_time、revision_order、
  required_filters、allowed_filter_values、unique_key、duplicate_policy、
  partitioning、storage_backend、query_policy、coverage_policy。
- **Schema 校验**（#35）：分区列豁免（hive 目录列不判 missing）。
- **schema_version**（#36）：数据集声明 schema 版本。
- **Session namespace**（#37）：`DataAccessSession(namespace=...)` /
  `namespace_scope` thread-local 覆盖。
- **Metadata Plane**（#38）：`store.metadata_plane()` 统一 manifest/coverage/
  PIT index/schema/contract/source epoch/mirror inventory。
- **语义覆盖审计**（#30）：`semantic_coverage_audit`（FULL_SEMANTIC/PHYSICAL_ONLY/
  AMBIGUOUS/BLOCKED）。

**写入 / 发布**
- **crash-safe overwrite**（#39）：write_arrow(overwrite) 写候选目录→校验→原子
  rename 替换，不再 `_clear_dir` 直写。
- **publish 冻结 source**（#40）：copy 前后比对 staging 文件清单，防 mixed
  generation。
- **publish 强校验**（#41）：文件清单+行数+schema hash 三方一致，不止 row count。
- **upsert TransactionManifest**（#42）：`.transactions.jsonl` 记录 committed/
  failed 与受影响分区。
- **upsert DuckDB COW**（#43）：不再全量 pandas merge，UNION ALL + row_number
  去重（new 覆盖 old）。
- **锁 lease**（#44）：mutation_lock 写 pid/host/process_start_time/
  transaction_id/acquired_at/lease_until；owner 死或 lease 超 hard_break 自动打破。

**COS / 覆盖**
- **max_staleness 执行**（#24）：coverage 判定 stale。
- **calendar 感知完整性**（#25）：mirror `_expected_dates` 按 calendar_domain
  用交易日历枚举期望 partition。
- **local+remote 混合**（#26）：auto+httpfs 下缺失 partition 走 s3://，已有走本地。
- **mirror inventory**（#27）：下载 manifest 含 remote_key/checksum/
  downloaded_at/verified；`load_mirror_inventory` 校验镜像正确性。
- **remote 通用化**（#29）：storage.source.type=cos 声明数据集（含
  ParametricDataset）可走 remote，不再绑定 StaticDataset+手工 mirror registry。

## 0.8.0 — 市场语义 + 物理布局 + 查询优化器统一（Phase 4）

按 `docs/PHASE4_ROADMAP_PLAN.md` 落地：Store 级 temporal model 强制、
财务 PIT `period_selection`、session 交易日历、物理计划 DAG、分钟时区/时段
修复、一次 scan 多聚合、美股财务 PIT 索引、Contract IR、serving 分层、
正确性 gates。

**P0 语义正确性**
- **Store 级 temporal model 强制**（#44）：`read/read_result/read_arrow_stream/
  scan_polars/read_joined` 所有读入口统一执行 COS 契约——EMPTY 拒绝、X0 须
  `allow_sparse=True`、E1/E2/RAW_EVENT 面板读 production 拒绝（mode=event/pit
  显式放行）、STATIC 面板读拒绝（mode=dimension）。`read()` 新增
  `mode=auto|panel|event|pit|dimension|sparse`。
- **财务 `period_selection`**（#45）：`TemporalJoinSpec` 新增
  latest_period/exact_period/annual/quarterly/ttm/all；`latest_period` 用
  running-max(period) 实现「先选报告期、再选该期最新 revision」——旧报告期
  晚修订不再回滚当前财务状态（实测对照：plain asof 会回滚到 999，latest_period
  保持 200）。
- **session 交易日历**（#46）：新增 `read/session_calendar.py`（A股 09:31–11:30
  +13:01–15:00 共 240 根；US 9:30–16:00）；`availability="session"` 用日历
  VALUES CTE 把 knowledge 编译成 available_from（decision >= 下一交易日），
  节假日/周末公告映射正确；`store.set_calendar/get_calendar` 可注入真实日历。
- **物理计划 DAG**（#4）：新增 `read/physical_plan.py`
  Scan→Filter→TemporalJoin→Aggregation→Normalize→Project；`DataRequest` 的
  `aggregations/transforms/field_params/pit/frequency` 真正编译进计划，
  `ReadPlan.explain()` 渲染 NODES，`execute()` 走一次 scan 多聚合。
- **分钟聚合时区/时段修复**（#5）：`AggregationSpec` 绑定 market/timezone；
  `QuoteTime`(UTC) 先 `timezone(tz, ...)`（naive 用 `AT TIME ZONE 'UTC'`）再取
  HH:MM，minute_at("09:31") 匹配北京 09:31；minute_of_day 用 session elapsed
  bar index（09:31=0 … 15:00=239），午休不多算 90 分钟。
- **一次 scan 多聚合**（#6）：`aggregate_minute_bundle` 单条 SQL（CTE + FILTER），
  同一分钟数据只扫一次产出几十列。
- **聚合治理**（#7）：聚合入口走 QueryBudget/deadline/audit，返回带 snapshot 的
  ReadHandle。
- **美股财务 PIT 索引**（#8）：`read/pit_event_index.py` + `store.pit_event_index/
  prune_pit_paths`——文件按 period_end 命名时仍能按 filing_date 精准裁剪；
  `docs/PIT_SERVING_LAYOUT.md` 说明 serving 分层。
- **allowed_filter_values 值校验**（#52）：Store 读路径统一校验过滤值
  （timeframe/IndustrySource 等），production fail-closed。
- **grain/unique_key/cardinality + fan-out 守卫**（#53）：契约新增
  grain/unique_key/cardinality/required_dimension_filters；read_joined exact
  join 拒绝静默行放大（TopTen/Industry 等 one_to_many）。
- **跨市场歧义 fail-closed**（#54）：`resolve_one` 无 market/dataset 且多市场
  候选时 production 抛 `AmbiguousSemanticFieldError`。

**P1 结构一致性**
- **Contract IR**（#12）：`read/contract_ir.py` + `store.contract_ir()/
  contract_ir_fingerprint()` 把 registry + COS 契约 + 语义字段编译成统一 IR；
  `scripts/audit_contract_ir.py` CI 对齐。
- **calendar_domain / coverage**（#13/#14）：契约新增 calendar_domain
  （StockList/Status/Industry/TopTen 等标 calendar_day）；`read/coverage.py` +
  `store.coverage()` 回答 complete/partial/unavailable。
- **RAW_EVENT + 事件未来 cutoff**（#15/#16）：`_MODELS` 加 RAW_EVENT（us_fact_news
  改用）；`enforce_event_cutoff`（effective_time_only 未来事件拒绝）+
  `TemporalJoinSpec.future_cutoff`（join 右表不读未来生效事件）。

**P2 serving 层 / 基准 / 正确性**
- **serving 分层 + 预聚合**（#30/#31）：`cos/serving.py`（Source/Serving/Semantic
  View 三层；`materialize_daily_aggregate` 高扇出→日频预聚合；
  `route_minute_storage` date-major/bucket-major 路由）。
- **查询结果缓存**（#25）：`read/query_cache.py` + `store.read_cached/
  enable_result_cache`（LRU+TTL，默认关闭，manifest epoch 失效）。
- **真实 workload Benchmark**（#32）：`scripts/benchmark_workloads.py` 补
  A股日频 1日/1年/5年、分钟全市场1日、美股5年、latest_period PIT、
  filing_date+timeframe PIT、Industry sw_l1、TopTen、US shares×close、
  FactNews、COS cold/warm + `--metrics`（scans/bytes/rows）。
- **正确性 gates**（#33）：新增 `tests/unit/test_correctness_gates.py`（16 个）：
  Return/10000 vs Ret 不除、X0/EMPTY 拒绝、timeframe 强制、latest_period
  不回滚、exact_period/annual、分钟 UTC→上海 + 午休 bar index、bundle、
  未来 dividend 不前视、跨市场歧义 fail-closed。

**回归**：dataaccess **507 通过 / 0 失败**；allowlist rc=0；语义审计 0 问题；
Contract IR 审计一致。

## 0.7.0 — A股/美股两本 COS 数据字典对照落地

对照 `COS_ashare_lqtp_data_dictionary.md` 与 `COS_us_massive_data_dictionary.md`，
把 dataaccess 的 schema / catalog / 契约对齐到两市场真实口径（清洗前后差异）。

**A股 schema 对齐字典**
- `ashare_stock_daily` 补 Factor/HighLimit/LowLimit/IsSuspend/PreClose
  （Factor 为后复权乘数：后复权价=Close×Factor）。
- `ashare_stock_status` 列名修正：`ListedState`→`PublicStatus`（字典无
  ListedState 列）；补 CompanyId/ChangeType 等。
- `ashare_stock_industry` 补 IndustrySource（不筛源行数×6）。
- Valuation/Indicator/Balance/Income/CashFlow/Dividend/Capital 各表 schema
  补齐字典真实列（% vs 倍数、PubDate 语义、ExDividendDate 等）。

**美股 schema 修正**
- `us_stock_valuation_daily` / `us_stock_indicator`：从错误的 A股 PascalCase
  列名修正为美股 snake_case 真实列（market_cap/price_to_earnings/
  return_on_equity/dividend_yield/...，X0 稀疏表禁止当全历史面板）。
- **`us_stock_capital_daily` 双 schema 拆分**：`us_stock_capital_split`
  （glob `[0-9]*.parquet`，拆分事件）+ `us_stock_capital_shares`
  （glob `shares_*.parquet`，稀疏 PIT 股本）——同目录两 schema 不再 glob 混读拼炸。
- 财务三表补 `timeframe` 列与 filing_date/period_end 语义（必须 filter
  timeframe）；dividend 补 currency/ex_dividend_date 等。
- 新增数据集：`us_ticker_shares_snapshot`（市值中性推荐源）、
  `us_security_master_daily_snap`、`us_fact_news`；datasets.yaml / mirror /
  cos_contract_us / cos_registry_runtime 四处同步。

**SemanticFieldCatalog 跨市场**
- `SemanticField` 新增 `market` 维度（ashare/us/any）；`_by_name` 改为
  name→候选列表，`resolve_one(name, dataset=)` 按数据集名前缀消歧。
- 解决跨市场单位串味：`ret`（A股 bp/10000 vs 美股小数不除）、`roe`（A股
  %→/100）与 `return_on_equity`（美股小数）、`total_assets`/`revenue`/
  `market_cap` 等按 dataset 命中对应市场字段。
- 美股财务字段 `required_filters=[timeframe]`（E2 必须 filter timeframe）。

**required_filters 扩展（#42）**
- 单表 `read()` 与 `read_joined` 都强制执行；满足途径从「仅 params」扩展为
  「params 或 filters / filters_by_dataset 列覆盖」（timeframe/IndustrySource
  是列过滤不是路径参数）。

**A股 Factor 契约修正**
- `cos_contract_ashare` 的 `adjustment_convention` 修正为 backward
  （字典实证：后复权价 = Close×Factor，旧「前复权=Close/Factor」已废止）。

**回归**：dataaccess 491 通过 / 0 失败（+14 个市场一致性测试）；allowlist
rc=0；语义审计 0 问题。

## 0.6.0 — 热路径性能 / PIT 语义级下沉 / FE 消费

**read_joined 重写（P0）**
- 右表 instrument_filter 下推（exact/asof/pit_asof 一律，不再只筛锚点）。
- 显式 join 列（`TemporalJoinSpec` 的 decision_time/knowledge_time），修复跨表列名 bug。
- PIT seed+window：asof 右表只读 `[start,end]` 窗口 + 每标的 start 前最后一条可见记录，
  语义与全历史 ASOF 逐字节一致（不再扫全历史）。
- revision 去重（QUALIFY ROW_NUMBER 按 revision_order），exact/asof 都不膨胀。

**语义级 PIT（#3/#4）**
- `TemporalJoinSpec`（`read/temporal_join.py`）：knowledge_time/period_time/revision_order/
  availability（next_trading_day 用严格大于实现 A 股盘后落地可见性）；catalog 字段自动推导。
- SemanticFieldCatalog 时间语义核对：财务字段 knowledge_time→PubDate、revision_order、
  `total_liabilities` 列名修正；datasets.yaml 补齐财务 schema；四方审计脚本
  `scripts/audit_semantic_consistency.py`。
- `required_filters` 强制执行（production fail-closed / research 告警）。

**DataRequest / universe（#7/#15）**
- DataRequest 新增 source_params/filters_by_dataset/join_specs/field_params/aggregations/
  transforms/time_varying_universe；universe 按 (date,instrument) 时变成员 INNER JOIN。

**Manifest O(1)（#17-#22）**
- `manifest_version` 去掉 O(N) glob；写路径 bump epoch、read path 信任；typed min/max 比较；
  `_time_key` 不再 `[:10]` 截断；空裁剪返回空 relation；snapshot 复用 manifest 文件元数据；
  row-group 统计持久化。

**成本/治理/引擎**
- ScanCost 真实 CBO（manifest 裁剪 + projection 字节 + EMA 校准）。
- RelationHandle 生产禁 `.relation` fetch，新增 schema/columns/types/sql_fragment。
- ReadHandle.rows 不触发 lazy collect；Deadline 连接池；`execute_reader(deadline_ms=)`。
- 分钟聚合下推（`read/aggregation.py`：minute_at/minute_range/minute_of_day）。

**FactorEngine 消费**
- DataAccessSource catalog 优先解析 + `normalize_units=True`（不再双重归一化）。
- 清直接 parquet 读（TurnoverBaseDaily/Intermediate → 登记数据集/因子湖）。
- SourceRef 批量 coalesce（`load_source_refs_batch`）。
- 分钟聚合 pushdown 接入；`ExecutionResourceManager`（n_jobs×threads≤cores）。

**回归**：dataaccess 477 通过 / 0 失败；allowlist rc=0；语义审计 0 问题。

## 0.5.0 — DataAccess ↔ FactorEngine 集成层

**字段语义单一事实源**
- 新增 `SemanticFieldCatalog`（`read/semantic_catalog.py` + `config/semantic_fields.yaml`）：
  logical field → dataset + physical column + dtype/frequency/grain/unit/scale/时间角色/
  PIT·join 策略/aliases/mining_allowed；`store.resolve_fields()` 解析，物理列反查拿 scale。
- 输出层单位归一化：`read()`/`read_joined()` 支持 `normalize_units=`（按 catalog scale
  做 percent→ratio、bp→decimal 等），默认 False 保持旧行为逐字节不变。

**统一数据请求 / 数据计划**
- 新增 `DataRequest` / `ReadPlan`（`read/data_request.py`）：`store.plan(request)` 编译
  逻辑需求（字段/时间/instruments/universe/PIT/join），`ReadPlan.explain()` 给计划、
  `ReadPlan.execute()` 执行（单数据集走 read，多数据集走 read_joined）。
- 字段按物理表 coalesce：同一 StockBalance/StockIncome 的多个字段只扫一次。

**多数据集批量 join**
- 新增 `store.read_joined(anchor, fields, joins=...)`：每张物理表一个子查询（投影 + 分区
  裁剪），DuckDB 内 `exact`（日频等值）或 `pit_asof`（ASOF，取最新可见记录）一次完成，
  返回 `ReadHandle`；多数据集合并 snapshot + 预算/审计。

**query-scoped snapshot token**
- `DatasetManifest` 新增 `dataset_version` / `partition_version`（写入 `_manifest.json`
  sidecar）；`store.manifest_version()` 几十微秒级读 token（不扫 footer）、
  `store.is_snapshot_stale()` 对比判断缓存过期。

**受控 Relation 句柄**
- 新增 `RelationHandle` + `store.sql_relation()`：在 scan 之上直接追加 SQL 表达式
  （scan + factor expression 融合），`.arrow()/.collect()/.pandas()` 强制
  QueryBudget(deadline) + audit + snapshot；`.relation` 只读逃生口。

**测试**
- 新增 `tests/unit/test_integration_layer.py`（15 个用例：catalog/plan/read_joined/
  manifest_version/sql_relation/单位归一化），全量 456 通过、0 失败。

## 0.4.0 — Universal Quant Data IO Layer

**读路径**
- 新增 `store.read()` / `store.read_uri()`，返回 `ReadHandle`（`.to_arrow()/.to_pandas()/.to_polars()/.to_lazy()/.stream()`）。
- 新增 FormatAdapter：`format:` 字段支持 parquet / csv / csv.gz / tsv / jsonl / arrow / feather；
  `_build_select_sql` 不再写死 `read_parquet`（`read/formats.py`）。
- 新增通用 Filter AST（`read/predicate_ast.py`）：Eq/Ne/Lt/Le/Gt/Ge/Between/In/NotIn/IsNull/NotNull/And/Or/Not，
  read API 支持 `filters=`；DuckDB / Polars / PyArrow 三编译器。
- 新增 Partition Planner（`read/partition_planner.py`）：按 time_range 展开 hive/日期路径。
- 新增 Dataset Manifest（`read/manifest.py`）：`_manifest.parquet` 文件级 min/max 裁剪 +
  `store.build_dataset_manifest()`；`dataset_read_stats` 优先走 manifest，避免全量 footer。
- 成本路由：`read_auto` / `read()` 按 `estimated_scan_cost`（rows/bytes/columns/files/remote/selectivity）路由。
- 因子批量读：`store.read_factors()`（UNION ALL / PIVOT）+ `FactorCatalog` + `refresh_factor_catalog()`。

**存储与引擎**
- 新增 `core/storage.py`：StorageBackend（local/s3/cos/...）统一解析与分派，COS 不再是 Store 特殊分支。
- COS/S3 DuckDB 接入迁移到 `CREATE SECRET`（Secret Manager），老版本回退 `SET s3_*`。
- 新增 `core/duckdb_capabilities.py`：版本能力探测，deprecated 的 `enable_object_cache` 自动跳过并告警。

**稳定性与治理**
- QueryBudget 超时**主动取消**：deadline + `conn.interrupt()`，抛 `DeadlineExceeded`（`core/engine.py`）。
- 新增 `ManagedBatchReader`：显式托管 reader+cursor 生命周期（正常/break/异常/GC 都释放）。
- 质量契约升级：schema 对齐 / null ratio / range / finite·inf·NaN / 时间单调 / 重复 timestamp /
  覆盖率 / 未来时间戳 / PIT 泄漏（`quality/contracts.py`），CLI 读侧走 `store.read_uri`。

**HTTP**
- 新增 `/v1/read_uri`、`/v1/factors`、`/v1/factors/read`。

**测试与工具**
- 新增 57 个单测（formats / predicate_ast / partition_planner / manifest / scan_cost / read_handle /
  read_uri / managed_reader / factors / storage / quality），全量 441 通过、0 失败。
- 新增 `benchmarks/bench_read.py` Benchmark Gate（相对默认引擎 ≤1.5x）。
- 修复既有 4 个失败：`sql_escape._format_literal` 支持 Timestamp、`adapter_options` 契约对齐、
  allowlist 静态检查通过（补登记 data_access 自身实现 + factor_engine 未迁移文件）。

## 0.3.1

- HTTP 团队统一密钥默认改为 `quantsociety`（`DATA_ACCESS_API_KEY` / `X-API-Key`）；文档与 docker-compose 同步。

## 0.3.0

- 包版本对齐 monorepo `dataaccess/`（Python 包名仍为 `data_access`）。
- HTTP：`/v1/datasets` 列表、读服务配置与客户端加固。
- 部署模板：`Dockerfile` / `docker-compose` / K8s / systemd（`deploy/`）。
- 质量 CLI：`data-access-quality`（`quality/`）。
- 文档：修正目录名漂移，补充总使用文档交叉链接与 COS PIT 索引。

## 0.2.0

- 同步 `quant_projects` 最新 COS fail-closed 思路到独立组织仓库。
- 新增 45 个 A 股/美股 COS 可执行语义契约。
- 修复 A 股收益率 bp 口径、行业源过滤、财报 `PubDate` 时间轴。
- 修复美股财报 `filing_date/period_end`、X0 稀疏表、EMPTY 表和拆股事件语义。
- 新增 latest-period PIT 状态算法，避免旧报告期晚修订造成状态回滚。
- 对无公告时间的分红/拆股事件启用 effective-time-only 保护。
- 修复 COS period/event/sparse 文件布局，禁止按决策日猜文件名。
- 将语义契约及物理补丁指纹纳入数据快照。
- 包版本升级至 0.2.0，安装地址改为 `HKUST-QUANT-SOCIETY/data_access`。
