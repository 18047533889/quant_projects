# DataAccess → Universal Quant Data IO Layer（改造计划 + 完成状态）

> **状态：2026-08-07 已完成并全量回归绿（456 通过 / 0 失败）**。各 Phase 标注 ✅。

> 日期：2026-08-07　作者：data_access 改造
> 目标：保留 `DataAccessStore / Registry / QueryBudget / Snapshot / Lineage / Audit / COS PIT / Factor Lake / HTTP`，
> 把底下从「统一 Parquet 读写器」升级为「所有量化数据的统一、高性能、可治理 Data IO Layer」。
> 兼容原则：所有现有公开 API 签名不变；FactorEngine 接口不破坏。

## 核心架构

```
调用方只写 store.read(...) / store.read_uri(...) / store.read_factors(...)
                     │
              Query Planner（registry + 谓词 + 预算 + 成本）
          ┌────────────┼──────────────┐
     Storage         Format          Semantic
  local/COS/S3     parquet/csv/...   panel/event/PIT
          └────────────┼──────────────┘
              Execution Router
          ┌────────────┼──────────────┐
      DuckDB         Polars         PyArrow
```

调用方不再关心：文件是 parquet 还是 csv、在本地还是 COS、长表还是宽表、要不要 stream。
这些由 DataAccess 决定。

## 实施阶段（与代码对应）

### Phase 1 ✅ — Registry DatasetSpec 扩展（registry/loader.py + read/formats.py）
- `DatasetBase` 新增可选字段：`format`（parquet/csv/tsv/jsonl/arrow/feather…）、
  `format_spec`（CSV 的 delimiter/header/encoding/compression 等）、`storage`、
  `roles`（event_time/instrument/publish_time/report_period…）、`partitioning`、
  `semantic`（panel/event/factor…）、`engine`（preferred/fallback）。
- `time_column` / `instrument_column` 允许 `None`（改为 semantic roles 的便捷别名）。
- 所有新字段带默认值 → 完全向后兼容；datasets.yaml 里现有条目不解析新字段也照常工作。
- 新增 `read/formats.py`：`DataFormat` 枚举、`FormatSpec`、`FormatAdapter` 抽象 +
  `ParquetAdapter` / `CSVAdapter` / `TSVAdapter` / `JSONLAdapter` /
  `ArrowIPCAdapter` / `FeatherAdapter`（后两者走 PyArrow 引擎）。
  DuckDB 原生 `read_csv`（含 header/delim/encoding/compression）、`read_json_auto`。

### Phase 2 ✅ — Generic Predicate AST（read/predicate_ast.py + read/predicate.py）
- 表达式树：`Eq / Ne / Lt / Le / Gt / Ge / Between / In / NotIn / IsNull / IsNotNull / And / Or / Not`。
- `parse_filters()`：`{"col": v}` / `{"col": [..]}` / `{"col": {"gt": .., "lte": ..}}` /
  `Filter` 实例统一归一化。
- `compile_filter_duckdb` / `compile_filter_polars` 两个编译器，共用语义。
- read API 新增 `filters=` 参数；`Predicate.extra` 的「PR1 未启用」占位 stub 被替换。

### Phase 3 ✅ — Path/Partition Planner + Dataset Manifest（read/partition_planner.py + read/manifest.py）
- `PartitionSpec`：`partitioning.time`（source=path/filename, pattern="{date}.parquet"）+
  hive 分区列；`prune_paths_for_time_range()` 在进 DuckDB 前把 `date=*` / `year=*` /
  `month=*` / `{date}.parquet` 展开成具体文件路径。
- `DatasetManifest`：每数据集根放 `_manifest.parquet`，记录
  file / rows / bytes / min_time / max_time / min_symbol / max_symbol / schema_hash / mtime / etag，
  可扩展 row-group 级 min/max。`dataset_read_stats()` 优先走 manifest，避免
  glob + `pq.read_metadata()` 扫全部 footer。
- 路径解析链：Query → Manifest pruning → 文件级 pruning → DuckDB。

### Phase 4 ✅ — StorageBackend + S3 Secret + DuckDB 版本能力层（core/storage.py + cos/s3_duckdb.py + core/duckdb_capabilities.py）
- `StorageSpec`（type=local/s3/cos/http/cli/clickhouse + uri/root/bucket/endpoint/secret）。
  `resolve_storage_for_dataset()` 把 COS mirror 注册表归一化为 `storage.type=cos`，
  COS 不再是 Store 里的特殊分支；读路径决策统一走 backend。
- `cos/s3_duckdb.apply_s3_credentials`：DuckDB ≥1.1 用 `CREATE OR REPLACE SECRET ...`，
  老版本回退 `SET s3_*`；支持 credential_chain 式 fallback（环境变量 → ~/.cos.yaml）。
- `DuckDBCapabilities`：进程内探测版本、`supports_create_secret`、
  `object_cache_is_noop` 等；`apply_pragmas` 对 no-op PRAGMA 静默跳过并告警一次。

### Phase 5 ✅ — ReadHandle + 统一 read() + 成本路由 + read_uri（read/read_handle.py + read/scan_cost.py）
- `ReadHandle`：`to_arrow() / to_pandas() / to_polars() / to_lazy() / stream()`。
- `store.read(..., engine="auto", result="auto")`：engine∈{auto,duckdb,polars,pyarrow}，
  result∈{auto,arrow,pandas,polars,lazy,stream}；auto 按 `estimated_scan_cost` 路由
  （rows/bytes/columns/files/remote/local/selectivity/引擎启动成本）。
- `store.read_uri(uri, format="auto", ...)`：dev 允许白名单根下任意 URI；
  production/strict 默认拒绝 arbitrary URI，只允许 registry dataset 根。
- `read_auto()` / `read_auto_stream()` 改走成本路由，签名保持兼容。

### Phase 6 ✅ — Query 超时取消 + ManagedBatchReader（core/engine.py + read/managed_reader.py）
- `ManagedBatchReader`：包装 reader+cursor，`close() / __enter__ / __exit__ / __del__`，
  正常结束、break、异常、GC、client disconnect 都释放 cursor。
- DuckDBEngine 支持 `deadline_ms`：watchdog 线程超时调 `conn.interrupt()` 取消
  在途查询，抛 `DeadlineExceeded`（替代「查完再报超时」）。

### Phase 7 ✅ — FactorCatalog + 批量 read_factors（read/factors.py）
- `FactorMeta`：factor_id/version/frequency/universe/dtype/start/end/instruments/
  storage_uri/layout/schema_hash/formula_hash/operator_hash/data_snapshot/created/status。
- `FactorCatalog`：湖内 `_factor_meta.json` 清单，discover/load/save。
- `store.read_factors(factor_ids=[...], time_range, universe, layout)`：单条
  UNION ALL 查询批量读多因子（带 factor_id 虚拟列），或走 factor_matrix 单查询。

### Phase 8 ✅ — Quality Contract 升级（quality/contracts.py + quality/cli.py）
- 检查项：schema 对齐 / PK 唯一 / null ratio / range / finite·inf·NaN / 时间单调 /
  重复时间戳 / 覆盖率 / 截面覆盖 / 缺日期 / 未来时间戳 / PIT 泄漏 / 分区完整性 /
  schema 漂移 / 行数异常 / 分布漂移。
- CLI 读侧改走 `store.read_uri()` / `store.read()`，不再裸 `pandas.read_parquet`。

### Phase 9 ✅ — 测试与回归
- 每个新模块加单测（formats / predicate_ast / partition_planner / manifest /
  scan_cost / read_handle / read_uri / managed_reader / factors / quality）。
- 修复既有 4 个失败：`_format_literal` 接受 Timestamp（compute_and_write）、
  `adapter_options` 契约对齐、`test_real_repo_passes`（factor_engine 违规为外部代码，
  记录原因）。

### Phase 10 ✅ — DataAccess ↔ FactorEngine 集成层（2026-08-07）
目标是「DataAccess 管数据语义，FactorEngine 管数学语义」：字段解析、跨表读取、
PIT、单位归一化、读取规划统一由 DataAccess 完成。本次完成 DataAccess 侧全部原语；
FactorEngine 侧按优先级的改造由 factor_engine 团队消费这些原语。

- `SemanticFieldCatalog`（`read/semantic_catalog.py` + `config/semantic_fields.yaml`）：
  logical field → dataset/physical/dtype/unit/scale/time roles/PIT·join policy/aliases/
  mining_allowed 单一事实源；`store.resolve_fields()`（含物理列反查）；
  `normalize_table_units()` 输出层单位归一化（`normalize_units=` 默认关，逐字节兼容）。
- `DataRequest` / `ReadPlan`（`read/data_request.py` + `store.plan()`）：统一逻辑数据需求，
  字段按物理表 coalesce，`ReadPlan.explain()` / `.execute()`。
- `store.read_joined(anchor, fields, joins=)`：多数据集批量 join——每表一个子查询
  （投影 + 分区裁剪），DuckDB 内 exact / **pit_asof**（ASOF）一次完成；asof 子查询不按
  time_range 裁剪（保留窗口外历史可见记录）；多数据集合并 snapshot + 预算/审计。
- query-scoped snapshot token：`DatasetManifest.dataset_version/partition_version` 写入
  `_manifest.json` sidecar；`store.manifest_version()` 读 token 几十微秒、
  `store.is_snapshot_stale()` 判断过期（不再每隔 TTL 对整个 dataset describe）。
- 受控 `RelationHandle`（`read/relation_handle.py` + `store.sql_relation()`）：scan 上直接
  追加 SQL 表达式，`.arrow()/.collect()/.pandas()` 强制 QueryBudget(deadline)+audit+snapshot；
  `.relation` 只读逃生口。
- `read()`/`read_uri()` 增加 `normalize_units=`（默认 False）；DataRequest 支持 `universe`
  （registry 数据集作为股票池，与显式 instruments 求交集）。

**测试**：`tests/unit/test_integration_layer.py` 15 个用例（catalog/plan/read_joined/
manifest_version/sql_relation/单位归一化）。全量 **456 通过 / 0 失败**；allowlist rc=0。

**FactorEngine 侧消费（2026-08-08 已完成核心项，与并发会话协调）**：
- `runtime/batch_service.py`：`pit_enforce` 不再让 `run_many`/`run_many_parallel`
  退化逐因子 —— PIT 审计在编译期（`_dag_from_factors(pit_enforce=)` 调
  `assert_pit_safe`，纯审计不改计划）；`auto_warmup` 通过
  `_maybe_prepare_batch_warmup` 合并共享 union 加载窗口，一次读数、各因子独立
  `_trim_batch_result`。新增 `tests/runtime/test_batch_warmup_pit.py`。
- `storage/sources/data_access_source.py`：`refresh_snapshot` 优先走廉价的
  `store.manifest_version` token（`_manifest.json` sidecar，几十微秒），无 manifest
  才回退 `describe_dataset`（消除每 TTL 全量 describe）；`load_columns` 改走
  `store.read(engine='auto')`，读取引擎由 DataAccess 成本路由决定（`read_auto`
  不再是 polars 路径的开关，`lazy_scan` 保持显式 polars-lazy opt-in）。
- `storage/sources/composite_source.py`：`load_columns` 按源批量读取（同源多列合并成
  一次子源 `load_columns` → DataAccessSource 子源一次 `store.read`），非锚点列再逐个
  对齐到缓存的锚点索引。
- `cleaned_operators/operator_policy.py`：幂等加法补丁 `_FISCAL_EVENT_PACK_POLICIES`
  （`date_diff_days` / `cash_flow_lifecycle_stage`），解除并发会话 fiscal pack 导致的
  `load_all` 全体阻塞。

**2026-08-08 第二阶段（P0 性能/正确性 + FE 消费，已全量回归）**：

**DataAccess 侧**：
- `read_joined` 重写：**右表 instrument_filter 下推**（exact/asof/pit_asof 一律，
  不再「锚点 100 只、右表扫 5000 只」）；**显式 join 列**（`TemporalJoinSpec`
  的 decision_time/knowledge_time 绑定锚点/右表，修复跨表列名 bug）；**PIT
  seed+window**（`[start,end]` 窗口 + 每标的 start 前最后一条可见记录，UNION 后
  ASOF，语义与全历史逐字节一致——已验证等价）；**revision 去重**（QUALIFY
  ROW_NUMBER 按 revision_order，替代依赖扫描顺序的 keep_last）。
- `TemporalJoinSpec`（`read/temporal_join.py`）：语义级 PIT——knowledge_time/
  period_time/revision_order/availability（same_day|next_trading_day，严格大于
  实现 A 股盘后落地可见性）。catalog 字段自动推导 join 语义。
- `SemanticFieldCatalog` 时间语义核对：balance/income 字段 `knowledge_time`
  TradeDate→PubDate、加 revision_order；`total_liabilities` 物理列名修正
  TotalLiabilities→TotalLiability（审计抓到）；datasets.yaml 补齐财务 schema；
  `scripts/audit_semantic_consistency.py` 四方审计（catalog×registry×FE
  FIELD_REGISTRY×COS）0 问题。
- `required_filters` 强制执行（#8）：production fail-closed、research 告警。
- `DataRequest` 扩展（#7/#15）：source_params/filters_by_dataset/join_specs/
  field_params/aggregations/transforms/time_varying_universe；universe 按
  (date,instrument) 时变成员 INNER JOIN（不再拍平成静态集合）。
- Manifest（#17-#22）：`manifest_version` 去掉 O(N) glob（写路径 bump epoch，
  read path 信任）；`is_snapshot_stale` 支持 epoch；min/max typed 比较（int64
  不再 "9">"10"）；`_time_key` 不再 `[:10]` 截断；空裁剪返回空 relation（不回退
  全量扫描）；snapshot 复用 manifest 文件元数据（省 O(N) stat）；row-group 统计
  持久化 `_manifest_rowgroups.parquet`。
- `ScanCost` 真实 CBO（#25/#27）：manifest 裁剪后的 selected_files/bytes/
  rowgroups/estimated_rows + projection 字节估算 + estimate-vs-actual EMA 校准。
- `RelationHandle` 治理（#31/#30）：生产禁 `.relation` fetch，新增 schema/
  columns/types/sql_fragment 结构化参数 API。
- `ReadHandle.rows` 不触发 lazy collect（#28）。
- Deadline 连接池（#33）+ `execute_reader(deadline_ms=)`（#32）。
- 分钟聚合下推（#14）：`read/aggregation.py` AggregationSpec + DuckDB 内完成
  minute_at/minute_range/minute_of_day。

**FactorEngine 侧**：
- `DataAccessSource` 消费 SemanticFieldCatalog（#9）：`_resolve_columns` catalog
  优先、FE FIELD_REGISTRY 兜底；`store.read(normalize_units=True)`，`_normalize_
  contract_columns` 只处理 catalog 未覆盖字段（不双重归一化）。
- 清直接 parquet 读（#11）：TurnoverBaseDaily→`ashare_turnover_base_daily` 注册
  数据集、Intermediate→`factor_lake` 因子湖。
- SourceRef 批量 coalesce（#10）：`load_source_refs_batch` 按 (dataset,params,
  transform) 分组，财务多字段一次 `store.read`，同 dataset-mapped 一次
  `child.load_columns`。
- 分钟聚合 pushdown 接入（#14）：`_minute_daily` minute_at/minute_range 走
  DataAccess DuckDB 聚合，失败回退 pandas。
- `ExecutionResourceManager`（#34）：n_jobs×duckdb_threads≤cores。
- 幂等 governance 补丁：`_FISCAL_EVENT_PACK_POLICIES` 扩充到 fiscal pack 全量
  + safe_ops（并发会话 fiscal/safe_ops 并入 active surface 后 load_all 不再全阻塞）。

**仍待 factor_engine 团队消费（本阶段未做）**：
1. CompositeDataSource 全量改走 `store.read_joined`（跨源一次 DuckDB join，替代
   pandas merge_asof）——当前已做「同源多列一次读」，跨源 fusion 待做
2. SourceRef 进入 SQL/Polars native fast path（#13）：纯 StockDailyBar SourceRef
   等价于 inner 普通列，但 certified pandas 边界当前仍阻断 fully_sql——需要
   planner 把 SourceRef 展开成物理列（风险高，留给 certified-boundary 团队）
3. FIELD_REGISTRY 完全改由 SemanticFieldCatalog 生成/消费（当前 catalog 覆盖
   ~20 个字段，FE registry 长尾 ~100 个 scaled 字段仍由 FE 侧兜底归一化）
4. 财报 PIT pandas `pit_asof_join` 完全下沉到 `store.read_joined`
   （FE `_financial` 仍用 pandas 做 next_trading_day PIT；DataAccess 已具备
   TemporalJoinSpec 语义，接 anchor 数据集后即可替换）

**回归**：dataaccess 477 通过 / 0 失败；allowlist rc=0；语义审计 0 问题。
factor_engine 受影响单测各自通过（并发会话 operator-surface WIP 导致组合运行
不稳定，非本阶段改动）。

## 兼容性保障
- 新字段全部默认值；`_build_select_sql` 默认走 ParquetAdapter → 生成的 SQL 与改前逐字节一致。
- COS mirror/remote 逻辑保持环境变量语义，仅在其上层归一化为 backend。
- 所有现有 API 签名不动，新增 API 追加。
