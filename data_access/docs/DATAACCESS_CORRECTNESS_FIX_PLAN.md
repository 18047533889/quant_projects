# DataAccess 0.8.0 正确性收口计划（DataAccess-only）

> 本计划只针对 `main` 上 DataAccess 自身（`/home/shw/quant_projects/dataaccess`），
> 不涉及 FactorEngine/资源治理/SourceRef 下沉。目标：把 44 项里会导致
> **读数正确性 / PIT 正确性 / 缓存正确性 / 发布一致性** 的 P0 全部修掉，
> 再处理治理与写入/发布层。并发的另一个 Claude 在改 `factor_engine/`，
> 与本计划文件交集仅 `data_access`（`read/scan_handle.py` 已被并发会话改动，
> 编辑前必须重读）。

## 执行状态（2026-08-08 完成度）
- ✅ #1/#2 Manifest 双 epoch + mutation 统一事务（`read/manifest.py` + `store.py`）
- ✅ #3/#4 缓存 key 完整 + byte-based LRU（`read/query_cache.py` + `read/predicate_ast.py`）
- ✅ #5 统一读前语义门禁（`store._prepare_read_request`）
- ✅ #6/#7 read_uri 契约回填 + sql_relation 沙箱（`store.py` + `read/sql_escape.py` + `read/relation_handle.py`）
- ✅ #8 exact join 跨表时间列（`store._read_joined_sql`）
- ✅ #9/#10 latest_period 默认 + pit=True 强语义（`read/semantic_catalog.py` + `store.plan`）
- ✅ #11 PhysicalPlanExecutor 聚合+join 组合（`read/physical_plan.py`）
- ✅ #12/#13 read_factors 版本门禁 fail-closed + matrix fallback 精确捕获（`store.py`）
- ✅ #14 snapshot 完整性 fail-closed（`store.py`）
- ✅ #15 PITEventIndex 权威元数据（`read/pit_event_index.py`）
- ✅ #23/#24 CoverageMatrix + max_staleness（`read/coverage.py`）
- ✅ #25/#26/#27/#29 日历感知完整性 + 混合读 + mirror inventory + remote 通用化（`cos/mirror.py` + `cos/remote.py`）
- ✅ #30/#31/#32/#33/#34/#35/#36/#37/#38 语义审计 + strict YAML + ContractIR 扩展 + schema_version + session namespace + metadata plane
- ✅ #39/#40/#41/#42/#43/#44 crash-safe overwrite + publish 冻结/强校验 + TransactionManifest + DuckDB COW + 锁 lease
- ✅ 回归：`tests/` 507 passed

> 注：`read/scan_handle.py` 已被并发会话修改；本次改动未触碰该文件，若后续需要
> 编辑请先 `git status` + 重读。`factor_engine/` 全部改动归并发会话，本计划未触碰。

## 执行原则
- mutation 一律走 `DatasetMutationTransaction`，禁止业务代码自己 `touch_manifest_epoch`。
- 无法证明相同 == 不相同（fail-closed）。
- 一个 Manifest 只允许在 `source_epoch == manifest_built_epoch` 时被 prune。
- 所有读入口先 `prepare_read_request()` 再过同一个语义门禁。

---

## 一、P0 读数/PIT/缓存/发布正确性

### 1. Manifest 双 epoch（source_epoch / manifest_built_epoch）—— `read/manifest.py` + `store.py`
现状：`manifest_epoch` 只有单值；`touch_manifest_epoch` 只改 `_manifest.json`，
`is_manifest_fresh` 只要看到 epoch 就 trust，但 `_manifest.parquet` 里的
min/max/rows 是旧的 → prune 可能返回错误数据。

改法：
- `DatasetManifest` / `_manifest.json` 增加 `source_epoch` 与 `manifest_built_epoch`。
- `is_manifest_fresh()` 改为：仅当 `source_epoch == manifest_built_epoch` 时 trust；
  老格式（无 split）回退文件名 glob 计数。
- `bump_manifest_epoch()` → 改名/改为 `bump_source_epoch()`，只递增 source_epoch，
  manifest 自动 dirty。
- 新增 `rebuild_manifest(store, dataset, params)`：mutation commit 后调用，重建
  manifest（读取 footer，O(files)），重建成功后 `manifest_built_epoch = source_epoch`。
- 保留 `build_manifest_for_dataset` 作为显式构建入口。

### 2. delete_rows 及所有 mutation 统一 manifest 失效 —— `store.py` + `write/`
现状：`write_arrow/upsert/publish` 都有 `touch_manifest_epoch`，但 `delete_rows`
没有；且各处直接调，无统一事务。
改法：新增 `store._DatasetMutationTx`（context manager）：
`mutation → bump source_epoch → 重建 manifest → commit`。`write_arrow/upsert/
delete_rows/publish_from_staging` 全部改用它。`delete_rows_from_dataset` 返回后由
store 统一失效。

### 3. `read_cached()` 缓存 key 缺 filters/limit —— `read/query_cache.py` + `store.py`
现状：key 含 dataset/params/time_range/instruments/columns/manifest_token，缺
`filters`、`limit`、`mode`、`allow_sparse`、`allow_effective_time`、`normalize_units`、
catalog fingerprint、contract IR fingerprint。
改法：`query_cache_key` 增加参数；Filter AST 用 canonical serialization + hash
（新增 `predicate_ast.canonical_filter_hash()`），不 `repr()`。

### 4. QueryCache 改 byte-based 约束 —— `read/query_cache.py`
现状：`capacity=128` 是 entry-count LRU。
改法：`max_bytes`（默认按 `table.nbytes` 估算）+ `max_entries` + `ttl_seconds` 联合
约束；`nbytes` 作为 eviction 权重。

### 5. 统一 PreparedReadRequest 语义门禁 —— 新增 `read/read_prepare.py` + 各读入口
现状：`read_result`（DuckDB）走完整 contract，但 `read_arrow_stream/scan_polars/
read_arrow` 不一致。
改法：新增 `prepare_read_request(dataset, *, time_range, instruments, filters,
mode, allow_sparse, allow_effective_time, ...) → PreparedReadRequest`，统一执行
temporal contract / required filters / allowed filter values / event cutoff /
schema / budget。`read/read_arrow/read_arrow_stream/scan/scan_polars/read_uri/
read_joined/sql/aggregation` 全部先调用。

### 6. production read_uri 禁止绕过契约 —— `store.py`
现状：production 只限制 URI 在已登记根下，随后构造 `_uri:parquet:xxx` 临时
Dataset，丢掉 PIT/required contract。
改法：production/strict 下 `read_uri` 先反查已登记数据集唯一匹配该 URI，命中则
用原契约；无法唯一匹配 → 拒绝。dev 保留现有宽松行为。

### 7. sql_relation 进 SQL sandbox —— `store.py` + `read/relation_handle.py`
现状：`sql_relation` 直接收 SQL，RelationHandle 直达 DuckDB。
改法：production/strict 下 `sql_relation` 必须声明 `read_datasets/view_columns/
read_params/read_time_ranges`，与 `store.sql()` 共用同一个 `sql_escape.run_sql`
治理（禁 INSERT/COPY/ATTACH/LOAD/INSTALL/read_parquet）。

### 8. read_joined exact join 跨表时间列 —— `store.py` + `read/temporal_join.py`
现状：exact join 生成 `a.{right_t_col} = b.{right_t_col}`，anchor 无该列时报错。
改法：exact join 用 `a.{decision_time} = b.{right_time}`（decision_time 缺省
anchor time_column，right_time 缺省右表 time_column），JoinSpec 一律显式
left_time/right_time。

### 9. latest_period 成为 financial 默认语义 —— `read/semantic_catalog.py` + `config/semantic_fields.yaml`
现状：`SemanticField.period_selection` 默认 "all"；YAML 里 financial 字段只写
`join_policy: pit_asof_backward` 没有 period_selection。
改法：`temporal_model in {financial_event, event}` 且未显式写 period_selection 时，
解析器默认 `latest_period`（用户显式 exact_period/annual/quarterly/all 除外）。
YAML 不强制改（向后兼容），默认在解析层生效。

### 10. DataRequest(pit=True) 强语义 —— `store.py` + `read/physical_plan.py`
现状：pit 只是 metadata 标记。
改法：`plan()` 时若 `pit=True`，对每个非 anchor 字段做 PIT 可证明性检查：
必须有 temporal contract 且 decision_time>=availability_time 可推导；缺失/歧义
production 拒绝。

### 11. PhysicalPlanExecutor 节点式执行 —— `read/physical_plan.py` + `store.py`
现状：`ReadPlan.execute()` 是 if/else 分支（aggregation/单表/多表）。
改法：新增 `PhysicalPlanExecutor`，把 Scan/Filter/Aggregate/TemporalJoin/
Universe/Normalize/Project/Limit 逐节点编译成一条 DuckDB SQL（或 Polars
Lazy/Arrow），可组合分钟聚合→财务 PIT→行业→normalize→filter→输出。先支持
DuckDB lowering，保留现有执行路径作为 fallback。

### 12. read_factors 版本门禁 fail-closed —— `store.py`
现状：`_check_factor_versions` 读取失败 `actual=None` 不报 mismatch；
`read_factors` 里 `except Exception: pass`。
改法：production/strict 下无法证明相同 == 不相同；`require_same_data_snapshot`
时任何因子读取失败即拒绝。

### 13. factor_matrix fallback 只捕特定异常 —— `store.py`
现状：`except ValidationError: pass` 太宽。
改法：新增 `MatrixUnavailable/MatrixCoverageMiss`，只捕这两个。

### 14. read_joined snapshot 完整性 fail-closed —— `store.py`
现状：`try: build snapshot except: continue`，可能 5 张表只记 4 张。
改法：参与数据集必须有 snapshot，否则 `SnapshotBuildError`（production 强约束）。

### 15. PITEventIndex 元数据 + 权威 prune —— `read/pit_event_index.py`
现状：sidecar 存在就 load、无 source epoch/snapshot、单文件读失败 continue、
写失败 pass、row_group 恒 None、prune O(N)。
改法：sidecar 加 `PITIndexMetadata{source_snapshot, manifest_epoch,
source_file_count, indexed_file_count, failed_files, schema_hash, complete,
created_at}`；仅 `complete AND source 匹配` 时 authoritative prune，否则 fail-open；
失败文件记录并置 incomplete。

---

## 二、语义 / Registry / 契约治理

### 16–22（原 review 第二部分中段，归档为以下条目）
- **16. `_enforce_read_contract` 覆盖到所有 read engine 入口**（并入 #5 PreparedReadRequest）。
- **17. 保留 manifest 空裁剪（#21 行为）**：`_prune_read_paths` 已实现，不回退全量扫描，
  维持。
- **18. `read_joined` 输出列冲突检查**：已实现（`seen_out` 冲突抛错），维持。
- **19. filter/query AST canonical hash**（并入 #3/#31）。
- **20. Snapshot/lineage 完整性**（并入 #14）。
- **21. QueryBudget 对所有引擎一致**（并入 #5）。
- **22. 审计完整性**：`sql_result` 已合并 snapshot，维持。

### 23. CoverageMatrix 全面覆盖矩阵 —— `read/coverage.py`
现状：只有 complete/partial/unavailable + 起止日期。
改法：新增 `CoverageMatrix`：expected trading days（用 session calendar）、
calendar days、monthly、quarterly、event-driven、missing internal partitions、
stale tail、per-column coverage、per-instrument coverage、partial universe。

### 24. max_staleness 真正执行 —— `read/coverage.py` + `cos_contract.py`
现状：只显示字段不判定。
改法：`compute_coverage` 用 calendar_domain 判定 stale（末个 partition 落后超过
max_staleness 交易/自然日 → status=stale）。

### 25. COS auto 完整性按 calendar_domain —— `cos/mirror.py`
现状：`_iter_dates` 按自然日。
改法：用 `calendar_domain/expected_cadence/missing_partition_semantics` 决定
哪些 partition 应存在（交易日型用 session calendar，自然日型用自然日）。

### 26. COS local+remote 混合计划 —— `cos/remote.py` + `read/partition_planner.py`
现状：本地齐全 → local，否则整区间 remote。
改法：`MultiLocationScan`：本地已有 partition + 远程缺失 partition，PhysicalPlan
统一表示。先实现"本地缺什么补什么"的最小混合（对缺的日期用远程路径）。

### 27. mirror inventory —— `cos/mirror.py`
现状：`Path.exists()` 判定 mirror 正确。
改法：mirror 目录写 `_mirror_inventory.json`：remote_key/etag/version_id/
remote_bytes/local_bytes/checksum/downloaded_at/verified；`ensure_local_mirror`
按 inventory 校验，中断/损坏/上游更新 → 重拉。

### 28. 单一 Dataset Registry 声明 storage —— `config/datasets.yaml` + `registry/loader.py`
现状：`DATASET_MIRROR_REGISTRY`（cos/remote.py 硬编码）与 datasets.yaml 两套。
改法：移除硬编码 registry，datasets.yaml 每个数据集声明 `storage.source/mirror/
serving`，ContractIR 编译出来。

### 29. remote backend 通用化 —— `cos/remote.py`
现状：`should_read_cos_remote` 主要针对 StaticDataset。
改法：remote backend 成为 Dataset storage 通用能力，ParametricDataset 也走
`prepare_cos_remote_paths`（factor lake / model outputs / features）。

### 30. Semantic Coverage Audit —— `read/semantic_catalog.py` + 新增脚本
现状：catalog 只覆盖核心字段。
改法：`semantic_coverage_audit(store)`：registry 物理字段 vs catalog，
输出 FULL_SEMANTIC / PHYSICAL_ONLY / AMBIGUOUS / BLOCKED。

### 31. Semantic YAML strict parsing —— `read/semantic_catalog.py`
现状：`_float_or_none` 非法 scale 变 None；`_bool_or_default` 用 `bool()`，
`mining_allowed: "false"` → True。
改法：strict：未知 key 报错、非法 boolean/enum/scale 报错、alias collision
报错、required filter 校验。

### 32. Dataset Registry bool parsing —— `registry/loader.py`
现状：`hive_partitioning = bool(raw.get(...))` 等。
改法：strict boolean 解析（"false" 字符串 → False，非 bool 类型报错）。

### 33. resolve_fields 全局回退 fail ambiguous —— `store.py`
现状：registry 里第一个含列的 dataset。
改法：候选=1 → resolve；候选>1 → `AmbiguousFieldError`（production 拒绝）。

### 34. ContractIR 扩展 —— `read/contract_ir.py`
现状：收 market/temporal/panel/pit/calendar/grain/cardinality/coverage/schema/
storage。
改法：加入 physical/logical field、dtype/unit/scale/frequency/knowledge_time/
effective_time/period_time/revision_order/availability/required_filters/
allowed_filter_values/unique_key/primary_key/duplicate_policy/join policy/
partitioning/storage backend/query policy/coverage policy。

### 35. Schema validation 全面化 —— `registry/schema_validation.py`
现状：不查 nullability/partition type/类型宽松。
改法：required/optional columns、nullable、logical vs physical dtype、safe cast
policy、unique key、partition schema、schema version。

### 36. Schema evolution 版本化 —— `write/upsert.py` + `read/schema` 契约
现状：upsert schema 不一致报错，evolution 靠 overwrite。
改法：Dataset Contract 支持 `schema_version/compatible_from/migration/
added_nullable_columns/deprecated_columns/renamed_columns`。

### 37. Namespace 作为 session context —— `core/namespace.py` + `registry/loader.py`
现状：loader 加载时替换 `${RUN_NAMESPACE}`。
改法：namespace 改为 request/session 上下文，路径 resolve 时再绑定；
新增 `DataAccessSession(namespace=...)`。

### 38. Metadata Plane 统一 —— 新增 `read/metadata_plane.py`
现状：manifest/coverage/PIT index/schema/rowgroup 各自 sidecar。
改法：`DatasetMetadataPlane` 聚合 manifest/rowgroup stats/coverage/PIT index/
schema/contract version/source epoch/mirror inventory/serving versions；
planner 只访问 plane。

---

## 三、写入 / 发布层

### 39. write_arrow(overwrite) crash-safe —— `store.py`
现状：`_clear_dir → write`，写一半崩溃即坏。
改法：写候选目录 → validate → 建 manifest → fsync → 原子 rename/pointer swap →
commit（与 publish 一致）。

### 40. Publish 冻结 source snapshot —— `write/publish.py`
现状：copy 期间 staging 可能被并发 upsert → mixed generation。
改法：`capture source snapshot → lock/freeze source → copy → verify source
unchanged`；或 immutable staging generations。

### 41. Publish 校验不止 row count —— `write/publish.py`
现状：只比 row count。
改法：比 file inventory / schema hash / partition inventory / row count /
primary-key count / manifest hash / source snapshot / contract IR；关键数据加
column checksum。

### 42. 多分区 upsert TransactionManifest —— `write/upsert.py`
现状：单分区原子、跨分区不保证。
改法：`TransactionManifest{transaction_id, affected_partitions, prepared,
committed, failed}`；全分区成功后 commit dataset version。

### 43. Upsert 不再全量 pandas merge —— `write/upsert.py`
现状：`pq.read_table().to_pandas() → concat → drop_duplicates → Arrow → parquet`。
改法：DuckDB MERGE-like COW 或 Arrow sort/dedup（factor lake / feature store 优先）。

### 44. 锁 lease / stale recovery —— `write/mutation_lock.py`
现状：已有 pid/host/created_at + stale_after + owner-dead 检测。
改法：补充 `process_start_time/transaction_id/acquired_at/lease_until`；
lease 过期但 owner 活（卡死）→ 允许打破。

---

## 落地顺序（依赖优先）
1. **manifest.py 双 epoch + store 写路径事务**（#1/#2/#39/#42 依赖它）
2. **query_cache key + byte LRU**（#3/#4）
3. **exact join / read_factors / snapshot 完整性 / PIT index**（#8/#12/#13/#14/#15）
4. **prepare_read_request 统一门禁**（#5/#6/#7）
5. **latest_period / pit=True / resolve_fields / YAML strict**（#9/#10/#31/#32/#33）
6. **ContractIR / schema / metadata plane**（#34/#35/#36/#38）
7. **write/publish 加固**（#39/#40/#41/#42/#43/#44）
8. **COS coverage / mirror / remote**（#23–29）
9. 回归测试 + 与并发 factor_engine 会话错峰

## 并发安全
- `factor_engine/` 文件一律不主动改（除非 store 调用链被并发会话改坏——编辑前重读）。
- `read/scan_handle.py` 已被并发会话修改；本计划如需动 scan，先 `git status` + 重读。
- manifest/query_cache 的测试文件属于本仓库，可安全更新。
