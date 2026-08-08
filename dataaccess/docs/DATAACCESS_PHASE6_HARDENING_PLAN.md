# DataAccess 第二轮收官加固计划（Phase 6）：新增能力之间的接口收口

> 依据：AI 对 DataAccess 的第二轮审计（仅含上一轮清单之外的新问题：P0-1~P0-56、
> P1-1~P1-26、P2-1~P2-5）。审计基线 commit `6d66e9c3bdfa57c6152374faf0bd49c5d0c013ac`。
> 核心结论：不需要新子系统，新问题主要是「新增能力之间的接口没有完全收口」——
> 新版 generic remote vs 旧 COS monkey patch；deadline connection pool vs
> ManagedBatchReader ownership；safe SQL vs semantic contract；runtime registry patch
> vs registry single source of truth；transaction/write lock vs delete/publish 特殊分支。
>
> 全部改动只在服务器本地（`/home/shw/quant_projects/dataaccess`），不 push GitHub。

---

## 一、优先级判定

本轮把 AI 建议的「最危险 20 项」作为 P0 先做，其余 P0/P1/P2 按「correctness / 数据损坏
/ 安全」优先顺延。**诚实原则**：高风险低活跃路径（完整 SQL AST sandbox、SQL AST 引擎级
white-list、ClickHouse streaming、minute 完整语义对齐）会以「guard + 文档」落地，不假装
完成——因为 AST parser 层重写属于单独 wave，本轮先建立 fail-closed 守卫。

---

## 二、P0 实现清单

### P0-1  COS monkey-patch API 签名同步 `cos_storage_runtime.py`
- `_patched_build_remote_paths` / `_patched_materialize_remote_via_cli` 增加
  `ds=None, params=None` 透传；`mirror.sync_dataset` 已兼容。
- `install_cos_storage_runtime` 后 `build_remote_paths(..., ds=, params=)` 不再
  TypeError，ParametricDataset generic remote 不被旧 patch 弄残。

### P0-2  generic `storage.source` 路径统一 normalize + authorize `cos/remote.py`
- 新增 `ResolvedStoragePath`（dataclass：`canonical_uri` / `scheme` / `bucket` /
  `key_prefix`）。
- 所有 remote 路径离开 planner 前经 `resolve_storage_path()`：URI normalize →
  credential profile resolve → `authorize_s3_path` → canonical `s3://` execution URI。
- `_remote_paths_from_storage` 与 `hybrid_cos_read_paths` 不再自行拼 `cos://`。

### P0-3  params/glob format 错误 fail-closed（不再 `**/*.parquet` 降级）`cos/remote.py`
- `_remote_paths_from_storage`：ParametricDataset `glob_template.format(**validated)`
  任何 `KeyError / ValueError / IndexError / FormatError` → `ValidationError`
  （missing/invalid param / unknown layout 一律拒绝，禁止宽泛 `except Exception →
  **/*.parquet`）。未知 layout 也不再 fallback 全库 glob。

### P0-4  hybrid 路径统一 `s3://` execution URI `cos/remote.py`
- `hybrid_cos_read_paths` 的 remote fragment 用 `_s3_table_base(spec)`（s3://），
  不再 `_cos_table_uri` 返回 `cos://`。executor 只见到一种 scheme。

### P0-5  S3 credential state 改为 per-connection `cos/s3_duckdb.py`
- `S3ConfigState` 用 `WeakKeyDictionary[conn, fingerprint]`；`ensure(conn)` 只看
  本连接指纹；`configure_fresh(conn)` 只记录本连接，**绝不**设置全局 `_configured`。
- `reset()` 清空。进程内残留全局 `_configured` 字段删除。

### P0-6  pooled 连接不交给 ManagedBatchReader 直接 close `core/engine.py`
- `_execute_isolated_reader_with_deadline`：`ManagedBatchReader` 不再拿 pooled conn
  当 `cursor`（pooled conn 的 `close()` 归 pool 所有）。改用 `on_close` 归还 + 健康
  检查；reader 内部 `cursor=None`（pooled 连接绝不 `.close()` 进池）。

### P0-7  interrupt/timeout 连接必须 discard，不重新进池 `core/engine.py`
- `_DeadlineConnectionPool.release(conn, healthy=True/False)`；deadline 触发 /
  interrupt / DuckDB fatal / credential error / 已关闭连接 → `healthy=False` →
  直接 `conn.close()`，不回到 idle 队列。只有 clean 成功执行才 `release` 可复用。

### P0-8  DeadlineConnectionPool 真正限并发 `core/engine.py`
- 增加 `_active` 计数 + `acquire(config, wait_seconds)`：达上限时等待（admission
  deadline 内），超时 `ResourceExhausted`，不再无限 overflow。
- 移除 `_overflow_ids` 无限溢出逻辑；`release` 按 healthy 分流。

### P0-9  deadline race 最终判定 `core/engine.py`
- `_execute_isolated_with_deadline` / reader 版本：查询正常返回后**再检查**
  `timed_out.is_set() or monotonic() >= absolute_deadline`，命中则 `discard_result()
  + raise DeadlineExceeded`。deadline 用 absolute monotonic，不各阶段重算。

### P0-10 `DuckDBEngine.close()` 关 deadline pool + EngineClosedError `core/engine.py`
- `close()` 先 `_deadline_pool.close()` 再 `self._conn.close()`；加 `_closed` flag，
  关闭后 `acquire/execute` 抛 `EngineClosedError`。

### P0-11  remote/deadline 用显式 `requires_remote_storage`，不 sniff SQL 字符串
- `PhysicalPlan` / `register_specs` 携带 `requires_remote_storage` + credential
  profiles；executor 按计划配置 fresh 连接；短期至少同时扫描 sql + 绑定 URI params
  （`execute_scoped_sql_arrow` / `_execute_isolated_*` 的 params 里含 `s3://` 也触发
  S3 配置）。

### P0-12  `store.sql()` 走 semantic gate，不绕过 temporal contract `read/sql_escape.py`
- `_prepare_sql_views` 内对每个 read dataset 先执行 `store._prepare_read_request`
  （temporal model / event cutoff / required filters / allowed values 强制），
  失败即拒绝，不再只 `registry.get + resolve_paths + build_select_sql`。
- 通过 `semantic_gate` 回调注入（保持 sql_escape 与 store 解耦）。

### P0-13  SQL 白名单 false-positive + AST 级 deny-by-default（阶段性）`read/sql_escape.py`
- 现有 `_FORBIDDEN_RE` 改走 `_reject_forbidden_tokens` 的字符串-aware 扫描
  （P2-2 一并解决：只扫 code 段，不扫字符串/注释）。
- 新增 `validate_sql_sandbox`：断言单条 SELECT/WITH；`deny-by-default` 的 table
  function 名单（`read_parquet/read_csv/glob/...`）。完整 AST engine white-list
  留待单独 wave（本轮 fail-closed 兜底 + 文档）。

### P0-14  QueryBudget production floor（take stricter）`read/query_budget.py`
- `resolve_query_budget`：production/strict 时与显式 budget 合并取**更严**
  （`merge_production_floor`）：`max_rows=min`、`require_columns=or`、
  `require_time_range=or`、`max_result_bytes/max_elapsed_ms/max_scan_files=min`。
  显式宽松 QueryBudget 不再能取消 production 默认。

### P0-15  eps 编译成 financial PIT（与 ROE 同类）`config/semantic_fields.yaml`
- `eps`（A股）：`temporal_model: financial_event`、`knowledge_time: PubDate`、
  `period_time: ReportPeriodEndDate`、`revision_order: [UpdateTime]`、
  `availability: next_trading_day`、`join_policy: pit_asof_backward`、
  `period_selection: latest_period`。

### P0-23 美股 X0 稀疏原始字段 `mining_allowed=false` `config/semantic_fields.yaml`
- `dividend_yield / return_on_equity / return_on_assets / price_to_earnings /
  price_to_book / us_market_cap`：`mining_allowed: false`（X0 稀疏原始字段不能当
  默认横截面挖掘字段）。

### P0-24 美股 market_cap 改 derived semantic `config/semantic_fields.yaml`
- `us_market_cap` 不再指向 `us_stock_valuation_daily.market_cap`（X0 稀疏）；
  改为 derived `us_market_cap_daily = us_stock_daily.Close ×
  us_ticker_shares_snapshot.weighted_shares_outstanding`，`mining_allowed: true`。

### P0-25  RAW_EVENT one-to-many 禁止 generic latest-asof `cos_event_runtime.py`
- `read_cos_events_asof`：contract 为 RAW_EVENT / one-to-many / event 时
  `ValidationError`（新闻非 state table；必须显式 window/aggregate 聚合）。

### P0-26  `read_cos_events/read_cos_panel` 生产下不被 view_columns 门禁卡住
- 改为直接走 `PreparedReadRequest` / `read_arrow`（走 `_prepare_read_request`），
  或在 `self.sql(...)` 传完整 `view_columns={dataset: required_internal_columns}`
  （含 PIT 必要列）。确保官方 semantic helper 在 production 可正常跑。

### P0-27 美股财务 `period_end` 统一 `_period_time = CAST(... AS DATE)` `cos_contract_us.py`
- scan boundary 生成 `_period_time` 派生列，所有 PIT period logic 只消费
  `_period_time`；`_period_time` 全量可转 DATE，否则 `DataQualityError`。
- 同样检查 `ex_dividend_date` / TickerShares `TradeDate` 等 string temporal 字段。

### P0-28  manifest 构建遇坏 parquet → fail-closed `read/manifest.py`
- `build_manifest_for_dataset`：任何 `pq.read_metadata` 失败 → 生产 manifest build
  失败（不再 `continue` 静默跳过 + 保存成 fresh manifest）。

### P0-29  manifest `bytes` 用真实文件大小 `read/manifest.py`
- `fbytes = stat().st_size`（remote 用 content_length）；`meta.serialized_size`
  单独记为 `footer_bytes`（新列，向后兼容）。

### P0-30  manifest parquet + JSON sidecar 原子 generation `read/manifest.py`
- `save()` 写 `_manifest.parquet` 与 `_manifest.json` 时先生成同
  `manifest_generation_id`（UUID），写 tmp → fsync → replace；两份都校验同一
  generation 才认为新鲜；不匹配视为 mixed → 不 fresh（fail-closed）。

### P0-31  `manifest_root_for_paths()` exact single-file → parent `read/manifest.py`
- path 无 glob 且是 `.parquet` 文件 → 返回 parent；glob → 静态目录前缀；
  directory → 本身。

### P0-32  publish manifest 在原子切换前写好并校验 `write/publish.py`
- candidate 内先生成 publish manifest → 校验 → 与数据一起 rename 成 target；
  `write_publish_manifest` 失败 → 不进行原子切换（杜绝「caller 失败但新版本已上线」）。

### P0-33  publish 单一 inventory snapshot `write/publish.py`
- `_file_inventory` 直接含 `(rows, bytes, schema_hash)`；`source_rows` 由 inventory
  sum 得出，不再单独 `_count_parquet_rows` 与 inventory 分离。

### P0-34  upsert new_table 重复 upsert key → reject `write/upsert.py`
- `_duckdb_merge` 前校验 new_table 对 `upsert_on` 唯一；重复 → `ValidationError`
  （不允许依赖输入/执行顺序的 tie-break）。

### P0-35  upsert_on ∩ partition_by 重叠时确定语义 `write/upsert.py`
- `local_merge_key = upsert_on - partition_by`（partition key 由目录固定）；
  重叠列从 merge key 剔除，行为确定。

### P0-36  delete_rows 坏 parquet → abort（production）`write/upsert.py`
- `delete_rows_from_dataset`：任何 candidate 文件读取失败 → 整个 delete abort
  （production）；`best_effort=True`（研究/手动）才允许 skip 并返回
  `failed_files / remaining_unverified_rows`。

### P0-37  delete_rows 两阶段 preflight `write/upsert.py`
- PHASE 1：扫描元数据/谓词计算完整 delete plan + 行数，验证 `max_rows`；
  PHASE 2：commit delete plan（max_rows 超限 → 0 行实际删除）。

### P0-38  strict typed 配置 spec `registry/loader.py`
- `partitioning/engine` 等 mapping 字段解析成 `StorageSpec/PartitionSpec/
  EnginePolicy`（strict enum/bool/unknown-key reject/required field/cross-field）。

### P0-45  `schema_version` 类型错 → 启动失败 `registry/loader.py`
- key 出现但非法（非字符串/空串）→ `ValidationError`，不再静默变 None。

### P0-46  StrictYAMLLoader（duplicate-key 保护）`registry/yaml_loader.py`
- 新增 `StrictYAMLLoader`（duplicate mapping key → 硬错误）；`loader.py` /
  `semantic_catalog.py` 统一使用。

### P0-47  unresolved `${ENV}` → 启动失败 `registry/loader.py`
- registry compile 完成后检查残留 `${...}` / `$VAR` → `ValidationError`。

### P0-48  `DATA_ACCESS_EXTRA_ALLOWED_ROOTS=/` production 拒绝 `registry/paths.py`
- production 下 extra root 拒绝 `/`、`$HOME`、过宽祖先；普通环境变量不作为
  部署级授权根。

### P0-49  ParametricDataset authorized root 用显式 `authorized_root` `registry/loader.py`
- 新增 `authorized_root` 字段（显式配置优先）；缺省时不再用「第一个 `{` 前字符串」
  当安全边界——从 path_segment 参数编译安全 prefix 或要求显式声明。

### P0-50  `ParamSpec.path_segment` strict bool + unknown-key reject `registry/params_validation.py`
- `bool(payload.get(...))` → 复用 `_strict_bool`；`ParamSpec.from_yaml_value` 增加
  unknown-key reject。

### P0-51/52  `QUANT_SCHEMA_CHECK` production floor `registry/schema_validation.py`
- production 下非法值 → 启动失败；显式 `off` → 拒绝（只有
  `BREAK_GLASS_SCHEMA_CHECK` + 审计才允许临时降级）。

### P0-53  默认 `partition_columns=()` `registry/loader.py` + `read/stats.py`
- `_parse_partition_columns` 缺省/空 → `()`；schema checker 不再对未显式声明
  partition 的数据集错误豁免。

### P0-54  `FormatSpec.extra` option whitelist `read/formats.py`
- 每种格式 `_ALLOWED_OPTIONS`；unknown key → `ValidationError`；禁止 raw option
  名进 SQL。

### P0-55/56  aggregation timezone/参数严格校验 `read/aggregation.py`
- timezone 只接受 `ZoneInfo(tz)` 可解析的 IANA；SQL 走 `sql_string_literal`；
  `period>0 / index>=0 / HH:MM regex / market enum / unknown-key reject`，禁止
  silent clamp。

---

## 三、P1 实现清单（本批落地项）

- **P1-1** recursive COS complete marker → `RemoteSnapshot`（含 object_count /
  inventory_hash / watermark），revalidate 支持。`cos_storage_runtime.py`
- **P1-3** production 只 `LOAD httpfs`，不自动 `INSTALL`；启动 health check 缺失
  → deployment failure。`cos/s3_duckdb.py` + `cos/remote.py`
- **P1-4/5** `retry_io`：`ExceptionClassifier`（TRANSIENT_IO / THROTTLED /
  NETWORK_RESET / DEADLINE / SCHEMA / CORRUPTION / AUTH 分类，仅前几种重试）+
  指数退避+jitter+absolute deadline；`gc_before_retry` 默认 False。`core/retry.py`
- **P1-6** 生产禁止自动 `EXPLAIN ANALYZE` 重跑；PROFILE 只走显式手动诊断。
  `read/telemetry.py`
- **P1-7/8** ManagedBatchReader：`read_next_batch()` EOF/异常自动 close；telemetry
  记录 `setup_ms / time_to_first_batch / stream_duration / total_duration / rows /
  bytes`。`read/managed_reader.py` + `core/engine.py`
- **P1-9** ReadHandle stream 消费状态保护：`stream()` 消费后 `to_arrow()` 拒绝
  （truncated 结果静默返回 → 报错）。`read/read_handle.py`
- **P1-16/17** `DatasetStatsSnapshot.column_null_ratio` → 真 null 率
  （`null_count/total`），字段改名 `column_null_ratio` 语义修正；
  sidecar 按 time_range 分桶 / 有 range 时不用全库 num_rows。`read/stats.py`
- **P1-18** CSV/JSON managed dataset：schema 声明时推给 reader（固定 dtype / date
  format / null rules / sample_size）。`read/formats.py`
- **P1-19/21** lock lease heartbeat + 统一 transaction lock（废弃裸 `_publish_lock`
  双重锁）。`write/mutation_lock.py` + `write/publish.py`
- **P1-23** 关键 publish/commit 写路径 fsync(file) + fsync(parent)。`write/publish.py`
- **P1-24** COW upsert partition split 走 DuckDB/Arrow，不整表 to_pandas。
  `write/upsert.py`
- **P1-26** cgroup/cpuset-aware 资源探测（memory.max / cpu.max / cpuset.effective）。
  `core/duckdb_config.py`

## 四、P2 实现清单（本批顺手清）

- **P2-2** forbidden-token 扫描字符串-aware（并入 P0-13）。
- **P2-3** `RelationHandle.explain()` 走安全 engine API，不经过被禁 `.relation`。
- **P2-4** `DatasetStatsSnapshot.from_dict` 缺省 `partition_columns=()`（与 P0-53 一致）。
- **P2-1** filter canonicalization 子节点完整 hash 排序（低风险，本轮 guard + 文档）。

## 五、留待单独 wave（诚实标注，不假装完成）

- 完整 SQL AST engine white-list（P0-13 完整版）。
- ClickHouse Arrow RecordBatch streaming。
- minute aggregation 完整 AggregationBundle 语义对齐。
- P1-2 多 credential profile → runtime secret（本轮单 profile + per-connection 收敛）。
- P1-20 Authoritative Commit Manifest 与 Operational Audit Log 分层。

## 六、验证

- 新增 `tests/unit/read/test_phase6_hardening.py` 覆盖：monkey-patch 签名、
  params fail-closed、s3:// canonical、per-connection S3、pool 并发上限、
  deadline race、sql semantic gate、QueryBudget floor、filter None、market mismatch、
  eps PIT、sparse mining_allowed、manifest fail-closed/bytes、delete preflight、
  duplicate YAML、unresolved env、path_segment bool。
- 全量回归：dataaccess tests 套件全绿；factor_engine 受影响子系统不回归。

## 状态

2026-08-08 开始实施。
