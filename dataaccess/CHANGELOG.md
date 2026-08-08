# Changelog

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
