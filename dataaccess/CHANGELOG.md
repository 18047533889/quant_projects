# Changelog

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
