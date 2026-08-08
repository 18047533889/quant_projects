# DataAccess 第四阶段计划：真实市场语义 + 物理数据布局 + 查询优化器统一

> 依据：AI 对 dataaccess 的第三轮审查 + `/home/shw/COS_ashare_lqtp_data_dictionary.md` +
> `/home/shw/COS_us_massive_data_dictionary.md` 两本真实数据字典。
> 目标：让 DataAccess 从「功能完整的 Universal Data IO Layer」进化成
> 「面向量化研究的 semantic query engine / physical data planner」。
> 部署面：会在各种服务器上部署（不只在当前机器），因此所有路径/镜像/日历/索引
> 都必须是显式可注入的，禁止依赖当前机器上的物理文件。

---

## 第一批（P0，语义正确性，必须完成）

### P0-1  Store 级 temporal model 强制（一等公民）  →  #44
- `cos_contract.py`：`_MODELS` 加 `RAW_EVENT`；补 `is_raw_event` 属性。
- `store.py` 新增 `_enforce_read_contract(ds, dataset, *, mode, allow_sparse, columns, ...)`，
  在所有读取入口调用：
  - `read_result/_read_dataset_object`（DuckDB 单表）、`read_arrow_stream`、
    `_scan_polars_with_paths`、`_read_pyarrow`、`read_joined`。
- 语义：
  - `EMPTY` → 一律拒绝。
  - `X0` → 必须 `allow_sparse=True`（显式确认稀疏读取），否则拒绝。
  - `E1/E2` → 普通 panel 读在 production/strict 下拒绝（必须走 PIT/event API，
    即 `mode="event"` / `mode="pit"`）；research 告警。
  - `STATIC` → 普通 panel（time×instrument）拒绝，`mode="dimension"` 放行。
- `read()`/`read_result()` 增加 `mode: str = "auto"`（auto=按契约推断）与
  `allow_sparse: bool = False`。

### P0-2  TemporalJoin `period_selection`（财务 PIT 正确性）  →  #45
- `temporal_join.py`：`TemporalJoinSpec` 增加 `period_selection:
  latest_period|exact_period|annual|quarterly|ttm|all`（默认 all 兼容旧行为；
  财务字段默认 latest_period）。
- `store._read_joined_sql`：对 asof/pit_asof 右表增加 period 语义包装：
  ```
  dedup → 计算 running_max(period_time) over (PARTITION BY inst ORDER BY knowledge_time)
  → 只保留 period_time == running_max 的行 → 再对 knowledge_time ASOF
  ```
  即「先选目标 period，再选该 period 最新 revision」，旧报告期晚修订不回滚。
  `exact_period`：过滤到指定 period；`annual/quarterly/ttm`：先按 timeframe
  类别过滤再 latest_period。
- 美股：`period_selection` 与 `timeframe` 组合（E2 必须 filter timeframe）。

### P0-3  MarketCalendar/SessionCalendar 编译 `available_from`  →  #46
- 新增 `read/session_calendar.py`：
  - `SessionSegment`（上午/下午）、`MarketSession`（ashare 09:31-11:30 + 13:01-15:00
    共 240 根；us 9:30-16:00 连续）、`MarketCalendar`（交易日历 + timezone）。
  - 交易日历来源：显式注入 `trading_days`，或从 registry 的
    `ashare_calendar`/`us_calendar` 数据集惰性加载；兜底周末休市 + 显式节假日表。
  - `available_from(market, knowledge_time, availability)`：`next_trading_day`
    编译为下一交易日 session 起点；`same_day` 为 knowledge_time 本身。
- `temporal_join.py`：`availability` 支持 `session`（按 session 边界判断
  同盘/次盘）；`_read_joined_sql` 在提供交易日历时把 `next_trading_day`
  从「严格 >」升级为「decision >= available_from」（trading-day VALUES CTE）。

### P0-4  DataRequest 编译成物理计划 DAG  →  #47
- 新增 `read/physical_plan.py`：节点
  `ScanNode → FilterNode → TemporalJoinNode → AggregationNode → NormalizeNode → ProjectNode`。
- `store.plan()` 把 `aggregations/transforms/field_params/pit/frequency` 真正编译进
  DAG（不再只当声明）；`ReadPlan.explain()` 渲染节点树；
  `ReadPlan.execute()` 按 DAG 执行：
  - `AggregationNode` → 走 `aggregate_bundle`（P0-6），一次 scan 多聚合。
  - `TemporalJoinNode` → read_joined（带 period_selection）。
  - `NormalizeNode` → normalize_units。
  - `ProjectNode` → 列投影 / transforms 结果命名。

### P0-5  分钟聚合时区/交易时段修复  →  #48
- `aggregation.py`：`AggregationSpec` 增加 `market`/`timezone`/`session`。
- SQL 侧：`QuoteTime`(UTC) 先 `timezone('Asia/Shanghai', t)` 再取 HH:MM；
  `minute_at("09:31")` 匹配北京 09:31 而非 UTC 09:31。
- `minute_of_day` 用 session elapsed bar index（09:31=0 … 11:30=119，
  13:01=120 … 15:00=239；`slot = (local_min - 571 - (90 if local_min>=780 else 0)) // period`），
  而不是自然时钟 `minute - 570`。

### P0-6  Multi-aggregation coalescing（一次 scan 多聚合）  →  #49
- `aggregation.py` 新增 `AggregationBundle`：`(field, spec, output_name)` 列表。
- `aggregate_minute_bundle(store, dataset, bundle, ...)`：单次 `FROM (scan)` +
  单次 `GROUP BY (date, inst)`，每输出列用 `agg(val) FILTER (WHERE <spec 条件>)`。
  一次性产出 morning_volume / morning_return / close_30m_vol / intraday_high 等几十列。
- 语义聚合映射保持不变（open→first, high→max, volume→sum …）。

### P0-7  分钟聚合走完整 QueryBudget/deadline/snapshot/audit  →  #50
- `aggregate_minute_to_daily` / `aggregate_minute_bundle` 增加 `query_budget`/
  `as_of` 参数：`execute_arrow(deadline_ms=budget.max_elapsed_ms)` +
  `enforce_arrow_budget` + `audit.record` + 返回带 snapshot 的句柄。

### P0-8  美股财务 PIT 物理索引 / Serving Layout  →  #51
- 新增 `read/pit_event_index.py`：
  - `PITEventIndex`：`(ticker, filing_date, period_end, timeframe, file_path, row_group)`。
  - `build_pit_event_index(store, dataset, ...)` → 落盘 `_pit_event_index.parquet`。
  - `prune_paths_by_filing_range(...)`：即使文件按 `period_end` 命名，也能按
    `filing_date` 精准裁剪文件路径。
- `store.pit_event_index(dataset)` / `store.prune_pit_paths(...)` 访问器；
  写 `docs/PIT_SERVING_LAYOUT.md` 说明 COS 原始布局不动、研究热路径走优化湖。

### P0-9  `allowed_filter_values` 下沉 Store 做值校验  →  #52
- `store._enforce_required_filters` 扩展：当过滤列命中契约 `allowed_filters`
  时校验值集合（timeframe∈{quarterly,annual,trailing_twelve_months}、
  IndustrySource∈{sw_l1,...}），production fail-closed / research 告警。
- `read()`/`read_result()`/`read_joined()` 的 filters 与 filters_by_dataset
  都统一校验。

### P0-10  grain / unique_key / cardinality 强契约 + fan-out 守卫  →  #53
- `cos_contract.py`：`COSDatasetContract` 增加 `grain` / `unique_key` /
  `cardinality` / `required_dimension_filters`。
- `store.read_joined`：join 前按 unique_key 判断 fan-out——exact join 且右表
  unique_key 不含 (time, instrument) 时，production 拒绝静默 fan-out；
  已知 one-to-many 表（TopTen/Industry/US capital shares）要求显式
  `duplicate_policy`/聚合策略。
- A股：`StockIndustry (TradeDate,Symbol,IndustrySource)`、财务
  `(Symbol,PubDate,ReportPeriodEndDate)`、TopTen（一股多股东）、IndexConstituent（+IndexSymbol）。

### P0-11  SemanticFieldCatalog 跨市场歧义 fail-closed  →  #54
- `semantic_catalog.py`：`resolve_one()` 在无 market/dataset 且存在多个非 any
  候选时：production 抛 `AmbiguousSemanticFieldError`，research 告警后取第一个。
- 新增异常类 `AmbiguousSemanticFieldError`（core/exceptions.py）。

---

## 第二批（P1，结构一致性）

### P1-12  Contract IR 编译器  →  #55
- 新增 `read/contract_ir.py`：`build_contract_ir(registry, cos_contracts, catalog)`
  把三套声明编译成一个 `ContractIR`（per-dataset: name/market/temporal_model/
  panel_policy/pit_policy/calendar_domain/grain/coverage/schema/字段）。
- `store.contract_ir()` + `contract_ir_fingerprint()`；`scripts/audit_contract_ir.py`
  CI 对齐：每个 COS 契约数据集必须在 registry（或显式 external），双向核对。

### P1-13  `calendar_domain` 契约  →  #56
- `cos_contract.py`：`COSDatasetContract.calendar_domain` =
  `trade_day|calendar_day|event_time|static`（按 model 给默认，显式覆盖）。
- 字典事实：StockList/Status/Industry/TopTen/ETFList/IndexList/IndexConstituent
  含周末自然日文件；行情/估值主要是交易日。

### P1-14  coverage / completeness / staleness 契约  →  #56
- 新增 `read/coverage.py`：`CoverageContract`（coverage_start/end /
  expected_cadence / max_staleness / missing_partition_semantics）。
- `store.coverage(dataset)`：manifest 观测 vs 声明；回答
  complete/partial/unavailable，而不是读出一堆 NaN 才发现。

### P1-15  `RAW_EVENT` 正式 temporal model  →  #57
- `us_fact_news` 改为 RAW_EVENT（PIT=published_utc、entity_list explode、
  event_id）；财务事件与新闻事件不再共用 E2 标签。
- `cos_event_runtime.py`/`resolve_event_clock` 支持 RAW_EVENT。

### P1-16  Event API 未来数据 cutoff  →  #57
- effective_time_only（A股/美股 Dividend）：读取时对 `event_column` 强制
  `<= min(time_range.end, now/指定 cutoff)`，禁止未来数据前视。
- `TemporalJoinSpec` 增加 `future_cutoff: bool = True`（事件右表在 join 前裁剪）。

---

## 第三批（P2，性能 / 治理 / 正确性 gates）

### P2-17~28  物理布局分层（从截断片段重建）
- 分钟级 serving layout：`Symbol%N` bucket（32/64/128），全市场日内扫描走
  date-major、窄股票池长历史走 bucket-major，Cost Router 自动选择。
- 日频数据不重分桶：`time_range → 具体日期文件路径`（直接路径生成）。
- Source Layer → Optimized Serving Layer → Semantic View Layer 三层；
  COS 原始布局可追溯性不动，Serving 层可改 partition/sort/rowgroup。

### P2-29  TopTen / Industry 预聚合 serving datasets  →  #58
- A股 TopTen/Industry 高扇出 → 日频聚合（top1_share/top10_concentration/
  institutional_share/sw_l1 code）落 `ashare_*_serving_daily` 物化视图。

### P2-30  真实 workload Benchmark Gate  →  #59
- `scripts/benchmark_workloads.py` 固定真实尺寸：A股日频 1日/1年/5年、
  A股分钟全市场1日 122.76 万行、100/1000 股多年、美股多年、A股 latest_period
  PIT、美股 filing_date+timeframe PIT、Industry sw_l1、TopTen 聚合、
  US shares×close、FactNews published_utc explode、COS cold/warm。
- 记录 `files opened / glob/list/stat/footer / row groups / bytes scanned /
  rows returned / row amplification / Arrow/Pandas bytes / physical scan count`。

### P2-31  正确性 gates（关键）  →  #59
- 固定测试：Return/10000、美股 Ret 不除；A/美 Factor 乘法后复权；PubDate vs
  filing_date；旧报告期 revision 不回滚；US timeframe 缺失报错；X0/EMPTY panel
  拒绝；A分钟 UTC→Asia/Shanghai、午休 bar index；dynamic universe；
  effective-only/future dividend 不前视；US missing halt partition 不得默认为 False。

---

## 依赖顺序（实施推进）
1. `session_calendar.py`（P0-3/5 地基）
2. `cos_contract.py` 扩展：RAW_EVENT / calendar_domain / grain / coverage（P0-1/10、P1-13/14/15）
3. `semantic_catalog.py` fail-closed（P0-11）
4. `store.py`：读入口契约强制 + allowed_filter_values 值校验 + fan-out 守卫（P0-1/9/10）
5. `temporal_join.py`：period_selection + session availability + future cutoff（P0-2/3、P1-16）
6. `aggregation.py`：时区/时段修复 + bundle + budget/audit（P0-5/6/7）
7. `physical_plan.py`：DAG + explain/execute（P0-4）
8. `pit_event_index.py`（P0-8）
9. `contract_ir.py` + audit 脚本（P1-12）
10. `coverage.py`（P1-14）
11. serving 分层 + 预聚合 + bucket 路由（P2）
12. benchmark_workloads + correctness gates 测试（P2）
13. 全量回归 + sync_install + 版本/CHANGELOG/文档
