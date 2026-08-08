# Changelog

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
