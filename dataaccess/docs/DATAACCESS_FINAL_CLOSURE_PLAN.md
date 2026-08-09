# DataAccess 最终收官整改计划

## 收官轮 closure ledger（0.9.7，2026-08-09）

真实数据 + 运行时 + 破坏性验收驱动；最终全量 **758 passed / 1 failed**（唯一
failed = `test_check_allowlist.py`，根因是 factor_engine/ 并发会话新脚本
`pd.read_parquet` 绕过 DataAccess，属于 factor_engine 领域——本会话按约束不动
factor_engine，标记 BLOCKED，待并发会话/用户授权后由 owner 修复）。

| ID | Area | Test | Dataset/Backend | Result | Bug root cause | Fix | Regression test | Status |
|----|------|------|-----------------|--------|----------------|-----|-----------------|--------|
| 1 | mutation lock | TOCTOU inode fencing（round7 遗留项核实） | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 2 | StorageSpec | scheme→backend 推断 | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 3 | PathAuthorizer | namespace 延迟解析 | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 4 | SemanticField | join_policy strict enum | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 5 | session_calendar | early-close next_bar | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 6 | market | canonicalizer fail-closed | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 7 | latency | exact-int only | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 8 | predicate | ordered sequence | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 9 | partition_planner | typed __post_init__ | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 10 | calendar cache | file-snapshot token | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 11 | QueryBudget | self-validating | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 12 | StorageSpec | deep-immutable options | 本地 | PASS | —（0.9.6 已修） | — | round8 | FIXED |
| 13 | atomic writer | 多进程 O_EXCL 唯一 tmp | 本地/4进程 | PASS | 旧共用 `.tmp` 互相 O_TRUNC | 唯一 tmp+full-write+fsync chain | round9 | FIXED |
| 14 | publish | success+audit failure → CommittedButAuditFailed | 本地 | PASS | audit durable 失败被当 ABORTED | 已提交+审计失败分开传播，COMMITTED 重建 manifest | round9 | FIXED |
| 15 | read_auto | polars 受控 collect | 真实A股 | PASS | 裸 scan_polars 无 revalidate/budget | scan→ScanHandle.collect_table | round9 | FIXED |
| 16 | ScanHandle | snapshot revalidation | 本地 | PASS | collect 前文件变化 lineage 失真 | collect 前 stat/fresh HEAD，strict fail-closed | round9 | FIXED |
| 17 | mirror | verified/corrupt/legacy 三态 | 本地 | PASS | 非空文件当完整 | _mirror_file_state+_local_file_usable strict 收紧 | round9 | FIXED |
| 18 | remote | 共享 expected-partitions | 本地 | PASS | remote 自己枚举自然日 | expected_partitions 共享编译器 | round9b | FIXED |
| 19 | remote | validate_params 唯一事实源 | 本地 | PASS | remote 绕过校验 .format() | resolve_remote_paths 复用 validate_params | round9 | FIXED |
| 20 | coverage | params 贯穿 manifest fast path | 本地 | PASS | manifest 硬编码 params={} | validated params 贯穿 | round9b | FIXED |
| 21 | coverage | strict 5t fail-closed | 本地 | PASS | 自然日×5/7 冒充交易日 | _trading_day_lag(strict) (None,False) | round7 | FIXED |
| 22 | read_joined | 空 universe 0 行 typed | 真实A股 | FIXED | **并发会话 `continue` 越级 → 全包无法 import**；空 plain glob 让 DuckDB "No files" 进 IO 重试 | `else:` 结构 + `_prune_read_paths` 空 glob→[] | round9/round10 | FIXED |
| 23 | PIT metadata | strict typed fuzz | 本地 | PASS | bool 串味/负数 counts | __post_init__ + load 校验 | round7 | FIXED |
| 24 | PIT builder | 并发 build current 有效 | 本地/2进程 | FIXED | unprotected 竞态（A prune 删 B gen） | build 走 mutation_lock + {new,previous} 保留 | round9（重写走真实入口） | FIXED |
| 25 | publish crash | 双 rename 崩溃 target 不缺失 | 本地/SIGKILL | FIXED | old→archive 与 candidate→target 间死亡 → target 缺失 | durable journal + 确定性 recovery（晋升 candidate/回退 archive） | round10 | FIXED |
| 26 | audit | 多进程行原子性 | 本地/12进程 | FIXED | 线程锁挡不住多进程 append | flock(fd) 兜底（不依赖单次 syscall） | round10 | FIXED |
| 27 | backend parity | date-only end / instruments=[] | 本地/3后端 | FIXED | **Polars `<= 00:00` 丢最后一天白天行** | 共享 expand_end_bound `< next_day` | round10 | FIXED |
| 28 | factor gates | 三 gate 正交 | 本地 | FIXED | version branch `continue` 豁免 snapshot/universe | 版本检查后不 continue，正交组合 | round10 | FIXED |
| 29 | factor gates | require_same_* fail-closed | 本地 | FIXED | 无 universe 列 → `continue` 静默放行 | schema 缺列直接拒绝 | round10 | FIXED |
| 30 | factor catalog | partial refresh merge | 本地 | FIXED | discover 子集后 save 删 B/C | merge with existing | round10 | FIXED |
| 31 | factor_matrix | 列探测失败 fail-closed | 本地 | FIXED | matrix_cols=None → columns=None 读全矩阵 | MatrixCoverageMiss → fallback factor-major | round10 | FIXED |
| 32 | read_auto | mode enum | 本地 | FIXED | 未知 mode 静默落到 arrow | 严格枚举校验 | round9 | FIXED |
| 33 | mirror | degraded 随结果返回 | 本地 | FIXED | module-global 线程互相覆盖；自然日回退非空列表致 degraded 检测失效 | (dates, used_calendar) 元组 + with_degraded 变体 | round10 | FIXED |
| 34 | RelationHandle | SQL 只能 FROM _sub | 本地 | FIXED | 仅查「SQL 含 FROM _sub」挡不住 JOIN 其它表 | assert_sql_from_scope parser 级 | round10 | FIXED |
| 35 | coverage | remote-only 不永久 unavailable | 本地/s3:// | FIXED | 无 manifest 直接 unavailable | partial + authority=declared_remote | round10 | FIXED |

## 收官轮第 2 批 closure ledger（0.9.8，FE 集成复查，2026-08-09）

沿「FactorEngine → DataAccess → factor lake」链路复查的 13 项。8 项 Freeze
blocker + 5 项顺手收口全落地。改动：`dataaccess/`（真 PyArrow backend / 统一
predicate / normalize_units 一致性 / 空 universe / read_factors bound）+
`factor_engine/` adapter（空股票池 / read_mode / semantic_filters / 二次 scale /
governed collect / PIT UNKNOWN fail-closed / 禁直写 factor lake）。

| ID | Area | Test | Backend | Result | Bug root cause | Fix | Regression test | Status |
|----|------|------|---------|--------|----------------|-----|-----------------|--------|
| FE-1 | FE adapter | instrument_filter=[] → 0 行（非全市场） | eager/lazy/long | FIXED | `list(x) if x else None` 把 `[]` 折叠成 None | None 保持 None、`[]` 保持 `[]` | FE integration | FIXED |
| FE-2 | PyArrow | 真 `engine="pyarrow"` 三引擎 parity | duckdb/polars/pyarrow | FIXED | `_read_pyarrow` 手写 bound/filter；`pyarrow_engine_read` 硬编码 IPC 连 parquet 都读不了；pyarrow 不做 string→time cast | 共用 `compile_predicate_arrow`（expand_end_bound + 空集恒假 + datetime bound）；`pa_ds.dataset` 自动检测格式 | round11 | FIXED |
| FE-3 | FE adapter | read_mode 贯穿到 DA mode= | eager/lazy/scan | FIXED | `store.read` 未传 mode → FE 声明 pit、DA 落 mode=auto 分层漂移 | 全部入口传 `mode=self.read_mode` | FE integration | FIXED |
| FE-4 | FE adapter | semantic_filters 真过滤 | 物理 WHERE | FIXED | filters 塞进 `params`（StaticDataset 拒绝多余 params → 只过门禁不过滤） | `filters=self.semantic_filters` 落到 SQL；conflict 检测无条件 | FE integration | FIXED |
| FE-5 | scale | A股 Return eager 无二次 scale / 且真归一化 | eager/lazy | FIXED | ①catalog eager 未 `normalized.add` → COS fallback 二次乘 return_scale；②更隐蔽：`normalize_units=True` 在 auto→stream/lazy result 静默失效 → eager 返回 raw BP | ①catalog 覆盖字段禁入 COS fallback；②ReadHandle lazy/stream 物化终点补 normalize 钩子 | FE integration + round11 | FIXED |
| FE-6 | governed terminal | polars-long collect 前 revalidation | 真实A股 | FIXED | 裸 `scan_polars()` LazyFrame 绕过 ScanHandle revalidate/budget/audit | `execute_polars_long_plan` 终端 collect 前 `revalidate_for_long_collect`（production fail-closed）+ budget | FE integration | FIXED |
| FE-7 | PIT | cleaned fundamentals PIT 禁用 | production | FIXED | `fundamentals_*`/`financials_ratios` period_end 无知识时钟，asof 前视 | `_NO_KNOWLEDGE_TIME_FUNDAMENTALS` deny-list + PIT gate 拒绝 | FE integration | FIXED |
| FE-8 | PIT | Four-layer gate UNKNOWN fail-closed | production | FIXED | contract/TableSpec None→True（「不知道」=「安全」） | `_dataset_pit_allowed`/`_table_pit_allowed` 三态；UNKNOWN production 拒绝、research 告警降级 | FE integration | FIXED |
| FE-9 | DataRequest | 静态 universe 空集 → 0 行 | duckdb | FIXED | `sorted(members) if members else None` → 空成分=全市场 | 返回 `[]`（空股票池） | round11 | FIXED |
| FE-10 | read_factors | date-only end 含最后一天 | duckdb | FIXED | `t_col <= ?` 丢 timestamp 最后一天白天 | 复用 `expand_end_bound` + ISO 字符串绑定 | round11 | FIXED |
| FE-11 | FactorCatalog | FE/DA catalog 统一 | — | 收口 | 双 catalog 语义未全对齐 | 列为 integration closure（本轮无 correctness 复现），factor lake 元数据 universe/frequency/snapshot roundtrip 已测 | round11 | INTEGRATION_CLOSURE |
| FE-12 | metadata | universe/frequency/snapshot 贯穿 | factor lake | PASS | — | factor meta `_factor_meta.json` → catalog → 读结果 snapshot 列贯穿 | round11 | FIXED |
| FE-13 | write | production 禁 direct-local factor lake 写 | production | FIXED | `LocalParquetWriteTarget` 直写绕过 staging→publish 原子发布 | production 拒绝 direct-local，强制 staging + `publish_factor_lake` | materialize tests | FIXED |

**本批验收**：三引擎真实 parity（`engine=duckdb/polars/pyarrow`，date-only end /
datetime end / 空股票池 / 单标的 4 组全等）；A股真实 eager==lazy==raw×1e-4；
`engine='auto'+normalize_units=True` 归一化与显式 duckdb 一致；read_mode/semantic_filters
经 spy 验证到达 `mode=`/`filters=`；PIT UNKNOWN production 构造即拒；polars-long
collect 前源替换 production reject；DataRequest 静态空 universe 0 行；read_factors
date-only end 含最后一天。

**真实数据验收**（A股 `ashare_stock_daily` 1815 文件 / 美股）：
- 三后端（DuckDB/Arrow/Polars）差分一致（88 行全等）。
- PIT no-lookahead property：04-20 见修订前 val=100、06-01 见修订后 95、Q2 不泄漏。
- determinism：值级一致；未排序 raw read 行序按设计不定。
- QueryBudget：deadline 活取消（0.26s 中断）；max_rows/bytes 物化后 guardrail；max_scan_files 预扫描。
- 资源：300 循环 FD 4→4 零泄漏；atomic writer + 4 reader 并发零错误。
- pruning：hive/目录-manifest 路径生效；plain `**/*.parquet` 无目录 manifest 依赖 DuckDB 谓词下推（POST-FREEZE BACKLOG 性能项，文件名修剪有正确性风险不做）。
- 真实数据 anomaly：A股 1815 文件全健康、schema 一致、无 null ticker、每日文件 1 个交易日。

**BLOCKED（非 DataAccess 代码）**：
- `test_check_allowlist.py::test_real_repo_passes`：factor_engine/scripts 并发会话
  新改两个 certify 脚本 `pd.read_parquet` 绕过 DataAccess。owner=factor_engine 并发
  会话（本会话按约束不动 factor_engine）。建议：这两个脚本是「按文件逐个读」的
  certification 语义，可走 `.data_access_allowlist.yaml` 豁免（有先例）或迁到
  DataAccess 正式 API——由 owner 决定。

## 收官轮第 3 批 closure ledger（0.9.8，3+1 排除式复查，2026-08-09）

按最新已提交 `a2dc80c` 做的排除式检查的「真正收官项」。外部 AI 结论：DataAccess
不再需要扩架构，补完这批即可 Core Freeze。改动只在 `dataaccess/` 树。

| ID | Area | Test | Backend | Result | Bug root cause | Fix | Regression test | Status |
|----|------|------|---------|--------|----------------|-----|-----------------|--------|
| R12-1 | plan | ReadPlan/CompiledDataRequest **真正 immutable** | duckdb | FIXED | `_deep_freeze_mapping` 返回普通 dict、`ReadPlan` 非 frozen + 可变 list/dict → `plan.datasets.clear()` / `plan.compiled.source_params["x"]["y"]=z` 能改编译结果，explain≠execute | `CompiledDataRequest` 深冻结成 MappingProxyType/tuple/frozenset；`ReadPlan` `@dataclass(frozen=True)` + `__post_init__` 冻结 datasets/per_dataset_columns/join_policies/snapshot_info/join_specs_effective/plan_snapshot_tokens/plan_pinned_files | round12 | FIXED |
| R12-2 | enum | `read()` engine/result strict enum | 所有入口 | FIXED | `_read_handle` dispatch 对未知 `engine="polarr"`/`result="lazzy"` 静默落到 duckdb 物化路径 | `validate_engine_result` 顶部门禁（auto/duckdb/polars/pyarrow × auto/arrow/pandas/polars/lazy/stream） | round12 | FIXED |
| R12-2b | typed | DataRequest typed fields | duckdb | FIXED | `compile_data_request` `pit=bool(request.pit)` → `pit="false"` 变 True；limit 未校验 | `__post_init__` 严格 bool（pit/normalize_units/time_varying_universe）、limit 非负 int\|None、engine/result/snapshot_policy enum + 小写归一 | round12 | FIXED |
| R12-3 | registry | factor_lake_wide contract 错位 | registry | FIXED | 宽表是 datetime × {asset} pivot，**asset 是列轴不是物理列**，declaring `instrument_column: asset` 会让 generic 读在假列上过滤 | `specialized_only` + `specialized_only_reason`（registry 新增字段，裸标记无 reason 拒绝）；`_prepare_read_request` 门禁覆盖所有 read/scan/sql 路径 | round12 | FIXED |
| R12-4 | lineage | ReadLineage None/[] + params canonicalize | duckdb | FIXED | `tuple(x) if x else ()` 把 None 与 [] 都折叠成 ()（provenance 丢失 None=全市场 vs []=空池）；stream 路径把 mutable dict 塞给 tuple 型 params | `instrument_filter: tuple\|None`（None=全市场，( )=空池）；`lineage_params()` canonicalize 成不可变 tuple | round12 | FIXED |

**本批验收**：plan() 后改 compiled.source_params / filters / plan.snapshot_policy /
datasets / per_dataset_columns / plan_snapshot_tokens 全部 TypeError，execute 仍按
plan 时刻语义执行；`engine="polarr"` / `result="lazzy"` / `DataRequest(pit="false")` /
`limit=-1` / `snapshot_policy="bogus"` 全部 ValidationError；factor_lake_wide generic
read/read_arrow/scan_polars/read_result 全拒绝（instrument_column 已从 asset 移除）；
ReadLineage None vs () 保留、lineage/stream params 不可变。

## 执行状态（2026-08-09 完成）

### 第七轮二阶回归收口（0.9.6，main=`608f602` 排除前 54 项后再审的 12 项）

用户排除前 54 项后再审，确认 12 个新增问题/二阶回归；前 7 个 P0/P0-P1 + 后 5 个
P1 全部一次性落地（Core Freeze 前最后一批）：
- ✅ **1. mutation lock TOCTOU（P0）**：续租改经 fd（inode 身份，`stat(path).st_ino
  == fstat(fd).st_ino`），旧 writer 绝不可能 truncate 新 inode；release 不再
  unlink，改经 fd 写 `released` 标记，`_can_break_lock` 视为立即可打破。锁 fd 改
  `O_RDWR`。
- ✅ **2. StorageSpec URI scheme→backend（P0）**：`type` 未显式时从 scheme 推导；
  `oss://` 与 scheme/type 矛盾拒绝，绝不 fallback LOCAL。
- ✅ **3. PathAuthorizer 延迟 namespace（P0）**：保存 unresolved root template，
  authorize 时按当前 session namespace 解析（`allowed_root_templates()`）。
- ✅ **4. join_policy strict enum（P0）**：拼写错 fail-closed，不再静默丢 PIT 语义。
- ✅ **5. next_bar early-close（P0）**：消费 `effective_segments_on(d)`，half-day
  收盘后正确跳下一交易日。
- ✅ **6. Market canonicalizer（P0）**：未知 market fail-closed，不再静默当 US。
- ✅ **7. latency exact int（P0/P1）**：`1.9`/`"1.9"` 拒绝，不再截断成 1。
- ✅ **8. strict_sequence 拆规则（P1）**：fields/order_by 只收 list/tuple；instruments
  接受 set 但 canonical 排序。
- ✅ **9. TimePartitionSpec `__post_init__`（P1）**：typo 构造即报错，不 fallback daily。
- ✅ **10. 无 manifest 日历缓存文件 snapshot（P1）**：path+size+mtime_ns 做数据版本
  token，不再用 registry_fingerprint（配置版本）。
- ✅ **11. QueryBudget 自身 invariant（P1）**：负数/NaN/Inf/bool 构造即拒绝。
- ✅ **12. StorageSpec.options 深冻结（P1）**：递归不可变映射，内部 mutation 从根上禁止。

回归：`tests/unit/test_final_closure_round8.py` 14 条；全量 **724 passed / 0 failed**；
ContractIR audit 71 数据集一致（fingerprint=b7d21b082127eb00）。
**Core Freeze 生效**：后续差分测试 → 并发/crash 测试 → 真实 A股/美股 golden →
benchmark → FactorEngine integration，不再由 AI 大范围审计新增功能。

### 第三批审计收口（0.9.5，main=`3550257f0` 补扫的 9 项）

用户按最新 main（仅改 FactorEngine，DataAccess 无变化，前两轮结论成立）又补扫出
9 项残留，全部落地（1–6 并入 Freeze blocker，7–9 P1 顺手清干净）：
- ✅ **1. Manifest generation 每 commit 全新**：`DatasetManifest.save()` 不再复用
  `self.manifest_generation_id`（从旧 manifest load 后重建会复用旧 gen）——每次
  logical manifest commit 都 `uuid4_hex()` mint 全新 generation，内存对象同步更新；
  row-group sidecar 绑定这次新 gen。彻底关掉「进程死在 parquet/JSON 两次 replace
  之间、新旧两代恰好同 gen → 双 generation 检查误通过」的 crash 窗口。
- ✅ **2. datasets.yaml 顶层 strict schema**：StaticDataset / ParametricDataset 各自
  完整 allowed-key 集合，**任何未知字段启动即失败**（`time_colum:` / `query_polcy:` /
  `formatt:` 等 typo 不再静默忽略走默认值）；static 用 `root_template` 等错位 key
  同样拒绝。
- ✅ **3. partitioning 唯一 schema**：`parse_partitioning` 成为唯一入口（顶层
  `time` + `hive`/`partition_columns`），**unknown key fail-closed**；loader 直接
  编译成 typed `PartitionSpec` 保存（不再存 raw dict）；旧 loader 允许但 planner
  不认的 `columns/partition_by/bucket/granularity/time_column` 全部拒绝——时间分区
  裁剪不再悄悄失效全量扫文件。
- ✅ **4. storage 唯一 schema**：新增 `StorageSpec.from_yaml()`（顶层 + 嵌套
  `source` 合并、顶层优先、unknown key 拒绝、`{source:{type:cos}}` 不再被默认成
  local）；registry 直接保存 typed `StorageSpec`；`core.storage` 提供
  `declared_storage_{type,uri,layout}` 读取 helper；`contract_ir` / `cos/remote`
  消费统一 helper，不再各自解析 raw dict。
- ✅ **5. TemporalJoinSpec 自身 invariant**：新增 `__post_init__`——直接
  `TemporalJoinSpec(policy="xxx", availability="whatever")` 也 fail-closed；
  `availability_latency` 拒绝负数和 bool；`_as_bool` 不再接受任意 int/float
  （`2`/`0.5` → 报错）；dict 入口 unknown-key reject。
- ✅ **6. early-close 加载 fail-closed**：`load_us_early_close_dates` 在
  production/strict 下「数据集未注册 / 缺时间列 / 读取失败 / 空结果」一律抛
  ValidationError——**不再把「没读到」当「没有 early close」**；research 保持宽容。
- ✅ **7. calendar 必须 trading-day flag**：`_load_calendar_from_registry` strict 下
  要求 `IsTradeDay/is_trading_day`（schema 或物理列 probe），缺标志 fail-closed——
  自然日/节假日不再被当权威交易日。
- ✅ **8. QueryPolicy strict typed config**：unknown key reject（`max_scan_file:`
  typo 不再让安全预算消失）；`_positive_finite_float` 拒绝 bool（`True` 不再变成
  `1.0ms` 预算）。
- ✅ **9. factor PIVOT + catalog fail-closed**：宽表 PIVOT 前新增唯一性门
  （`build_factor_duplicate_check_sql`，strict 下命中 `(datetime, asset, factor_id)`
  重复即抛，不再静默 `USING first(value)`）；`FactorCatalog.load/discover` strict 下
  损坏 JSON / 非法 `_opt_int` ⇒ `DataError`/`ValidationError`（production 权威目录
  不许静默缺因子）；`save()` 按 `factor_id` 排序（确定性落盘）。

**回归**：全量 `tests/` **688 passed / 0 failed**（前轮 669 + 本轮新增
`test_final_closure_round6.py` 19 条）；真实 `datasets.yaml` 71 个数据集照常加载；
FE 侧 `test_catalog_us.py` + `test_data_access_catalog_errors.py` 15 passed、
`tests/market + tests/storage` 230 passed / 1 failed（唯一失败是 FE 并发会话 WIP：
`operator_market_capabilities.json` 未随新 operator 重新生成，与本批无关）。未碰
GitHub。

### 本会话 54 项 Closure Ledger 收口（0.9.4 补充，含 0.9.5 未覆盖项）

在并发会话 0.9.5 批次之上，本会话完成用户最终 54 项 ledger 的剩余项（全量回归
**710 passed / 0 failed**，ContractIR 71 数据集一致）：
- ✅ **写侧 fencing**：mutation_lock heartbeat 续租 + owner-only release + PID
  reuse 检测（item 2）。
- ✅ **namespace 请求级**：load 不再烘焙 `${RUN_NAMESPACE}`，read/write 时按
  当前 context 解析；static_prefix 屏蔽占位符修 `$` 截断；session namespace 原样
  返回不汇聚（items 3 / 46）。
- ✅ **snapshot pin 执行消费 pinned file set**（physical_scope 注入，item 6）。
- ✅ **availability 冲突 fail-closed**（item 13）+ **日历右边界 strict 报错**
  （item 14）。
- ✅ **PIT generation-directory + 原子指针**（item 18，crash 保留旧 gen）。
- ✅ **SQL parser 级 allowlist**（item 31）：单语句证明 + `duckdb_functions()`
  注册表按 function_type 分类，只允许 scalar/aggregate/window，table/file/network
  自动拒绝，未知函数 strict fail-closed。
- ✅ **strict 判定统一**（item 47）：paths/key_policy/telemetry/s3_duckdb/remote。
- ✅ **P1**：Filter AST 完整 canonical 排序（44）、date-only end `< next_day`（45）。
- ✅ **fsync durable-write**（52）：`core/atomic.py` 统一 tmp→fsync(fd)→replace→fsync(dir)。
- ✅ **coverage**（53）：empty_ok=complete、5t 真实交易日历、remote-only 标注。
- ✅ **COS mirror 三态**（54）：verified/legacy-unverified/corrupt，strict 不把
  非空当完整。
- ✅ **设计决策**：upsert 契约写死 partition-level atomicity；audit durable
  acknowledgement（publish 写失败抛 `AuditWriteError`）。
- ✅ 回归：`tests/unit/test_final_closure_round7.py` 22 条。

**Core Freeze 生效**：本批即最终 Closure Ledger 收官。不再由 AI 大范围审计新增
功能，按路线进入 differential → concurrency/crash → 真实 A股/美股 golden →
benchmark → FactorEngine integration；新发现按正常 bug maintenance 处理。
未提交 GitHub（用户要求只留服务器本地）。

### 最终收官轮（0.9.4，Core Freeze 前最后一轮 closure，HEAD=`a6e21d6`）

用户给出的「最终收官轮」清单 1–11 全部落地（1–9 是 Freeze blocker，10–11 顺手
收口），并新增组合场景 DoD 测试（`tests/unit/test_final_closure_dod2.py`，15 条）：
- ✅ **1. 写路径统一事务**：`_dataset_mutation` 在 dataset-root 级 `mutation_lock`
  内完成「bump epoch → mutate → rebuild → unlock」；`mutation_lock` 同线程同 root
  可重入（正文的 target 锁共用）；rebuild 失败不再被吞（write caller 能看到）。
  顺带修掉 `_crash_safe_overwrite` 每次 overwrite 清掉 manifest sidecar 的真 bug
  （sidecar 是数据集元数据，overwrite 时从旧目录带过来）。
- ✅ **2. CompiledDataRequest 深冻结**：filters/joins/join_specs/source_params/
  field_params/transforms/aggregation 项全部 `deepcopy`；`ReadPlan.anchor` /
  `explain()` 只读 `compiled`；`store.plan()` 用 compiled 编译物理计划 DAG。
- ✅ **3. snapshot_policy 物理 pin**：`pin` 绑定 `manifest_generation` + 逐文件
  (path, size, mtime_ns / etag / version_id)，execute 时实际 stat（不复用 manifest）；
  `fail_if_changed` 在「plan 时有 manifest、execute 时消失」→ fail。顺带修
  `manifest_root_for_paths` 把 `part-*.parquet` 的根错算成 `part-` 目录的 bug。
- ✅ **4. 组合执行 budget parity**：composed 用 `plan.join_specs_effective`（不再
  重推）；merged budget 覆盖全部参与数据集；物化分支 `enforce_arrow_budget`，
  stream 分支累计 `max_rows/max_result_bytes`；逐 dataset `max_scan_files`。
- ✅ **5. 统一 AvailabilityCompiler**：`compile_available_from`（same_instant /
  next_bar / next_session_open / next_trading_day / after_close_next_open /
  session + latency）；`next_bar` 算**当前 session 的下一根 bar**（盘中+1、午休
  跨段、收盘/周末/节假日下一交易日）；`MarketCalendar.available_from` 委托；
  事件 helper 消费同一 IR + latency；字段 availability 冲突不再静默回退 same_day
  （strict 抛 AmbiguousSemanticFieldError）。
- ✅ **6. PITEventIndex fail-closed**：glob 枚举失败进 `failed_files` +
  `glob_failed`（complete=False）；null ticker / 空 filing → 拒绝构建；sidecar
  tmp+fsync+atomic replace。
- ✅ **7. derived 字段 fail-closed**：catalog load 拒绝 `mining_allowed=true` 的
  derived 字段；Planner 遇 derived 字段给清晰 ValidationError；config 里
  `us_market_cap_daily` 改为 `mining_allowed: false`。
- ✅ **8. 语义歧义 + 重复列**：`resolve_by_physical` 市场过滤后仍多身份 →
  production 抛 `AmbiguousSemanticFieldError`；`normalize_table_units` 按列位置
  重建（`Table.from_arrays`），合法重复列不丢。
- ✅ **9. 聚合治理**：aggregation 路径强制 `max_scan_files`；ReadLineage 写入
  field / time_range / instruments（不再 `columns=(), time_range=None`）。
- ✅ **10. FormatSpec strict typing**：顶层 unknown-key reject（`delimeter:` 不再
  静默忽略）；extra 每 option 值类型校验；顶层 compression 与 extra.compression
  重复拒绝；`columns` 结构值渲染成 DuckDB STRUCT literal（不再 str() 化）。
- ✅ **11. TIMESTAMP/TIMESTAMPTZ 正式分离**：`timestamp`/`datetime` 只匹配 naive
  （TIMESTAMP_NS/MS，**不**匹配 WITH TIME ZONE / TIMESTAMPTZ）；审计真实数据后
  `ashare_stock_{daily,balance,industry,valuation_daily}.UpdateTime` 改为
  `timestamptz`（实际是 `timestamp[ms, tz=UTC]`）。

**回归**：全量 `tests/` 669 passed / 0 failed；allowlist 7 passed；新增
`test_final_closure_dod2.py` 15 条组合场景 DoD。FE 侧 `test_catalog_us.py` /
`test_data_access_catalog_errors.py` 全绿；FE 其余 5 个失败均为并发会话既有 WIP
（operator arity mismatch / ListedState schema / FE 自身 catalog 归一化 test-code
mismatch），与本批无关，未碰 GitHub。

### 第五轮二阶边界收口（0.9.3，Core Freeze blockers）

按第二轮深扫的 12 个「二阶边界 bug」收口（P0 1-7 + P1 8-12）。这轮没有新增
架构/subsystem，全部是已有架构没有完全贯穿到边界分支的问题：
- ✅ **cache**：`query_cache_key` 严格区分 `[]`（空池）与 `None`（全市场）；
  `read_cached` gates 前置（`gates → cache key → hit/miss`），缓存命中不绕过
  semantic/budget gate。
- ✅ **COS helper `[] = empty`**：panel/events 的 `instrument_filter=[]` 生成
  `WHERE FALSE`，不再静默全市场。
- ✅ **ReadHandle**：one-shot stream 一旦消费，第二终点 fail-closed
  （`buffer=True` 显式固化例外）；lazy 第一次 terminal collect 缓存 canonical
  Arrow，绝不重复执行底层 LazyFrame。
- ✅ **remote meta**：`_remote_meta_cache` 30s TTL（失败 None 不永久缓存）；
  `_remote_object_meta` 先 `cos_uri_to_s3_uri` 再切 bucket/key（cos:// 长度错位
  消除）。
- ✅ **PIT index**：新增统一 `validate_current_source`（authoritative + epoch +
  source_snapshot + schema_hash），MetadataPlane / prune / 复用三处共用。
- ✅ **sql() lexical**：`RelationHandle` 的 `?` 计数改 quote/comment-aware 扫描器。
- ✅ **stats sidecar**：legacy 无 source identity 在 production/strict 视为 stale。
- ✅ **asof 列冲突**：`allocate_unique_column_name` 统一（`x_event_2` 递增），
  空 decisions 分支同策略。
- ✅ **strict sequence parser**：DataRequest / SemanticField 拒绝裸 str/bytes 与
  非法元素。
- ✅ **顺带**：publish 首次发布误归档修复（`_target_has_published_content`）；
  test_phase6_hardening 泄漏恢复。
- ✅ 回归：`tests/unit/test_final_closure_round5.py` 21 条；全量 `tests/` 通过
  （除并发会话 WIP 的 3 个失败，非本批）。

**Core Freeze 生效**：本批完成后不再由 AI「大范围审计新增功能」，正式进入
维护态——differential tests → concurrency/crash tests → 真实 A股/美股 golden
tests → benchmark → FactorEngine integration。

### 第四轮最终 P0 收口（0.9.2，HEAD=`6c1453f` 之上）

14 个最终 P0 全部落地（10 项为已修复项补回归锁住，4 项为真实代码改动）：
- ✅ **Publish 契约门**：`_validate_candidate_contract`（candidate 实际数据 vs
  target 声明 schema：缺列 / 类型不符 / mixed schema → fail-closed）；manifest
  `base_dir="."`（相对路径恒有效）；symlink 复核（copytree 前拒绝）。
- ✅ **asof 统一 availability**：`read_cos_events_asof` 与 read_joined 共用
  `availability_uses_calendar/availability_strict_next`——有日历用
  `available_from <= decision`，无日历按 strict-next `<` 回退；financial
  next_trading_day 真正生效。
- ✅ **ContractIR 时间轴硬化**：event_time 只来自契约时钟列 / registry
  time_column，不再从值字段 time_role 推导（`event_time="Close"` bug 消除）；
  修 `roe` 缺失 availability → audit 重新 71 一致。
- ✅ **Manifest rowgroup 持久化**：`_save_row_groups` 改列数组字典构造（旧
  `pa.table(list_of_dicts)` 在 pyarrow 25 直接崩，sidecar 从未成功写过）；
  save/load 双侧 generation 校验。
- ✅ 回归：`tests/unit/test_final_closure_round4.py` 22 条 + 全量 `tests/` 626
  passed + ContractIR audit 71 一致。
- 并发协调：并发会话把 `us_stock_dividend` 升级为 strict-PIT；3 处旧语义测试
  改指 `us_stock_capital_split`，验证目标不变。

### 第三轮增量收口（0.9.1，HEAD=`6d66e9c` 之上）

第三层审计清单（P0 1-23 / P1 24-40）全部落地：
- ✅ **计划=执行**：`CompiledDataRequest` 冻结请求语义（plan 后改 req 不影响
  execute；plan 不再写回 request.anchor）；`snapshot_policy` latest/
  fail_if_changed/pin；组合执行器 honor `result="stream"` + 静态 universe；
  execute 把 plan 编译的 `join_specs_effective` 原样交 read_joined。
- ✅ **Snapshot 真实**：`_expand_glob_paths` 冻结精确文件清单（TOCTOU）；
  ScanHandle.collect 前 revalidate + strict fail-closed；`FileVersion` 加
  etag/version_id/content_length/last_modified（s3 对象头 best-effort）。
- ✅ **日历**：available_from 先转交易所本地时区；非交易日不生成当天开盘；
  缓存 key 含 source token（更新自动失效 / store-local）；next_day O(N) zip。
- ✅ **契约门**：required_filters/allowed_filter_values 升级为 PredicateConstraint
  （Ne/IsNotNull/OR-部分分支不再算已约束）；strict_read 完全共享 production
  fail-closed。
- ✅ **read_uri**：精确 URI physical scope + 对象 URI 不再用 pathlib + 格式匹配修正。
- ✅ **Serving**：`materialize_daily_aggregate` 一次扫描 + 行级谓词；
  Polars 按 FormatSpec 选 scan；CBO arrow/feather → pyarrow。
- ✅ **写事务**：`_dataset_mutation` 显式 PREPARED/COMMITTED/ABORTED（mutate
  前置 mark dirty；ABORTED 绝不 build fresh）。
- ✅ **Spec/IR/Storage 硬化**：TemporalJoinSpec bool/类型 fail-closed；ContractIR
  external + required_filters 完整合并 + 冲突检测；StorageSpec 非法 type 构造即抛；
  COS 解析失败 fail-closed；能力矩阵。
- ✅ **Query cache**：无权威 manifest 不缓存；plan 按 dataset 传 source_params。
- ✅ 回归：`tests/unit/test_final_closure_round3.py` 18 条 + 全量 420 passed +
  ContractIR audit 71 数据集一致。

### 第二轮收官（0.9.0，HEAD=`5073edf`）

**P0 全部落地**：
- ✅ P0-1/P0-2 统一 JoinCompiler（删除 `_join_aggregated_anchor`，聚合锚点走同一
  `_read_joined_sql`；`_effective_join_specs` 全链路共享）
- ✅ P0-3 financial seed PIT 状态（Q3 不被晚到 Q2 修订回滚）
- ✅ P0-4/5/6/7/8 Calendar：IsTradeDay 过滤 / fallback 不污染权威缓存 /
  session 前瞻 / 细粒度 availability / US early close
- ✅ P0-9/10 多时钟 TemporalAxes + pruning clock 安全 / dataset 级契约门
- ✅ P0-11..16 `is_strict_semantics()` / read_uri / event cutoff / fanout 单值 /
  PIT index（文件级剪枝、文件计数、截断 incomplete、真 schema hash）
- ✅ P0-17/23/24/25 factor matrix（gate 前置、精确投影、DISTINCT 版本检查）
- ✅ P0-18..22/35/36 执行治理（stream 直路由 + deadline 前置、governed lazy、
  Scanner pushdown、sql_relation FROM 绑定、cache 列序/默认列）
- ✅ P0-26..31 写事务（DatasetTransaction、upsert prepare-then-promote、
  publish unique_key gate + content_hash、mutation lock host 感知）
- ✅ P0-32/33/34 namespace contextvars / ContractIR issues 字段 / Metadata Plane
  epoch 感知

**P1 部分落地**：P1-14 CBO 不重复计 selectivity；P1-17 row-group 去重；
P1-21 storage backend/layout/format 分离；P1-23 PIT schema hash；P1-28 order_by。
其余 P1（CoverageMatrix 内部孔洞、CBO 校准粒度、mirror registry 并入 datasets.yaml、
QueryProvenance、InstrumentKey、incremental manifest 等）列为后续批次。

**P2 / DoD / CI**：17 条 DoD 回归（`tests/unit/test_final_closure_dod.py`）+ 永久
CI（`.github/workflows/dataaccess-final-closure.yml`）。API 收敛到
`read(DataRequest)/scan/sql/write` 的统一 compiler 列为下一批次。

**回归**：`tests/` 380 passed。

> 基线：`main` @ `5073edf`（dataaccess 本轮整改在 HEAD=`6d66e9c` 之上）。
> 目标：消灭所有"第二套语义实现"。同一个 DataRequest 无论走 minute aggregation、
> read_joined、Polars、PyArrow、stream、factor matrix 还是 remote/mirror，都必须由
> 同一个 **ContractIR → PreparedReadRequest → MetadataSnapshot → PhysicalPlan →
> 统一 Semantic Compiler → executor** 决定语义。backend 只能改变执行方式，不能改变
> 数据含义。
>
> 并发安全：本计划**只改 `dataaccess/` 树**。`factor_engine/` 全部改动归并发会话。
> `read/scan_handle.py` 已被并发会话修改，需要编辑时先 `git status` + 重读。

---

## 收口链路（唯一事实源）

```
读:
DataRequest
  → ContractIR
  → PreparedReadRequest
  → MetadataSnapshot
  → PhysicalPlan
  → 统一 Semantic Compiler (compile_temporal_join)
  → DuckDB / Polars / PyArrow Executor
  → ReadHandle
  → QueryProvenance

写:
WriteRequest
  → DatasetTransaction (PREPARED/COMMITTED/ABORTED)
  → immutable candidate generation
  → schema/key/data validation
  → manifest + metadata commit
  → atomic generation switch
```

---

## P0 — 必须修完才能宣布正确性收官

### 1. 统一 JoinCompiler（P0-1 / P0-2）
- 删除 `PhysicalPlanExecutor._join_aggregated_anchor` 的第二套 join 语义。
- aggregation 结果 → `DerivedRelationSource` → **同一个** `compile_temporal_join`。
- `compile_temporal_join` 支持 `PhysicalSource / DerivedRelationSource / MaterializedSource`。
- 生成 `EffectiveJoinSpec`：`SemanticField 默认 → 同 dataset 字段语义合并 → COS Contract
  默认 → request.joins/join_specs 覆盖`。
- PIT validator / PhysicalPlan / read_joined / cache key / lineage / explain 全部消费
  EffectiveJoinSpec。
- 验收：`minute aggregation + financial PIT + industry` 结果 == 手工 aggregate 再走完整
  read_joined，逐行逐列一致。

### 2. Financial seed / latest_period（P0-3）
- seed 必须计算"截至 start 的 PIT financial state"（同 instrument+period 取最新 revision，
  再在所有可见 period 中取 max(period)），而不是"start 前最后一条 event"。
- 修复 Q3 已发布后被晚到 Q2 revision 回滚的问题。专项回归测试。

### 3. Calendar 正确性（P0-4 / P0-5 / P0-6 / P0-7 / P0-8）
- Calendar loader 过滤 `IsTradeDay`/`trade_date` flag（不再把自然日当交易日）。
- Calendar 用 `mode="dimension"` 读（STATIC 契约 production/strict 下不被拒）。
- production 真实 calendar 不可用 → **fail closed**，不静默退回 business day。
- 全局 `_calendars[market]` 缓存 key 至少含 `market + calendar_source + snapshot/version`，
  fallback calendar 不得污染 authoritative calendar。
- `_session_avail_sql` lookahead ≥1 个交易日（映射窗口延伸到 query end 之后的下一交易日）。
- availability 拆分：`same_instant / next_bar / next_session_open / next_trading_day /
  after_close_next_open / effective_date_only` + `availability_latency`。
- US early close（如 13:00）作为 date-specific session override 进入 engine。

### 4. 多时钟时间轴 + dataset 级契约门（P0-9 / P0-10）
- ContractIR 增加 `TemporalAxes`（partition_time / event_time / knowledge_time /
  effective_time / period_time / decision_time / storage_timezone / semantic_timezone）。
- PreparedRead 明确 `predicate_clock / pruning_clock / join_clock`；manifest 无对应 clock
  统计 → 禁止基于另一根时间轴 prune（宁可多扫，不能 false negative）。
- Dataset 级 required filters（timeframe / IndustrySource / IndexSymbol）**无论用户选哪个
  字段都强制执行** —— 两层门禁 `DatasetContractGate` + `FieldContractGate`。

### 5. Strict 语义统一 + read_uri + cutoff + fanout + PIT index（P0-11..16）
- 单一 `is_strict_semantics() = production OR strict_read`，全链路统一。
- `read_uri` production/strict 必须唯一反查已登记 dataset（不能用 prefix 根绕过）。
- effective-event 硬 cutoff：无 end 的 Dividend/split 查询不得读未来生效事件。
- fanout 单值证明：`IndustrySource IN ('sw_l1','sw_l2')` 仍判 fanout。
- PITEventIndex：文件级 min/max prune（不 O(records)）、真 schema hash、`limit` 构建
  `complete=False`、indexed_file_count 必须等于源文件数。

### 6. Factor matrix 正确性（P0-17 / P0-23 / P0-24 / P0-25）
- 一致性 gate 移到 matrix route **之前**（版本 / data_snapshot / universe）。
- matrix 精确投影请求的 fids，缺一个 → `MatrixCoverageMiss`。
- `FactorArtifactManifest`（factor_version / data_snapshot_id / universe_id / date 范围 /
  row_count / schema_hash / source_lineage_hash）取代 `limit=1` 行探测。

### 7. 执行治理（P0-18..22 / P0-35 / P0-36）
- `result="stream"` 在 execution 前直接路由到真正 reader，禁止"先 materialize 再 stream"。
- stream deadline 作用在第一批之前；generator `finally` 里 interrupt/close cursor、
  归还连接、结束 watchdog。
- `GovernedLazyHandle`：production 不暴露 raw LazyFrame，只有 `collect()/stream()` 受控
  终点；research 用 `unsafe_scan_polars()`。
- PyArrow executor 改 `Scanner`：projection / filter / partition pruning pushdown +
  真实 time_range。
- sql_relation production 走 `SafeSQLRequest → read_datasets → scoped temp views →
  rewrite query`（token sandbox 只是辅助，不当作数据边界）。
- QueryCache key：columns 不排序（[A,B] != [B,A]）、default columns 显式展开、含 ordering。

### 8. 写事务原子性（P0-26..31）
- overwrite → `.generation_uuid/` immutable candidate → validate → manifest → atomic
  generation switch（禁止 clear-dir-then-write）。
- `DatasetTransaction` PREPARED/COMMITTED/ABORTED：metadata 只在 COMMIT 前进，
  ABORTED 不进位（修掉 finally 无条件 bump）。
- 多分区 upsert → transaction manifest 全成功才 commit。
- publish 只发布 immutable staging generation（capture source snapshot + lock + copy +
  verify unchanged）。
- publish 强校验：schema hash / file inventory / partition inventory / row count /
  primary-key count / manifest hash / source generation；关键 dataset 加 column checksum。
- mutation lock：host-aware stale reclaim（跨机器不得用本地 os.kill 回收活锁）。

### 9. Namespace / ContractIR / Metadata Plane（P0-32..34）
- Namespace → `contextvars.ContextVar`，enter 存 token / exit restore；registry 保留
  namespace token，path resolve 时才绑定。
- **ContractIR `issues: list[str]` 字段缺失是直接 bug**（`audit()` 引用 `d.issues` 会
  AttributeError）→ 补字段；`store.contract_ir().audit()` 必须在完整 registry 上通过。
- Metadata Plane 改 immutable `MetadataSnapshot(dataset, params, source_epoch)`，accessor
  检测 epoch 变化即清 derived metadata。

---

## P1 — 元数据 / CBO / COS 收口

- CoverageMatrix（expected trading days、internal hole → partial）。
- 物理源感知 CBO：`logical_source=COS` vs `physical_source=local_mirror/s3_httpfs/cli_cache`；
  不重复计 selectivity；`calibrated_score` 换成真实 elapsed/memory/network/spill 预测；
  校准粒度 dataset+backend+engine+query shape+projection bucket+selectivity bucket+remote/local。
- mirror registry 并入 datasets.yaml storage 声明（删除硬编码 DATASET_MIRROR_REGISTRY）。
- mirror freshness 在 deep verify / publish / suspected corruption 时重算 checksum；
  full cos sync 后为每个文件生成 inventory sidecar。
- PITEventIndex 文件级 serving 结构（file_path/min_filing/max_filing/timeframe set/bloom）。
- FactorArtifactCatalog（factor_id/version/coverage/snapshot/universe/frequency/schema/
  storage generation/lineage）。
- QueryProvenance：request_hash / filter_ast_hash / effective_join_spec_hash /
  physical_plan_hash / contract_ir_hash / semantic_catalog_hash / calendar_hash /
  universe_snapshot / normalization_version / engine / engine_version / query_budget /
  code commit。
- remote object 版本（etag / version_id / content_length / last_modified）。
- 审计升级：query_id / request_id / transaction_id / operator / namespace / host / pid /
  snapshot / plan hash / 成功失败 / runtime / bytes / rows；redaction allowlist。
- 输出顺序契约：unordered raw read 文档化；`order_by=[time,instrument]` 显式请求，
  进 cache key / lineage / plan。
- InstrumentKey（market, security_id）PIT TickerMap，Symbol/Ticker 只作显示 key。
- schema evolution：required/nullable/safe_cast/enum/domain/timezone/primary_key/unique_key/
  partition schema/schema_version/compatible_from/deprecated/renamed/migration；key
  uniqueness 进 publish gate。
- upsert → DuckDB COW merge（不整 partition→pandas→merge）。
- incremental manifest mutation（ADD/REMOVE/REPLACE partition，commit 时 O(changed files)）。

---

## P2 — 平台整理

- 统一公开 API：`store.read(DataRequest)` / `store.scan(DataRequest)` /
  `store.sql(SafeSQLRequest)` / `store.write(WriteRequest)`；其余入口保留 compatibility
  wrapper 全部 lower 到同一 compiler。
- empty policy 统一、deterministic ordering 契约。

---

## 落地顺序（依赖优先）

1. **JoinCompiler 统一**（P0-1/P0-2）— 依赖面最大，先做。
2. **ContractIR 修复**（P0-33 issues 字段 + TemporalAxes + EffectiveJoinSpec 承载）。
3. **Financial seed**（P0-3）。
4. **Calendar 正确性**（P0-4..8）。
5. **Temporal axes + dataset gates**（P0-9/P0-10）。
6. **Strict/PIT/cutoff/fanout**（P0-11..16）。
7. **Factor matrix**（P0-17/23/24/25）。
8. **执行治理**（P0-18..22/35/36）。
9. **写事务**（P0-26..31）。
10. **Namespace/Metadata Plane**（P0-32/34）。
11. **P1**（元数据/CBO/COS）。
12. **P2 + DoD 回归 + CI**。

## 并发安全
- 只改 `dataaccess/`；`factor_engine/` 不碰。
- `read/scan_handle.py`（并发会话已改）需要编辑时先重读。
- 测试：`tests/` 属本仓库，可安全更新。

---

## 第四批审计收口（0.9.5 二期，main=`0cf8e18` 的 12 项二阶/故障注入）

用户按最新 main（新增 atomic write / PIT generation-directory / SQL sandbox /
COS mirror 校验）又扫出 12 项「实现边界 / 二阶竞态」，全部落地（1–7 并 Core
Freeze blocker，8–12 P1 收尾）：

- ✅ **1. atomic 并发 writer 安全**：`atomic_write_bytes/file` 临时文件改为
  `.<name>.tmp.<pid>.<uuid8>` + `O_EXCL`（不再共享固定 `.<name>.tmp` 互 O_TRUNC）；
  `_write_all` full-write loop；`durable=True` 的 fsync 错误向上传播（`durable=False`
  best-effort 吞）；异常清理自己的 tmp。
- ✅ **2. publish commit+audit 事务分离**：新增 `CommittedButAuditFailed`（
  `AuditWriteError` 子类）——publish body 已成功提交（ok=True）但 durable audit 落盘
  失败时抛它；`_dataset_mutation` 对这类异常仍按 **COMMITTED** 重建 manifest（数据
  确实发布了，manifest 必须追平），再单独向上报告；body 真失败 + 审计失败 → 原始
  错误不被审计失败掩盖。
- ✅ **3. read_auto polars 受控 collect**：polars 分支改走 `self.scan()`（ScanHandle）
  + `collect_table()`，不再拿裸 LazyFrame 塞 `collect_polars_with_budget`；同时
  `mode` 未知/空/非字符串 fail-closed（`_normalize_read_mode`）。
- ✅ **4. ScanHandle 远程 snapshot revalidation**：s3:// / cos:// 路径不再被跳过——
  collect 前 `fresh=True` 强制 re-HEAD 并对比 etag/content_length/version_id；strict
  下变化或无法验证（无凭证/网络/未记录身份）⇒ fail-closed。
- ✅ **5. auto/hybrid 消费 mirror 三态**：`local_mirror_complete_for_range` /
  `_local_daily/hive_date/year_complete` / `hybrid_cos_read_paths` 一律走
  `_local_file_usable`（verified/corrupt/legacy 三态）——本地截断/checksum 错的
  parquet 不再被裸 `.exists()` 误当「齐全」而绕过 remote/resync。
- ✅ **6. 共享 expected-partitions 编译器**：`mirror.expected_partitions()` 成为
  local/mirror/remote/hybrid 唯一「期望 partition」事实源（daily/hive_date 走
  trade_day 日历、calendar_day 自然日；hive_year 按年）——remote 不再自造一套自然日
  逻辑，交易日型数据跨周末不再误切 remote / 请求不存在的周末对象。
- ✅ **7. ParametricDataset remote 复用 validate_params**：`_remote_paths_from_storage`
  走 `validate_params`（type/range/allowed/regex/路径遍历防护）再 format，local 与
  remote 参数 parity 一致，参数不再能扩大 glob 范围；新增 public `resolve_remote_paths`。
- ✅ **8. coverage manifest fast path 修好 + 贯穿 params**：`_observed_from_manifest`
  用与主流程相同的 validated params（不再硬编码 `{}`）；顺带修掉它读
  `manifest.min_time_key/max_time_key`（**不存在**的属性）被 `except Exception` 静默
  吞掉的旧 bug——改从 `manifest.files` 的 min_time/max_time 推导，fast path 首次真正
  生效。
- ✅ **9. strict 5t 无日历 fail-closed**：`_trading_day_lag` 返回 `(lag, authoritative)`；
  strict 下日历不可用 ⇒ `(None, False)` ⇒ status 降级 `partial` + `authority=approximate`
  （绝不把自然日近似当权威交易陈旧度）；research 才允许近似并显式标记。
- ✅ **10. read_joined 空 universe ⇒ 0 行**：universe 分支 `not upaths`（含 glob 无
  匹配文件，`_paths_have_files`）复用 typed empty 分支（`_empty_branch_sql`）——
  不再 `upaths[0]` IndexError / 不把空 glob 塞 DuckDB。
- ✅ **11. PITIndexMetadata typed invariant**：`__post_init__` 拒绝 bool 串味
  （`complete="false"` / `glob_failed="no"`）、负数 counts、`indexed > source`、
  `complete=True` 但 `glob_failed/failed_files`；load 路径把 JSON 原始值直接传入（不
  `bool()` 预转换）并额外要求 complete 索引能证明 generation + source identity，
  缺失 ⇒ 非权威。
- ✅ **12. PIT index build lock + generation fencing**：`build_pit_event_index` 全程持
  `mutation_lock(.pit_index)`（含锁内重检复用）；`_commit_index_generation` 的指针
  切换 + 清理也在 commit lock 内，指针走 `atomic_write_text`（唯一 tmp，不再共享
  `.current.tmp`）；**只有 authoritative（complete）构建才切 current**（不完整保留
  上一代）；`_prune_old_generations` 保留 `{new, previous}`——并发 builder 的 current
  永远指向有效 generation。

**回归**：全量 `tests/` **758 passed / 0 failed**（含并发会话共享 DoD
`test_final_closure_round9.py` 11 条 + 本会话补充 `test_final_closure_round9b.py`
13 条；两者互补覆盖全部 12 项）；真实 `datasets.yaml` 71 个数据集照常加载。
协调并发会话：为其 `cos_storage_runtime` 补了 `uninstall_cos_storage_runtime()` +
`test_cos_storage_runtime.py` autouse 还原 fixture（否则其全局 patch 污染整个 suite）；
为其 `round9.py` 修了一处 `ParamSpec(enum=…)` 测试 bug 与 PIT 指针 `.current.tmp`
共享名竞态（两进程并发 `_commit_index_generation`）。FE 侧
`test_catalog_us.py` + `test_data_access_catalog_errors.py` 15 passed；
`tests/market + tests/storage` 其余失败均为并发会话 FE WIP（operator manifest 未
重新生成 / R7-229 override reason），与本批无关。未碰 GitHub。

---

## 第五批审计收口（DA+FE Integration Freeze，main @ `ecffd56`）

外部 AI 复查确认 **DataAccess Core 本身已可停止新增架构**；剩余为 DA+FactorEngine
联合链路的最后一批 integration root issue（6 项）+ 2 个 incomplete-fix bypass +
1 个 P1。全部本地实现，未碰 GitHub。

### 新发现 6 项

- ✅ **1. source-level 四层 PIT 真正挂到 production runtime**：`DataSourceBuildContext`
  的 `pit_enforce` 之前只进 build_context、factory 不取、`DataAccessSource` 不存、
  `_ensure_field_plans()` 只自动跑 mining/coverage gate——「有安全门、主通道没经过
  安全门」。现在 factory 把 `pit_enforce`（ctx 默认，source 可覆盖）传给
  `DataAccessSource`（新增 `self.pit_enforce`），`_ensure_field_plans()` 在
  `pit_enforce` 时自动 `assert_four_layer_pit(plans)`（UNKNOWN 层 production
  fail-closed / research 告警降级，与 mining/coverage gate 同为 opt-in）。
- ✅ **2. Secondary SourceRef child 继承父级 execution policy**：
  `LQTPLogicalDataSource._child()` 原来不传 run_mode/production/mining/PIT，生产
  anchor 的二级 SourceRef child 会退回 research/fail-open。现在 `_child()` 从
  `self.inner` 继承 run_mode / production / strict_unknown_fields /
  enforce_mining_gate / snapshot_now_only / mining_coverage_threshold / pit_enforce。
- ✅ **3. CompositeDataSource temporal join authority 定界**：Composite 只认识
  exact/asof_backward/tolerance 并自行 `merge_asof`——E1/E2/RAW_EVENT/revision/
  period-selection 不能交给它。新增 `_enforce_join_authority`：非 exact join +
  PIT 敏感源（read_mode event/pit、`_NO_KNOWLEDGE_TIME_FUNDAMENTALS` 财务、
  COS pit_policy==strict）→ production fail-closed、research 告警放行；
  exact 保留（child 已把 temporal 语义对齐好）。
- ✅ **4. Composite snapshot cache coherence**：`_invalidate_if_snapshot_changed`
  原来只读 child.`data_snapshot_id` 不主动 `refresh_snapshot()`，底层 A→B 后仍命中
  Composite 自己的 `_column_cache`。现在统一 child snapshot protocol：先调用每个
  子源 `refresh_snapshot()`，再读新增的 `DataAccessSource.snapshot_token`
  （`_manifest_token or data_snapshot_id`，弥补廉价 manifest token 不更新
  `data_snapshot_id` 的盲区），变化才清缓存；production 下 refresh 失败 →
  fail-closed 清缓存强制重读。
- ✅ **5. data_scope 区分 None / [] / LIST**：`compute_data_scope` 原来 `if
  instrument_filter:` 让 None 和 [] 都进不了 scope（全市场结果可污染空 universe）。
  现在显式编码 `instrument_filter_kind ∈ {ALL, EMPTY, LIST}`，三种身份缓存
  namespace 互不相同；`pit_enforce` 也纳入 scope（改变实际读语义）。
- ✅ **6. FE adapter 拒绝裸 str instrument_filter**：`DataAccessSource.__init__`
  复用 DA `strict_sequence`（`_strict_instrument_filter`）——None / list / tuple /
  set[str] 接受；裸 str/bytes（"AAPL"→["A","A","P","L"]）、dict、generator、
  非 str 元素全部构造期拒绝。顺带修掉 DA `strict_sequence` 一个 docstring 撒谎：
  generator 原来被 `tuple()` 静默消费，现在按声明拒绝。

### 两个 incomplete-fix bypass

- ✅ **A. production direct-local 写 guard 上提**：guard 原来只在
  `LocalParquetWriteTarget`，`ParquetMaterializer.materialize()` 主路径直接
  `_upsert_partition` 绕过。现在 `materialize()` 在**任何 side effect 之前**统一
  消费 `write_targets.normalize_write_target()` 严格枚举（`"locla"` 拼写错误 → 在
  注册因子/写文件/更新水位线之前 `ValueError`，不再「metadata 说已提交、物理数据
  不存在」），且 production + 含 local 的目标直接拒绝（必须 staging→publish）。
- ✅ **B. native Polars long 与 governed-lazy 同源**：`_store_scan_polars_native`
  原来优先 `store.scan_polars()` 拿裸 LazyFrame，绕过 ScanHandle 的 collect-time
  revalidation / budget / audit。现在与 `scan_dataset_columns` 同走
  `store.scan()`（ScanHandle 绑定 snapshot+budget），composition 链拿到真实
  LazyFrame，collect 前由 `revalidate_for_long_collect` + `enforce_arrow_budget`
  受控；strict 下 `native_lazyframe()` 抛错 = 要求受控 collect 的 fail-closed。

### P1（顺手清）

- ✅ **streaming_bars registry contract**：声明 `instrument_column: symbol` 但
  schema 无 symbol（symbol 在 `{symbol}/{YYYY-MM-DD}/{symbol}.parquet` 文件名里）。
  标记 `specialized_only: true` + reason，generic read/scan 一律拒绝（不再在假的
  asset 列上过滤）；`instrument_column` 置空。专用读取需 writer 侧 hive 化。

### 权威水位线状态机（staging-only）

staging-only（不写本地 lake）物化强制 `watermark_deferred=True`——不再在
materialize() 推进正式 FactorCatalog 水位线（增量调度器不会误以为已正式提交）。
`publish_factor_lake` 在 `publish_from_staging` **成功**后调用
`advance_published_watermark()` 计算 published lake 的 min/max datetime + row_count
推进水位线（CALCULATED → STAGED → PUBLISHED）；publish 失败抛异常 → 水位线不动。

### 回归

FE 新增 `tests/integration/test_final_closure_fe_integration2.py`（22 条 adversarial，
覆盖全部 6+2+1 项）+ 既有 `test_final_closure_fe_integration.py`（16 条）全绿；
`tests/runtime + tests/storage`、`tests/backend + tests/integration`、materializer /
production-gate / R10 identity 全绿。DA 侧 `tests/unit/test_final_closure_round12.py`
+ round11（specialized_only / strict_sequence）23 条全绿。按新 contract 更新了
`test_r10_factor_identity.py` 与 `test_materializer.py::test_staging_write_target_*`
（production+local 由 guard 拒绝；staging-only 水位线 defer）。预存失败：
`tests/runtime/test_logical_source_production_contracts.py::test_minute_sessions_never_collide_across_lunch_boundary`
（V2 `_session_slots` lunch 边界 bug，stash 验证为提交时已存在，与本批无关）。
