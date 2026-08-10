# 变更记录（shw）

本文档按版本记录与 **WorldQuant BRAIN 风格算子库** 相关的代码与文档改动。每条均标注 **第 N 版更改-shw**，便于追溯。

---

## 2026-08-10 R25 DataAccess 全平台最终收口（第 38 版）

**内容**：DataAccess 全平台收口——权限 / PIT / 物理分区 / Source Snapshot /
一致性 / 跨市场语义 / 并发 / 缓存 / 服务化 / FactorEngine 协作（详见
`dataaccess/docs/R25_FULL_PLATFORM_CLOSURE_REPORT.md`）。基线
`main@7b15a5e7`，20 个 P0 + ContractIR v2 + 12 条不变量：

- **Physical（P0-001/002/016）**：`dataaccess/contract/` 新增 PhysicalPartitionSpec/
  PhysicalLayout/MissingPartitionSemantics；US finance 编译成 PERIOD_END_FILE
  （partition_clock=period_end，{period_end}.parquet，query_clock=filing_date）；
  StockCapital split/shares 用 file_selector（{date}.parquet / shares_{date}.parquet）；
  event/period 不再被 trade-day enumerator 建模。
- **Filters（P0-003/004）**：FilterRequirement（panel/event/dimension scope、
  exactly_one）+ `_enforce_runtime_contract_filters` 覆盖所有读面；timeframe
  进入 RuntimeDatasetContract（event/exactly_one）。
- **PIT（P0-005/017/018）**：AvailabilityResult（authoritative/degradation_reason）
  + CalendarUnavailableError；filing_date date_label 不转时区；A UpdateTime
  revision_availability_time=None。
- **Snapshot（P0-009/010/011/012/013/014）**：`dataaccess/snapshot/` 新增
  ResolvedObject/ResolvedSourceSnapshot/SourceSnapshotResolver/SnapshotVerifier；
  production 拒绝 unresolved wildcard；QueryBudget v2（max_scan_objects/bytes）；
  content_digest（etag/versionId）；mirror generation 原子发布（staging→rename→pointer）。
- **Security（P0-006/007/008/019 + R24 保留）**：production 禁 ~/.cos.yaml；
  cache principal 隔离；derived factor 权限继承（FE security/access.py）。
- **Automated Research / FE 边界（P0-020）**：RunMode（interactive/automated/
  production）；automated_research 走 is_strict_semantics()（INV-07）；GovernedFrame
  （production 裸 dataframe → UnknownProvenanceError）。
- **Resource/Cache/Service**：GlobalResourceGovernor（admission 超限拒）、
  CacheManager（pin/GC/配额，pinned 永不删）、HTTP extra=forbid + 长度/基数限制、
  /ready 真正 readiness、/v1/read_uri admin-only。
- **DQ/CI**：unit_sentinel（单位漂移检测）、Production startup gate、
  ContractIR drift audit（scripts/audit_r25_contract_drift.py）、allowlist 登记
  3 个 FE 认证/校准脚本。

**测试**：729 unit + 143 contract + 3 concurrency = **875 passed**（新增
test_r25_physical_layout/filters/availability/snapshot/mirror/resource_schema/
security_fe/dq/startup_gate 9 个文件 58 个测试）。Freeze 判定见报告 §L。

---

## 2026-08-10 R24 DataAccess 安全 / PIT / 跨市场语义专项整改（第 37 版）

**内容**：DataAccess 权限边界 / 数据泄露 / COS 多服务器分级权限 / 财务 PIT /
A股与美股单位与会计语义 / DataAccess↔FactorEngine 契约漂移（详见
`dataaccess/docs/R24_SECURITY_PIT_CROSSMARKET_CLOSURE_REPORT.md`）。7 阶段：

- **安全**：`dataaccess/security/` 薄层（CredentialProvider / DataPrincipal /
  AccessPolicy / redaction / api key→principal）；production 禁解析 `~/.cos.yaml`、
  不写 os.environ secret；S3Credentials STS + repr 脱敏；store 读/写入口统一授权门；
  `read_uri` privileged；403→`AccessDeniedError` 不 fallback；secure cache 按
  principal 隔离 + 0700/0600 + O_EXCL temp + symlink 拒绝；`propagate_authorization`
  防吞授权错误；gitleaks CI + `.gitignore` secrets；audit 安全分类。
- **PIT**：US filing_date=date_label（不转纽约提前一天）→ next_session_open；
  A UpdateTime 仅 dedup tiebreaker（`pit_fidelity=knowledge_date_pit`）；US
  `vintage_pit`；timeframe exactly-one；strict 无日历 hard fail。
- **跨市场**：US dividend 拆 local/usd + currency；`UnitSpec` +
  `cross_market_compatible`（CNY vs USD 禁静默混合）；net_income
  consolidated/attributable 分开；`flow_semantics` 机器可读；instrument 加 market
  namespace。
- **派生因子权限继承**：`factor_engine/security/access.py`（新文件）——
  `derive_derived_access_tags` / `require_declassification_approval`（禁止自动降密）；
  DataAccess `FactorMeta` 带 access tags；read_factors 按 tag 授权。

**测试**：`test_r24_security_2026_08.py`（13）+ `test_r24_pit_crossmarket_2026_08.py`
（15）+ 存量回归 → **785 passed**。

## 2026-08-09 FactorEngine R14 第三轮 production DataEvent 收口（第 36 版）

**内容**：外部 AI 复查最新 main@44b2946 确认 **DataAccess Core Freeze + 普通
FE↔DA Integration Freeze 正式成立**；production DataEvent 链还差 2 个 P0，本轮
全部收口（详见 `docs/R14_PRODUCTION_INTEGRATION_CLOSURE_REPORT.md` §G）：

- **P0-1（方案 A feature gate）production DataEvent 自动发布默认关闭**：逐 factor
  `publish_factor_lake` 不是 visibility transaction——f1 成功、f2 失败会产生 mixed
  published state（上一轮测试已自己暴露：f_a 已发布、f_b 未发布、event rejected）。
  新增 `runtime/production_policy.production_data_event_auto_publish_enabled()`
  （`DATA_EVENT_PRODUCTION_AUTO_PUBLISH`，默认 off）：production 事件直接拒绝 +
  `ProductionEventAutoPublishDisabled`（begin→reject→raise，不落任何
  staging/published → published 湖零 mixed）；production 一律**强制**
  `write_target="staging"`——不再 `setdefault`，调用方无法覆盖成
  `staging_clickhouse`/`clickhouse`（stage-all 阶段产生 published/CH side effect）。
  显式启用仅实验性，真正 event visibility transaction 留待后续。research 不变。
- **P0-2 DataEventLedger fencing token**：`begin()` 返回 `(status, ReservationToken)`
  ——`{event_id, attempt_id, fencing_epoch, owner, expires_at}`；takeover 单调递增
  fence；同 worker 同 attempt 重入幂等续租。`commit`/`reject`/`renew` 在锁内校验
  `(owner, attempt_id, fencing_epoch)` 三元组，不匹配 → `DataEventLeaseLostError`
  （stale worker 绝不能再 publish/commit/reject）。`owner`/`attempt_id`/
  `fencing_epoch` 进 `DataEvent.to_dict()` + `normalize_data_event`（dict/API
  round-trip 不再丢）。调度器 stage 后逐 factor `renew` 心跳（长计算 > lease 不被误
  takeover）、publish 前 `renew` 做 fencing 门、commit/reject 带 token；renew 移到
  逐 factor try 外（lease 丢失 = 事件级中止，不算 factor 失败）。
- **P1（非 blocker，deferred）**：DA `_read_factor_matrix` 读取后重算
  `current_generation()` 写 audit，读取期间 generation 翻转可能 audit 失真；后续小修
  成一次性 `MatrixResolvedSnapshot`。

**新增测试**：`test_r14_event_fencing_2026_08.py`（9：stale commit/reject/renew 被拒、
takeover fence 单调、无 token fail-closed、renew 心跳、dict round-trip、
scheduler 级 stale publish 门零 publish）+ `test_r14_event_atomic_publish` 新增 3
（production 默认禁用零 mixed、无 event_id 也禁、强制 staging target）。存量对齐：
data_event_ledger / ledger_lease / round13 incremental 的 begin→(status,token) +
commit/reject 带 token、append 失败测试改 monkeypatch。回归全绿。

## 2026-08-09 FactorEngine R14 第二轮 production integration 收口（第 35 版）

**内容**：外部 AI 复查最新 main@64ad308 确认 DataAccess Core Freeze 继续成立；
Integration Freeze 还需 4 P0 + 1 条件 P0 + 1 P1，全部收口（详见
`docs/R14_PRODUCTION_INTEGRATION_CLOSURE_REPORT.md` §F）：

- **#1 DataAccess factor_matrix 跟随 generation 指针**：`ParametricDataset`
  新增 `generation_pointer` + `current_generation()`；`resolve_paths` 只解析
  `manifest.json` 的 `generation` 指向那一代（绝不 `generation/*` 通配混读
  current+previous）；manifest 指向缺失目录 → `DataError` fail-closed；动态因子列
  从 `manifest["factors"]` 取（`_matrix_available_columns`，不再永远
  MatrixCoverageMiss fallback）；generation 进 audit。factor_matrix 显式声明
  `authorized_root: ${FACTOR_MATRIX_ROOT}`（组件级 `path_is_under` 过不了
  `universe=` 前缀）。
- **#2 matrix generation fail-closed**：`load_matrix` 对「manifest 有 generation
  但目录缺失」抛 `FactorMatrixCorruptionError`（不再 legacy fallback 捞
  previous/orphan）；materialize 对旧代缺失同样 hard fail；已发布 generation 文件
  损坏 → 只复制到 quarantine 留证、**不移动**原文件（immutable，在途 reader）。
- **#3 MaterializationDelta 进入 ClickHouse 主链**：`execute_materialize` 构造唯一
  delta（upserts + tombstones=deleted_keys + full semantic digest + snapshot），CH
  经 `_series_with_tombstones` 把删除键写 NaN——源行删除后 Parquet 与 CH 同键一致。
- **#4 event rebuild 恢复 canonical scope**：`factor_from_catalog_info` 挂
  `FactorExecutionScopeHint`（market/universe/frequency/calendar/decision_policy）；
  `_verify_factor_semantic_identity` 对 scope 字段 fail-closed（declared 有值 +
  actual 缺失 → production 拒绝）；**修真 bug**：Expr 直接 `compute_ir_hash` 报
  `.attrs` 错 → 先 `Analyzer().lower()` 再 hash（否则含 ast_hash 的 production
  event 无法证明 identity）。
- **#5 DataEvent 两阶段原子发布 + lease**：production 事件全因子先写
  `write_target="staging"`，全部 stage 成功才逐因子 `publish_factor_lake`；任一失败
  reject + `PartialIncrementalFailureError`，published 湖零 mixed，同 event 重试幂等
  收敛只 commit 一次。`pending` 预留带 `owner/attempt_id/reserved_at` + lease 过期
  takeover（`DATA_EVENT_LEDGER_LEASE_SECONDS`）；`in_flight` 不再塌缩成
  "duplicate"。

新增测试：R14 第二轮 5 个文件 21 条（storage failclosed 4 / delta→CH 2 / event
rebuild scope 5 / ledger lease 5 / event atomic publish 4）+ dataaccess 侧 5 条
（generation resolver）。存量回归全绿（见 §F）。

## 2026-08-09 FactorEngine R14 production integration 收官（第 34 版）

**内容**：外部 AI 复查确认 **DATAACCESS CORE FREEZE** 成立，但
**FACTORENGINE ↔ DATAACCESS PRODUCTION INTEGRATION FREEZE 暂缓一轮**——修掉
production 集成链 5 个 blocker + 1 个 P1（详见
`docs/R14_PRODUCTION_INTEGRATION_CLOSURE_REPORT.md`）：

- **#1 factor_matrix 全局 crash-atomic**：整个 `(universe, freq)` 当一个
  generation——完整写 `generation/<gid>/`（未触达分区 copy-on-write 硬链接）+ 全部
  validate 后，一次 `os.replace` 原子切 `manifest.json` 的 `generation` 指针；
  `load_matrix` 只读该代。任何崩溃窗口 reader 只能见完整 old/new generation，无
  mixed（旧「先 manifest 再逐分区 os.replace」在切换一半时崩溃会 mixed）。GC 保留
  当前 + 上一代。
- **#2 matrix version gate 自动绑定**：`matrix_service` 先校验 matrix 声明的
  universe/frequency 与 factor 实际执行 scope（`_scope_from_factor`，优先
  semantic_identity）一致；再按真实执行 scope+analysis+source contract 自动算
  `semantic_digest`（与 execute_materialize 同源）；production 缺 version 直接拒绝。
- **#3 语义身份统一消费**：execute_materialize 只算一次 canonical scope，把
  frequency/universe/market/calendar/decision_time_policy 同时喂给
  `_build_semantic_identity` + `_build_full_factor_definition`（decision_policy 不再
  写死 None）——执行/cache scope/factor_version/full definition/rebuild 全消费同一
  scope，杜绝「执行 5m、落库 1d」。
- **#4 ClickHouse version parity**：orchestrator 只算一次 canonical factor_version
  （semantic digest 前缀），显式传 ClickHouse 双写；data_snapshot_id 改用
  effective_data_source_config。与并发会话 NEW-P0-59 unified digest 兼容。
- **#5 DataEventLedger 事务语义**：last_committed_snapshot 取 sequence 最大 chain
  head（修「返回第一个匹配」bug）；sequence 乱序校验；损坏 fail loud（仅尾部半行
  截断恢复）；append OSError 上抛；begin/commit/reject 在 flock 内 reload+append，
  并发 worker 同事件只有一个 begin（pending 预留 → in_flight）。
- **#6 Composite execution_spec**：任一 child 无法序列化 → production 抛
  `UnreconstructableDataSource`、research 返回 None，不再伪造缺 dataset 的
  `{"type":"data_access"}`。

新增测试 27 条（R14 4 个文件）；存量回归（r11 15 / materializer 48 / r13 matrix
17 / round13 incremental 10 / r10 scheduler + composite 27 / phase21 matrix roundtrip）
全绿。只改 factor_engine/ orchestration+storage 层，不碰 cleaned_operators/ir/scripts。

---

## 2026-08-09 FactorEngine R11 orchestration 收官（第 33 版）

**内容**：外部 AI 复查确认 DataAccess Core 不再扩架构后，把焦点转到 FactorEngine
orchestration 层，8 项收官（7 必修 + 1 补测试）：

- **#2 修订窗口**：scheduler 以 `affected_start/end` 为重算基准（历史修订不再从事件
  抵达日期漏算）；`plan_updates_from_data_event` 的 since=affected_start、end=affected_end。
- **#3 删除打通**：`DataEvent.deleted_keys → FactorUpdatePlan → materialize_incremental →
  execute_materialize → ParquetMaterializer.materialize(deleted_keys=…)`；纯删除事件也写
  tombstone（`is_valid=0`），旧有限值不再残留。
- **#4 物理源 edge**：`_edges_from_analysis` 以 `spec.dataset`（DataAccess 物理 dataset）
  为 edge `source_dataset`，不再从 anchor 猜；edge 表加 `logical_table`。
- **#5 full definition 原子持久化**：`execute_materialize` 成功落盘后经
  `DependencyCatalog.record_factor_manifest` 原子写完整重建规格；`Factor` 增
  surface/dialect/dialect_version。
- **#6 语义身份版本**：`factor_registry` 增 `factor_version`（= `FactorSemanticIdentity`
  digest 前缀）；production 同 factor_id + 语义身份变化 → `FactorSemanticIdentityMismatchError`。
- **#7 factor_matrix 安全 merge**：分区写改 flock 互斥 read-merge-write，partial
  因子/日期不再覆盖整月；tmp 名带 pid+uuid。
- **#8 原子替换**：`record_factor_manifest` 单 `BEGIN IMMEDIATE` 事务写 legacy+edges+
  full spec；`delete_factor` 补删 edge/full_definition 表。
- **#1** 事件服务 normalize 别名确认已修复。

见 [`R11_ORCHESTRATION_CLOSURE_REPORT.md`](R11_ORCHESTRATION_CLOSURE_REPORT.md)。

---

## 2026-07 文档清理

**内容**：删除已完成/过期的规划类文档（`enterprise_factor_engine_roadmap.md`、`performance_platform_plan.md`、`STRUCTURE.md`、`lqtp_vs_factor_engine_operators.md`）；总览并入 [`FactorEngine完全指南.md`](FactorEngine完全指南.md)；根 `README.md` 移除冗长分版脚注。

---

## 2026-07 Backend 与文档同步（第 32 版）

**内容**：SQL 下推 **58** 算子 + `column`/`literal`（共 **60** 可编译节点）；registry sql backend **62**（CI 门禁）；新增 `cs_resid`/`cs_regression` SQL；统一 Polars **325** 文档；修正 `neutralize`（组内去均值）与 `cs_resid`（截面 OLS）语义混用。

- **[`sql_pushdown_coverage.md`](sql_pushdown_coverage.md)**：CI 自动生成 SQL 清单  
- **[`operators_semantics.md`](operators_semantics.md)** / **[`dsl_operators_reference.md`](dsl_operators_reference.md)** / **[`算子与导入教程.md`](算子与导入教程.md)**：Backend §、OLS vs group_neutralize、`is_nan`/`is_finite` float64、`bfill` 因果  
- **[`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md)**：§5.4 SQL 下推 + 修正 neutralize 描述  
- **`cleaned_operators/docs/operator_doc_semantics.py`**：`neutralize` 文档对齐 dedupe 别名  

---

## 2026-06 文档精简

**内容**：删除重复或过时文档，统一以 [`算子与导入教程.md`](算子与导入教程.md) + [`dsl_operators_reference.md`](dsl_operators_reference.md) + [`miner_delivery_spec.md`](miner_delivery_spec.md) 为挖掘侧必读；详见 [`docs/README.md`](README.md) §5「已移除文档」。

---

## 2026-06 文档对齐（挖掘对接 — 仅文档）

**内容**：第 31 版 cleaned 迁移后，统一修正 **文档 vs `build_dsl_allowlist()`** 不一致处；新增 [`dsl_operators_reference.md`](dsl_operators_reference.md) 作为投递白名单人类索引。

- **[`dsl_operators_reference.md`](dsl_operators_reference.md)**：可投递算子分类 + **37 个待迁回 cleaned 的名字** + 替代写法  
- **[`operators_semantics.md`](operators_semantics.md)**：区分「白名单内」与「待迁移」  
- **[`miner_delivery_spec.md`](miner_delivery_spec.md)** §4：美股字段组合、算子表、禁止 `pasteurize`/`bucket` 等误用  
- **[`canonical_data_fields.md`](canonical_data_fields.md)** §3.1：美股价量物理列 vs DSL 组合  
- **[`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md)** / **`.txt`**：LLM 勿生成非白名单算子  
- 根 **[`README.md`](../README.md)**、`runtime/README.md`、教程与 examples 索引同步  

**说明**：历史第 10–19 版 changelog 条目中「已实现」算子，若未出现在当前白名单，以 **`dsl_allowlist.json`** 为准。

---

## 第 30 版更改-shw

**内容**：为 **根目录与各包 `README.md`** 统一增加 **「协作者速览（约 5 分钟）」** 文首小节（与 **`backtest/README.md`** 已有 **「新人 5 分钟上手」** 同思路），便于协作者快速理解 **该目录职责、上下游边界、从哪读起**；**[`docs/README.md`](README.md)** 导航句已注明这一约定。

---

## 第 29 版更改-shw

**内容**：**多资产回测文档与实现对齐**，便于协作者理解 **执行层 → 滞后 → 收益/成本** 与 **`FACTOR_BACKTEST_EXECUTION_ENGINE`** 语义。

- **[`backtest/README.md`](../../../backtest_layer/single_asset_backtest/README.md)**：文首增加 **「给协作者」** 摘要；**§4.5 / §4.6 / §8–§9 / §13–§14** 与 [`runner.py`](../../../backtest_layer/single_asset_backtest/runner.py) 中 **`run_multi_asset_backtest`**、**`_multi_asset_fingerprint(feeds, executed_weights)`** 一致；修正 **§9** 步骤顺序（先 **`asset_return` + 执行层**，再 **`shift`** 与 **`gross`/`net`**）。
- **[`docs/adr_backtest_target_position.md`](adr_backtest_target_position.md)**：重写 **§13** 多资产前视段落；新增 **§14**（执行输出、指纹列、**requested/resolved**）。
- **[`README.md`](../README.md)**：多资产 **`portfolio_execution_engine`** 说明与 **第 29 版** 摘要；ADR 索引补充 **§14**。
- **[`runtime/README.md`](../runtime/README.md)**：`perf_config` 小节补充 **`FACTOR_BACKTEST_EXECUTION_ENGINE`**。

---

## 第 1 版更改-shw

**内容**：确立算子分类与目录约定，不新增与现有 `expr` 重复的模块名。

- **`expr/`**：在既有 `arithmetic.py`、`logical.py`、`ts.py`、`cs.py` 上扩展表达式节点；**不**新增 `time_series.py` / `cross_sectional.py`；新增同级文件 `vector.py`、`transformational.py`、`group.py`（占位 Expr）。
- **`api/operators/`**：由单文件 `api/operators.py` 改为**包**，子模块与 expr 对齐：`arithmetic.py`、`logical.py`、`ts.py`、`cs.py`、`vector.py`、`transformational.py`、`group.py`，`__init__.py` 统一 re-export。
- **新增** [`api/operator_registry.py`](../api/operator_registry.py)：`build_dsl_allowlist()` 生成 DSL 白名单；`STUB_IR_OPS` 标明仅编译、Pandas 未实现的算子。
- **对外命名**：DSL/API 优先 WQ 蛇形名（如 `ts_std_dev`、`ts_delay`、`ts_max`、`ts_min`）；保留 `ts_std`、`delay` 等兼容别名。

---

## 第 2 版更改-shw

**内容**：编译链与 DSL 行为对齐新算子。

- **`ir/analyzer.py`**：为全部新 `Expr` 类型增加 `visit` 分支，输出 `IRNode`；滚动窗口属性统一使用 **`d`**（与 WQ 文档一致）；`TsStdDev` / `Delay` 等与旧类型兼容。
- **`api/dsl_parser.py`**：白名单改为从注册表构建；支持 **`ast.Compare`**（仅允许单段比较，不支持链式）；支持 **call 的 kwargs** 写入 IR `attrs`；逻辑算子在 Python 语法下使用 **`and_` / `or_` / `not_`**。

---

## 第 3 版更改-shw

**内容**：`PandasBackend` 扩展与占位策略。

- **`backend/pandas_backend.py`**：实现 Arithmetic / Logical / Time Series / Cross Sectional 下大量算子的 kernel；**`ts_max`、`ts_min`** 已实现；`ts_std_dev` 为主实现并注册 `ts_std` 别名；`nary_add` / `nary_mul` / `nary_sub` 及 `densify` 等。
- **依赖**：截面/时序 `quantile`（Gaussian）依赖 **scipy**，已写入 `pyproject.toml` 的 `[project.optional-dependencies] pandas`。
- **占位**：`vec_*`、`bucket`、`trade_when`、`group_*`、`ts_step`、`hump` 等在注册表中存在，执行时统一 **`NotImplementedError`**（说明见 `STUB_IR_OPS`）。  
  *（注：第 7 版起 **`group_*` 已在 Pandas 实装**，上句为第 3 版当时状态；当前占位集以仓库内 `STUB_IR_OPS` 为准。）*

---

## 第 4 版更改-shw

**内容**：测试与共享工具。

- **新增** [`tests/helpers.py`](../tests/helpers.py)：`InMemorySeriesSource`，供多测试复用。
- **新增** `tests/__init__.py`（包标识）。
- **新增** `test_operators_arithmetic.py`、`test_operators_logical.py`、`test_operators_ts.py`、`test_operators_cs.py`、`test_operators_stub.py`。
- **`test_pandas_backend.py`**：改为从 `tests.helpers` 引用 `InMemorySeriesSource`。

---

## 第 5 版更改-shw

**内容**：文档与 README 同步（本条及后续小节）。

- **新增** [`docs/operators_semantics.md`](operators_semantics.md)：Bar/日历语义、DSL 限制、scipy 依赖、占位算子列表。
- **新增** 本文档 [`docs/changelog_shw.md`](changelog_shw.md)。
- **更新** [`README.md`](../README.md)：`支持的算子` 与 `项目结构` 反映 `api/operators/` 包、注册表与新 expr 文件；增加文档索引链接。
- **更新** [`docs/factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md)：文首增加文档修订说明与算子文档指向。
- **更新** [`docs/massive_parquet_data_dictionary.md`](massive_parquet_data_dictionary.md)：文首增加与因子算子变更的交叉引用（数据字典本身无算子逻辑变更）。

---

## 第 6 版更改-shw

**内容**：为核心源码补充 **中文模块/类/关键函数注释**（`expr/`、`api/`、`ir/`、`planner/`、`backend/pandas_backend` 与 `debug_backend`、`runtime/engine` 等），便于快速理解职责与数据约定（MultiIndex、时序/截面分组）。

---

## 第 7 版更改-shw

**内容**：对照《算子库扩展与加速规划》落地 **路线图文档**、**清洗/技术指标/上下文** 算子、**group_* Pandas 实现**、**子树缓存 MVP**，并同步依赖与文档。

- **新增** [`docs/operators_roadmap.md`](operators_roadmap.md)：研究报告主题 ↔ 算子名 ↔ 实现状态表；可选加速依赖说明。
- **新增** [`docs/adr_context_benchmark.md`](adr_context_benchmark.md)：`change_instrument` 基准列 MultiIndex 与按日聚合约定。
- **`api/operator_registry.py`**：扩展 `BrainCategory`（`TECHNICAL` / `CLEANING` / `CONTEXT` 及远期枚举）；`STUB_IR_OPS` **移除** `group_*`（已可执行）。
- **新增** `expr/cleaning.py`、`expr/technical.py`、`expr/context.py` 与对应 `api/operators/*.py`；**`ir/analyzer.py`** 增加 visit 分支。
- **`backend/pandas_backend.py`**：实现上述 op 的 kernel；**`group_*`** 全套 kernel；**`ExecutionContext.cache`** 启用时按 **JSON 结构键** 做子树结果缓存（与 `CacheManager` 集成）。
- **`pyproject.toml`**：可选依赖组 **`accel`**（bottleneck、numba）、**`talib`**（TA-Lib）。
- **测试**：新增 [`tests/test_operators_extension.py`](../tests/test_operators_extension.py)；调整 `test_operators_stub.py`（`group_*` 不再属 stub）。
- **文档/README**：更新 [`operators_semantics.md`](operators_semantics.md)、[`README.md`](../README.md) 算子表、项目结构树与文档索引。

---

## 第 8 版更改-shw

**内容**：补齐与第 7 版实现不同步的文档与测试，避免「代码已实装、Prompt/历史条目仍写旧状态」。

- **[`docs/factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md)**：新增 **§5.4**（Group / 清洗 / 技术 / 上下文）与 **§5.5** 算术（原 5.4 顺延）；修正 **性能** 小节中关于 `enable_cache` 的表述（子树缓存 MVP）。
- **[`docs/factor_engine_llm_prompt.txt`](factor_engine_llm_prompt.txt)**：文首 **NOTE（第 7 版）**、算子字典新增同名扩展段、性能说明与 `.md` 对齐。
- **[`docs/changelog_shw.md`](changelog_shw.md)**：为 **第 3 版** 中「`group_*` 占位」补 **脚注**（标明为历史快照，当前以 `STUB_IR_OPS` 为准）。
- **[`README.md`](../README.md)**：`operator_registry` 一行注释改为 **STUB_IR_OPS** 表述，避免「全是占位」歧义。
- **测试**：[`tests/test_dsl_parser.py`](../tests/test_dsl_parser.py) 增加 `test_parse_expr_new_operators_v7`，覆盖新 DSL 函数解析。

---

## 第 9 版更改-shw

**内容**：**Bottleneck 可选滚动加速**（阶段 1）：`ts_mean` / `ts_max` / `ts_min` 在已安装 Bottleneck 且未设置 `FACTOR_ENGINE_DISABLE_BOTTLENECK=1` 时走 `move_*`，与 pandas `rolling` 结果对齐。

- **[`backend/pandas_backend.py`](../backend/pandas_backend.py)**：`_bottleneck_mod`、`_ts_roll_via_bottleneck`；上述三算子优先尝试 Bottleneck。
- **测试**：[`tests/test_bottleneck_ts_roll.py`](../tests/test_bottleneck_ts_roll.py)（`importorskip("bottleneck")`）。
- **[`docs/operators_roadmap.md`](operators_roadmap.md)**：Bottleneck 行改为「部分/可选（已接线）」。
- **[`README.md`](../README.md)**：可选依赖 `accel` / 环境变量说明一句。

---

## 第 10 版更改-shw

**内容**：**`bucket` / `trade_when` / `ts_step` / `hump`** 在 Pandas 实装；**`STUB_IR_OPS`** 仅余 `vec_*`；新增 ADR 与测试。

- **[`backend/pandas_backend.py`](../backend/pandas_backend.py)**：`_op_bucket`、`_op_trade_when`、`_op_ts_step`、`_op_hump`；`_as_bool_mask_scalar`。
- **[`api/operator_registry.py`](../api/operator_registry.py)**：从 `STUB_IR_OPS` 移除 `bucket`、`trade_when`、`ts_step`、`hump`。
- **[`expr/ts.py`](../expr/ts.py)** / **[`api/operators/ts.py`](../api/operators/ts.py)**：**`ts_step(d, anchor)`** 须显式传入 `anchor` 以对齐 MultiIndex。
- **ADR**：[`adr_trade_when.md`](adr_trade_when.md)、[`adr_ts_step_hump.md`](adr_ts_step_hump.md)。
- **文档**：[`operators_semantics.md`](operators_semantics.md)、[`factor_engine_llm_prompt` *.md/*.txt](factor_engine_llm_prompt.md)。
- **测试**：[`test_operators_transformational.py`](../tests/test_operators_transformational.py)、[`test_trade_when_ts_hump.py`](../tests/test_trade_when_ts_hump.py)；[`test_operators_stub.py`](../tests/test_operators_stub.py) 仅保留 `vec_avg` 占位编译测。

---

## 第 11 版更改-shw

**内容**：**`vec_avg` / `vec_sum` 路径 B**（阶段 5）— 暂不实现标量列上的假向量语义；在路线图与语义文档中写明 **需 Parquet 向量列 / object ndarray 契约** 后再实装。

- **[`docs/operators_roadmap.md`](operators_roadmap.md)**：新增「向量算子」行（路径 B）。
- **[`README.md`](../README.md)**：支持的算子段补充一句。

---

## 第 12 版更改-shw

**内容**：**PolarsBackend 核心子集**（阶段 6）：长表执行 `column` / `literal` / `add` `sub` `mul` `div` / `rank` / `ts_mean`，结果转回 pandas MultiIndex Series。

- **[`backend/polars_backend.py`](../backend/polars_backend.py)**：重写递归求值与 `_series_to_long` / `_long_to_series`。
- **测试**：[`tests/test_polars_backend.py`](../tests/test_polars_backend.py)（`importorskip("polars")`）。

---

## 第 13 版更改-shw

**内容**：**远期数据层算子占位**（阶段 7）：微观结构 / 基本面 / 另类各 1 个 Expr + IR + DSL + Pandas stub，**不假装有 LOB/PiT 数据**。

- **新增** `expr/microstructure.py`、`expr/fundamental.py`、`expr/alternative.py`；[`api/operators/future_data.py`](../api/operators/future_data.py)；`STUB_IR_OPS` 增加 `lob_ofi_stub`、`fundamental_ttm_stub`、`alt_sentiment_stub`。
- **[`ir/analyzer.py`](../ir/analyzer.py)**、[`operator_registry.py`](../api/operator_registry.py) 接线。
- **[`docs/operators_roadmap.md`](operators_roadmap.md)**：标明「仅接口 stub」行。

---

## 第 14 版更改-shw

**内容**：**`sin` / `cos`** 与 **Joblib 多因子并行示例**（阶段 8）；**Welford / RLS 显式算子**仍延期（与内部滚动优化合并评估）。

- **算术**：`expr/arithmetic.py` `Sin`/`Cos`；`api/operators/arithmetic.py`；`PandasBackend` 一元 `np.sin`/`np.cos`；**Polars** 同步子集。
- **示例**：[`examples/run_factors_joblib.py`](../examples/run_factors_joblib.py)。

---

## 第 15 版更改-shw

**内容**：为 **难理解算子** 加长 **中文 docstring / 模块说明**（不改数值语义），便于阅读源码与 IDE 悬停即懂；行为与测试不变。

- **Expr**：[`expr/transformational.py`](../expr/transformational.py)（`Bucket` / `TradeWhen`）、[`expr/ts.py`](../expr/ts.py)（`TsStep` / `Hump` / `TsRegression` / `TsQuantile` / `TsDecayLinear` / `KthElement` / `LastDiffValue` / `TsBackfill`）、[`expr/group.py`](../expr/group.py)（各 `Group*`）、[`expr/cs.py`](../expr/cs.py)（`CsQuantile` / `Scale`）、[`expr/logical.py`](../expr/logical.py)（模块说明 / `IfElse`）、[`expr/context.py`](../expr/context.py)（`Orthogonalize` / `ChangeInstrument`）、[`expr/microstructure.py`](../expr/microstructure.py) 等远期占位。
- **API**：[`api/operators/transformational.py`](../api/operators/transformational.py)、[`api/operators/ts.py`](../api/operators/ts.py)、[`api/operators/cs.py`](../api/operators/cs.py)、[`api/operators/group.py`](../api/operators/group.py)、[`api/operators/future_data.py`](../api/operators/future_data.py)。
- **后端**：[`backend/pandas_backend.py`](../backend/pandas_backend.py) 中 `_as_bool_mask*`、`_op_bucket` / `_op_trade_when` / `_op_ts_step` / `_op_hump` 的说明性 docstring。

---

## 第 16 版更改-shw

**内容**：**研究导向算子扩充**：技术指标（通道、波动、动量、成交量、Overlap 均线族）+ 截面 **`neutralize`** + 时序 **`ts_skew` / `ts_kurt`**；TA-Lib 优先、pandas/numpy 退化；DSL 白名单与测试同步。

- **Expr**：[`expr/technical.py`](../expr/technical.py)（`TsAtr`/`TsNatr`/`TsTrange`/`TsDonchian`/`TsKeltner`/`TsMaEnvelope`/`TsMacd`/`TsCci`/`TsStoch`/`TsWillr`/`TsRoc`/`TsObv`/`TsMfi`/`TsDema`/`TsWma`/`TsKama`）；[`expr/cs.py`](../expr/cs.py) `Neutralize`；[`expr/ts.py`](../expr/ts.py) `TsSkew`/`TsKurt`。
- **IR / API**：[`ir/analyzer.py`](../ir/analyzer.py)；[`api/operators/technical.py`](../api/operators/technical.py)、[`cs.py`](../api/operators/cs.py)、[`ts.py`](../api/operators/ts.py)、[`operator_registry.py`](../api/operator_registry.py)、[`api/operators/__init__.py`](../api/operators/__init__.py)。
- **后端**：[`backend/pandas_backend.py`](../backend/pandas_backend.py)（`_wilder_tr_arr`/`_wilder_atr_arr`/`_eval_hlc_series` 与各 `_op_ts_*`、`_op_neutralize`）。
- **测试**：[`tests/test_operators_extension.py`](../tests/test_operators_extension.py)、[`tests/test_dsl_parser.py`](../tests/test_dsl_parser.py)。
- **文档**：[`operators_semantics.md`](operators_semantics.md)、[`operators_roadmap.md`](operators_roadmap.md)、[`README.md`](../README.md)。

---

## 第 17 版更改-shw

**内容**：**远期 stub 大规模扩充**（仍不实现数值内核）：基本面 / 另类 / 微观结构统一为 **单 child + IR `op` 字符串**，`STUB_IR_OPS` 与 `expr` 侧 `*_STUB_OPS` 常集对齐；`PandasBackend` 对 `STUB_IR_OPS` 整集注册 `_stub`。

- **Expr**：[`expr/fundamental.py`](../expr/fundamental.py) `FundamentalStub` + `FUNDAMENTAL_STUB_OPS`；[`expr/alternative.py`](../expr/alternative.py) `AlternativeStub` + `ALTERNATIVE_STUB_OPS`；[`expr/microstructure.py`](../expr/microstructure.py) `MicrostructureStub` + `MICROSTRUCTURE_STUB_OPS`（取代原先各单类 Stub 的 IR 分支写法）。
- **API / IR**：[`api/operators/future_data.py`](../api/operators/future_data.py) 工厂函数全集；[`ir/analyzer.py`](../ir/analyzer.py)；[`api/operator_registry.py`](../api/operator_registry.py)；[`api/operators/__init__.py`](../api/operators/__init__.py) 动态 re-export。
- **后端**：[`backend/pandas_backend.py`](../backend/pandas_backend.py) `for op in STUB_IR_OPS`。
- **测试**：[`tests/test_operators_stub.py`](../tests/test_operators_stub.py) 对 `STUB_IR_OPS` 参数化；[`tests/test_dsl_parser.py`](../tests/test_dsl_parser.py) 增补解析样例。
- **文档**：[`operators_semantics.md`](operators_semantics.md)、[`operators_roadmap.md`](operators_roadmap.md)、[`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) / [`.txt`](factor_engine_llm_prompt.txt)、[`README.md`](../README.md)。

---

## 第 18 版更改-shw

**内容**：**技术指标第二波**：ADX/Aroon、Chaikin `ts_ad`/`ts_adosc`、`ts_sar`（无 TA-Lib 时为简化 PSAR）、`ts_cmo`、`ts_ppo`/`ts_apo`、`ts_ultosc`、`ts_stochrsi`、`ts_tema`/`ts_trima`/`ts_t3`；TA-Lib 优先，pandas/numpy 退化。

- **Expr / API**：[`expr/technical.py`](../expr/technical.py)；[`api/operators/technical.py`](../api/operators/technical.py)；[`ir/analyzer.py`](../ir/analyzer.py)；[`api/operator_registry.py`](../api/operator_registry.py)；[`api/operators/__init__.py`](../api/operators/__init__.py)。
- **后端**：[`backend/pandas_backend.py`](../backend/pandas_backend.py)（`_numpy_adx_line`、`_numpy_aroon`、`_chaikin_ad_from_hlcv`、`_numpy_sar`、`_numpy_ultosc`、`_numpy_t3_close` 与各 `_op_ts_*`）。
- **测试 / 文档**：[`tests/test_operators_extension.py`](../tests/test_operators_extension.py)、[`tests/test_dsl_parser.py`](../tests/test_dsl_parser.py)；[`operators_semantics.md`](operators_semantics.md)、[`operators_roadmap.md`](operators_roadmap.md)、[`README.md`](../README.md)。

---

## 第 19 版更改-shw

**内容**：**技术指标第三批**：`ts_bop`（open 用上一根 close 近似）、`ts_mom`、`ts_stochf`（`line=fastk|fastd`）、`ts_trix`、`ts_adxr`、`ts_dx`、`ts_rocr` / `ts_rocr100`、`ts_linearreg_slope` / `ts_linearreg_angle`；TA-Lib 优先，pandas/numpy 退化。

- **Expr / API**：[`expr/technical.py`](../expr/technical.py)；[`api/operators/technical.py`](../api/operators/technical.py)；[`ir/analyzer.py`](../ir/analyzer.py)；[`api/operator_registry.py`](../api/operator_registry.py)；[`api/operators/__init__.py`](../api/operators/__init__.py)。
- **后端**：[`backend/pandas_backend.py`](../backend/pandas_backend.py)（`_dmi_wilder_di_dx`、`_numpy_adxr`、滚动线性回归辅助与各 `_op_ts_*`）。
- **测试 / 文档**：[`tests/test_operators_extension.py`](../tests/test_operators_extension.py)、[`tests/test_dsl_parser.py`](../tests/test_dsl_parser.py)；[`operators_semantics.md`](operators_semantics.md)、[`operators_roadmap.md`](operators_roadmap.md)、[`README.md`](../README.md)、[`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) / [`.txt`](factor_engine_llm_prompt.txt)。

---

## 第 20 版更改-shw

**内容**：**华泰 GPT 因子工厂 2.0** 算子（研报图表 9/11）与引擎 **WQ 风格 DSL** 的 **对照目录 + 去重 ADR**；不新增重复 DSL 名；分钟频 `Agg_*` / `Agg_Explode_*` 保持「无对应 / 远期数据层」说明。

- **新增** [`docs/huatai_factor_factory_operator_catalog.md`](huatai_factor_factory_operator_catalog.md)：华泰名 → 规范名 / 组合式、映射类型（等价/近似/仅 stub/无对应）、频率与数据契约、来源引用。
- **新增** [`docs/adr_huatai_factor_factory_operators.md`](adr_huatai_factor_factory_operators.md)：命名策略、去重判定顺序、第二阶段可选 intraday stub 的边界。
- **更新** [`docs/operators_roadmap.md`](operators_roadmap.md)：研究主题表一行 + 相关文档链接。
- **更新** [`docs/factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) / [`.txt`](factor_engine_llm_prompt.txt)：第 20 版 NOTE（华泰名须经 catalog 映射）。
- **更新** [`README.md`](../README.md)：文档索引、第 20 版摘要、`docs/` 结构树。

---

## 第 21 版更改-shw

**内容**：华泰《GPT 因子工厂 2.0》对照 **落地到 Python**：新增机器可读映射模块与单测；**不**向 DSL 白名单增加第二套华泰函数名。

- **新增** [`api/htsc_factor_factory_reference.py`](../api/htsc_factor_factory_reference.py)：`HtscOperatorRef`、图表 9/10/11 结构化条目、``canonical_allowlist_keys_for_validation()``（供测试）。
- **新增** [`tests/test_htsc_factor_factory_reference.py`](../tests/test_htsc_factor_factory_reference.py)：等价映射键须存在于 `build_dsl_allowlist()`。
- **更新** [`api/operator_registry.py`](../api/operator_registry.py)、[`expr/fundamental.py`](../expr/fundamental.py)、[`api/operators/future_data.py`](../api/operators/future_data.py)：模块说明互指华泰文档与上述模块。
- **更新** [`docs/huatai_factor_factory_operator_catalog.md`](huatai_factor_factory_operator_catalog.md)：维护节说明与 Python 同步。
- **更新** [`docs/factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) / [`.txt`](factor_engine_llm_prompt.txt)：第 21 版 NOTE。
- **更新** [`README.md`](../README.md)：第 21 版摘要、项目结构树。

---

## 第 22 版更改-shw

**内容**：华泰算子 **融入** 现有 `api/operators` / `expr`（**华泰对照** docstring）；**删除** [`api/htsc_factor_factory_reference.py`](../api/htsc_factor_factory_reference.py) 与 [`tests/test_htsc_factor_factory_reference.py`](../tests/test_htsc_factor_factory_reference.py)；新增 **`exp`**（`Exp` Expr，华泰图表 9/11 `Exp(X)`）；新增 **`INTRADAY_STUB_OPS`** / [`expr/intraday.py`](../expr/intraday.py) / [`api/operators/intraday.py`](../api/operators/intraday.py)（华泰图表 11 `Agg_*` / `Agg_Explode_*` / `Tp_Sample` 蛇形 stub）；[`BrainCategory.INTRADAY`](../api/operator_registry.py)；`ir/analyzer` / `PolarsBackend` 扩展；测试 [`tests/test_intraday_stub.py`](../tests/test_intraday_stub.py)、算术 `test_exp`。

- **更新** [`docs/huatai_factor_factory_operator_catalog.md`](huatai_factor_factory_operator_catalog.md)、[`adr_huatai_factor_factory_operators.md`](adr_huatai_factor_factory_operators.md)、[`README.md`](../README.md)、[`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) / [`.txt`](factor_engine_llm_prompt.txt)。

---

## 第 23 版更改-shw

**内容**：**向量化与执行路径增强**——多因子 **CSE**、可选 **Modin** / **Polars Lazy**、**Numba** 滑动均值可选路径、**perf** 环境变量与剖析/对比脚本；不改动 DSL 白名单语义。

- **新增** [`planner/plan_hash.py`](../planner/plan_hash.py)、[`planner/cse.py`](../planner/cse.py)：结构化哈希与 `apply_cse`；[`runtime/engine.py`](../runtime/engine.py)：`compile_many` / `_dag_from_factors` / `run_many` / `run_many_parallel`。
- **新增** [`backend/pandas_compat.py`](../backend/pandas_compat.py)：`FACTOR_ENGINE_USE_MODIN` 与 **`build_backend("pandas_modin")`**；[`backend/factory.py`](../backend/factory.py) 别名 **`pandas_modin`**、**`polars_lazy`**。
- **扩展** [`backend/polars_backend.py`](../backend/polars_backend.py)：更多算子与 **`PolarsBackend(use_lazy=True)`** / **`FACTOR_ENGINE_POLARS_LAZY`**。
- **新增** [`backend/numba_kernels.py`](../backend/numba_kernels.py)（可选）、[`runtime/perf_config.py`](../runtime/perf_config.py)；脚本 [`scripts/profile_pandas_backend.py`](../scripts/profile_pandas_backend.py)、[`scripts/bench_pandas_vs_modin.py`](../scripts/bench_pandas_vs_modin.py)。
- **依赖** [`pyproject.toml`](../pyproject.toml)：`[modin]` optional extra；pytest marker **`modin`**。
- **测试** [`tests/test_cse_run_many.py`](../tests/test_cse_run_many.py)、[`tests/test_pandas_compat.py`](../tests/test_pandas_compat.py)；**更新** [`README.md`](../README.md) 第 23 版摘要与项目结构树。

---

## 第 24 版更改-shw

**内容**：新增 **Backtrader 单标回测子系统**，冻结研究员 C→D 的 `target_position` 对接协议（权重 `[-1,1]`、缺失 `ffill`），并统一回测输出 schema。

- **新增目录** [`backtest/`](../../../backtest_layer/single_asset_backtest/)：
  - [`config.py`](../../../backtest_layer/single_asset_backtest/config.py)：`BacktestConfig`（`initial_cash` / `commission` / `slippage_perc` / `rebalance_threshold` / `enforce_target_bounds`）。
  - [`contracts.py`](../../../backtest_layer/single_asset_backtest/contracts.py)：`target_position` 校验与标准化、时间索引约束、`ffill` 与边界处理。
  - [`strategy.py`](../../../backtest_layer/single_asset_backtest/strategy.py)：消费目标仓位并调仓的策略轨迹记录。
  - [`runner.py`](../../../backtest_layer/single_asset_backtest/runner.py)：`run_single_asset_backtest(...)` 执行入口（Backtrader `order_target_percent`）。
  - [`report.py`](../../../backtest_layer/single_asset_backtest/report.py)：统一输出 `returns` / `metrics` / `summary`。
  - [`io.py`](../../../backtest_layer/single_asset_backtest/io.py)：CSV/Parquet 输入加载与规范化。
- **依赖**：[`pyproject.toml`](../pyproject.toml) 新增 optional extra **`backtest = ["backtrader>=1.9.78.123"]`**。
- **示例**：新增 [`examples/backtest_single_asset.py`](../../../backtest_layer/examples/backtest_single_asset.py)（现位于 monorepo `backtest_layer/examples/`）。
- **测试**：新增 [`tests/test_backtest_contracts.py`](../../../backtest_layer/tests/test_backtest_contracts.py)、[`tests/test_backtest_single_asset.py`](../../../backtest_layer/tests/test_backtest_single_asset.py)、[`tests/test_backtest_report_schema.py`](../../../backtest_layer/tests/test_backtest_report_schema.py)（现位于 monorepo `backtest_layer/tests/`）。
- **ADR**：新增 [`docs/adr_backtest_target_position.md`](adr_backtest_target_position.md) 冻结接口口径。
- **文档**：[`README.md`](../README.md) 增补第 24 版摘要、可选依赖 `backtest` 说明与结构树条目。


## 第 25 版更改-shw

**内容**：回测模块从 MVP 升级为 **D-3 策略可追踪 + 分层指标体系**，保持 D-1/D-2 协议兼容。

- **D-3 策略入库**：新增 [`backtest/strategy_registry.py`](../../../backtest_layer/single_asset_backtest/strategy_registry.py)（`StrategySpec` / `StrategyRegistry`）与 [`backtest/strategy_library.py`](../../../backtest_layer/single_asset_backtest/strategy_library.py)（内置 `target_position@1.0`）。
- **执行入口扩展**：[`backtest/runner.py`](../../../backtest_layer/single_asset_backtest/runner.py) 支持 `strategy_name` / `strategy_version` / `strategy_params`；回测摘要新增 `strategy_name`、`strategy_version`、`strategy_params`、`strategy_instance_id`。
- **指标工业化**：新增 [`backtest/metrics.py`](../../../backtest_layer/single_asset_backtest/metrics.py)，[`backtest/config.py`](../../../backtest_layer/single_asset_backtest/config.py) 增加 `metrics_profile`（`core`/`standard`/`industrial`）；[`backtest/report.py`](../../../backtest_layer/single_asset_backtest/report.py) 接入分层指标计算。
- **对外导出**：[`backtest/__init__.py`](../../../backtest_layer/single_asset_backtest/__init__.py) 导出 `StrategyRegistry` / `StrategySpec` / `build_strategy_registry`。
- **测试**：新增 [`tests/test_backtest_strategy_registry.py`](../../../backtest_layer/tests/test_backtest_strategy_registry.py)、[`tests/test_backtest_metrics_extended.py`](../../../backtest_layer/tests/test_backtest_metrics_extended.py)；更新 [`tests/test_backtest_single_asset.py`](../../../backtest_layer/tests/test_backtest_single_asset.py)、[`tests/test_backtest_report_schema.py`](../../../backtest_layer/tests/test_backtest_report_schema.py)。
- **示例/文档**：更新 [`examples/backtest_single_asset.py`](../../../backtest_layer/examples/backtest_single_asset.py)、[`README.md`](../README.md)、[`docs/adr_backtest_target_position.md`](adr_backtest_target_position.md)。

## 第 26 版更改-shw

**内容**：回测 **防前视配置**、**指纹语义** 与 **README / ADR** 对齐。

- **配置**：[`backtest/config.py`](../../../backtest_layer/single_asset_backtest/config.py) 新增 `target_lag_bars`（单资产，默认 `0`）、`portfolio_weight_lag_bars`（多资产，默认 `1`，禁止 `0`）。
- **执行**：[`backtest/runner.py`](../../../backtest_layer/single_asset_backtest/runner.py) 单资产在对齐目标后应用 `shift(target_lag_bars)`；多资产用可配置滞后替代写死的 `shift(1)`；`data_fingerprint` 与滞后后的有效输入一致。
- **ADR**：[`docs/adr_backtest_target_position.md`](adr_backtest_target_position.md) 新增 **§13**（信号时间语义、基准与 ffill）。
- **文档**：[`README.md`](../README.md) 增补第 26 版摘要、回测章节（ADR 链接、`target_lag_bars` / `portfolio_weight_lag_bars`、`data_fingerprint` 说明、`PYTHONPATH` + 回测 pytest 示例）。
- **测试**：更新 [`tests/test_backtest_single_asset.py`](../../../backtest_layer/tests/test_backtest_single_asset.py)、[`tests/test_backtest_multi_asset.py`](../../../backtest_layer/tests/test_backtest_multi_asset.py)。

---

## 第 27 版更改-shw

**内容**：为各顶层包目录补充 **`README.md`**（`api/`、`api/operators/`、`backend/`、`expr/`、`ir/`、`planner/`、`runtime/`、`storage/`、`scripts/`、`tests/`、`examples/`、`examples/configs/`、`docs/`），并在 [`docs/README.md`](README.md) 建立文档索引；根目录 [`README.md`](../README.md) 文档索引增加指向。

## 第 28 版更改-shw

**内容**：**大幅扩写**各包 `README.md`（增加架构图、文件逐项说明、数据流、环境变量/契约、测试与延伸阅读）；新增 [`docs/_refs/README.md`](_refs/README.md)；[`backtest/README.md`](../../../backtest_layer/single_asset_backtest/README.md) 增加 **§0**（与主因子链路关系 + 章节导航）。根目录 [`README.md`](../README.md) 增加 **第 28 版** 摘要。

---

*若后续继续迭代算子实现，请在本文件追加「第 31 版更改-shw」及之后条目。*

---

## 第 31 版更改-shw

**内容**：**`cleaned_operators` 全量接入** — 删除 **`api/operators/`** 与强类型 **`expr/*`**；`PandasBackend` 委托 **`cleaned_bridge`**；`build_dsl_allowlist()` 仅收录 cleaned 已注册算子；**`STUB_IR_OPS` 恒为空**。

- **新增 / 主路径**：[`cleaned_operators/`](../cleaned_operators/)、[`backend/cleaned_bridge.py`](../backend/cleaned_bridge.py)、[`api/cleaned_ops.py`](../api/cleaned_ops.py)
- **删除**：`api/operators/` 整包；`expr/arithmetic.py` 等强类型节点；`runtime/registry.py`
- **测试**：[`tests/test_cleaned_operators_comprehensive.py`](../tests/test_cleaned_operators_comprehensive.py)、[`tests/test_cleaned_integration.py`](../tests/test_cleaned_integration.py)
- **文档**：[`api/README.md`](../api/README.md)、[`cleaned_operators/README.md`](../cleaned_operators/README.md)、[`operators_semantics.md`](operators_semantics.md) 第 31 版节

---

## 挖掘对接与路径优化（2026-06，非「第 N 版更改-shw」编号）

**内容**：挖掘投递校验、canonical 字段修复、路径 `~/quant_projects`、文档统一。索引见 [`docs/README.md`](README.md)。

- **新增**：[`api/mining_integration.py`](../api/mining_integration.py)、[`workspace_paths.py`](../workspace_paths.py)、投递校验 CLI、[`docs/dsl_operators_reference.md`](dsl_operators_reference.md)
- **修复**：`build_canonical_fields.py` US OHLCV `forbidden` 误判；`cleaned_bridge` TypeError 重试；`ParquetSource` 列名回退
- **规范**：[`docs/miner_delivery_spec.md`](miner_delivery_spec.md) v2.14（组员必读唯一正文）

---

## 第 40 版更改-shw（2026-07）

**内容**：**DuckDB SQL 下推 Tier4（77 → 88 算子）**。

- 新增：`WMA` · `ewm_std`/`ewm_var` · `cum_std`/`expanding_std` · `floor`/`ceil`/`inverse` · `count` · `Slope` · `ts_argmax`/`ts_argmin`
- `test_sql_pushdown_tier4`；CI `min-sql` 95

---

## 第 39 版更改-shw（2026-07）

**内容**：**DuckDB SQL 下推 Tier3 扩展（58 → 77 算子）**。

- 新增 SQL 下推：`power` · 比较 `gt/lt/eq/ge/le/ne` · 逻辑 `and_/or_/not_` · `group_std` · `ts_cov` · `ts_quantile` · `ts_product` · `ts_skew` · `ts_regression` · `cum_sum/cum_max/cum_min`
- `test_sql_pushdown_tier3`；CI `min-sql` 77；`sql_pushdown_coverage.md` 自动更新

---

## 第 38 版更改-shw（2026-07）

**内容**：**底层栈默认启用 hybrid（DuckDB + auto）**。

- YAML 未指定 `backend` 时默认 **`auto`**（SQL 下推 + Polars + Pandas fallback）
- `prod` profile 显式 `backend: auto`
- 投递 / smoke 模板改 `auto`；新增 `data_access_auto_smoke` / `data_access_duckdb_sql_smoke` 示例
- `test_backend_stack_defaults`；README「底层栈与 backend 选择」矩阵

---

## 第 37 版更改-shw（2026-07）

**内容**：**Phase 18 Pipeline 覆盖与增量批量**。

- `PipelineConfigOverrides`：CLI `--dq-check` / `--write-target` / 增量参数等统一传入 engine batch
- `config_run_batch_key` / `config_materialize_batch_key` 支持 `pipeline=` 分组
- `materialize_many_from_config_parallel`；batch 物化路径支持 `run_many_parallel`
- `materialize_incremental_many_from_config`；`run_config_directory --incremental` 目录 batch
- `write_targets.LocalParquet` lazy import 解循环依赖
- `test_enterprise_p6`；`test_pipeline_batch` 扩展 CLI 覆盖 batch 用例

---

## 第 36 版更改-shw（2026-07）

**内容**：**Phase 17 pipeline 收敛 + allowlist 修复**。

- `.data_access_allowlist.yaml` 登记 `schema_migration.py`（CI allowlist 门禁通过）
- `pipeline._execute_config` / `materialize_incremental_from_config` 统一 `to_*_materialize_kwargs`
- `run_config_directory` 在 eligible 时委托 `run_many_from_config` / `materialize_many_from_config`
- `ResolvedRunKwargs.to_run_kwargs`；batch API 支持 `profile` / `enable_cse`
- `test_pipeline_batch`；795 factor_engine 测通过

---

## 第 35 版更改-shw（2026-07）

**内容**：**Phase 16 物化批量与配置映射**。

- `ResolvedMaterializeKwargs.to_engine_materialize_kwargs` / `to_run_kwargs`；修复 `resume_materialize` / `isolate_partition_failures` 传参遗漏
- `materialize_many_from_config(batch_run=True)`：同 scope 共享 `run_many` + `execute_materialize_from_resolved`
- `run_many_from_config(parallel=True)` + `run_many_from_config_parallel`
- `test_enterprise_p5`；789+ 测通过

---

## 第 34 版更改-shw（2026-07）

**内容**：**Phase 15 批量编排与写目标收敛**。

- `config_run_batch_key` + `run_many_from_config` 按 run kwargs 子分组（production auto_warmup 可 batch）
- `ClickHouseWriteTarget.write_factor_series`；`dual_write_service` 统一经 write target
- `storage` 公开 export `resolve_write_target` / Local / Staging / CH targets
- `test_enterprise_p4`；CI `check_enterprise_docs`

---

## 第 33 版更改-shw（2026-07）

**内容**：**企业级 P0/P1/P2 加固** — `factor_engine` + `data_access` 生产路径。

- **data_access**：`QueryBudget` + read/sql/stream 审计；`delete_rows` dry_run/max_rows；`factor_lake` metadata schema；`sql_stream` + `apply_sql_row_limit`
- **factor_engine**：`factor_schema` 单点契约；production `auto_warmup`；lineage `source_expr`；`dual_write_service` / `warmup_service` / `lineage_service` / `materialize_service`；`SessionBarCalendar` 分钟 warmup；`FactorWriteTarget`（local/staging/CH）；`run_many_from_config` 按 data_scope 分组；`materialize_many_from_config`；`schema_migration` + CLI；composite `join_reports` → lineage
- **测试**：`test_enterprise_p0/p1/p2/p3`；782+ factor_engine / 168 data_access unit 通过
- **文档**：企业级路线图（已删除，见 changelog 2026-07 文档清理）；[`dataaccess/README.md`](../../dataaccess/README.md) PR8+
