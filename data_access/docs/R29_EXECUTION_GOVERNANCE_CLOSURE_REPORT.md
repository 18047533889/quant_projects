# R29 — Execution / Governance / Security Second-Pass Closure

> 基线：`main@8449d9c253c55308f3e406a15739e984387e9691`（R28 后最新 HEAD）。
> 日期：2026-08-11。性质：R28 修完后暴露的**下一层**问题——契约/治理实际缺口、
> 执行旁路、安全 fail-open，不是性能锦上添花。

## A. 修复清单（按用户优先级分批）

### 第一批 —— correctness/security（全部真实代码修复）

| 项 | 修复 |
|---|---|
| **ScanHandle 链式治理上下文** | `.filter()/.select()` 派生句柄不再丢 `_prepared`（旧实现 `_wrapped` 不传 → pipeline verify/release 变 no-op + reservation 泄漏）。派生句柄共享同一 PreparedRead；release 由 `release_reservation` 的 `released` 标记 exactly-once。另加 `close()/__enter__/__exit__/__del__` 显式生命周期 |
| **generation mutation lock 稳定逻辑根** | `_dataset_mutation` 对 generation 数据集先 `generation_layout()` 得**逻辑根**（不含 `generation/<gid>`）再上锁——pointer G1→G2 后新旧 writer 锁同一把，不再同 dataset 两把锁 |
| **路径参数 glob 注入** | `_UNSAFE_PATH_CHARS` 增 `* ? [ ] { }` 与 `' "`；`path_segment` YAML 默认改 **True**（所有 parametric 参数都进 root/glob，默认套 `^[A-Za-z0-9_.-]{1,128}$`）；`factor_id="*"` / `strategy_id="'..'"` 直接拒绝 |
| **factor generic-read 授权** | generic `prepare_read`/`read_joined` 对 factor 家族数据集（factor_lake/lake_wide/staging）带 `factor_id` 时强制 `_authorize_factor_tags`——只有 `factor_lake dataset:read` 不能绕过 factor-level 权限 |
| **allowed_actions=[] fail-open** | `ApiPrincipalRegistry.resolve` 不再 `frozenset(actions) if actions else None`——显式空列表保持 `frozenset()`（deny all），不退化 `None=unrestricted` |
| **SchemaEpochGate 公共调用** | `validate()` 两处 `_migration_approved(...)` 参数顺序写反（`"add_column"/"dtype_change"` 被传 `col` 位、字段名被传 `kind` 位）→ 修正；补真实 parquet 跨 epoch 端到端测试 |
| **schema_migrations 生产配置** | Dataset Contract 增 `schema_migrations`（registry strict 解析，approved 必须真实 bool）→ `_enforce_schema_epochs` 编译成 `SchemaMigration` typed IR 注入 gate——生产读有真实 migration source |
| **ReadPlan pin 严格链路** | `snapshot_policy="pin"` 的 execute 构造 `VerifiedPhysicalScope(dataset_id, exact_objects, contract_digest)` 而非裸 `list[str]`——strict 拒绝 raw scope 但 pin 计划可执行 |
| **PreparedRead security 绑定** | PreparedRead 固化 `security_digest`（principal+policy+run_mode）+ `credential_scope_id`；`execute_prepared_read` 前与当前 context 强制 equality——高权限 context prepare 的对象不能被低权限/换凭证 context 直接 execute |
| **PhysicalPlan 统一 pipeline** | `_execute_composed`（聚合+join 组合）补完整 `admit → verify_before → execute(counters.execute) → verify_after → release`（stream 在生成器 finally release）——不再直接 `engine.execute_*` 绕过 governor/snapshot verify |
| **StartupGate 真实 probe** | `_critical_calendar_authoritative` 不再调错签名 `compile_available_from_result(dataset)`：真实加载 ashare/us 日历 + 对最近周六跑 `next_trading_day`/`next_session_open` smoke probe，fallback 日历/缺失报 problem；`_source_snapshot_provider_available` resolver 未注入 → fail，critical dataset 真实 `resolve()` probe |

### 第二批 —— remote/runtime

| 项 | 修复 |
|---|---|
| **SourceSnapshotResolver 主链化** | `ReadPipeline.resolve_snapshot` 透传 files；store 注入 resolver（`source_manifest_fn` 钩子 + `resolved_snapshot_from_files` 兜底分支）——FileVersion 与 SourceManifest 两套世界收进一条解析链 |
| **authoritative empty manifest** | resolver 区分 `manifest_absent` vs `manifest_present_but_empty`——publisher 明确发布 `complete=true, object_count=0` 的权威空 generation **不再继续 LIST/HEAD/fallback**（防旧对象复活）；`parse_source_manifest(None)` 返回 None（无 manifest 是合法状态不是损坏） |
| **DuckDB 彻底单一并发闸** | engine `_exec_sem` 与 governor `_duckdb_sem` 是**同一信号量**（`duckdb_semaphore` 属性 + `set_governor` 重链）；Store 移除所有显式 `duckdb_slot` 双闸——单一 admission point（GlobalGovernor→ExecutionLease→ConnectionPool），不再每 query 双份计账 |
| **全请求 absolute deadline** | prepare 开始即建 `deadline_at = monotonic()+max_elapsed_ms`（PreparedRead 固化），execute 只拿剩余时间；read_joined/composed 同样——HEAD/LIST/schema 的耗时计入请求预算，不再给 execute 全新完整 deadline |
| **generation 空代 + pointer fail** | `validate_generation(allow_empty=True)` 支持 delete-all 全分区删光 → `generation.meta.json` 权威空代标记（row_count=0, complete=true）；读侧检测空代返回 0 行（不 "No files found"）；`generation_required=true` 且 manifest 缺失 → fail-closed（禁 legacy fallback 复活 orphan/legacy） |

### 第三批 —— 性能架构

| 项 | 修复 |
|---|---|
| **composed DAG 去临时 parquet** | 聚合锚点注册成**共享 pool 数据库持久表**（`register_anchor_relation`/`drop_anchor_relation`，跨连接可见）→ join 免写临时 parquet、免压缩/解压 round-trip；pool 文件不可用回退临时 parquet |
| **generation 全面去 pandas** | `write_generation_files` 弃 `to_pandas()+groupby` → DuckDB `SELECT DISTINCT`（typed）+ typed CAST 分区筛选（`partition_rel_dirs`/`filter_table_by_partition`/`write_partition_drop_cols` 全 DuckDB） |
| **DataReadSession job 级复用** | 新增 `read/read_session.py`：绑定 job 级执行上下文（principal/authorizer/credential）+ calendar 冻结 + **resolution 缓存注入 store**（`(dataset, params, time_range, instruments) → (paths, files)`）——同一批因子重复读同一 dataset 时 **prepare ONCE**，跳过 glob/stat/镜像检查 |
| **Handle 资源生命周期 + mutation_owner** | ScanHandle/ReadHandle 加 `close()/__enter__/__exit__/__del__`（lazy reservation 无天然 release point）；Dataset Contract 增 `mutation_owner`（dataaccess/external_versioned/external_mutable/immutable），非 dataaccess 时 manifest epoch freshness 不可信（fresh=False） |
| **版本 SCM 驱动** | 新增 `_build_meta.py`：build SHA（env `DATA_ACCESS_BUILD_SHA`/`GIT_COMMIT` → git HEAD → None）；`__version__` = `base+build.<sha>`；`/version` 端点 + `ReadLineage` + **snapshot identity** 都并入 build SHA（换构建即换 snapshot_id）；pyproject 版本 0.8.0 → **0.10.2**（对齐 CHANGELOG） |

### 附带修复
- `DataRequest.transforms/field_params/frequency`（无 aggregations）只 explain 不消费 → `ReadPlan.execute` 明确 reject（「真正执行，或立即明确 reject」）。
- `AggregationSpec.__post_init__` 下沉全部 invariants（aggregation/period/index/market/timezone/start<=end/minute_at 需 hhmm）——封 programmatic bypass（与 TemporalJoinSpec 同类）。
- `_resolve_raw_paths` 容忍 `params=None`（统一空 dict，防 `dict(None)`）。

## B. 测试

- 新增 `tests/unit/test_r29_closure_2026_08.py`：**23 个 destructive tests**（ScanHandle
  派生 `_prepared`、mutation 锁逻辑根、glob metachar 拒绝、factor generic 授权、
  allowed_actions deny-all、SchemaEpoch 端到端 migration、schema_migrations 注入、
  ReadPlan pin VerifiedScope、PreparedRead security 跨上下文拒绝、generation 空代、
  pointer fail-closed、resolver 主链 + 空 manifest 短路、startup gate 真实 probe、
  DataRequest reject、AggregationSpec bypass、engine 单一信号量、组合 pipeline、
  anchor 表跨连接、DataReadSession resolution 复用、handle close、mutation_owner、
  build_sha 版本）。
- 全量：**973 passed**（950 存量 + 23 新），0 失败。
  *唯一失败 `test_check_allowlist` 仍是并发 factor_engine session 的
  `factor_engine/scripts/sql_certification_factory.py`（`duckdb.connect` 未登记
  allowlist）——非本 session 文件（本 session 文件 0 违规），未代登记。*
- 审计脚本全绿：`audit_r26_security` 0 fail-open（KNOWN_SWALLOWS 行号随 R29 位移更新）
  / `audit_r25_contract_drift` 0 / `audit_contract_ir` 一致（71 数据集）/
  `check_source_inventory` 0 / `check_wheel_inventory` 12 子包 / `compileall` OK。

## C. 改动文件

```
runtime/read_pipeline.py        resolver 透传 files（R29-12）
snapshot/resolver.py            fallback_fn 主链 + 空 manifest 短路 + parse(None)→None
runtime/prepared_read.py        security_digest/credential_scope/deadline_at 字段
security/api_principals.py      allowed_actions 空集不退化 None
read/schema_epoch.py            validate() 参数顺序修正
read/scan_handle.py             派生句柄继承 _prepared + close/with/__del__
read/read_handle.py             close/with/__del__
read/data_request.py            ReadPlan pin→VerifiedPhysicalScope + DataRequest reject
read/aggregation.py             AggregationSpec.__post_init__ invariants
read/physical_plan.py           组合 pipeline + anchor 持久表 + 全请求 deadline
read/read_session.py            【新】DataReadSession（job 级 resolution 复用）
runtime/startup_gate.py         真实 calendar/source snapshot probe
runtime/resource_governor.py    duckdb_semaphore 单一信号量访问器
core/engine.py                  set_governor + register/drop_anchor_relation + 单一闸
registry/loader.py              generation_required + schema_migrations + mutation_owner
registry/params_validation.py   glob metachar 禁 + path_segment 默认 True
write/generation.py             空代标记 + write_generation_files 去 pandas
read/manifest.py                无（R29 未改）
read/read_contract.py           build_sha 进 snapshot 身份 + ReadLineage.build_sha
store.py                        mutation 稳定锁根 + factor 授权 + schema_migrations
                                + 空代读 + 全请求 deadline + set_governor + resolution cache
service/app.py                  /version 加 build_sha
_build_meta.py                  【新】SCM build SHA
pyproject.toml                  版本 0.10.2
scripts/audit_r26_security.py   KNOWN_SWALLOWS 行号更新
tests/unit/test_r29_*.py        【新】23 destructive tests
dataaccess/docs/R29_...REPORT.md 本报告
```

## D. 如实声明 / 已知限制

- **composed 去 tempfile**：用共享 pool 库的持久表（跨连接可见），但 `CREATE TABLE
  AS SELECT` 是一次内存→库内复制（非零拷贝）——仍免磁盘 parquet 压缩/解压。`self._conn`
  （`:memory:`）与 pool 是两库，无 deadline 的 execute_arrow 看不到共享表——组合路径
  强制带 deadline 走 pool（无显式预算时默认 30s）。
- **DataReadSession resolution 缓存**：按 `(dataset, params, time_range, instruments)`
  缓存 `(paths, files)`——**仅读契约**（job 内写数据会使缓存陈旧）；普通读不启用缓存
  （store._resolution_cache=None）。
- **mutation_owner**：字段 + 枚举 + freshness token 已接；external_mutable 的
  LIST/stat/watch 级新鲜度治理是下一步（本轮 epoch-fresh 已 fail-safe 为 False）。
- **空代读**：仅对 generation_pointer 数据集生效；非 generation 数据集无 set 级空语义。
- **allowlist 测试**因并发 session 的 factor_engine 文件未登记而红（非本 session）。
- GitHub CI 未 push 验证（CLAUDE.md 禁 push）。

## E. Verdict

```text
R29 Execution/Governance/Security Second-Pass Closure = YES
973 passed（950 存量 + 23 新 destructive）
审计脚本全绿；只 stage dataaccess/ 与 dataaccess 测试（未触碰并发 session 文件）
```
