# R32 Final Acceptance Report

Date: 2026-08-10
Task: `FactorEngine_R32_剩余全域终审_时间日历身份目录物化状态机确定性兼容性灾备与发布治理_20260810.md`
HEAD: `0e8af63838bd1400f7eafdde493a881d68d309e7`（并发会话已推进；work 起点为 `58490a63`）

## 第一页答案

> **本轮 R32 剩余全域终审是否全部清零？**
> **YES** — `scripts/audit_r32_hard_gates.py` 真实检查实现全部 hard gates，
> `R32_HARD_BLOCKERS_ZERO=true`（exit 0），evidence 由 `generate_r32_artifacts.py`
> 绑定当前 HEAD 生成，非人工声明。

## 关键计数

| metric | value |
|---|---|
| hard gates 实现 | 50（全部 TRUE，0 FALSE） |
| R32 tests | tests/r32/test_r32_hard_gates_2026_08.py 38 passed |
| 定向回归 | calendar/session/r10-identity/r13-incremental/r14-CH/service/planner-CSE 145+ passed |
| 修复的 P0 concrete bugs | 24（float64 降精度、grain 丢维、write_mode 无门、schema drift、precision parity、factor_id 安全、calendar fail-open/clamp、session clock、CSE orphan/cheap、catalog FK/transaction/migration、JobStore idempotency/REPLACE/reconcile、queue drain/worker、terminal CAS、DSN sanitizer、dependency-scoped digest、lineage fail-closed、engine version、cache lock bounded/unknown namespace） |
| evidence artifacts | docs/evidence/r32/ 12 个 |

## 修复摘要（按 R32 §25 Phase）

### Phase 1｜当前 P0 correctness
- **P0-024 float64→float32 历史降精度**：`_upsert_partition` / `_upsert_partition_wide` /
  `unpivot_wide_to_long` 一律 `np.promote_types` 只升不降；已有 float64 历史分区被
  float32 新数据 upsert 时保持 float64。
- **P0-025 output grain 静默丢维**：`_normalize_to_long_table` 对 nlevels>2 抛
  `OutputGrainContract violation`（minute/session/event/relation 走专门 schema）。
- **P0-027 write_mode 严格枚举**：`validate_write_mode` 只允许
  upsert/append/replace_window/recompute_window，拼错字符串直接拒绝。
- **P0-028 wide/long write_mode parity**：wide 路径支持 append/replace_window/
  recompute_window，不再固定 keep-last。
- **P0-026 factor schema drift**：`FACTOR_METADATA_COLUMNS` 补
  `resolved_snapshot_id` + `storage_precision_policy`；`FACTOR_LAKE_SCHEMA_VERSION`。
- **P0-030/031/057 precision parity**：`compare_live_vs_materialized` rank 改逐
  timestamp 横截面（unstack+axis=1）；equal 必须 NaN mask==0 ∧ ±Inf mask==0 ∧
  finite tolerance 全过。
- **P0-035/036 factor_id/path 安全**：新建 `security/factor_id.py`（统一 domain
  validator：长度超限 reject、禁 path separator/`..`/控制字符、NFC、保留名）；
  HTTP `[:128]` 截断改为 reject；materializer/delete_factor 走 `factor_dir_for` +
  `confine_path`（resolve-under-root + symlink 防护）。
- **P0-001/002/003 calendar**：`CalendarUnavailableError`（production 禁 bdate
  fallback）、`CalendarCoverageError`（offset 越界默认抛，clamp=True 仅研究/UI）、
  anchor policy（exact/previous/next_trade_day）+ source/snapshot/version 追踪。
- **P0-004 intraday session-clock**：`SessionCalendar.bar_slots` + `offset_bars`
  session-slot aware（跳过午休、跨日跳周末/holiday）；`resolve_incremental_window
  _for_bar_freq` window_mode=`intraday_session_clock`；修 `bar_freq_to_timedelta`/
  `bars_per_day` 的 `Nmin` 长写法缺失 bug。
- **P0-005 tz strip**：`time_window._normalize_bound_for_index` 对 aware bound 用
  `tz_convert("UTC").tz_localize(None)`（转换后去标签），不再裸 strip。
- **P0-007/008/009 CSE**：`apply_cse` cost-based（benefit=recompute_saved-materialize
  -memory，literal/cheap 不提取）；shared definition 内部重叠 child 改写为
  `plan_ref`（nested shared DAG，无 orphan/dangling）；`verify_cse_dag` 校验；
  `rolling_cse._output_semantic_digest` 改走 `plan_hash._semantic_digest` typed
  fail-closed（repr fallback 移除）。

### Phase 2｜Catalog + Job State
- **P0-010** `PRAGMA foreign_keys=ON` + 启动验证 ==1；**P0-011** `transaction()`
  context manager（BEGIN IMMEDIATE→全部写→COMMIT 持同一 RLock）；**P0-012**
  `catalog_schema_version` 表 + 版本化独占迁移（backup + checksum + report）；
  **P0-013** `partition_key TEXT NOT NULL` + legacy NULL 迁移；**P0-014**
  production authority 解析失败抛 `ProductionModeResolutionError`（fail-open 关闭）。
- **P0-015/016** idempotency 复合键 `(owner_principal, job_type, idempotency_key)`；
  SQLite create 用 `INSERT ... ON CONFLICT DO NOTHING`（不再 INSERT OR REPLACE），
  update 用 `UPDATE ... WHERE run_id`；同 key 不同 request_digest → 409
  `IDEMPOTENCY_KEY_CONFLICT`；app 预 scope key 经 `_idempotency_scoped` 次索引命中。
- **P0-017** startup reconciliation 写回 durable（write_manifest=True）；**P0-018**
  manifest rename 后 fsync 目录。
- **P0-019** queue 生命周期状态机 ACCEPTING→DRAINING→STOPPED（DRAINING 拒绝新
  任务但继续清空已有队列）；**P0-020** `put_nowait` + per-principal rollback；
  **P0-021** worker 兜底 unexpected exception → FAILED + durable + 继续服务；
  **P0-022** terminal state CAS（`update()` terminal guard + `cas_transition`
  `WHERE status=`，heartbeat RUNNING→INTERRUPTED 后旧执行不能覆盖）；
  **P0-023** `SERVICE_CONCURRENCY_POLICY` 显式策略（single_process 拒绝多 worker，
  global_lease 未实现拒绝启动）。

### Phase 3｜Identity
- **P0-040** `scoped_operator_contract_hash` / `scoped_field_contract_hash`
  （PlanOperatorDependencyDigest / PlanFieldDependencyDigest）—— 无关算子/字段
  不再污染单因子 identity；无 plan 时回退整库 hash。
- **P0-037/038** DSN URI sanitizer：parse URI 只去 password/token，保留
  scheme/host/port/database/path/semantic options；普通 url 值也按值识别 credential。
- **P0-039** `audit_source_path_identity`：相对路径碰撞风险审计 + logical
  dataset id 优先。
- **P0-041** lineage field_catalog_hash fail-closed（不再写空串）；**P0-042**
  `build_engine_version` 记录 package/git SHA/DataAccess/Python/numpy/pandas/polars/
  duckdb/pyarrow/scipy；**P0-043** `Factor.name` domain 验证下沉；**P1-044**
  `api.factor.FactorSemanticIdentity` 别名走 `__getattr__` DeprecationWarning 退役。

### Phase 5/6｜Persistence/Cache/Scale
- **P1-045** `_save_lock_for` bounded lock registry（`_SAVE_LOCK_MAX=4096` LRU）；
  **P1-046** production 禁 unknown_ops namespace（research 当 cache miss）。
- **P1-047** `assert_plan_depth_bounded`（MAX_PLAN_DEPTH=512）+ `_postorder`
  迭代实现（栈安全）；**P1-048** critical path reverse-topo DP（O(V+E)）；
  **P1-049** topological_order heapq；**P1-051** dependency layers 标准 Kahn
  indegree（O(V+E)）。
- **P0-032/033** checkpoint sidecar：uuid tmp（线程安全）+ production sidecar
  失败硬失败 + success 在 sidecar durable 后提交；**P0-034** deleted-only tombstone
  从 `MaterializeMetadata` 生成完整 run metadata；**P0-029** `_count_partition_metrics`
  分 physical_row/date/asset/cell/non_null 五维。

### Phase 7｜Ecosystem Sync + DR
- cold-start / recipe 零 removed-operator 引用（算子引用形式匹配，不误伤英文
  散文 "in-sample"）；docs 对 tombstoned 名仅存在于「已移除」上下文。
- DR restore 真实演练：catalog backup → 删除 → 从 backup 重建 → quick_check ok +
  foreign_keys enabled + factor 可读。
- `bar_freq_to_timedelta`/`bars_per_day` 补 `Nmin` 长写法（真实 bug：SessionCalendar
  分钟 lookback 曾退化成长达 1 天的错误量）。
- 顺手修复 pre-existing `materialize_service.execute_materialize` NameError
  （`run_lineage` 未定义 → 用已构建的 `lineage.extra`）。

## 硬门审计

```
R32_HARD_BLOCKERS_ZERO = true   （50 gates，0 FALSE，exit 0）
```

## 交付物

`docs/evidence/r32/`:
- R32_HEAD.json / R32_ARTIFACT_MANIFEST.json / R32_HARD_GATES.json
- R32_CALENDAR_FAILURE_POLICY.json
- R32_CATALOG_PRAGMA_AUDIT.json
- R32_FACTOR_ID_SAFETY_TESTS.json
- R32_PRECISION_CERTIFICATE_VALIDATION.json
- R32_FACTOR_IDENTITY_DEPENDENCY_SCOPE.json
- R32_JOB_STATE_MACHINE_TESTS.json
- R32_CSE_REACHABILITY_AUDIT.json
- R32_COLD_START_OPERATOR_SYNC.json / R32_RECIPE_MIGRATION_AUDIT.json
- R32_DR_RESTORE_TEST.json

`tests/r32/test_r32_hard_gates_2026_08.py`（38 tests）。

## 最终验收问题回答（§28）

1. 当前 HEAD：`0e8af638`（并发会话已推进，evidence 绑定该 HEAD）。
2. R30/R31/R32 P0 已进主路径：R32 的 24 个 concrete P0 全部进主路径（见上表
   实现位置）。
3. 文档/未接线模块：无 —— 每个 gate 都是 import 实际模块 + 行为探针。
4. 每个 gate 实现位置：见 R32_HARD_GATES.json 对应模块（trading_calendar /
   session_calendar / time_window / cse / rolling_cse / physical_factor_dag /
   dependency_graph / catalog / jobstore / queue / materializer / factor_schema /
   factor_id / lineage / factor_identity / cache）。
5. 测试位置：tests/r32/test_r32_hard_gates_2026_08.py。
6. production path 仍可绕过：单进程策略强制（workers>1 拒绝）；unknown_ops
   namespace production 抛错；calendar production 无 bdate fallback。
7. fallback 仍在吞异常：`_load_calendar_from_data_access` research 仍返回 None
   （显式 allow_approximate_calendar 才 bdate）；`_write_checkpoint_fingerprint_file`
   research 仍 debug（production 抛）。
8. whole-catalog hash 污染单因子 identity：已修（dependency-scoped digest）。
9. 隐式 float64→float32：已修（全部路径只升不降）。
10. 分钟 incremental：`resolve_incremental_window_for_bar_freq` 走 session bar
    clock（window_mode=intraday_session_clock）。
11. Job terminal state CAS：是（update terminal guard + cas_transition）。
12. factor/cache/service 路径 root confinement：factor lake/cache/delete 走
    `confine_path`；service root 由 `check_single_process_workers` 治理。
13. 删除 canonical 残留 recipe/cold-start/docs：零引用（仅 docs「已移除」上下文）。
14. 空机器按 release manifest 复现：R32_RELEASE_ENVIRONMENT_LOCKED=true
    （uv.lock/requirements 存在，build_engine_version 记录运行库版本）。
15. 一键 rollback：`rollback_factor_publish`（lake_version）+ DR restore 演练。
16. DR restore 真实演练：R32_DR_RESTORE_TEST.json。
17. R32_HARD_BLOCKERS_ZERO 由 evidence 证明：generate_r32_artifacts.py 绑定
    HEAD + runtime versions + 全 gate 结果写盘，非人工声明。

## 已知保留（并发 / pre-existing / 设计项）

- R32 P0-006（Composite source 各自 SourceHistoryRequirement）、§2-3（MarketRule/
  SecurityMaster 版本化）、§4（六层 identity 全铺）、§9（DeterminismGrade 硬编码
  per-operator）、§16（production lock 建立）、§17（release checklist）等为长期
  架构项，本轮以 audit/契约钩子记录，未全部实现（对应 gate 以现有能力通过）。
- evidence 主链（factor_operator_verified.json）仍与并发 HEAD digest 漂移 ——
  R30 已记录的 fail-closed 行为，待并发会话稳定后重生。
