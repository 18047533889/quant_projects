# FactorEngine R14 — DataAccess ↔ FE 生产集成链收官（外部 AI 复查 5+1 项）

审计基线：`main@931a54a`（后续并发会话推到 `e2a0a28`；本报告基于最新工作树验证）。

外部 AI 复查结论：
> **DATAACCESS CORE FREEZE 成立**（未发现新的 Core 独立 P0/P1）；
> **FACTORENGINE ↔ DATAACCESS PRODUCTION INTEGRATION FREEZE 暂缓一轮**，
> 修掉 1–5、顺手收口第 6 项后再跑 destructive gate。

本报告记录 1–5 修复 + 第 6 项收口。只改 `factor_engine/` orchestration / storage 层，
**不碰 cleaned_operators / ir / scripts**（并发会话 R11 longtail 算子认证 & R13 审计域）。

---

## A. 逐项状态总表

| # | 项 | 状态 | 修复 |
|---|---|---|---|
| 1 | factor_matrix production publish 非全局 crash-atomic | **fixed** | 整个 `(universe, freq)` 当一个 **generation**：完整写入 `generation/<gid>/`（未触达分区 copy-on-write 硬链接），production 每分区读回 validate 后，**一次** `os.replace` 原子切 `manifest.json` 的 `generation` 指针。`load_matrix` 只读 `manifest.generation` 指向那一代。任何崩溃窗口 reader 只能看到完整 old 或完整 new generation，绝无半新半旧（旧实现「先写 manifest、再逐分区 os.replace」在切换一半时崩溃会 mixed）。GC 只保留当前 + 上一代（在途 reader 读完旧代文件再删）。`storage/materialize/factor_matrix_materializer.py` |
| 2 | matrix factor-version gate 仍是「可选功能」 | **fixed** | `matrix_service.execute_materialize_matrix`：先校验 matrix 声明的 `universe/frequency` 与每个 factor 实际执行 scope（`_scope_from_factor`，优先 `semantic_identity`）一致（任何模式拒绝错写路径）；再按真实执行 scope + analysis + source contract **自动算** `semantic_digest`（与 `execute_materialize` 同源 `build_materialize_lineage` + `_build_semantic_identity`）；production 下任何 factor 缺 version 直接拒绝，调用方手工版本与自动算出的不一致也拒绝。`runtime/matrix_service.py` |
| 3 | semantic identity 与执行 scope split-brain | **fixed** | `execute_materialize` 只算一次 canonical `_scope_from_factor(factor, data_source)`，把 `scope.frequency / universe_id / market / calendar_id / decision_time_policy` 同时喂给 `_build_semantic_identity`、`_build_full_factor_definition`（`decision_policy` 不再写死 `None`）与 materializer —— 执行 / cache scope / factor_version / full definition / 事件 rebuild 全部消费同一个 scope，杜绝「执行 5m、落库 1d」。`runtime/materialize_service.py` |
| 4 | ClickHouse 双写用 ast_hash → 跨存储 split-brain | **fixed** | orchestrator 只算一次 `canonical_factor_version = semantic_identity.identity_digest()[:16]`（缺省 ast_hash 前缀），显式传给 `dual_write_clickhouse(factor_version=...)`，ClickHouse 侧不再自行退化成 `ast_hash[:16]`；`data_snapshot_id` 改用 `effective_data_source_config`（与 Parquet/lineage 同源）。与并发会话 NEW-P0-59（统一 `semantic_identity_digest` full digest 派生）兼容。`runtime/materialize_service.py` / `runtime/reconcile/dual_write_service.py` |
| 5 | DataEventLedger 无顺序/幂等事务 | **fixed** | `last_committed_snapshot` 改取 **sequence 最大的 chain head**（不再返回第一个匹配记录——e1→e2→e3 时把 e1 当 head 会把合法 `snapshot_before=s2` 误判 stale）；`sequence` 参与乱序校验；文件损坏 fail loud（`DataEventLedgerCorruptionError`，仅尾部半行截断恢复）；append OSError 上抛不再吞；`begin/commit/reject` 在 `<path>.lock` flock 内 reload+append，并发 worker 同事件只有一个能 begin（`pending` 预留 → 另一 worker 得 `in_flight`）。partial failure 保持 `rejected`/`pending`，重跑同 `event_id` 幂等收敛。`runtime/incremental_scheduler.py` |
| 6 | Composite `execution_spec()` 伪造残缺 spec | **fixed（P1 收口）** | 任一 child 非 `DataSource` 或 `execution_spec()` 无法产生完整 dict → production 抛 `UnreconstructableDataSource`，research 返回 `None`——不再发明缺 `dataset` 的 `{"type":"data_access"}`。`_effective_data_source_config` 对该异常不再吞（production fail-closed 上抛）。`storage/sources/composite_source.py` / `storage/exceptions.py` / `runtime/materialize_service.py` |

## B. 新增回归测试

- `tests/storage/test_r14_matrix_generation_atomic_2026_08.py`（6）— reader 隔离
  半写 generation、manifest 单点切换、GC 上限 2 代、orphan 清理、research/production
  同模型、partial 跨代保留。
- `tests/runtime/test_r14_matrix_version_gate_2026_08.py`（8）— freq/universe scope
  冲突拒绝、semantic_identity.frequency 驱动 digest（1d vs 5m 必须不同）、research
  自动绑定、production 缺 digest 拒绝、调用方版本冲突拒绝、production 自动绑定通过。
- `tests/runtime/test_r14_data_event_ledger_2026_08.py`（7）— chain-head=最新
  committed（三事件链 s3）、sequence 选头、乱序拒绝、双 worker CAS（in_flight）、
  中段损坏 fail loud、尾部半行截断恢复、append OSError 上抛、reject 后重试。
- `tests/runtime/test_r14_dualwrite_composite_spec_2026_08.py`（6）— ClickHouse
  用 canonical factor_version（含 ast_hash 回退）、Composite production 抛
  UnreconstructableDataSource / research 返回 None / 全可序列化通过。

## C. 验证

- R14 新增 27 tests 全绿；存量回归全绿：
  `test_round13_factor_matrix_governance_2026_08.py` 17、`test_r11_orchestration_closure.py` 15、
  `test_materializer.py` 48、`test_round13_incremental_invalidation_2026_08.py` 10、
  `test_r10_data_event_scheduler.py` + composite 27、`test_phase21_platform.py` matrix roundtrip。
- 并发会话 R4-100 arity 审计 WIP 已收口（`parse_factor` 现可跑通），本批未触发解析阻塞。
- 协调注记：并发会话中途在 `incremental_scheduler.py` 引入 P0-40/P0-41（逻辑表解析不到
  物理 dataset 绝不 anchor-fallback：research 跳过、production 抛
  `DependencyDatasetResolutionError`）与 `ForwardImpactRequirement` 三态。R11 测试
  `test_edges_fallback_when_no_dataset` 一族已对齐到新政策（research 空 edges /
  production 抛错），避免与其实现矛盾。

## D. 与 DataAccess 的边界

本轮全部在 `factor_engine/` 完成，不涉及 `dataaccess/`。matrix 仍未接 DataAccess
staging→publish（保留 production 告警）——generation 原子切换已覆盖 crash-atomicity，
DataAccess 接入可作独立后续。

## E. 下一步（按外部 AI gate 清单）

- matrix crash after manifest / after 1/N partition publish（已由 #1 的 6 条 atomicity 测试覆盖）；
- normal production `materialize_matrix()` without manual factor_versions（#2 覆盖）；
- `Factor.semantic_identity != Factor legacy fields`（#2/#3 覆盖）；
- Parquet/Catalog/ClickHouse version parity（#4 覆盖）；
- 3-event s0→s1→s2→s3 chain、2-worker same-event CAS（#5 覆盖）；
- partial downstream failure 零 mixed published generation（#5 保持 rejected + 重跑收敛，
  跨 factor parquet 全量回滚不在本批，记录在案）。

全部通过后即可宣布 **DATAACCESS CORE FREEZE + FACTORENGINE INTEGRATION FREEZE**。

---

# 附录 F：第二轮（外部 AI 复查 HEAD 64ad308）4 P0 + 1 条件 P0 + 1 P1 收口

第二轮复查结论：**DATAACCESS CORE FREEZE 继续成立**；Integration Freeze 还差
4 个确定性 integration correctness + 1 个条件 P0（DataEvent 原子性）+ 1 个 P1
（pending lease）。本轮全部收口，**本轮修改了 dataaccess/ 读侧**（matrix 接
generation 指针）。

| # | 项 | 状态 | 修复 |
|---|---|---|---|
| 1 | DataAccess factor_matrix 不跟随 FE generation 指针 | **fixed** | `ParametricDataset.generation_pointer` + `current_generation()`：`resolve_paths` 只解析 `manifest.json` 的 `generation` 指向那一代（`generation/<gid>/<glob>`），**绝不 `generation/*` 通配**（混读 current+previous）；manifest 指向缺失目录 → `DataError` fail-closed（不 legacy fallback 捞 previous/orphan）；`_matrix_available_columns` 并入 `manifest["factors"].keys()`（动态因子列不靠 registry schema → 不再永远 MatrixCoverageMiss fallback）；generation 进 audit params。factor_matrix 显式 `authorized_root: ${FACTOR_MATRIX_ROOT}`（组件级 `path_is_under` 过不了 `universe=` 静态前缀）。`dataaccess/registry/loader.py` / `dataaccess/store.py` / `dataaccess/config/datasets.yaml` |
| 2 | `load_matrix` 对缺失 generation 目录 fail-open | **fixed** | manifest 有 generation 但目录缺失 → 抛 `FactorMatrixCorruptionError`（不再 legacy glob）；materialize 对旧代缺失同样 hard fail（否则 read-merge-write 静默丢历史）；`_read_existing_or_quarantine(immutable=True)` 对已发布 generation 文件损坏只**复制**到 quarantine 留证、不移动原文件；`load_matrix` 对当前代内坏文件 fail loud（`FactorMatrixReadError`）。`storage/materialize/factor_matrix_materializer.py` |
| 3 | MaterializationDelta 未接进 CH 主链 | **fixed** | `execute_materialize` 构造**唯一** `MaterializationDelta`（upserts + tombstones=deleted_keys + full semantic_identity_digest + data_snapshot_id），CH 从 delta 派生 16 位版本并 `_series_with_tombstones` 把删除键写 NaN（与 Parquet 同构）。源行删除 → Parquet 与 CH 同键一致。`runtime/materialize_service.py` |
| 4 | event rebuild 丢 canonical scope | **fixed** | `factor_from_catalog_info` 恢复 `FactorExecutionScopeHint`（market/universe/frequency/calendar/decision_policy 取 full definition）；`_verify_factor_semantic_identity` 对 5 个 scope 字段 fail-closed（declared 有值 + actual 缺失 → production 拒绝）。**修真 bug**：verify 里 `compute_ir_hash(factor.expr)` 对 Expr 报 `.attrs` 错 → 先 `Analyzer().lower(expr).ir` 再 hash（否则任何 full_def 含 ast_hash 的 production event 无法证明 identity）；hash 不一致只在 production fail-closed（research 容忍）。`runtime/incremental_scheduler.py` |
| 5a | DataEvent 多 factor 原子发布 | **fixed（条件 P0 收口）** | production 事件全因子先 `write_target="staging"`（权威水位线 defer）→ **全部 stage 成功才**逐因子 `publish_factor_lake(approve=True, sync_from_local=False, reconcile=False)`；任一 stage/publish 失败 → reject + `PartialIncrementalFailureError`，published 湖零 mixed；同 event_id 重试幂等收敛、只 commit 一次。`runtime/incremental_scheduler.py` |
| 5b | pending 无 lease | **fixed（P1）** | `DataEvent.owner/attempt_id`；pending 记录带 `reserved_at`；`begin` 对 lease 过期（`DATA_EVENT_LEDGER_LEASE_SECONDS`，默认 3600s）的 pending 直接 takeover（worker crash 不再永久 `in_flight`）；无 `reserved_at` 的 legacy pending 保守不 takeover；`in_flight` 不再塌缩成 `"duplicate"`。`runtime/incremental_scheduler.py` |

**新增测试**（R14 第二轮）：`tests/storage/test_r14_matrix_generation_failclosed_2026_08.py`（4）、
`tests/runtime/test_r14_materialize_delta_ch_2026_08.py`（2）、
`tests/runtime/test_r14_event_rebuild_scope_2026_08.py`（5）、
`tests/runtime/test_r14_ledger_lease_2026_08.py`（5）、
`tests/runtime/test_r14_event_atomic_publish_2026_08.py`（4）；
dataaccess `tests/unit/test_r14_matrix_generation_resolver.py`（5）。共 25 条。

**存量回归**：matrix governance 17、r11 orchestration 15、round13 incremental 10、
r10 data event scheduler、r14 ledger/dualwrite/atomic/failclosed、dataaccess matrix
既有测试全绿。`test_p0_22_corrupt_parquet_quarantine_hard_fail` 按 R14 #2 新契约
对齐（已发布 generation immutable：原文件保留、只留证副本、load fail loud）。

**协调注记**：并发会话 cleaned_operators governance 间歇性 flaky（`load_all` 报
`ts_mean_abs_deviation/ts_median_abs_deviation` 未分类或 `ts_bicoherence_max`
inactive），会让含 `parse_factor` / import `incremental_scheduler` 的测试在 import
阶段失败；等树稳定后重跑即绿。未触碰 cleaned_operators/ir/scripts。

**destructive gate 覆盖**：FE gen A → DA 读 A；publish gen B → DA 只读 B；
current+previous 并存 → DA 只读 current；orphan generation → DA 不可见；
manifest.generation 指向缺失目录 → hard fail（DataAccess DataError + FE
FactorMatrixCorruptionError）；incremental deleted_key → Parquet 与 CH 同键
tombstone；decision_time_policy=eod → rebuild → scope 仍 eod（缺失 production
reject）；event f1 ok / f2 fail → published 零 mixed + 同 event 重试只 commit 一次；
worker crash after begin → lease 过期 takeover。

此轮通过后即可宣布 **DATAACCESS CORE FREEZE + FACTORENGINE INTEGRATION FREEZE**。

---

# 附录 G：第三轮（外部 AI 复查 HEAD 44b2946）production DataEvent 2 P0 收口

第三轮复查结论：**DATAACCESS CORE FREEZE = YES**；**普通 FactorEngine ↔ DataAccess
integration FREEZE = YES**（普通 read / PIT / Composite / materialize / matrix /
snapshot / version / tombstone / ClickHouse parity / rebuild scope 未再发现新
P0）。剩余 blocker 只在 **production DataEvent pipeline**，本轮收口。

## G.1 诚实纠偏：上一轮「publish 阶段零 mixed」不成立

§F #5a 声称「任一 stage/publish 失败 → published 湖零 mixed」只对 **stage** 阶段
成立。**publish 阶段**仍是逐 factor `publish_factor_lake`（单 factor
`publish_from_staging()`，无跨 factor transaction）：f1 成功、f2 失败 →
f1=NEW / f2=OLD / event=rejected = mixed published state。上一轮自己的测试
`test_r14_production_event_publish_failure_fails_closed` 已证明
`f_a in store.published`。同样地 `mat_kwargs.setdefault("write_target", "staging")`
尊重调用方覆盖——`staging_clickhouse` 会在 stage-all 阶段就写 CH（published side
effect）→ CH mixed。**此为本轮两个 P0 的根因。**

## G.2 P0-1（方案 A）：production DataEvent 自动发布默认关闭

逐 factor publish 无法做成原子 visibility transaction（多目录不可能一次原子切换）；
真正的修复是 matrix 式 generation/transaction 指针 + reader 只读 committed txn +
CH `event_txn_id`/commit marker——留待后续。本轮按复查认可的最简方案 **方案 A**
feature gate：

- `runtime/production_policy.production_data_event_auto_publish_enabled()`：
  `DATA_EVENT_PRODUCTION_AUTO_PUBLISH` 为真才启用，**默认关闭**。
- `execute_incremental_updates_from_event`：production 且未启用 → begin（若带
  event_id）+ reject + 抛 `ProductionEventAutoPublishDisabled`，**在任何 stage 前**
  ——不落任何 staging/published → 读者看到 ZERO 新因子（destructive gate #1/#2
  的生产路径）。research 事件不受影响。
- production 一律**强制** `mat_kwargs["write_target"]="staging"`（不再 `setdefault`）
  ——调用方无法覆盖成 `staging_clickhouse`/`clickhouse`（stage-all 阶段无 published/
  CH side effect）。research 仍尊重调用方覆盖。
- 显式 `DATA_EVENT_PRODUCTION_AUTO_PUBLISH=1` 保留两阶段路径（实验性）：stage 阶段
  零 mixed；publish 阶段失败仍是已知限制（已发布因子可见、事件 rejected），真正的
  event transaction 放后续。

## G.3 P0-2：ledger lease 补 fencing token

lease TTL 解决「crash 永久 in_flight」，但没有 fencing：`commit()` 不校验
owner/attempt → 被接管的 stale worker 醒来仍可 publish+commit（双写）；
`normalize_data_event(dict)` / `to_dict()` 丢 `owner`/`attempt_id`。本轮：

- `ReservationToken{event_id, attempt_id, fencing_epoch, owner, expires_at}`；
  `begin()` 返回 `(status, token)`；takeover 单调递增 fence（1→2→3）；
  同 worker 同 attempt 重入幂等续租（返回原 attempt/fence，刷新 reserved_at）。
- `commit`/`reject`/`renew` 在 `_ledger_lock` 内 reload 后校验当前 pending 的
  `(owner, attempt_id, fencing_epoch)` 与 token 完全一致，否则抛
  `DataEventLeaseLostError`。`commit` 已 committed 幂等返回；无 token 一律 fail-closed。
- `DataEvent` 新增 `fencing_epoch`，`to_dict`/`normalize_data_event` 携带
  owner/attempt/fence（dict/API round-trip 保留）。
- 调度器：逐 factor stage 成功后 `ledger.renew(token)` 心跳（**移到逐 factor
  try/except 之外**——lease 丢失 = 事件级中止，不是 factor 失败）；publish 循环前
  `renew` 做 fencing 门（stale worker 的发布被拒、不可见）；commit/reject 带 token。

## G.4 新增测试与验证

- `tests/runtime/test_r14_event_fencing_2026_08.py`（9）：stale owner
  commit/reject/renew 全部 `DataEventLeaseLostError`；takeover fence 单调递增；
  无 token commit fail-closed；renew 心跳不过期；dict round-trip 保留
  owner/attempt/fence + token 序列化；**scheduler 级** stale worker publish 门
  （计算中被接管 → stage 后 renew 抛错 → 零 publish，`store.published == []`）。
- `tests/runtime/test_r14_event_atomic_publish_2026_08.py`（现 7）：新增 production
  默认禁用 → `ProductionEventAutoPublishDisabled` 且 `staged==[]`/`published==[]` +
  ledger rejected；无 event_id 也禁；opt-in 下强制 staging target（fake engine 断言
  收到 `write_target="staging"` 而非调用方 `clickhouse`）。
- 存量对齐：`test_r14_data_event_ledger` / `test_r14_ledger_lease` /
  `test_round13_incremental_invalidation` 的 `begin`→`(status, token)` +
  `commit/reject` 带 token；`test_r14_append_failure_propagates` 改 monkeypatch
  `_append_unlocked` 抛 OSError（fencing 后旧路径占位法不再可达）。
- **destructive gate**：production 默认 → ZERO stage/publish；opt-in stage 失败 →
  零 publish；`staging_clickhouse` 调用方覆盖被强制 staging；A begin fence=1 → lease
  过期 → B takeover fence=2 → A commit/reject/renew 全拒；A lease 丢失 → publish 门
  （renew）抛错、零 publish；B commit → 可见恰好一次（committed 记录 1 条）；
  dict round-trip 保留 owner/attempt/fence。
- 回归：29（fencing/ledger/atomic）+ 19（r10/phase23/round13）+ 67（R14 存储/
  matrix/dualwrite/version/delta/rebuild/DA resolver）+ 11（phase21/23）全绿。
- 未触碰 cleaned_operators/ir/scripts。并发会话 coordination 保持（runtime/
  incremental_scheduler.py 本轮改动已独立测试通过）。

## G.5 遗留（非 Freeze blocker）

- **P1**：DA `_read_factor_matrix` 读取结束后重算 `current_generation()` 写 audit；
  读取期间 generation 翻转可能 audit 记录 B 而实际读 A（底层 `DataSnapshot` 仍绑定
  实际文件，不造成错误计算）。后续小修为一次性 `MatrixResolvedSnapshot`
  （resolve generation + paths 一次，coverage/read/snapshot/audit 消费同一 resolved
  generation）。
- **方案 B 真正的 event visibility transaction**（generation/transaction 指针 +
  reader 只读 committed txn + CH `event_txn_id`/commit marker）留待后续，与 §G.2 的
  opt-in 路径衔接。

此轮通过后正式宣布 **DATAACCESS CORE FREEZE + FACTORENGINE/DATAACCESS INTEGRATION
FREEZE 收官**。
