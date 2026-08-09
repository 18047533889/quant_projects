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
