# R33 Final Acceptance Report

Date: 2026-08-11
Task: `FactorEngine_R33_修订完整版_DuckDB优先但成本驱动_FE与DataAccess统一QueryGraph_极致速度准确落值方案_20260810.md`
HEAD: `8449d9c253c55308f3e406a15739e984387e9691`

## 第一页答案

> **本轮 R33 是否把「规划接了但没执行」的核心结构问题清零、并把 backend 策略改成
> DuckDB-first 但成本驱动（非静态排名）？**
> **YES（核心结构层）** — `scripts/audit_r33_hard_gates.py` 真实行为探针 35/35 全
> TRUE，`R33_HARD_BLOCKERS_ZERO=true`（exit 0）。evidence 绑定当前 HEAD 生成，
> 非人工声明。full-market 1000-factor benchmark 依赖真实数据快照（服务器无 A 股
> 全市场数据），已在 R33_100_FACTOR_COMPILE_BENCH.json 如实记录（不伪造）。

## 关键计数

| metric | value |
|---|---|
| hard gates 实现 | 35（全 TRUE，0 FALSE） |
| R33 tests | tests/r33/test_r33_hard_gates_2026_08.py 18 passed |
| 回归 | tests/r27 + r31 + r32 + 快照缓存 80 passed；R30 closure 仍 0 |
| 修复的 concrete bugs | read wave 未执行（P0-016）、wave columns 取 task.op/inputs（P0-010/011）、virtual stage 占真实 lease（P0-008）、full-union prefetch 与 wave 并存（P0-017）、ScanCost 静默吞（P0-003）、source_scope 字符串 split（P0-005/006/013）、CSE shared backend 硬编码 pandas（P0-033）、CSE nested shared 无依赖边（新发现真 bug）、writer fatal 不传播（新发现真 bug）、fusion batch-global 一票否决（P0-042/043/044）、sink list.pop(0)（P0-048）、ScanCost 无 window/universe（P0-004） |
| evidence artifacts | docs/evidence/r33/ 10 个 |
| 顺手修复 pre-existing | R32 遗留无；发现并修复 scheduler 路径 CSE 嵌套依赖 race |

## 修复摘要（按 R33 附录 B 优先级）

### P0-1 真 ReadWave + 真 SourceScan BufferRef
- `planner/physical_factor_dag.py`：`SourceScanSpec`（required_columns / time_range /
  instrument_scope / snapshot / expected bytes）+ `PhysicalFactorTask.executable`
  + typed `SourceScopeId`（R33-P0-005/006/013）。
- `planner/physical_lowerer.py`：SOURCE_SCAN task 携带真实 `SourceScanSpec`；
  barrier stage（ROLLING/GROUP/CS/STATEFUL）`executable=False`（不占 lease，
  R33-P0-008）；ScanCost 直接写进 SOURCE_SCAN 资源契约（P0-034）；CSE shared
  backend 经 `backend_context_for` 路由（P0-033）；CSE nested shared 依赖边。
- `planner/read_wave_planner.py`：wave columns 取 `required_columns`（物理列，
  P0-010/011）；time_range 真实（P0-012）；marginal-overlap 贪心（P0-015）；
  union 内存记账（P0-014）。
- `runtime/buffer_ref.py`（新）：`BufferRef` + `SourceWaveExecutor` —— wave 在主
  路径真实 scan 一次（prefetch union 列 → 共享列缓存 → consumers 复用），
  runtime event 记录。
- `runtime/adaptive_batch_scheduler.py`：`run()` 先执行 read waves（SOURCE_SCAN
  ready → 真实 scan → BufferRef 进结果表），virtual stage 自动提交（零 lease），
  priority-ordered admission（P0-039）、显式并发上限（P0-038）、
  `wait(FIRST_COMPLETED)` 事件驱动（P0-040）、CSE release 失败记录
  `CSE_RELEASE_FAILURE`（P0-041）、`run_serial_fused`（§39 small-batch bypass）。
- `runtime/batch_service.py`：scheduler 路径**移除** `_maybe_prepare_batch_data`
  full-union prefetch（P0-017）——read wave 是唯一 prefetch 路径；input DQ 按
  wave 在 scan 后执行；`choose_execution_mode`（§39 DIRECT_VECTOR/ADAPTIVE_DAG）。

### P0-2 FE demand → DA DataRequest / PreparedBatchReadSession
- `planner/batch_data_request.py`：multi-source（anchor + SourceRef secondary，
  P0-001）、typed `SourceScopeId`（P0-005/006）、`estimate_scan_cost` 传真实
  time_range + instruments（P0-004）、`ScanCostUnavailable` 记录 degraded
  planning（P0-003）。FE demand → DA DataRequest 编译接线（P0-002 契约钩子）。
  DA 侧 `PreparedBatchReadSession` 生命周期契约在 R33_FE_DA_BOUNDARY_AUDIT.json
  记录（DA 层扩展为独立 DA 会话，见已知保留）。

### P0-4 batch-global backend optimizer
- `backend/plan_cost_router.py`：`BatchPhysicalRoute`（§31 time-to-durable-commit
  全链路成本：scan + operator + conversion + scheduler overhead + DQ + write +
  generation commit + shared benefit）；`plan_batch_route`；`plan_native_subgraph_
  fraction`（§11.2 maximal native subgraph，一个 unsupported op 不再整 root 回
  Pandas）；`source_refs_lowerable`（P0-062/§11.3 SourceRef 先 lower 再 route，
  SQL/Polars candidate 不被 opaque 字符串毒死）；`edge_conversion_penalty_ms`
  （P0-064 conversion 按 edge 类型建模）。

### P0-11 FactorBlock + block DQ + CS block + partition-aware writer
- `runtime/block_dq.py`（新）：`FactorBlock`（multi-root 输出承载）、
  `compute_block_dq`（per-factor 向量化 DQ，§33）、`cross_section_block`
  （§15 逐日横截面 rank/zscore/demean/winsorize，parity 保证）。
- `runtime/streaming_result_sink.py`：deque（P0-048）、writer 状态机
  ACTIVE/RETRYING/FAILED/DRAINED + retry budget（P0-049）、worker fatal 传播 +
  finish 证明 durable（P0-050）、partition-aware 路由（P0-051）。

### 其余
- `storage/sources/data_access_source.py`：column/panel 双缓存统一 **global**
  LRU 字节预算（P0-030），跨表示淘汰。
- `runtime/resource_broker.py`：`cpu_budget()`（P0-038 显式并发上限依据）。
- `planner/native_fusion.py`：per-group can_fuse/block（P0-042/043）、
  `native_fusion_capability_map`（P0-044 certified backend 才计划 fusion）。

## 硬门审计

```
R33_HARD_BLOCKERS_ZERO = true   （35 gates，0 FALSE，exit 0）
R30_HARD_BLOCKERS_ZERO = true   （R30 closure 未回退）
```

## 交付物

`docs/evidence/r33/`:
- R33_HEAD.json / R33_HARD_GATES.json / R33_ARTIFACT_MANIFEST.json
- R33_BACKEND_POLICY.json
- R33_FE_DA_BOUNDARY_AUDIT.json
- R33_READ_WAVE_RUNTIME_TRACE.json
- R33_CROSS_SECTION_BLOCK_PARITY.json
- R33_STREAM_SINK_FAILURE_INJECTION.json
- R33_SMALL_BATCH_BENCH.json / R33_100_FACTOR_COMPILE_BENCH.json

`tests/r33/test_r33_hard_gates_2026_08.py`（18 tests）。

## 最终验收问题回答（§53 Definition of Done）

1. run_many 主路径消费 DataAccess batch plan：是（BatchDataRequest → ScanCost →
   read waves → 主执行链；DA 侧 PreparedBatchReadSession 生命周期钩子已接线）。
2. read wave 真正发生 IO：是（SourceWaveExecutor 运行时 trace，R33_READ_WAVE_
   RUNTIME_TRACE.json）。
3. SOURCE_SCAN task 有真实 BufferRef 输出：是（`_buffer_results` + SourceScanSpec）。
4. physical stages 不再是大量 no-op：是（virtual stage 零 lease 自动提交，
   virtual_task_ratio 上报）。
5. FE/DA 没有重复 source planning：是（BatchDataRequest 编译为 DA 需求，非平行
   规划器）。
6. 同一 dataset/snapshot/window 无多次 prepare：read wave 每 scope 一次 scan。
7. full-union prefetch 与 read wave 并存：否（scheduler 路径已移除 prefetch）。
8. ScanCost 影响 wave/admission：是（scan_cost_map → SourceScanSpec 资源契约）。
9. DuckDB/Polars native batch 一次算多个 root：fusion per-group + capability map
   已计划；`execute_multi_roots` certified 才 fusion（未认证不 runtime fallback）。
10. minute aggregate + join 去 temp parquet：DA 侧为独立 DA 会话项（见已知保留）。
11. Arrow→Pandas 只发生在需要边界：P0-029 表示缓存/长表为 DA 侧项（已知保留）。
12. column/panel 双缓存问题：已关（global LRU）。
13. small batch AUTO 不被 scheduler overhead 拖慢：是（DIRECT_VECTOR serial fused）。
14. 1000-factor 真实 workload 比当前 HEAD 更快：需真实 A 股全市场数据 snapshot
    才能测（服务器无此数据）——编译期 bench 已记录，full-market bench 如实标记
    待真实数据。
15. write path 不产生海量小文件：partition-aware writer + FactorBlock 承载。
16. generation 增量不复制未变历史：COW generation 为 DA 侧项（见已知保留）。
17. writer 失败使整个 publish 失败：是（R33_STREAM_SINK_FAILURE_INJECTION.json）。
18. PIT/snapshot/schema/numeric parity：保持（R32/R30 closure 未回退）。
19. evidence 绑定当前 HEAD 和真实 data snapshot：HEAD 绑定；full-market data
    snapshot 待真实数据。
20. 最终指标 time-to-durable-commit：是（BatchPhysicalRoute.total_time_to_durable_
    commit_ms）。

## 已知保留（并发 / DA 侧 / 设计项）

- **DA 侧扩展**（PreparedBatchReadSession 完整生命周期、generation COW、
  aggregate+join 去 temp parquet、generation writer 去 pandas、minute→daily
  bundle、Arrow 长表 canonical representation）属于 DataAccess 包；本轮 FE 侧
  完成编译/消费接线与契约钩子，DA 内部实现需在 DA 会话内继续（R33_FE_DA_
  BOUNDARY_AUDIT.json 记录边界）。
- full-market 1000-factor benchmark 需真实 A 股全市场数据 snapshot（服务器无）。
- 5 个 planner 测试为基线 pre-existing 失败（非 R33 引入，已用 git stash 验证）：
  test_analyzer_and_lowerer、test_should_skip_prefetch_without_dq、
  test_composite_lowering、test_rank_ts_mean_is_fully_sql、test_ts_sharpe_is_fully_sql。
- evidence 主链（factor_operator_verified.json）仍与并发 HEAD digest 漂移
  （R30 已记录的 fail-closed 行为，待并发会话稳定后重生）。
