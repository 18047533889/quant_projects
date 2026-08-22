# FactorEngine R11 — 增量调度 / 依赖 Catalog / Factor Identity / factor_matrix 收官报告

外部 AI 复查（基于 a2dc80c）把扫射焦点从 DataAccess Core 移到 **FactorEngine
orchestration 层**，给出 8 项：7 项收官前必解 + 1 项（dependency edge 原子替换）
至少补测试。本报告为这 8 项的执行记录。

审计基线：`main@432f0c7`（并发会话持续提交；本报告改动基于最新工作树验证）。
本轮只改 `factor_engine/` 树；**不碰 cleaned_operators**（并发会话 R11 longtail
算子认证 WIP）。

---

## A. 逐项状态总表

| # | 项 | 状态 | 修复 |
|---|---|---|---|
| 1 | `incremental_event_service.normalize_data_event` 自递归 | **已存在修复，确认** | 该文件已用 `import ... normalize_data_event as _normalize_data_event` 别名再 `def normalize_data_event` 转发，不再覆盖 import 名。`runtime/incremental_event_service.py:14-25`；新测试确认不递归。 |
| 2 | scheduler 忽略 `affected_start/affected_end` | **fixed** | `plan_updates_from_data_event` 改以 `recompute_start = affected_start or updated_date`、`recompute_end = affected_end or end_date` 作为 watermark/since/end——历史修订（如 2026-08-09 收到、实际改 2024-06-01）不再从事件抵达日期起重算，lookback 由 `build_incremental_plan` 从 affected_start 往前推。`runtime/incremental_scheduler.py` |
| 3 | `deleted_keys` 未消费 → 删除修订留 stale factor value | **fixed** | `DataEvent.deleted_keys → _coerce_deleted_keys → FactorUpdatePlan.deleted_keys → execute_incremental_updates_from_event → materialize_incremental → execute_materialize → ParquetMaterializer.materialize(deleted_keys=…)` 全链路打通。裸 key 以 affected_start 为删除日期归一成 `(datetime, asset)` 键对；纯删除事件（窗口重算空结果）也写 tombstone（`_empty_factor_series` 兜底，不再 skip）。`incremental_scheduler.py` / `runtime/engine.py` / `runtime/materialize_service.py` |
| 4 | dependency edge `source_dataset` 从 anchor 猜，事件查不到真因子 | **fixed** | `_edges_from_analysis` 改为从 **resolved physical source manifest** 写 edge：`spec.dataset`（DataAccess 物理 dataset）> `spec.table`（FactorEngine 逻辑表）> anchor。edge 新增 `logical_table`（+`join_policy` 列）；`factors_for_event` 按 `source_dataset == event.dataset` 的匹配现在能命中复合/二级源。`incremental_scheduler.py` / `dependency_catalog.py` |
| 5 | `factor_full_definition` 未接入主 materialization path | **fixed** | `execute_materialize` 成功落盘后在依赖写入时用 `_build_full_factor_definition` 原子持久化完整规格（expression/surface/dialect/dialect_version/market/universe/frequency/data_source_config/run_mode/calendar/backend/pit_enforce/ast_hash）。`Factor` 增 surface/dialect/dialect_version 字段；`parse_factor` 透传。`materialize_service.py` / `api/factor.py` / `api/dsl_parser.py` |
| 6 | Factor ID / factor_version 只绑 AST，不绑 `FactorSemanticIdentity` | **fixed** | `factor_registry` 增 `factor_version` 列；`register()` 记 `semantic_identity_digest` 前缀（缺省 ast_hash 前缀）；production 下同 factor_id + 语义身份变化 → `FactorSemanticIdentityMismatchError`（research 允许覆盖并记新版本）。落盘行 `factor_version` 也改用 `(identity_digest or ast_hash)[:16]`。`storage/catalog.py` / `storage/materialize/materializer.py` / `storage/exceptions.py` |
| 7 | factor_matrix 直写覆盖整月 data.parquet | **fixed（数据完整性）** | 分区写改为 flock 互斥内的 **read-merge-write**：读旧分区、与增量在 `(datetime, asset)` 上 outer-merge（keep=last，新值覆盖、旧因子/旧日期保留），tmp 文件名带 pid+uuid，`os.replace` 原子替换；production 下告警未走 DataAccess staging→publish 治理。`storage/materialize/factor_matrix_materializer.py` |
| 8 | `factor_dependency_edge` replace 非事务 + delete_factor 残留 | **fixed** | 新增 `DependencyCatalog.record_factor_manifest`：legacy upsert + 删旧 edge + 插全部新 edge + 写 full definition 在单个 `BEGIN IMMEDIATE` 事务内提交（`_begin_atomic`/`_atomic_apply`，SQLITE_BUSY 重试）。`record_factor_edges` 委托该原子写。`FactorCatalog.delete_factor` 补删 `factor_dependency_edge` + `factor_full_definition`。`dependency_catalog.py` / `storage/catalog.py` |

## B. 本批新增回归测试

`factor_engine/tests/runtime/test_r11_orchestration_closure.py`（14 tests，刻意不 import
`api.dsl_parser` / 不解析公式，避免踩并发会话的 R4-100 审计 WIP）：

- #1 事件服务 normalize 不递归；
- #2 affected 日期窗口 + 无 affected 时回退 updated_date；
- #3 `_coerce_deleted_keys` 键对/裸 key + plan 携带 deleted_keys + materializer 落盘
  NaN tombstone（`is_valid=0`，旧有限值不再残留）；
- #4 edge 物理 dataset + 逻辑表回落；
- #5 `record_factor_manifest` 写 full definition + 同步 data_source_json；
- #6 register factor_version（digest / ast_hash 前缀）+ production 冲突拒绝 + research 覆盖；
- #8 edge 原子替换 + delete_factor 补删两个新表；
- #7 matrix 分区 partial 更新 merge 不覆盖（旧因子/旧日期保留、新因子/新日期写入）。

## C. 验证

- 新增 `test_r11_orchestration_closure.py`：**14 passed**。
- `tests/storage/test_materializer.py`：**48 passed**（tombstone/register/version 路径）。
- matrix 平台测试（不解析公式的部分）：7 passed。
- **既有失败 = 并发会话 cleaned_operators R4-100 arity 审计**（`report_filing_delay_surprise`
  / `ts_transfer_entropy_peak_lag` / `fin_roe_cash_gap` 等 polars 缺参数），任何触发
  `load_all()` 的测试（parse_factor）都会被挡——非本批范围，属并发会话 R11 longtail
  算子认证 WIP。

## D. 下一步（按外部 AI 分组）

第一组（必修）已收口；建议继续第二组（PIT policy 继承 → Composite join authority →
Composite snapshot coherence → write-target/watermark 状态机 → governed lazy bypass），
然后转真实数据六类 destructive/property tests（历史 revision、row deletion、factor
source change、partial matrix update、multi-source event invalidation、crash/concurrency、
restart/rebuild reproducibility），通过后冻结 **DataAccess + FactorEngine integration core**。
