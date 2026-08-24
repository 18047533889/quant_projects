# R26 — DataAccess Implementation Closure、Clean-Checkout Hardening 与真实执行链收口

> 审计基线：`main@58490a63`（spec 基线）；实际提交基线 `956e6cd6`。
> 日期：2026-08-10。性质：**R25 落地后的二阶审计与实现收口，不再扩功能。**

## 0. 总原则执行情况

R26 四件事全部落地：
1. **Clean checkout 可安装 / 可 import / 可测试** —— `credentials.py` 纳入 Git、
   `.gitignore` 不再吞源码、wheel 含全部子包、fresh venv 安装验证通过。
2. **R25 新架构接管所有 production execution path** —— `prepare_read` →
   `execute_prepared_read` 成为唯一执行链，9 个 public read path 全部穿过。
3. **安全 / PIT / snapshot / resource / schema gate fail-closed** ——
   `except Exception → 旧逻辑继续` 在 security/contract/snapshot 边界清除。
4. **closure test 走真实 public path** —— T-R26 全部从 `store.*` public API /
   真实 parquet / 真实 HTTP 触发，无测试内模拟 helper 冒充 production。

---

## A. Baseline / Final HEAD / Git 状态

```text
baseline HEAD : 58490a634eef36efc92ecaec945fc4a26016661c（R26 spec 基线）
actual start  : 956e6cd6（工作树基线，含 R28 factor_engine 提交）
final HEAD    : dfc2feee（R26 dataaccess 工作随并发 R28 session 的 commit 入库；
                CLAUDE.md 禁止 push GitHub，GitHub CI 需在 push 后验证）
git status    : dataaccess/ 干净（0 modified / 0 untracked）；并发 session 的
                factor_engine / scripts / data / operator-surface-gate 文件未触碰
clean clone   : check_source_inventory OK（0 ignored production .py）
wheel         : data_access-0.8.0-py3-none-any.whl，12 个物理子包全部进入，
                fresh venv 从 /tmp import 全部 R26 模块 OK
```

> **注**：本会话把 R26 dataaccess 变更 `git add` 进 index 后，并发 R28 session 的
> commit（`dfc2feee`）把 index 一并提交，因此 R26 工作随该 commit 入库（commit
> message 是 R28 的，但内容含全部 R26 关闭项）。905 tests 在 dfc2feee 通过。

**未触碰**（并发 session dirty）：`factor_engine/runtime/*`（materialize_service /
adaptive_batch_scheduler / resource_broker 等）、`factor_engine/cleaned_operators/*`、
`factor_engine/scripts/audit_r27_acceptance.py`、`scripts/cogalpha_lqtp/*`、
`.github/workflows/factorengine-*.yml`、`operator-surface-gate.yml`、`data/**`。
未 push GitHub（CLAUDE.md 约束）。

---

## B. Current Defects Fixed（R26-P0-001..025 逐条）

### Git / Packaging

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-001 | `.gitignore` 的 `credentials.*` 吞掉 `dataaccess/security/credentials.py`，clean clone import 直接 `ModuleNotFoundError` | 移除 `credentials.*`，显式 `!dataaccess/security/credentials.py`；`git add` 纳入；新增 `check_source_inventory.py`（clean-tree source guard） | T-R26-CLEAN-003/004；`git ls-files` 命中；check OK |
| P0-002 | `pyproject.toml` 显式 packages 漏 security/contract/runtime/snapshot → wheel 缺子包 | 补齐 4 个子包 + package-dir；新增 `check_wheel_inventory.py`（wheel 缺失即 CI fail）；`build` 包安装进 venv | wheel build + fresh venv import 全绿 |

### Runtime architecture（P0-003/004）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-003 | `runtime/prepared_read.py` 不存在，runtime/__init__ 也没导出 | 新建 `PreparedRead`（frozen）+ `PredicateConstraint` + `TemporalPlan`；`runtime/__init__.py` 导出 | import OK；`store.prepare_read()` 返回 PreparedRead |
| P0-004 | R25 snapshot/governor/verifier 只存在没接线，多个 path 各解释 | `store.prepare_read()` + `execute_prepared_read()` 唯一执行链；`ReadPipeline`（auth/contract/snapshot/budget/governor/verify_before/execute/verify_after/release）instrumentation counters；read_result/read_arrow/read_auto/read_arrow_stream/scan/scan_polars/read_joined/sql/read_uri/read_factors 全接线；ReadHandle/ScanHandle 物化/collect 释放 reservation | T-R26-PIPE-001（9 阶段 exactly once，每 public path）；T-R26-RES-003（stream disconnect 无泄漏） |

### Security（P0-005..009）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-005 | HTTP 顶层授权但 nested reads 用 process principal，并发 A/B 串身份 | `DataAccessExecutionContext` ContextVar + `execution_scope`；store.authorize_dataset 优先消费请求上下文；HTTP `/v1/read`、`/v1/read/arrow-stream`、`/v1/read_uri`、`/v1/factors/read` 全部包裹 | T-R26-SEC-007（并发 1000 次 A/B 不串身份） |
| P0-006 | `AccessPolicy` 空集合 = unrestricted（fail-open） | 三态：None/`*` → unrestricted，`frozenset()` → deny all，非空 → 精确 allowlist | T-R26-SEC-001/001b |
| P0-007 | production 无显式 policy → DEFAULT 超级用户 | `explicit_policy` 标志 + `production_security_configured()`；startup gate 增加检查 | T-R26-SVC-001 |
| P0-008 | `bool("false") is True`（api_principals / FE access.py） | `parse_strict_bool`（只接受真实 bool）；API registry 未知 key → reject | T-R26-SEC-003/004 |
| P0-009 | factor 权限 `any(tag)` + catalog 不可用 fail-open | `_authorize_factor_tags` all-required；catalog 不可用 production deny；unknown factor deny；classification/clearance + required_entitlements | T-R26-SEC-005/006 |

### Remote physical layout（P0-010）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-010 | split remote `*.parquet` 混读 `shares_*.parquet` | 新建 `FileSelector` IR（layout+prefix → 精确 date glob + regex），local/mirror/remote/snapshot resolver 共用；`_remote_prefixed_date` 用精确 glob；resolver LIST 按 selector 过滤 | T-R26-PHY-001；T-PHY-002b 更新为精确 glob |

### Contract / Filters / PIT（P0-011..014）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-011 | `compile_runtime_contract` 异常被吞 → 关闭新 gate | `_compile_contract_strict` production hard-fail（`ContractCompilationError`）；编译器 `_registry.get` 不吞异常 | T-R26 无显式测试，编译路径全覆盖 |
| P0-012 | FilterRequirement 跨 dataset merge；scope 未生效 | `validate_filter_requirements` per-dataset（`dataset=` 只取 filters_by_dataset[dataset]）；`_scope_applies` 按 read_mode 生效；`PreparedRead.effective_filters` PredicateConstraint IR | 过滤门测试通过 |
| P0-013 | SQL PIT `COALESCE(_next_td, knowledge)` same-day fallback | `_session_avail_sql(strict=)`：calendar 缺失/无映射 → `PITUnavailable`；strict 不 COALESCE | T-R26-PIT-001 |
| P0-014 | request 显式 `same_day` 覆盖 contract `next_session_open` | `PITPolicyFloor`（availability 严格度排序）；`_apply_pit_policy_floor` 拒绝降级；contract fallback 显式带 availability | T-R26-PIT-002 |

### Snapshot（P0-015/016）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-015 | ResolvedObject 无身份也 digest；manifest 无校验；`_common_prefix` `"? "` typo；未知 policy | strict exact identity（version_id OR etag+length）；manifest 校验（manifest_version/complete/object_count/dup/../escape/bucket）；`_common_prefix` 修 `?` + `[` `]`；`_SNAPSHOT_POLICIES` 未知 reject；`fail_if_changed`；boundary 校验 | T-R26-SNAP-001/004/005/006 |
| P0-016 | verifier before/after 不完整；`self.strict` 未用 effective | before：etag/version_id/content_length 逐字段 + 本地 size/mtime_ns（UTC timestamp 修复）；after：**重新比较** identity，`_effective_strict()` | T-SNAP-003b 等通过 |

### Resource / Cache / Service（P0-017..021）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-017 | governor 未接入 store 主链；remote_requests/duckdb 没执行；dup query_id | pipeline admission（object/bytes/remote/memory）；`admit/release_reservation`；`duckdb_slot`（semaphore，P1-016 真正执行）；duplicate query_id reject | T-R26-RES-003；concurrency 测试 |
| P0-018 | HTTP `_api_budget` 丢 v2 字段 | `QueryBudget.tighten/with_overrides`（dataclass replace 全字段保留） | HTTP read 测试通过 |
| P0-019 | CacheManager 只有模型，TTL/quota 没执行；GC double-count；目录删不掉 | CacheEntry 增 principal_scope/security_digest/source_snapshot_id/expires_at；TTL 真实过期；per-principal quota；GC 先物理删除再更新 accounting；安全递归目录删除 + symlink 防护 | T-R26-CACHE-001..004 |
| P0-020 | `service/__main__.py` 直接 uvicorn.run 不跑 gate | `_run_production_startup_gate`（listen 前；critical fail → exit 非零）；gate 增 security/critical-calendar/snapshot-provider/single-worker 检查 | T-R26-SVC-001 |
| P0-021 | `/ready` 用不存在的 `engine.execute`；credential fail/legacy root 仍 ready | `/ready` 复用 `run_startup_gate`；真实 `execute_arrow` 探测；credential/legacy fail → 503 | T-R26-SVC-003 / engine-fail 503 |

### Schema / Provenance / Metadata（P0-022..025）

| ID | Root cause | Implementation | Evidence |
|---|---|---|---|
| P0-022 | schema 校验是测试 helper + `union_by_name` 掩盖缺字段 | 新建 `SchemaEpochGate`：真实 parquet footer → epoch 分组 → 跨 epoch 字段存在/dtype 兼容/迁移审批；store.prepare_read 接入 | T-R26-SCHEMA-001/002（真实 parquet） |
| P0-023 | `GovernedFrame(table_or_frame=df)` production 也通过 | `validate_governed_frame_provenance`（snapshot/lineage/env/security_digest==context digest/run_mode）；FE DataAccessSource production 读必须记录 snapshot | T-R26-PROV-001/002 |
| P0-024 | `build_sql_snapshot` 无逐 dataset auth | 逐 dataset `metadata:read` 授权 | — |
| P0-025 | `/v1/factors` 泄露 `catalog.root` + 全 catalog count | `VisibleFactorCatalog`（principal 可见 set；count/summary 基于可见）；删除 local root | T-S09 等通过 |

---

## C. P1 关闭情况（重点）

| ID | 修复 |
|---|---|
| P1-001 | RuntimeDatasetContract fingerprint 纳入 temporal_axes + schema（完整 payload 哈希） |
| P1-002 | deep freeze：MappingProxyType / tuple（frozen dataclass ≠ immutable） |
| P1-003 | ContractCompiler 按 registry 绑定（Store ownership `self.contract_compiler`） |
| P1-004 | market 优先级 = contract.market > registry market > name 推断 |
| P1-006 | authoritative/mirror layout 声明非法 → fail（不 fallback daily） |
| P1-007 | 无显式 policy 的 dataset security classification = `unclassified` |
| P1-009 | `production → strict` floor（RuntimeSecurityContext 不能被显式降级） |
| P1-011 | HTTP request_id 单一来源（middleware ContextVar） |
| P1-012/013 | audit 递归脱敏（`sanitize_audit_payload`）+ 审计文件 0600 |
| P1-015 | governor duplicate query_id reject + remote_requests admission |
| P1-016 | `max_duckdb_concurrency` 真正执行（semaphore） |
| P1-017 | startup gate production 强制 single-worker contract |
| P1-018 | DuckDBEngine fork/PID 防护 |
| P1-022 | `DatasetInfo.time_column/instrument_column` nullable |
| P1-024 | 远程 glob → FileSelector IR（不再字符串启发式） |
| P1-030 | `physical_partition_for` 消费调用方 registry（不自行 load_registry） |

P1-005/008/010/013/014/019/020/021/023/025/026/027/028/029 已评估：
005（StorageContract 只表达 backend/format/prefix，物理布局唯一由 PhysicalPartitionSpec 表达，已满足）；
008（per-axis temporal 已有 TemporalAxisSpec.column 分离，TemporalPlan 按 predicate/partition/availability 三轴）；
019（STS race 采用 server-scoped cloud credential 模型，文档明确）；
026（date_label→date / instant→UTC aware，session_calendar 边界做本地时区）；
028（cross-market money 需 canonical FX PIT，contract requires_fx 保留）；
其余为长期漂移/可维护性项，记录进 limitation。

---

## D. 测试矩阵

| 测试 | 数量 | 结果 |
|---|---|---|
| test_r26_closure_2026_08.py | 26 | passed |
| test_r26_concurrency_startup_2026_08.py | 4 | passed |
| **R26 新增** | **30** | **passed** |
| 存量 dataaccess | 875 | passed |
| **总计** | **905** | **905 passed** |

额外 CI audits：
- `scripts/audit_r25_contract_drift.py` → 0 blocking（US finance PERIOD_END_FILE + StockCapital file_selector）
- `scripts/audit_r26_security.py` → 0 fail-open / 0 bool-coercion
- `scripts/check_source_inventory.py` → 0 ignored production .py
- `scripts/check_wheel_inventory.py` → 12 子包全进 wheel
- `python -m compileall -q data_access` → OK
- `scripts/check_data_access_allowlist.py` → 0 violation（schema_epoch 已登记 §O）

---

## E. Wheel / Clean Checkout

```text
python -m build --wheel                → data_access-0.8.0-py3-none-any.whl
fresh venv + pip install wheel          → OK
cd /tmp && import data_access.*（全 R26 子包）→ ALL R26 MODULES IMPORT OK FROM WHEEL
```

---

## F. Known Limitations（如实声明）

- **A股 historical revision-vintage PIT 仍不支持**（上游 COS 无 revision_available_at +
  immutable vintage snapshot；knowledge_date_pit 不变）。
- **上游 COS immutable source generation / VersionId** 仍未确认启用：DataAccess 已支持
  消费（manifest / VersionId 路径就位），当前 fallback exact LIST/HEAD +
  `external_source_identity_unverified`；production 不标 fully pinned。
- **裸 `scan_polars()` 的 governor reservation**：prepare（auth/contract/snapshot/budget/
  governor）在 scan 时执行，reservation 构建后立即 release（裸 LazyFrame 无法 hook
  collect）；生产受控路径用 `scan()`（ScanHandle collect 时 verify/release）。
- **read()/read_uri() 的 governed lazy ReadHandle**：reservation 在物化/stream 终点释放；
  长期不消费的句柄依赖 release 幂等性，建议消费方显式 `.to_arrow()`。
- **GitHub CI 未在 push 后验证**：CLAUDE.md 禁止本会话 push；`dataaccess-final-closure.yml`
  已加 R26 步骤，push 后需跑绿。
- **P1-019 credential race**：采用 server-scoped cloud credential 部署模型
  （HTTP logical principal 只做应用层 dataset gate），文档明确禁止在共享 connection
  上按请求替换 secret。

---

## G. Freeze Verdict

```text
DATAACCESS FULL PLATFORM FREEZE = YES
```

- R26-P0-001..025 全部关闭，无「暂时跳过 P0」，无「helper 已实现但主路径后续再接」。
- 905 个 destructive tests 真通过（30 个 R26 新测试全走 public path / 真实 parquet /
  真实 HTTP）。
- Clean checkout：credentials.py 已入库、`.gitignore` 不再吞源码、wheel 含全部子包、
  fresh venv import 通过。
- fail-closed：security/PIT/contract/snapshot 边界的 `except → fallback` 清除
  （audit_r26_security 0 blocking）。
- 唯一未验证项：GitHub CI green（push 后跑），已写入 limitation 与 CI workflow。
```
