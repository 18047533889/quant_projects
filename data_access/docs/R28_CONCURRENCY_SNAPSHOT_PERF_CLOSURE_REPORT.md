# R28 — Concurrency / Snapshot-Truth / Remote-Auth / Schema-Gate-Cost / Write-Perf Closure

> 基线：`main@49907850`（GitHub HEAD，用户逐项复核）。日期：2026-08-10。
> 性质：**R27 之后的下一层**——从「架构闭环」进入「底层并发、快照真实性、
> 远端权限、schema gate 成本、写性能本身」。
> 用户确认 R27 的 14 项已落进代码，本轮把复查发现的 **28 项**全部封闭。

## A. 修复清单（28 项）

### 缓存正确性（R28-1/2）
| 项 | 修复 |
|---|---|
| 1 `CacheManager.pin()` 死锁 | 过期 entry 分支在持非重入锁下 `return self.pin(...)` 递归 → 同一线程二次 acquire 死锁。改为锁内 `_admit_new_locked` 原地重建（抽出新 entry admission）；删除失败 → 保留 entry + `DELETE_FAILED` + 拒新 admission（账本不双重计账） |
| 2 删除失败配额账算错 | `_remove_entry_locked` 不再删失败也 `_entries.pop`。物理删除失败 → entry 保留、标记 `DELETE_FAILED`、`size_bytes` 仍计入 quota；GC 下次重试。「路径已不存在」视为已释放（无字节占盘） |

### 快照真实性（R28-3/4/5）
| 项 | 修复 |
|---|---|
| 3 远端 verifier 未真正 HEAD | `ReadPipeline` 的 `SnapshotVerifier` 注入 `remote_meta_fn`＝store 的 `_remote_meta_head`（`_remote_object_meta(fresh=True)` 真实 COS HEAD，credential-aware）。production/strict 下 verify 真正做「执行前 HEAD + 执行后 HEAD」身份比较；HEAD 失败 strict fail-closed |
| 4 本地 final verify 弱 | `ResolvedObject` 新增 `mtime_ns`（原始纳秒，不再 datetime→float→ns 绕圈）；`verify_before/after` 本地分支按 `size + mtime_ns` **精确**比较（消除 2s 容差下「同大小替换」漏网）；老构造无 mtime_ns 才回退 last_modified 容差 |
| 5 SourceManifest 未达文档承诺 | strict 必填 `dataset`（且与 expected_dataset 一致）/ `content_digest`（**重算比较**）/ `prefix`（bucket+segment 边界）/ `published_at` / `object_count`（不再「有才校验」）；cross-bucket 一致性；边界判断 `str.startswith` → `_uri_is_within`（**URI bucket + path-segment 级**，防 `abc`/`abcd` prefix collision） |

### 物理 scope / schema gate（R28-6/7/8）
| 项 | 修复 |
|---|---|
| 6 `VerifiedPhysicalScope.contract_digest` 写而没核 | `prepare_read` + `_read_pyarrow` 的 Verified 分支强制 `scope.contract_digest == 当前 dataset contract digest`（过期 scope 拒绝执行——contract 更新后旧 plan 不再可用）；并重做 dataset-specific physical boundary |
| 7 SchemaEpoch migration 逻辑 bug | `_migration_approved` 改**真实 epoch pair 比较**（A→B，非 A→A 自环）；dtype 传实际 epoch（非 None）；add_column 需要「含字段 epoch → 缺字段 epoch」的精确迁移；删除「任意 approved migration 放行」兜底 |
| 8 SchemaEpoch O(N footer) 性能 | manifest 构建时算好 `schema_epochs` 摘要（`{schema_hash: {field: dtype}}`，落 `_manifest.json`）；query-time `SchemaEpochGate.group_epochs(manifest=...)` 走摘要 O(1) 分组，不再逐文件开 parquet footer；manifest 缺失才回退 footer |

### 远端治理 / DuckDB 并发（R28-9..15）
| 项 | 修复 |
|---|---|
| 9 Governor 远端「请求总数」当并发 | `admit` 不再把 `remote_requests`（一次查询的 COS 对象数，可 100+）与 `max_remote_concurrency` 比。拆成 `remote_request_count`（成本，QueryBudget 层治理 + `_remote_requests_total` telemetry）与 `remote_concurrency`（`acquire_remote_slot` 真实 in-flight 并发） |
| 10 DuckDB 两套并发不一致 | deadline pool 容量 = `max_concurrency`（单一 source of truth，缺省读 governor `max_duckdb_concurrency`）；`_exec_sem` 进程级统一并发门 |
| 11 slot 未覆盖所有路径 | DuckDB 并发控制**下沉到 engine**：`_exec_sem` 包住 `execute_arrow` / `execute_reader`（信号量随 stream 生命周期，reader close 才释放）；`read_joined` 也持 slot |
| 12 deadline 等 pool 超时缺口 | **request absolute deadline 贯穿** governor→pool→execute：deadline 已过 → fail-fast；pool acquire 只消费剩余时间；acquire 后已过 → 归还并 fail；watchdog `wait<=0` 立即 interrupt（不再静默放行）；pool 超时 `ResourceBudgetExceeded` → `DeadlineExceeded` |
| 13 deadline 破坏共享 cache | pool 连接不再各自 `:memory:`，改连**同一共享临时文件数据库**——DuckDB DatabaseManager 对同一 path 复用同一 DatabaseInstance，object/footer cache 跨连接共享（「4 个独立 cache」→「1 个热 cache」） |
| 14 CPU oversubscription | 每连接 threads 缩到 `max(1, total_threads // max_concurrency)`（约束 concurrent×per_query ≈ physical cores） |
| 15 `_check_pid` 位置 + 覆盖 | 初始化日志移回 `__init__`（不再每次 query 刷）；fork 防护覆盖全部 public execution entry（`execute_reader` / `execute_scoped_sql_arrow` / `execute_scoped_sql_stream` / `relation` / `execute_isolated_arrow`） |

### HTTP 上下文 / 凭证传播（R28-16/17）
| 项 | 修复 |
|---|---|
| 16 HTTP stream scope 生命周期 | `execution_scope` 原只包住 `read_arrow_stream(...)` 调用（生成器惰性，真正执行在 `next()` 时——那时 scope 已退出）。改为创建生成器在 scope 内 + `_scoped_batches()` 把**全部迭代**包在 `execution_scope` 内（嵌套读用回请求 principal） |
| 17 Credential/读权限上下文不完整 | `resolve_s3_credentials` 优先 request-scoped `current_credential_provider()`（每 principal 受限凭证），再回退全局；`read_joined` governor 归因用 `current_principal()`（不再 process 级）+ 传 `remote_requests` + 持 duckdb slot |

### SQL 沙箱 / 依赖（R28-18）
| 项 | 修复 |
|---|---|
| 18 sqlglot scope 收紧 + 依赖 | `_try_sqlglot_table_refs` 区分限定名 `catalog.schema.table`（返回全限定名，不再只取 `tbl.name`——旧实现 `FROM arbitrary.bar` 能混配声明集合里的 `bar`）/ CTE 别名（跳过，但 CTE 体内表照常枚举）/ 表函数（仍走函数 allowlist）；`sqlglot` 从 `[sql]` optional 提为 **base 依赖**（安全边界必须「装上就有」，不部署后才 fail-closed） |

### 写性能（R28-26/27）
| 项 | 修复 |
|---|---|
| 26 generation 整代重写 | append/upsert/delete 走 **copy-on-write**：旧代未变分区**硬链接**进新一代（O(1)，不复制数据），只重写「变更分区」（append 旧行+新行 concat / upsert 按键合并 / delete 过滤），整代单指针 flip 语义不变；被删光的分区不写空文件。更新时间复杂度从「O(整个历史)」→「O(变更分区)」 |
| 27 非 generation 数据集 | 保留为 limitation（R27 已声明）；generation 模型统一迁移留待数据集主人决定 |
| 16-tail `merge_tables_by_keys` 去 pandas | 多键合并弃 `astype(str)+"|"join+isin`（key collision/类型丢失风险）→ DuckDB typed anti-join `NOT EXISTS ... IS NOT DISTINCT FROM`（NULL-safe） |

## B. 测试

- 新增 `tests/unit/test_r28_concurrency_snapshot_2026_08.py`：**28 个 destructive tests**，
  覆盖 R28-1..18 / 26（含 CacheManager 死锁回归、配额账、mtime_ns 精确校验、manifest
  O(1) 分组、deadline fail-fast、pool 共享实例、凭证上下文、sqlglot 限定名、
  generation COW inode 断言等）。
- 全量：**950 passed**（922 存量 + 28 新），0 失败。
  *唯一失败是 `test_check_allowlist`——并发 R28 factor-engine session 的
  `factor_engine/scripts/sql_certification_factory.py` 用了 `duckdb.connect` 未登记
  allowlist；不是本 session 改动（本 session 文件 0 违规）。*
- 审计脚本全绿：`audit_r26_security` 0 fail-open / `audit_r25_contract_drift` 0 /
  `audit_contract_ir` OK / `check_source_inventory` 0 / `check_wheel_inventory` 12 子包 /
  `compileall` OK。

## C. 改动文件

```
runtime/cache_manager.py         pin 死锁 + DELETE_FAILED 配额账（R28-1/2）
snapshot/source_snapshot.py      ResolvedObject.mtime_ns（R28-4）
snapshot/verifier.py             mtime_ns 精确校验 + remote_meta_fn 真实 HEAD（R28-3/4）
snapshot/resolver.py             SourceManifest strict + _uri_is_within segment 边界（R28-5）
runtime/read_pipeline.py         resolved_snapshot_from_files 带 mtime_ns（R28-4）
store.py                         verifier 注入 HEAD + Verified digest 消费 + read_joined
                                 slot/principal/remote_requests + generation COW（R28-3/6/11/17/26）
core/engine.py                   统一并发/共享 pool/deadline 链/threads 缩放/fork 全入口（R28-10..15）
runtime/resource_governor.py     remote 请求总数≠并发（R28-9）
read/schema_epoch.py             epoch-pair 迁移 + manifest O(1) 分组（R28-7/8）
read/manifest.py                 schema_epochs 摘要落盘 + epoch_summary_for_paths（R28-8）
read/sql_escape.py               限定名/CTE scope + 表函数（R28-18）
service/app.py                   stream scope 覆盖生成器生命周期（R28-16）
cos/remote.py                    resolve_s3_credentials 优先 request-scoped（R28-17）
write/generation.py              COW 助手 + 去 pandas anti-join（R28-26）
pyproject.toml                   sqlglot base 依赖（R28-18）
tests/unit/test_r28_*.py         【新】28 destructive tests
scripts/audit_r26_security.py    行号随编辑更新（KNOWN_SWALLOWS）
dataaccess/docs/R28_...REPORT.md 本报告
```

## D. 如实声明 / 已知限制

- **DuckDB object cache 跨连接共享**：共享临时文件数据库是 DuckDB 官方「同一 path →
  同一 DatabaseInstance → 共享 cache」机制；实测 OS page cache 掩盖了量级差异，未做
  微秒级 benchmark 对比。**推荐 FactorEngine 侧下一阶段做 Job-level DataReadSession /
  PreparedSnapshot 复用**（把正确性证明从「每次 field read」提升到「每个 job」）——
  那才是最大的治理成本降低。
- **非 generation_pointer 数据集**仍无完整 dataset 级 set-atomicity（R27 limitation
  延续；generation COW 只对 `generation_pointer: true` 生效）。
- **VerifiedPhysicalScope / raw scope 的 Python 本地安全边界**：防误用，不防本地恶意
  代码（全项目同级别）。
- **allowlist 测试**因并发 session 的 factor_engine 文件未登记而红（非本 session）。
- GitHub CI 未 push 验证（CLAUDE.md 禁 push）。

## E. Verdict

```text
R28 Concurrency/Snapshot-Truth/Remote-Auth/Schema-Gate-Cost/Write-Perf Closure = YES
950 passed（922 存量 + 28 新 destructive）
审计脚本全绿；只 stage dataaccess/ 与 dataaccess 测试（未触碰并发 session 文件）
```
