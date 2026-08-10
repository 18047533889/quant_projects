# R27 — Cache / Write-Transaction / Unsafe-Surface Closure

> 基线：`main@58490a63`（沿用 R26 基线）。日期：2026-08-10。
> 性质：**R26 之后的下一层收口**——execution path 已唯一化，本轮封闭的是
> **旁路 API 与写路径**没有纳入同一 Cache / 事务 / Snapshot 世界的问题。

## 0. 触发背景

R26 收口后复查发现：即便读执行链已唯一，仍存在
`read_cached` 跨 principal 缓存泄露、`normalize_units` 真实调用 bug、
`physical_scope` raw path 逃生口、写路径无逻辑授权、append/upsert/delete
非 dataset 级原子、stream 二次 resolve、裸 `scan_polars` 旁路、SQL 逗号连接
绕过、calendar 运行中可变、registry 可被外部 mutate 等 14 项。全部纳入本轮。

## A. 修复清单（14 项）

### Cache（R27-A/B/C）
| 项 | 修复 |
|---|---|
| A 跨 principal 缓存泄露 | `read_cached` **cache lookup 之前先 `authorize_dataset`**；cache key 纳入 `security_scope`（principal + policy digest + clearance/entitlements + run_mode）+ classification；restricted/premium 数据**默认不进共享缓存**（`classification_cache_blocked`） |
| B `normalize_units` bug | 旧代码把 `normalize_units` 透传给无该参数的 `read_arrow` → 掉进 `**params`。现走 `read(..., normalize_units=True, result="arrow")` 真实 normalize 路径；`read_cached` miss 走 `read_result`（带 snapshot/lineage） |
| C hit 丢审计/provenance | cache 条目带 meta（snapshot_id/source_generation/security_scope/principal）；命中记录 `cache_hit=true` 审计 + provenance；新增 `read_cached_result()` 返回 `CachedReadResult`（命中/未命中都带 provenance） |

### 物理边界（R27-D/E）
| 项 | 修复 |
|---|---|
| D `physical_scope` raw path 逃生口 | 新增 `VerifiedPhysicalScope(dataset_id, exact_objects, contract_digest)`——只有内部 `prepare_read`/`read_uri` 构造；`prepare_read` 校验 `scope.dataset_id == dataset`；**production/strict 下 raw str/list physical_scope 一律拒绝**（research 允许但强制 dataset boundary）；`_read_pyarrow` 同样收口 |
| E dataset-specific path boundary | `PathAuthorizer` 是全局 roots 白名单（A 的路径可落在 B 的 root 下）。新增 `_enforce_dataset_path_boundary`：**默认解析**（无 read_root 覆盖）产生的路径必须落在 dataset 自己 `authorized_root`/`root` 内；跳过 COS cache 根 / env 白名单 / s3://；跨 dataset relocation 必须独立登记/迁移 |

### 写路径（R27-F/G）
| 项 | 修复 |
|---|---|
| F 写路径逻辑授权 | 新增 action：`dataset:write/delete/publish`、`metadata:write`、`factor:write`、`factor:catalog:write`；`write_arrow`→`dataset:write`、`upsert`→`dataset:write`、`delete_rows`→`dataset:delete`、`publish_from_staging`→`staging 读 + target publish`；policy 未含写 action 的 principal 一律拒绝 |
| G dataset 级原子写 | `generation_pointer` 数据集（factor_matrix）走**原子代写**：写完整新一代 `generation/<gid>/`（hive 分区 `data.parquet`，与上游约定一致）→ **原子 flip `manifest.json.generation` 单指针**。overwrite/append（旧代+新行 concat）/upsert（整代合并）/delete_rows（整代过滤）全部走新代 + 单指针 flip——读者只看到完整一代，**绝无半写 / mixed generation / 两次 rename 缺失窗口**。非 generation 数据集保留 mutation_lock+candidate-rename 语义（limitation） |

### Stream / Polars / SQL / Calendar / Registry（R27-H..L）
| 项 | 修复 |
|---|---|
| H stream 二次 resolve + fail-open | `read_arrow_stream(_return_meta=True)` 复用**同一次** prepare_read 的 snapshot/lineage；`_read_handle` result=stream 不再二次 resolve（TOCTOU 消除）、`except: stream_snapshot=None` 删除（snapshot 缺失 → fail-closed 抛错） |
| I `set_calendar` 运行中可变 | calendar 首次读后冻结（`_calendars_locked`）；`set_calendar` 冻结后拒绝；`calendar_snapshot_id()` 进查询缓存 key / EnvironmentIdentity |
| J 裸 `scan_polars` 旁路 | production/strict 直接禁止裸 `scan_polars()`（collect 绕过 budget/snapshot/audit）；research 放行但审计标记 `scan_polars_unsafe`；受控路径用 `scan()` + `ScanHandle.collect()` |
| K SQL 逗号连接绕过 | `_query_from_tables`/`assert_sql_from_scope`/`assert_sql_tables_declared` 改用 **sqlglot AST** 枚举全部表引用（逗号连接/子查询/CTE 全覆盖）；sqlglot 缺失 → production/strict **fail-closed**（`_require_provable_table_scope`） |
| L registry 真不可变 | `DatasetRegistry.__init__` 把 `schema/roles/engine/params_schema/param_specs` 冻结为 `MappingProxyType`——`store.get_dataset('x').schema['k']=v` 不再能改共享 registry |

## B. 测试

- 新增 `tests/unit/test_r27_cache_write_atomic_2026_08.py`：**17 个 destructive tests**，
  全走 public path / 真实 parquet / 真实 DuckDB / 真实 HTTP 语义。
- 存量 dataaccess：905 → **922 passed**（+17），0 失败。
- 审计脚本全绿：`audit_r26_security` 0 fail-open / `audit_r25_contract_drift` 0 /
  `check_source_inventory` 0 ignored / `check_wheel_inventory` 12 子包 / `compileall` OK。
- allowlist：`write/generation.py` 已登记（与 publish/upsert 同类豁免）。

## C. 改动文件

```
dataaccess/read/query_cache.py          CachedReadResult + _security_scope_digest +
                                        classification_cache_blocked + cache meta
dataaccess/read/sql_escape.py           sqlglot AST 表引用（逗号连接）+ fail-closed
dataaccess/runtime/prepared_read.py     VerifiedPhysicalScope
dataaccess/registry/loader.py           MappingProxyType 冻结
dataaccess/security/principal.py        write/delete/publish 等 action 常量
dataaccess/store.py                     read_cached 重写 + 写路径授权 + generation 写 +
                                        dataset boundary + stream 单 snapshot +
                                        scan_polars strict + calendar 冻结 + physical_scope
dataaccess/write/generation.py          【新】generation 原子代写模块
dataaccess/tests/unit/test_r27_*.py     【新】17 destructive tests
.data_access_allowlist.yaml             generation.py 豁免登记
dataaccess/docs/R27_...CLOSURE_REPORT.md【本报告】
```

## D. 如实声明 / 已知限制

- **非 generation_pointer 数据集**（绝大多数）：append/upsert/delete 仍为
  mutation_lock + per-partition 原子、非 dataset 级 set-原子。完整 dataset 级
  原子性需要把那些数据集也声明 `generation_pointer: true`（读侧已支持），或
  统一迁移 generation 布局——记入 limitation，不在本轮强改。
- **`VerifiedPhysicalScope` 的 class 本身可被任何 Python 代码构造**：防护是
  public API 拒绝 raw path + `dataset_id` 绑定校验；与全项目其它 Python 安全
  边界同级别（防误用，不防本地恶意代码）。
- **read_uri 非 strict 分支**：临时虚拟 dataset 读仍是 glob 全局白名单治理
  （原行为）。
- **registry 顶层字段已冻结**；`engine.options` 嵌套 dict 未 deep-freeze
  （residual）。
- GitHub CI 未在 push 后验证（CLAUDE.md 禁 push）。

## E. Verdict

```text
R27 Cache/Write-Transaction/Unsafe-Surface Closure = YES
922 passed（905 存量 + 17 新 destructive）
审计脚本全绿；未触碰并发 session 的 factor_engine / scripts / data 文件
```
