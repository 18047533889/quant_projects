# DataAccess 第四轮全目录扫尾审计 —— Phase 7 执行计划

> 基线：`main@17020a84f9c13cd2fa3a582b60b153df4335e1b4`
> 规则：只处理本清单新增事项，不重复实现此前 DataAccess 收官清单（P0-1..P0-56 / P1-1..P1-26 / P2-1..P2-5 已在 Phase 6 完成）。
> 所有改动在服务器本地 `/home/shw/quant_projects`，不 push GitHub。

---

## 一、P0（正确性 / 数据完整性，必须修）

| # | 问题 | 修复方案 | 涉及文件 |
|---|---|---|---|
| P0-1 | governed lazy 的 `_collect_lazy()` 返回 `pa.Table`，`to_arrow()` 却 `.to_arrow()`、`to_polars()` 直接当 polars df | 拆 `_collect_lazy_arrow() -> pa.Table`；`to_arrow()` 直接返回；`to_polars()` 用 `pl.from_arrow()`；`stream()` 用 `.to_batches()`；加 3 终点测试 | `read/read_handle.py` |
| P0-2 | `instrument_filter=[]` 被 truthiness 当「不过滤」→ 全市场 | `None`=不限；`[]`=空股票池→`WHERE FALSE`（1=0）；贯穿 DuckDB/Polars/Arrow/partition/bucket pruning | `read/predicate.py`、`read/manifest.py`、`read/partition_planner.py`、`registry/layout_policy.py`、store 边界 |
| P0-3 | 空 hive filter `{"year": []}` 被跳过 → 全年份扫描 | 空 list/空 set → `1 = 0`（WHERE FALSE）；`None` 值跳过 | `read/predicate.py` |
| P0-4 | `time_range=(None, None)` 生成空 WHERE 且各 backend 解释不一 | Predicate 构造时归一化 `(None, None) → None` | `read/predicate.py` |
| P0-5 | 字符串被当 Sequence（`instrument_filter="AAPL"` → 逐字符） | 新增 `ensure_sequence_arg()`；在 instrument_filter/columns/partition_by/upsert_on/order_by/factor_ids 边界拒绝 str/bytes | `read/predicate.py`、`store.py` 读写入口、aggregation、cos runtime |
| P0-6 | Filter AST `isnull: false` 仍生成 `IS NULL` | boolean-aware：`isnull:true→IsNull`、`isnull:false→IsNotNull`、`isnotnull:true→IsNotNull`、`isnotnull:false→IsNull`；非 bool 拒绝 | `read/predicate_ast.py` |
| P0-7 | `aggregate_minute_*` public API 绕过 PreparedReadRequest / semantic gate / required filters | 两个聚合入口顶部调 `store._prepare_read_request(dataset, columns=[field], time_range=..., params=...)`（mode="auto"），与 read() 同 gate | `read/aggregation.py` |
| P0-8 | bundle 多 item 时 timezone/market 可能被全局值覆盖 | 解析各 item spec 的 (market, timezone)，不一致 → ValidationError；未声明 item 用全局默认 | `read/aggregation.py` |
| P0-9 | 聚合多输出 `output_name` 不唯一 | compile 前 `effective_output_name` 唯一性断言 | `read/aggregation.py` |
| P0-10 | `RelationHandle.sql()` 参数顺序：外层 `?` 在 `FROM _sub` 前时错位 | 按「`FROM _sub` 之前的 `?` 数」切分外层参数：`outer[:n] + inner + outer[n:]` | `read/relation_handle.py` |
| P0-11 | `_sub` 检查在替换后做，恒通过 | 替换前先断言原 SQL 含 `\bFROM\s+_sub\b`，否则 ValueError | `read/relation_handle.py` |
| P0-12 | `manifest_generation_id` 未被 JSON loader 读回，mismatch 检测失效 | `_read_manifest_meta_json` 返回该字段；load 时缺失任一侧 → 判 mixed | `read/manifest.py` |
| P0-13 | `_manifest_rowgroups.parquet` 不绑 generation，重建不删旧 sidecar | rowgroups parquet 写 generation metadata；load 时与 manifest generation 不一致 → 忽略；`include_row_groups=False` 重建时删旧 sidecar | `read/manifest.py` |
| P0-14 | `pit_event_index(force=False)` 直接返回旧 index，不核对 source | reuse 前验证：`is_authoritative` + 当前 manifest_epoch/source_snapshot/schema_hash 与 metadata 一致，stale 则重建 | `read/pit_event_index.py` |
| P0-15 | PIT index 局部 time_range 构建被当全局 authoritative | metadata 记录 `filing_scope_min/max`、`timeframe_scope`；请求超出 scope → fail-open（返回空） | `read/pit_event_index.py` |
| P0-16 | timeframe_filter 局部 index 冒充全局 | 同上统一 IndexScope 判定 | `read/pit_event_index.py` |
| P0-17 | PIT index parquet 与 JSON 仍非同 generation | 复用 manifest generation 模型：index.parquet metadata + json 双写，load 不一致 → 非权威 | `read/pit_event_index.py` |
| P0-18 | PIT 自定义列 override 未进 identity | metadata 记录 `columns_used`，reuse 时一致才复用；schema_hash 并入列 override | `read/pit_event_index.py` |
| P0-24 | Upsert UNION ALL 按位置对齐，列序不同会错位 | 显式按 existing 列序 `SELECT` 对齐后 UNION ALL | `write/upsert.py` |
| P0-25 | Upsert 只比列名不比 dtype/nullability | `_assert_schema_compatible`：逐字段名/序/类型/nullable 严格比对，不符 reject | `write/upsert.py` |
| P0-26 | `delete_rows()` 无范围 → 全删 footgun | start/end/after 全 None 且未显式 `delete_all=True` → ValidationError | `write/upsert.py` |
| P0-27 | `delete_rows()` 遇到无 time_column 文件静默跳过 | 缺谓词列 → 非 best_effort 时 abort 整个事务；best_effort 记 failed_files | `write/upsert.py` |
| P0-28 | Publish manifest 记 candidate 绝对路径，上线后失效 | `files[].path` 存相对路径（相对 target_dir）；manifest 含 base_dir | `write/publish_manifest.py`、`write/publish.py` |
| P0-29 | Publish unique_key 校验 `except: return` fail-open | 校验异常 → DataError fail-closed（DuckDB 坏/缺列/SQL 错/parquet 坏一律拒绝发布） | `write/publish.py` |
| P0-30 | 多文件 unique-key SQL 字符串拼路径 | `read_parquet(?)` + `params=[文件列表]`（list 绑定） | `write/publish.py` |
| P0-31 | publish 未验证 staging schema vs target 契约 | `_validate_publish_pair` 增加 schema_version / declared schema / partition_columns / format 一致性 | `write/publish.py` |
| P0-32 | `_copy_tree(symlinks=False)` 注释与语义相反（实际跟随） | `_assert_no_symlinks(staging_dir)` 先拒绝任何 symlink 组件 | `write/publish.py` |
| P0-33 | `read_cos_events(columns=None)` 只返回 PIT 必需列 | columns=None → view_cols = registry 完整 declared schema ∪ 必需列 | `cos_event_runtime.py` |
| P0-34 | optional event filter 未进 view projection → Binder Error | view_cols 并入所有 filter 引用列 | `cos_event_runtime.py` |
| P0-35 | `read_cos_panel(columns=None, filters=...)` 只返回轴列 | columns=None → view_cols = 完整 schema ∪ filter 列 | `cos_panel_runtime.py` |
| P0-36 | `normalize_returns=True, columns=None` 时 return 列不在 view | columns=None + normalize_returns → view_cols 并入 return_column | `cos_panel_runtime.py` |
| P0-37 | `read_cos_events_asof` 状态机未复用 AvailabilityCompiler | **留 gap**（统一 AvailabilityCompiler 是独立架构项，见五）；本轮补 period_selection 严格校验 | `cos_event_runtime.py` |
| P0-38 | deadline pool `apply_pragmas()` 失败泄漏 active slot | acquire 里 pragma 包 try/except → discard + active-=1 + notify + re-raise | `core/engine.py` |
| P0-39 | `on_before_read` 抛 timeout 可能不 close | before_read+read+after 同一 try/except/finally，异常 → had_error + close | `read/managed_reader.py` |
| P0-40 | `reader.close()` 自身失败连接仍判 healthy | close 内 reader/cursor close 异常 → `_had_error=True` | `read/managed_reader.py` |
| P0-41 | `sql_stream()` 未把 `max_elapsed_ms` 下推流式执行 | `execute_scoped_sql_stream` 加 `deadline_ms` + watchdog interrupt；`run_sql_stream` 传 budget.max_elapsed_ms | `core/engine.py`、`read/sql_escape.py` |

## 二、P1（完整性 / 可复现 / 监控）

| # | 问题 | 修复 | 文件 |
|---|---|---|---|
| P1-42 | `RelationHandle.arrow()` 失败无审计 | try/except/finally audit ok=False | `read/relation_handle.py` |
| P1-43 | 聚合 lineage 没记 AggregationSpec | audit extra 并入 spec/output/market/timezone；ReadHandle lineage 尽量补 | `read/aggregation.py` |
| P1-44 | ScanHandle 链式变换后 lineage 不变 | **留 gap**（Provenance DAG 架构项）；记录 transform 名到 lineage extra | `read/scan_handle.py` |
| P1-45 | Coverage `LIMIT 50000` 截断仍判 complete | glob 命中达上限 → truncated → 不得 complete | `read/coverage.py` |
| P1-46 | Coverage 单 root 失败仍判 complete | 任一 glob 失败 → 不得 complete（partial/unknown） | `read/coverage.py` |
| P1-55 | JSONL/Parquet accepted options 未真正执行 | `scan_options` 渲染 `spec.extra` + compression（parquet/jsonl） | `read/formats.py` |
| P1-56 | production 允许 `ignore_errors=true` | strict 语义下 `format.extra.ignore_errors=true` → ValidationError | `read/formats.py` |
| P1-57 | 只验 union schema，单文件缺列不查 | parquet 数据集逐文件 footer 校验 declared 列存在（每文件 required-column 检查） | `registry/schema_validation.py` |
| P1-58 | TIMESTAMP 与 TIMESTAMPTZ 区分不够 | 新增 `timestamptz`/`timestamp_tz` 别名精确匹配 `TIMESTAMP WITH TIME ZONE`；`timestamp` 只匹配 naive | `registry/schema_validation.py` |
| P1-59 | schema cache key 没绑 declared schema identity | key 并入 `declared_schema_hash`（schema_hash_from_decl） | `registry/schema_validation.py`、`store.py` |
| P1-60 | break-glass 降级无专门 audit | `_resolve_mode` 触发 break-glass 时 audit.record | `registry/schema_validation.py` |
| P1-61 | namespace fallback 跨 host 冲突 | fallback 并入 hostname + 进程启动/run_uuid | `core/namespace.py` |
| P1-62 | sanitizer 不 injective | 显式 namespace 非法字符 → ValidationError（不再静默替换） | `core/namespace.py` |
| P1-63 | 坏 namespace 清洗成 anon 仍被当显式 | 显式路径先 `validate_namespace_chars`，非法即拒 | `core/namespace.py` |
| P1-64 | operator 仍进程/env 级 | 增加 `operator_scope()` ContextVar，`resolve_operator` 优先 context | `core/namespace.py` |
| P1-65 | audit JSON 序列化遇 datetime/Path/Decimal 整条丢失 | `_canonical_default` canonical serializer | `core/audit.py` |
| P1-66 | production 判定不统一（audit 只看 QUANT_PRODUCTION_MODE） | audit 用 `is_strict_semantics()`（含 FACTOR_ENGINE_RUN_MODE） | `core/audit.py` |
| P1-67 | env 检查顺序 | 已正确（expand 后检查）；补回归测试 | `registry/loader.py` |
| P1-68 | `${RUN_NAMESPACE}` 与 env gate 冲突 | 已豁免 + 展开顺序正确；补测试 | `registry/loader.py` |
| P1-69 | `expand_env` docstring 承诺 `$VAR` 但未实现 | 结尾 `os.path.expandvars()` | `registry/paths.py` |
| P1-70 | PathAuthorizer symlink TOCTOU | **留 gap**（dirfd/openat 是存储层架构项）；写路径已逐层拒 symlink | `registry/paths.py` |
| P1-71 | stats sidecar 未绑 source generation | snapshot 加 `source_epoch`；router 读 sidecar 前校验新鲜 | `read/stats.py`、`read/read_auto_router.py` |
| P1-72 | stats sidecar 缺 strict validation | `from_dict` 校验 num_rows/num_files>=0、null_ratio∈[0,1]、类型；加 version | `read/stats.py` |
| P1-73 | mirror ensure 部分只查 `exists()` | single_full/root_file/daily 改用 `_local_file_fresh`（verify manifest） | `cos/mirror.py` |
| P1-74 | data 与 `.manifest.json` 非 atomic pair | `_write_download_manifest` 改 tmp+os.replace 原子写 | `cos/mirror.py` |
| P1-75 | `layout_policy` 非 strict | 未知 key 拒、count 严格正整数、column 非空 | `registry/layout_policy.py` |
| P1-76 | bucket hash 不进存储契约 | BucketLayoutPolicy 加 `hash_algorithm`/`hash_version`，stable_bucket 支持 | `registry/layout_policy.py` |

## 三、P2（收尾）

| # | 问题 | 修复 | 文件 |
|---|---|---|---|
| P2-77 | ContractIR `file_format` 回退不走 format_spec | `_storage_of` fmt = `ds.format_spec.type` | `read/contract_ir.py` |
| P2-78 | 多字段 availability 冲突留 None 而非 issue | `len(avail)>1` → issues 追加 | `read/contract_ir.py` |
| P2-79 | semantic catalog env 自定义路径缓存不一致 | 按解析路径缓存（freeze path+fingerprint+catalog） | `read/semantic_catalog.py` |
| P2-80 | `max_age_days=1.9` 静默截断 | 严格整数（拒 bool/float） | `cos_event_runtime.py` |
| P2-81 | `fundamental_staleness_days` 保留输出名冲突 | 冲突时改名 `_event` 后缀 | `cos_event_runtime.py` |

## 四、新增/更新测试

`tests/unit/test_phase7_final_audit.py`（新增）+ 更新既有测试：
governed-lazy 三终点 / instrument_filter=[] / 空 hive filter / isnull:false / bundle clock 冲突 / output_name 重复 / relation sql 参数序 / _sub 校验 / manifest gen load / rowgroup gen / PIT reuse 校验 / PIT scope / upsert 列序 / upsert dtype / delete 无范围 / delete 缺列 / publish manifest 相对路径 / unique_key fail-closed / symlink reject / cos_events columns=None / event filter projection / read_cos_panel projection / apply_pragmas 泄漏 / before_read timeout close / reader.close 异常 / sql_stream deadline / coverage truncated / ignore_errors production / 每文件 required 列 / cache key / break-glass audit / namespace host / operator_scope / audit serializer / expand_env $VAR / stats 绑定 / layout strict / bucket contract / max_age int / 保留列冲突。

## 五、Honest gaps（本轮不强行硬改，写清楚留给专门 wave）

1. **P0-37 AvailabilityCompiler 统一**：read_joined PIT / read_cos_events_asof / PhysicalPlan 三条 as-of 语义路径共用同一个 availability 编译器是独立架构项（涉及 `session_calendar` + contract availability 声明），不在一轮内塞进 helper。
2. **P1-44 LogicalTransformLineage / Provenance DAG**：ScanHandle 链式 transform 的真 lineage DAG。
3. **P1-70 dirfd/openat O_NOFOLLOW**：破坏性路径的 TOCTOU 根治需要存储层句柄化。
4. **P0-7 完整 AggregationCompiler 管线**：本轮先补 semantic gate + clock 一致性 + 命名唯一，完整 `AggregationRequest → PreparedReadRequest → GovernedSourceRelation → AggregationCompiler` 管线单独 wave。
5. **P1-76 bucket hash 全链路写/读**：本轮先记录 hash 契约（writer/reader 共享实现入口），存量已落盘 bucket 不改写。

## 六、验证

- `pytest dataaccess/tests -q` 全绿（基线 548 过）。
- FE 语义消费侧（test_catalog_us / test_field_catalog_alignment_2026）回归。
- 语法检查全部改动文件。

---

## 完成状态（2026-08-08）

### 已实现并验证（dataaccess 全量 **604 passed / 0 failed**，基线 548）

**P0 全部 41 项（除 19–23 未出现在粘贴文本中 + P0-37 标注 gap）**
- read_handle / predicate / relation_handle：P0-1 governed-lazy 三终点、P0-2 空股票池 `WHERE FALSE`、P0-3 空 hive filter、P0-4 `(None,None)` 归一化、P0-5 str/bytes 序列拒绝（instrument_filter/columns/partition_by/upsert_on 边界）、P0-6 isnull boolean-aware、P0-10/11 RelationHandle 参数序 + `_sub` 前置校验、P0-7 聚合入口 semantic gate、P0-8 bundle clock 一致性、P0-9 output_name 唯一。
- manifest / PIT：P0-12 generation 读回 + 单侧缺失判 mixed、P0-13 rowgroups generation 绑定 + 重建删旧 sidecar、P0-14 reuse 前源校验、P0-15/16 IndexScope（filing/timeframe scope fail-open）、P0-17 PIT generation 双写、P0-18 columns_used 身份。
- upsert/publish/delete：P0-24 列序对齐 UNION、P0-25 dtype/nullable 严格、P0-26 delete 无范围拒绝（delete_all opt-in）、P0-27 缺时间列 abort、P0-28 publish manifest 相对路径、P0-29 unique_key fail-closed、P0-30 list 参数 read_parquet、P0-31 staging schema/format/partition 契约、P0-32 symlink 拒绝。
- cos runtime：P0-33/34/35/36 columns=None 全 schema + filter 进 projection + return_column。
- engine/reader：P0-38 apply_pragmas 泄漏修复、P0-39 before_read 异常 close、P0-40 close 异常 unhealthy、P0-41 sql_stream deadline 下推（watchdog interrupt）。
- 新增测试：`tests/unit/test_phase7_final_audit.py`（36 个）。

**P1**：P1-42 relation audit、P1-43 聚合 lineage、P1-45/46 coverage 截断/失败、P1-55 extra/compression 真正渲染、P1-56 ignore_errors production 拒绝、P1-57 逐文件 required 列、P1-58 timestamptz 精确别名、P1-59 cache key 绑 declared schema、P1-60 break-glass audit、P1-61 hostname fallback、P1-62/63 显式非法 namespace 拒绝、P1-64 operator_scope ContextVar、P1-65 audit canonical serializer、P1-66 read audit 统一 strict 判定、P1-69 `$VAR` 展开、P1-71/72 stats sidecar 绑 source + strict bounds、P1-73/74 mirror verified/atomic manifest、P1-75 layout strict、P1-76 bucket hash 契约。

**P2**：P2-77 file_format 走 format_spec、P2-78 availability 冲突成 issue、P2-79 semantic catalog 路径缓存、P2-80 max_age_days 严格整数、P2-81 保留列冲突改名。

### Honest gaps（plan 五节保持一致）
1. P0-37 AvailabilityCompiler 统一（独立架构项）。
2. P1-44 LogicalTransformLineage/Provenance DAG。
3. P1-70 dirfd/openat 根治 TOCTOU。
4. P0-7 完整 AggregationCompiler 管线（本轮补了 gate/clock/命名）。
5. P1-76 存量已落盘 bucket 数据不改写。
6. P1-58 `timestamp` 保持宽松匹配（向后兼容 registry 既有声明）；`timestamptz` 精确别名新增——强制区分（timestamp 拒 TIMESTAMPTZ）需逐一确认各数据集实际时区，单独 wave。

### FE 回归
- `test_catalog_us.py` 全过；`test_field_catalog_alignment_2026.py` 56 过 1 失败（`valuation_growth_mismatch` 算子 arity——并发会话 operator-certification WIP，与本次 dataaccess 改动无关）。
- 整个 FE tests/ 树 collection 报 38 个 backend signature arity 错误，同为并发会话 `operator_signatures_phase2.py` WIP 导致，非本次改动回归。
