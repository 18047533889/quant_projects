# DATA_ACCESS_IO_REPORT — DataAccess 读链路实测（100k GO §105）

> 生成：2026-09-04 · data_access 基线 commit 见各节。数据源：`data_access/docs/evidence/r30/`（B01–B04 基准 + FE_DA_1000/10000_FACTOR_TRACE）、`data_access/docs/R28_CONCURRENCY_SNAPSHOT_PERF_CLOSURE_REPORT.md`（R28-9..15 并发/共享缓存）、`R29/R32` governor 基线。
> 对应：GO_PROMPT §105 交付物清单 `DATA_ACCESS_IO_REPORT.md` —— DataAccess 读链路（snapshot 解析 / remote 展开 / 成本治理 / DuckDB 并发 / 写原子）的实测汇总。

## 硬件 / 运行时环境（BENCHMARK_ENV.json）

- 平台：Linux-5.15.0-171-generic-x86_64-glibc2.35 · Python 3.10.12
- CPU：INTEL(R) XEON(R) PLATINUM 8576C（基准机 8 vCPU / 30.5 GB）
- 版本：data_access 0.10.2+build.`a2dc3a1` · duckdb 1.5.4 · polars 1.42.1 · pyarrow 25.0.0 · pandas 2.3.3

## 1. R30-P0-001 基准套件（B01–B04，全 PASS）

| workload | 内容 | verdict | wall_ms | rows/s | scan_count |
|---|---|---|---|---|---|
| B01 | 10y daily panel（full + pruned） | PASS | 805.5 | 6,000,629 | 1 |
| B02 | minute read（单日/月/年）+ minute→daily 聚合 | PASS | 143.8 | 288,039 | 2 |
| B03 | daily + fundamental PIT join + universe | PASS | 173.2 | 465,592 | 1 |
| B04 | 100 factors 共享 source | PASS | 1,729.2 | n/a | **100** |

B01 细节：ttfc 262.7ms / ttdc 301.8ms / resolution 98.0ms / bytes_scanned 16.7MB（= bytes_returned，物理 scan 1 次）。
B04 暴露旧瓶颈：100 factors 各扫一次 source（scan_count=100、bytes 161MB）→ 需要请求合并（见 §3）。

## 2. 因子批量 trace：1000 / 10000 factors 共享 scan（FE_DA_1000/10000_FACTOR_TRACE.json）

| 指标 | 1000 factors | 10000 factors |
|------|-------------|---------------|
| 唯一概念 | 11 | 11 |
| source_group_count | 7 | 7 |
| **physical_scan_count** | **7** | **7** |
| **scan_amplification** | 0.007 | **0.0007** |
| coalescer input_requests | 1850 | 18500 |
| coalescer merged_requests | 7 | 7 |
| coalescer saved_scans | 1843 | 18493 |
| field_union_total | 32 | 32 |
| acceptance（source_group < factor_count） | ✅ | ✅ |

**结论**：请求合并（coalescer）把 18500 个 factor 数据请求压到 7 次物理 scan —— 10000 factors 不放大 scan 数，scan_amplification 0.0007。

## 3. Snapshot 真实性 / 解析（R28-3/4/5）

- **远端 verifier 真 HEAD**：`ReadPipeline.SnapshotVerifier` 注入 `remote_meta_fn` = store `_remote_meta_head`（真实 COS HEAD，credential-aware）；production/strict 下 verify 做「执行前 HEAD + 执行后 HEAD」身份比较，HEAD 失败 strict fail-closed。
- **本地 final verify**：`ResolvedObject.mtime_ns` 原始纳秒比较（消除 2s 容差同大小替换漏网）。
- **SourceManifest strict 必填**：dataset（与 expected 一致）/ content_digest（重算比较）/ prefix（bucket+segment 边界）/ published_at / object_count；URI 边界判断升到 `_uri_is_within`（防 `abc`/`abcd` prefix collision）。
- **verified physical scope**：`scope.contract_digest == 当前 dataset contract digest` 才执行（过期 scope 拒绝）。

## 4. Remote wildcard 主读链展开（R58 #1，2026-09-04）

`s3://…/*.parquet` 不再原样直传 DuckDB —— `_expand_glob_paths` 调 `_expand_remote_wildcard`：
1. LIST prefix（glob 所在目录，`rfind("/",0,marker)` 不截到第一个 marker）；
2. fnmatch 过滤 → 精确 `s3://` 对象列表（**2593 对象回填真实 size** 实测）；
3. strict 下 LIST 失败抛 `RemoteMetadataUnavailable`（含 dataset/prefix/provider/cli_fallback/retryable）；research 保留原 pattern。

## 5. 成本治理：整数 scan-bytes + 跨 domain COS（R57/R58 #3/#6）

- **只接受明确整数 scan-bytes**：`estimated_scan_bytes=None` 不再 coerce 到保守上界 `(1<<62)-1`（避免把「元数据缺失」伪装成「scan bytes 超限」）；strict/production 抛 `RemoteMetadataUnavailable`，research 可显式豁免（`DATA_ACCESS_ALLOW_UNKNOWN_REMOTE_SCAN_COST=1`）。
- **跨 domain（跨账号）COS bucket**：boto3 首选 → 落回 **COS CLI 网关**（`cos_cli_ls`/`cos_cli_head` 解析人类可读表格；coscli ls prefix 无尾斜杠返回 DIR 行 → 必须补 `/` 枚举对象；SIZE 列人类可读 ceil 换算宁高估不低估）。
- **governor 实测**：resolver snapshot total_bytes=782562（fidelity REMOTE_VERSION_ID）；httpfs 读路径 governor gate 放行（此前 ResourceAdmissionError）；cli backend 全链 `read_arrow("ashare_stock_daily", columns=(TradeDate,Symbol,Close), time_range=2016-01-04, instrument_filter=[000001.SZ])` 无 TypeError / ResourceAdmissionError，10193 rows，schema/backend 可观测。

## 6. DuckDB 并发 + 共享缓存（R28-9..15）

- **deadline pool 容量 = max_concurrency**（单一 truth，缺省读 governor）；`_exec_sem` 进程级统一并发门。
- **DuckDB 并发控制下沉到 engine**：`_exec_sem` 包住 `execute_arrow`/`execute_reader`（reader close 才释放 slot）；`read_joined` 也持 slot。
- **absolute deadline 贯穿**：deadline 已过 fail-fast；pool acquire 只消费剩余时间；watchdog `wait<=0` 立即 interrupt（不再静默放行）。
- **共享临时文件数据库**（替代各自 `:memory:`）→ 同一 path 复用同一 DatabaseInstance，object/footer cache 跨连接共享（「4 个独立 cache」→「1 个热 cache」）。
- **CPU oversubscription 防制**：每连接 threads = `max(1, total_threads // max_concurrency)`。
- governor 上限：`max_total_scan_bytes_inflight` 默认 `max_total_reserved_memory × 0.5`。

## 7. 写路径原子性（R27/R28-1/2）

- `CacheManager.pin()` 死锁修复（锁内 `_admit_new_locked` 原地重建）；删除失败保留 entry + `DELETE_FAILED` + 拒新 admission（账本不双重计账）。
- 物理删除失败 → entry 保留、标记 DELETE_FAILED、size_bytes 仍计入 quota，GC 下次重试（「路径已不存在」视为已释放）。

## 8. 结论

DataAccess 读链路在 snapshot 真实性（R28）、remote wildcard 展开 + 整数成本治理（R58）、跨 domain COS 网关（R57）、共享 scan 因子批量（R30 FE_DA trace）四个维度均有实测：10000 factors scan_amplification 0.0007、R30 四档基准全 PASS、1929 全套测试绿（R58 收口）已 push。与 FE 侧 `compute_many` 单 scan（见 PERFORMANCE_BENCHMARK.md）叠加构成 100k 生产读链路的完整证据。
