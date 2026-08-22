# DataAccess 运维手册（current-state）

> 故障场景逐条：**诊断 → 修复 → 验证**。只描述当前系统的行为与运维手段。
> 错误类型见 `data_access.core.exceptions`；审计/自检脚本见 `scripts/`。

---

## 场景 1 — COS 403（credentials / scope）

**症状**

- 读 COS-backed 数据集（`ashare_stock_daily` 等）抛 `AccessDeniedError` /
  `AuthorizationError`；
- DuckDB httpfs 报 `HTTP 403` / `AccessDenied`；
- production 模式下不自动解析 `~/.cos.yaml`，受限 wrapper 下拿不到更高权限。

**诊断**

1. 确认当前执行身份与凭证作用域：
   ```python
   from data_access import get_store
   store = get_store()
   print(store._effective_security_digest())   # 身份折叠
   print(store._effective_credential_scope())  # 凭证作用域
   ```
2. 确认运行模式：`DATA_ACCESS_RUN_MODE`（production 禁 `~/.cos.yaml` 自动解析；
   research 需 `DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1`）。
3. 确认服务器运行在**受限角色**（`clean-cos-ro` wrapper）下，且 IAM/CAM 对该
   数据集路径有读权限。403 是**能力/授权**错误，不得 fallback 到更高身份
   （`security/policy.py`：AccessDenied 不 fallback）。

**修复**

- IAM/CAM 为该角色授予数据集路径读权限；或更换到权限匹配的部署角色。
- 使用显式 `CredentialProvider`（STS 临时凭证 / 注入的 server-scoped env），
  不依赖 base credential 自解。
- 检查 STS 临时凭证是否过期（`expires_at`），过期则轮换。

**验证**

```python
store.read_arrow("ashare_stock_daily", time_range=("2024-01-01", "2024-01-05"))
```
成功返回；`security_digest` 稳定；日志无 403 重试链。

---

## 场景 2 — manifest stale（build_epoch 过期）

**症状**

- `store.is_snapshot_stale(...)` 返回 `True`；
- 读路径失去 manifest 文件级裁剪，退化为 `**/*.parquet` 全量 glob + 逐文件 footer
  （读变慢）；
- `manifest_version` 里的 `manifest_epoch`/`source_epoch` 与实际文件不一致。

**诊断**

```python
store.manifest_version("ashare_stock_daily")
# 对比 _manifest.parquet 的 mtime 与数据文件 mtime；is_fresh 用文件名级计数比对
```

`read/manifest.py`：`is_fresh_epoch` 要求 `source_epoch == built`；计数不一致视为
过期。构建是**显式操作**，读路径只消费，不会自动重建。

**修复**

```python
store.build_dataset_manifest("ashare_stock_daily", include_row_groups=False, force=True)
```
或按需 CLI / `read/manifest.py: bump_manifest_epoch(root)`。

**验证**

- `manifest_version` 的 epoch 更新为最新；
- `store.is_snapshot_stale` 返回 `False`；
- 读路径重新按 time_range / instrument_filter 做文件级裁剪（日志可见 prune 命中）。

---

## 场景 3 — snapshot mismatch（snapshot_token 对不上）

**症状**

- execute 阶段抛 `SourceSnapshotChanged`（对象在 resolve 后被改写）或
  `SourceSnapshotUnavailable`（无法解析精确对象集）；
- `verify_before`/`verify_after` 失败：etag / versionId / content_length / mtime_ns
  不匹配。

**诊断**

- 确认 resolver policy：`latest` / `pin(snapshot_id)` / `fail_if_changed`
  （`snapshot/resolver.py`）；
- 对比 `ResolvedSourceSnapshot` 的对象集与当前 COS LIST/HEAD 的
  etag/versionId/size/mtime_ns（`snapshot/verifier.py`）——mtime_ns 是**原始纳秒**，
  消除 2s 容差下「同大小文件替换」漏网；
- 确认是否有**并发 writer** 在读取期间改写源对象（写路径应走 generation 原子代，
  不改 live 对象）。

**修复**

- `latest`/`fail_if_changed`：重新 resolve 到最新对象集后再执行；
- `pin(snapshot_id)`：恢复被 pin 的对象版本，或显式升级 pin；
- 消除并发改写源（检查 cron / 上游 sync 是否与读重叠）。

**验证**

```python
store.prepare_read("us_stocks_sip_day_aggs", ...)      # resolve + verify
store.execute_prepared_read(prepared)                  # execute + verify_after
```
全程无 `SourceSnapshotChanged`；对象集与 resolve 结果逐字节一致。

---

## 场景 4 — calendar unavailable（strict fail-closed）

**症状**

- 读 PIT/join 相关路径抛 `CalendarUnavailableError`；
- `store.get_calendar("ashare")` 返回 `None`；
- production/strict 下**绝不 fail-open**（宁可拒绝，不引入未来数据）。

**诊断**

```python
print(store.get_calendar("ashare"))        # None → 日历缺失
print(store.calendar_snapshot_id())        # 会话冻结的日历世界
```
- 确认 `ashare_calendar` / `us_calendar` 数据集在 `config/datasets.yaml` 已登记且
  manifest 可用；
- 确认 `read/session_calendar.py` 的进程内单例未因加载失败被清空。

**修复**

- 恢复/重建日历数据集（见场景 2 重建 manifest）；
- 若日历在远端：确认 `snapshot/resolver` 能解析到精确日历对象；
- 重启会话（`DataReadSession` 在 job 首读冻结日历世界，坏了就重开 session）。

**验证**

```python
cal = store.get_calendar("ashare")
assert cal is not None
store.read_asof(...)   # PIT 读成功
```
`calendar_snapshot_id()` 在单 job 内稳定，跨 job 重新冻结。

---

## 场景 5 — disk full（磁盘满）

**症状**

- 写路径抛 `DataError` / `ResourceAdmissionError`；缓存 GC 无法释放；
- DuckDB spill / COS mirror sync 失败；`df -h` 显示满载。

**诊断**

```bash
df -h /path/to/data && du -sh /path/to/cache /path/to/_archive 2>/dev/null
```
- `runtime/cache_manager.py`：high_watermark 触发 LRU 逐出到 low_watermark；
  无法释放 → 新 admission fail（**不等磁盘 100%**）；
- 检查 `write/publish.py` 的 `_archive/` 旧版本是否堆积。

**修复**

- 触发缓存 GC 到 low_watermark（CacheManager GC / 逐出 evictable 条目）；
- 清理 `_archive/` 旧 published 版本（保留最近 N 版，见 `publish_manifest`）；
- 扩容或清理无关数据；确保 DuckDB `memory_limit` 与磁盘 ratio 合理。

**验证**

- GC 后缓存字节 ≤ low_watermark；
- 写入 / publish 成功；governor 重新能 admit 新请求；
- `df -h` 有足够余量。

---

## 场景 6 — cache corruption（缓存层损坏）

**症状**

- 读缓存路径抛 `CacheSecurityError` / 条目进入 `QUARANTINED`（损坏或权限不符）；
- `CacheManager` 隔离损坏条目且**不允许自动恢复**（防把错误数据当正确数据返回）；
- per-principal quota / TTL 失效表现异常。

**诊断**

- 查看 `runtime/cache_manager.py` 条目状态（`unpinned / pinned / evictable /
  stale / quarantined`）；
- 确认 `CacheEntry.security_digest` 与当前 principal 匹配（跨 principal 隔离，
  不共享缓存项）。

**修复**

1. 关闭查询结果缓存并重建：
   ```python
   store.enable_result_cache(False)
   store.enable_result_cache(True)   # 新缓存目录/条目从零开始
   ```
2. 清理损坏的 cache 目录（确认无 in-flight pin 后再删）；
3. `quarantined` 条目不自动恢复——删除后由下次读重建。

**验证**

```python
store.read_cached("ashare_stock_daily", columns=["Close"], time_range=...)
```
返回正确数据；缓存命中且 `security_scope` 匹配；无 `CacheSecurityError`；条目状态
回到 `pinned/evictable` 而非 `quarantined`。

---

## 附：通用自检

```bash
python3 scripts/audit_dataaccess_source_contracts.py   # registry contract 一致性
python3 scripts/audit_semantic_concept_coverage.py     # 语义字段概念覆盖
python3 scripts/audit_dataaccess_perf_paths.py         # 热路径 pandas 物化检测
```
三个静态审计脚本 0 违规时 exit 0（只做静态检查，不连接数据源）。
生产启动前跑 `runtime/startup_gate.py` 的启动门检查（registry/契约/凭证/日历权威/
快照可达）。
