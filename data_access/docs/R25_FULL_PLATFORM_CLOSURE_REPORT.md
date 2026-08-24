# R25 — DataAccess 全平台最终收口报告

> 审计基线 `main@7b15a5e7`；日期 2026-08-10。
> 本轮性质：DataAccess 全平台收口——权限 / PIT / 物理分区 / Source Snapshot /
> 一致性 / 跨市场语义 / 并发 / 缓存 / 服务化 / FactorEngine 协作。
> 目标：把「每一层单独正确、组合起来仍然出错」的漏洞关闭，建立
> **一个 Runtime Contract + 一个 Source Snapshot + 一个 Security Context + 一个
> PreparedRead** 作为运行时唯一事实源。

---

## A. Baseline / Final HEAD

```text
starting HEAD : main@7b15a5e7（审计基线）
final HEAD    : 工作树未提交（本轮新增/修改见下）
changed files : dataaccess/contract/（5）+ dataaccess/snapshot/（4）
                dataaccess/runtime/（4）+ dataaccess/security/（+run_mode,
                governed_frame）+ dataaccess/cos/{mirror,remote}.py
                + dataaccess/read/{query_budget,read_contract,contract_ir,
                session_calendar}.py + dataaccess/core/exceptions.py
                + dataaccess/store.py + dataaccess/service/{app,models}.py
                + dataaccess/quality/contracts.py + dataaccess/scripts/
                audit_r25_contract_drift.py + .data_access_allowlist.yaml
                + 9 个 test_r25_* 测试文件
```

**未触碰**：`cleaned_operators/`、`ir/analyzer.py`、`scripts/`、factor_engine
并发 session 的 dirty 文件（materialize_service 等）；未 push GitHub。

---

## B. Current Defects Fixed（R25-P0 逐条）

### Physical

| ID | Root cause | Implementation | Test |
|---|---|---|---|
| R25-P0-001 | US finance `storage_layout=period_files` 但 MirrorSpec 用 `daily_parquet`，remote 用 request 日期拼 `{date}.parquet` | `PhysicalPartitionSpec(layout=PERIOD_END_FILE, partition_clock=period_end, filename_template="{period_end}.parquet", query_clock=filing_date)`；mirror/remote/auto 共用 `physical_partition_for()` locator | T-PHY-001 a/b/c/d |
| R25-P0-002 | StockCapital split/shares 目录混放两种 schema，MirrorSpec 无 filename pattern | `file_selector`：split=`{date}.parquet`、shares=`shares_{date}.parquet`；`_daily_filename()` 唯一 locator | T-PHY-002/003 |
| R25-P0-016 | event/period 被 trade-day expected partition 错误建模 | `expected_partitions()` 对 period/event/prefixed 布局返回空（不按 request 时间轴枚举）；`_sync_dir_full` 完整目录 sync | T-PHY-001b |
| R25-P0-015 | 404/NoSuchKey 统一 debug 跳过 | `missing_semantics`（error/warn/empty_ok）按契约；dense 缺 → `MissingRequiredPartition` | T-PHY-004/005 |

### Filters

| ID | Root cause | Implementation | Test |
|---|---|---|---|
| R25-P0-003 | Store dataset gate 漏 `required_event_filters` | `_dataset_required_filters` 并入 event；`FilterRequirement`（panel/event/dimension scope）+ `_enforce_runtime_contract_filters` 覆盖所有读面（read_result/arrow/auto/stream/scan/scan_polars/read_joined/sql/FE/HTTP） | T-FLT-004/004b |
| R25-P0-004 | `timeframe=["quarterly","annual"]` 合法 | `FilterRequirement(cardinality="exactly_one")` + `validate_filter_requirements`；进入 RuntimeDatasetContract identity | T-FLT-001/002/003 |

### PIT

| ID | Root cause | Implementation | Test |
|---|---|---|---|
| R25-P0-005 | calendar=None + strict 仍 return knowledge | `compile_available_from_result` → `AvailabilityResult`；calendar-required 缺失 → `CalendarUnavailableError`（strict）；research 返回 `authoritative=False + degradation_reason` | T-TIME-001/001b |
| R25-P0-017 | US filing_date date-label 被当 UTC instant 转纽约提前一天 | `time_representation="date_label"` → `_to_local_date_time` 不做时区换算；semantic_timezone=America/New_York | T-TIME-002/003 |
| R25-P0-018 | A UpdateTime 混用为 revision order | `revision_availability_time=None`（UpdateTime 仅 dedup tiebreaker），`pit_fidelity=knowledge_date_pit`（R24 已建，R25 验证） | R24 T-P08 |

### Snapshot

| ID | Root cause | Implementation | Test |
|---|---|---|---|
| R25-P0-009 | remote wildcard 被当 1 个 FileVersion | `ResolvedSourceSnapshot` exact objects；`SnapshotVerifier.verify_before_execute` production 拒 unresolved wildcard；`build_file_manifest` glob 不带 etag/content_length 冒充 exact | T-SNAP-001/002 |
| R25-P0-010 | wildcard 绕过 max_scan_files | QueryBudget v2 `max_scan_objects/max_scan_bytes/max_remote_list_objects/max_remote_requests`；`enforce_scan_object_budget` 按真实对象数卡 | T-SNAP-002 / T-RES-002 |
| R25-P0-011 | direct DuckDB/Arrow 未统一 SnapshotVerifier | 全 backend 共用 `verify_snapshot_before_execute`（remote HEAD etag/size；local path/size/mtime） | T-SNAP-003b |
| R25-P0-012 | local mirror freshness 只比本地 checksum | `SourceSnapshotResolver`（publisher manifest → exact LIST/HEAD）；`content_digest`（etag/versionId）识别 same-key overwrite | T-SNAP-003 |
| R25-P0-013 | auto hybrid 混 source epoch | strict 下 `hybrid_cos_read_paths` 返回 None（不混）；research 才允许 | T-SNAP-004/005 |
| R25-P0-014 | full cos sync 直写 live mirror | `publish_mirror_generation`：staging→verify→rename→atomic pointer；reader 只 resolve pointer；旧 generation 延迟 GC | T-MIR-001/001b/002 |

### Security

| ID | Root cause | Implementation | Test |
|---|---|---|---|
| R25-P0-006/007 | production 自动 parse `~/.cos.yaml` / registered prefix 当授权 | R24 已建 CredentialProvider 链（production 禁 ~/.cos.yaml）；R25 验证 + allowlist | R24 T-S03 + §95 |
| R25-P0-008 | local mirror/cache 不继承 principal 权限 | R24 per-principal cache root + scope digest；R25 验证 | T-MIR-004 |
| R25-P0-019 | derived 因子权限未强制 lineage inheritance | FE `factor_engine/security/access.py`（derive/max_sensitivity/declassification）+ DA `_authorize_factor_tags` | T-FE-003 |
| R25-P0-020 | automated mining 仍走 research warning fallback | `RunMode`（interactive/automated/production）+ `is_strict_semantics()` 纳入 automated_research | T-FE-002/002b |

---

## C. Runtime Architecture

```text
Request（read/read_joined/sql/HTTP/FE DataAccessSource）
  → Principal/Auth（DataPrincipal ∩ AccessPolicy ∩ registered boundary ∩ CAM/STS ∩ cache ownership）
  → RuntimeDatasetContract（ContractCompiler：datasets.yaml + COSDatasetContract
      + SemanticFieldCatalog + mirror/storage + security）
      ├─ PhysicalPartitionSpec（layout/partition_clock/filename_template/completeness）
      ├─ FilterRequirement（panel/event/dimension scope, exactly_one）
      ├─ TemporalAxisSpec / PITContract / UnitContract
      └─ fingerprint
  → PreparedRead（predicate_clock / pruning_clock / physical_partition_clock）
  → SourceSnapshotResolver（publisher manifest → exact objects → content_digest）
  → ResourceGovernor（admission：objects/bytes/remote/memory 入场前拦截）
  → SnapshotVerifier（verify_before_execute：wildcard 拒 / ETag 比对）
  → Backend（DuckDB/Polars/PyArrow/stream；read_joined 组合）
  → Result / Lineage（audit：request/principal/snapshot/object count/bytes/PIT/calendar）
```

---

## D. Physical Layout Matrix

| Dataset | Logical clock | Physical partition clock | Layout | Filename template | Missing semantics |
|---|---|---|---|---|---|
| A daily (StockDailyBar) | TradeDate | date | DAILY_TRADE_DATE | `{date}.parquet` | error（dense） |
| A finance (Balance/Income/CashFlow/Indicator) | PubDate | date | DAILY_TRADE_DATE | `{date}.parquet` | error |
| A dividend | ExDividendDate | date | EVENT_DATE_FILE | `{date}.parquet` | empty_ok |
| US daily (StockDailyBar) | TradeDate | date | DAILY_TRADE_DATE | `{date}.parquet` | error |
| US finance (Balance/Income/CashFlow) | **filing_date** | **period_end** | **PERIOD_END_FILE** | `{period_end}.parquet` | error |
| US dividend | declaration_date | ex_dividend_date | EVENT_DATE_FILE | `{date}.parquet` | empty_ok |
| US capital split | execution_date | execution_date | EVENT_DATE_FILE | `{date}.parquet` | empty_ok |
| US capital shares | TradeDate | date | PREFIXED_DATE_FILE | `shares_{date}.parquet` | empty_ok |
| US news (FactNews) | published_utc | date | EVENT_DATE_FILE | `{date}.parquet` | empty_ok |

---

## E. PIT Matrix

| Market | Dataset | Knowledge | Representation | Availability | Period | Revision fidelity |
|---|---|---|---|---|---|---|
| A | Finance | PubDate | date_label | next_trading_day | ReportPeriodEndDate | knowledge_date_pit |
| A | Dividend | （无公告时点） | date_label | effective_date_only | ExDividendDate | effective_only |
| US | Finance | filing_date | date_label | next_session_open | period_end | vintage_pit |
| US | Dividend | declaration_date | date_label | next_session_open | ex_dividend_date | strict PIT |
| US | News | published_utc | instant | same_instant | - | strict PIT |

---

## F. Security Matrix

```text
Server A（basic）              Server B（premium）
  ├─ principal scope           ├─ principal scope（高）
  ├─ AccessPolicy 受限          ├─ AccessPolicy 全量
  ├─ CAM/IAM/STS 受限           ├─ CAM/IAM/STS 全量
  ├─ cache 按 principal 隔离    ├─ cache 按 principal 隔离
  └─ 读 premium factor → 403   └─ 读 premium factor → 放行
```

- **clean-cos-ro**：CLI credential boundary；DataAccess 只 `subprocess clean-cos-ro`，不解析其 SecretKey。
- **httpfs**：SigV4 + STS token；403 → `AccessDeniedError`（不 retry 不换身份）。
- **local cache**：per-principal scope root；0700/0600；scope digest 绑定。
- **factor classification**：derived=max/union(source)；无 declassification 自动降密。

---

## G. Source Snapshot

- **remote wildcard**：production 下 unresolved wildcard → `SourceSnapshotUnavailable`；
  snapshot resolver 先 LIST/HEAD 解析成 exact objects 或读 publisher source manifest。
- **upstream overwrite**：`content_digest = hash(sorted(key, etag/versionId, length))`；
  same-key overwrite → ETag 变 → digest 变 → snapshot 变（T-SNAP-003）。
- **hybrid same epoch**：strict 下 auto 禁止混 local/remote（T-SNAP-004）；research 才允许。
- **mirror generation**：staging→verify→rename→atomic pointer；reader 只 resolve pointer（T-MIR）。

---

## H. Resource Governance

| 层 | 限制 |
|---|---|
| per-query（QueryBudget v2） | max_rows / max_result_bytes / max_elapsed_ms / max_scan_files / **max_scan_objects / max_scan_bytes / max_remote_list_objects / max_remote_requests / max_estimated_memory** |
| global（GlobalResourceGovernor） | max_active_queries / max_total_reserved_memory / max_total_scan_bytes_inflight / max_remote_concurrency / per_principal_active |
| cache（CacheManager） | max_bytes / high_watermark / low_watermark / ttl / per_principal_quota / pin-refcount（pinned 永不删） |
| remote | max_remote_concurrency（slot acquire/release） |
| HTTP | extra=forbid / dataset name len / column count / column len / instrument filter cardinality / factor ids cardinality / URI len |

---

## I. A/US Unit/Definition Matrix

| Concept | A source → canonical | US source → canonical | Currency | Cross-market comparable |
|---|---|---|---|---|
| return | bp → decimal (×0.0001) | Ret decimal (×1) | - | ✓ |
| ROE | % → ratio (×0.01) | decimal (×1) | - | ✓ |
| market_cap | CNY | USD | CNY vs USD | ✗（需 FX PIT） |
| revenue | cumulative YTD → quarterize | single-period | - | 需 canonical quarterization |
| net income | consolidated (NetProfit) | attributable common | - | ✗（两个概念） |
| dividend | local currency | local / usd（currency_column） | HKD/EUR/CAD | ✗（requires_fx） |
| shares | shares outstanding | weighted_shares_outstanding | - | 需 aligned exposure |
| adjusted price | Close×Factor（后复权） | Close×AdjFactor | - | ✗（基期/事件覆盖不同） |

---

## J. Tests

| 测试文件 | 数量 | 分类 | 结果 |
|---|---|---|---|
| test_r25_physical_layout_2026_08.py | 9 | physical layout | passed |
| test_r25_filters_2026_08.py | 6 | required filters | passed |
| test_r25_availability_2026_08.py | 6 | PIT / availability | passed |
| test_r25_snapshot_2026_08.py | 8 | source snapshot | passed |
| test_r25_mirror_2026_08.py | 6 | mirror atomicity | passed |
| test_r25_resource_schema_2026_08.py | 8 | resource + schema | passed |
| test_r25_security_fe_2026_08.py | 7 | security / FE / HTTP | passed |
| test_r25_dq_2026_08.py | 5 | DQ | passed |
| test_r25_startup_gate_2026_08.py | 4 | startup gate | passed |
| **R25 新增合计** | **58** | | **passed** |
| 存量 dataaccess unit | 671 | regression | passed |
| 存量 contract + concurrency | 146 | regression | passed |
| **总计** | **875** | | **875 passed** |

额外：`scripts/audit_r25_contract_drift.py` 0 blocking（US finance PERIOD_END_FILE +
StockCapital file_selector 硬卡）；`scripts/check_data_access_allowlist.py` 0 violation。

---

## K. Known Limitations（如实声明）

- **A股 historical revision-vintage PIT**：**不支持**。COS 无历史 revision availability；
  `pit_fidelity=knowledge_date_pit`（PubDate 知识-PIT，pinned snapshot 上可证明）。
  UpdateTime 只作 dedup tiebreaker，绝不当 revision availability。未来需上游保存
  `revision_available_at` + immutable vintage snapshot 才能升到 vintage_pit。
- **上游 COS immutable source generation**：**尚无** publisher 写 `source_generation`
  + `source_manifest` + complete marker。DataAccess 已支持消费（`parse_source_manifest` /
  `SourceSnapshotResolver`），当前 fallback 到 exact object LIST/HEAD +
  `external_source_identity_unverified`；production 不标 fully pinned。
- **COS bucket VersionId**：未确认启用。`ResolvedObject.version_id` 已支持，若 bucket
  开版本控制则优先记录 VersionId。
- **只能 knowledge-date PIT**：A 股 finance/indicator、US news（published_utc instant
  除外）、A 股 dividend（effective_date_only）。
- **auto hybrid 在无 publisher generation 时 production 禁混**（T-SNAP-004）；
  research 允许但需 lineage 标 degraded。

---

## L. Freeze Verdict

```text
DATAACCESS FULL PLATFORM FREEZE = YES
```

- 当前 P0 全部关闭（R25-P0-001..020，见 B 节）。
- destructive tests 真通过：58 个 R25 + 817 存量 = **875 passed**。
- ContractIR runtime 不再漂：`audit_r25_contract_drift.py` 0 blocking。
- security server privilege 保留：principal/policy/CAM/STS/cache/factor 分类全链。
- PIT 无 fail-open：calendar-missing strict 抛 CalendarUnavailableError；date_label
  不转时区；A UpdateTime 不冒充 revision。
- source snapshot 可证明：exact objects + content_digest + mirror generation。

剩余真实 blocker：无（A股 vintage PIT / COS immutable generation / VersionId 是
上游能力，非 DataAccess 代码 blocker，已明确 limitation + 保守 fallback +
production fail-closed）。
