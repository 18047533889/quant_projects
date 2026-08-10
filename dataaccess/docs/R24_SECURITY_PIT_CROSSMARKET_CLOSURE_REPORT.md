# R24 — DataAccess 安全 / PIT / 跨市场语义专项整改 收官报告

> 审计基线 `main@7b15a5e7`；日期 2026-08-10。
> 范围：权限边界 / 数据泄露 / COS 多服务器分级权限 / 财务 PIT / A股与美股单位与会计语义 / DataAccess↔FactorEngine 契约漂移。

**最终不变量（本轮达成）**：生产环境里，任何一个 DataAccess read 或 FactorEngine factor value，
如果不能证明「有权限、已可知、语义一致、版本可追溯」，就必须失败，而不是猜、降级、换身份或静默继续。

---

## A. Changed Files

### DataAccess

| 文件 | 修改内容 | 为什么 |
|---|---|---|
| `security/__init__.py` | 新增安全薄层导出 | §26 薄层 |
| `security/principal.py` | `DataPrincipal` / `AccessPolicy` / 动作常量 / `DATASET_SCOPED_ACTIONS` | P0-S2 §4 逻辑授权身份与策略 |
| `security/policy.py` | `DatasetAuthorizer` / `DefaultAuthorizer`（strict 下 uri:read 拒绝、AccessDenied 不 fallback） | P0-S2 §4 / P1-S9 §25 |
| `security/credentials.py` | `CredentialProvider` / `CredentialMaterial` / `EnvCredentialProvider` / `CosCliConfigProvider` / `allow_coscli_config_parse` | P0-S1 §3 凭证边界 |
| `security/redaction.py` | `redact_secret` / `redact_uri` / `redact_authorization_header` | P0-S1 §3.5 / P1-S6 §8 |
| `security/runtime.py` | `RuntimeSecurityContext` / `resolve_runtime_context` | P0-S5 §7 统一运行时上下文 |
| `security/api_principals.py` | `ApiPrincipalRegistry`（api key hash → principal/policy） | P0-S5 §7 key→principal |
| `cos/remote.py` | `S3Credentials` 加 session_token/expires_at/principal/scope + repr 脱敏；`resolve_s3_credentials` 走 provider 链；production 禁解析 `~/.cos.yaml`；废弃 `load_cos_cli_credentials_into_env`（不再写 env）；`cos_cache_root` 按 principal 隔离；`ensure_cache_root_secure`（0700/0600/symlink/world-readable fail-closed）；删除重复 `raise`（P1-2） | P0-S1 §3 / P0-S3 §5 |
| `cos/s3_duckdb.py` | STS session token 注入（CREATE SECRET SESSION_TOKEN + SET s3_session_token） | P0-S1 §3.4 STS |
| `cos/mirror.py` | 删除重复 `_expected_dates`（P1-1）；`_sync_cos_file` 用 O_EXCL 随机 temp + 0600 + symlink 拒绝；manifest 增加 principal_scope_id/access_policy_digest/credential_scope_id | P0-S3 §5.3/§5.5 / T-S08 |
| `core/exceptions.py` | `AuthorizationError` / `AccessDeniedError` / `TemporalContractError` / `FinancialRevisionAmbiguityError` + `propagate_authorization` | P1-S9 §25 |
| `core/engine.py` | `_wrap_query_error`：403/AUTH 分类为 `AccessDeniedError`（不吞不 fallback）；错误信息脱敏 | P1-S9 / P1-S6 |
| `core/audit.py` | `record(category=...)` + `record_security_event`（AUTHENTICATION_FAILED / AUTHORIZATION_DENIED / PIT_REJECTED / DATA_NOT_FOUND / DATA_CORRUPTION / QUERY_INVALID） | P1-5 §35 |
| `store.py` | 读/写入口统一 `authorize_dataset`/`authorize_uri` 门；`read_uri` privileged；`_authorize_factor_tags`；`_resolve_universe_instruments`/`manifest_version` 等 broad except 处 `propagate_authorization`；`_validate_allowed_filter_values` 加 exactly-one 卡 | P0-S2 / P1-S8 / P1-S9 / P0-PIT5 |
| `cos_contract.py` | contract 加 `time_representation` / `time_precision` / `storage_timezone` / `semantic_timezone` / `pit_fidelity` / `revision_availability_time` / `filter_cardinalities`；`_validate_filters` 加 exactly-one；`market_scoped_instrument` | P0-PIT2/4/5 §11/13/14 / P1-XM6 §22 |
| `cos_contract_ashare.py` | 财务契约：`date_label` / `pit_fidelity=knowledge_date_pit` / `revision_availability_time=None`（UpdateTime 仅 tiebreaker） | P0-PIT4 §13 |
| `cos_contract_us.py` | 财务契约：`filing_date` date_label / next session / `filter_cardinalities=exactly_one` / `pit_fidelity=vintage_pit`；dividend 声明 currency 动态；FactNews instant | P0-PIT1/3/5 §10-12/14 / P0-XM1/2 |
| `read/semantic_catalog.py` | `SemanticField` 加 time_representation / time_precision / pit_fidelity / revision_availability_time / dimension / currency / currency_column / cross_market_comparable / requires_fx / flow_semantics；`UnitSpec` + `cross_market_compatible` | P0-PIT2 / P0-XM3/5 |
| `read/temporal_join.py` | `TemporalJoinSpec` 加 time_representation / time_precision | P0-PIT2 §11 |
| `read/session_calendar.py` | `_to_local_date_time` 加 date_label（不转时区）；`compile_available_from` 加 time_representation + strict 无日历 hard fail | P0-PIT2/6 §11/15 |
| `read/contract_ir.py` | ContractIRDataset 加 R24 字段；audit 加 knowledge_time/time_representation/pit_fidelity/revision_availability_time 契约↔字段漂移检测 | §33 |
| `service/app.py` | api key→principal scopes；production 恒认证（T-S12）；`/v1/datasets` 按可见性过滤；`/v1/read_uri` 默认 403；`/v1/factors` 脱敏摘要；外部错误脱敏 | P0-S5 §7 / P1-S6 §8 |
| `read/factors.py` | `FactorMeta` 加 source_access_tags / derived_access_tags / declassification | P0-S4 §6 |
| `config/semantic_fields.yaml` | US 财务 availability→next_session_open + date_label；A 财务 date_label；US dividend 拆 local/usd + currency；net_income 拆 consolidated/attributable；flow_semantics；money 字段 currency + cross_market_comparable=false；adjusted_price_backward canonical | P0-XM1/2/3/4/5 / P1-XM7 |
| `tests/unit/test_r24_security_2026_08.py` | 新增（T-S01..T-S13，13 个） | §30 |
| `tests/unit/test_r24_pit_crossmarket_2026_08.py` | 新增（T-P01..T-P11 + T-X01..T-X07，15 个） | §31/§32 |

### FactorEngine

| 文件 | 修改内容 | 为什么 |
|---|---|---|
| `security/__init__.py` | 新增安全薄层导出 | P0-S4 §6 |
| `security/access.py` | `access_tag_level` / `max_sensitivity` / `derive_derived_access_tags` / `DeclassificationApproval` / `require_declassification_approval`（禁止自动降密） | P0-S4 §6 |

### 其他

| 文件 | 修改内容 |
|---|---|
| `.gitignore` | `.cos.yaml` / `cos.yaml` / `*.credentials` / `*.secret` / `*.key` / `*secret_*` |
| `.github/workflows/secret-scan.yml` | gitleaks working tree + history（P1-S7 §9） |

---

## B. 最终 Security Model

```
请求者 / 当前服务器身份（DATA_ACCESS_PRINCIPAL_ID / api key hash → principal）
        │
        ▼
① DataAccess 逻辑授权（DataPrincipal ∩ AccessPolicy ∩ action）
    dataset:read / factor:read / uri:read / metadata:read ...
        │
        ▼
② Registry 路径边界（PathAuthorizer：请求只能落到已注册 dataset / prefix）
        │
        ▼
③ COS CAM/IAM/STS（云端最终拒绝/允许；CredentialProvider 链，不解析 ~/.cos.yaml）
        │
        ▼
④ 本地 mirror/cache 权限（per-principal cache root、0700/0600、scope digest 绑定）
```

- **server identity 从哪来**：`DATA_ACCESS_PRINCIPAL_ID` + `DATA_ACCESS_SERVER_ID`（部署注入），或 HTTP `X-API-Key` → `DATA_ACCESS_API_PRINCIPALS`（key hash → principal）。
- **AccessPolicy 从哪来**：env `DATA_ACCESS_ALLOWED_DATASETS` / `DATA_ACCESS_API_PRINCIPALS`；未配置时 DEFAULT（放行已注册 dataset、uri:read 仅 research、production 拒绝）。
- **COS credential 从哪来**：CVM/容器绑定角色 / STS 临时凭证 / 部署注入 env（`EnvCredentialProvider`）/ 显式 `CredentialProvider`；production **绝不**自动解析 `~/.cos.yaml`；research 需 `DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1`。
- **clean-cos-ro 负责什么**：CLI = credential boundary。DataAccess 只调用 `clean-cos-ro cp/ls`，不解析其 SecretKey、不猜 profile、不换更高权限 profile、403 不换 base credential。
- **httpfs 负责什么**：DuckDB 直读 s3://（SigV4 签名 + endpoint），凭证只来自 provider 链；403 → `AccessDeniedError`（不 retry、不换身份）。
- **本地 cache 怎么隔离**：`cos_cache_root` = per-principal scope 子目录；0700/0600；manifest 带 principal_scope_id/access_policy_digest/credential_scope_id；scope 不匹配的旧缓存不得复用。
- **权限拒绝怎么处理**：`AuthorizationError`/`AccessDeniedError` 原样传播（`propagate_authorization`），绝不降级成空 universe / 换身份 / 静默继续。

---

## C. PIT Matrix

| Market | Dataset | Knowledge time | Representation | Precision | Availability | Period | Revision fidelity |
|---|---|---|---|---|---|---|---|
| A | StockBalance/Income/CashFlow/Indicator | PubDate | date_label | date | next_trading_day | ReportPeriodEndDate | knowledge_date_pit（COS 无历史 revision vintage；UpdateTime 仅 dedup tiebreaker） |
| A | StockDividend | （无可靠公告时点） | date_label | date | effective_date_only | ExDividendDate | effective_only |
| US | StockBalance/Income/CashFlow | filing_date | date_label | date | next_session_open | period_end | vintage_pit（revision_availability_time=filing_date） |
| US | StockDividend | declaration_date | date_label | date | next_session_open | ex_dividend_date | strict PIT；missing declaration → unavailable；cash_amount 币种动态 |
| US | FactNews | published_utc | **instant** | timestamp | same_instant | - | strict PIT（UTC→America/New_York 时区换算） |

**timeframe 契约**：US 财务必须 exactly-one（quarterly/annual/trailing_twelve_months），missing/multiple/invalid 一律拒绝，进入 contract identity。

---

## D. Unit Matrix

| Concept | A source unit | A canonical | US source unit | US canonical | Currency issue | Cross-market comparable |
|---|---|---|---|---|---|---|
| Return | bp | decimal (×0.0001) | Ret（已 decimal） | decimal (×1) | - | 可（无量纲） |
| ROE | % | ratio (×0.01) | return_on_equity（已 decimal） | ratio (×1) | - | 可 |
| MarketCap | CNY | CNY | USD | USD | CNY vs USD | **否**（需 FX dataset + FX PIT；`cross_market_comparable=false`） |
| Dividend cash | dynamic currency | local（value+currency） | dynamic | usd（仅 USD） | HKD/EUR/CAD | 否（local requires_fx=true） |
| Revenue | cumulative YTD | quarterize 后 | quarterly single-period | single-period | - | 需 canonical quarterization |
| Net income | NetProfit（consolidated） | - | attributable common | - | - | consolidated ≠ attributable（两个概念） |
| Instrument | Symbol | ashare:000001.SZ | Ticker | us:AAPL | - | (market, instrument) 主键 |

---

## E. Tests

| 测试文件 | 数量 | 结果 |
|---|---|---|
| `tests/unit/test_r24_security_2026_08.py`（T-S01..T-S13） | 13 | passed |
| `tests/unit/test_r24_pit_crossmarket_2026_08.py`（T-P01..T-P11 + T-X01..T-X07） | 15 | passed |
| 存量 DataAccess 回归（`tests/unit/` + `tests/contract/`） | 753 | passed（含 security 改动后） |
| 合计 | **781** | **781 passed** |

---

## F. Known Limitations

> **A股当前是否真正支持 historical revision-vintage PIT？**

**No.** Current system can prove PubDate knowledge-PIT on a pinned source snapshot,
but cannot prove full historical revision-vintage PIT.

- COS 未保存历史 revision 的真实 availability；A股 `UpdateTime` 是供应商 freshness，
  不是市场可知修订时点。`pit_fidelity=knowledge_date_pit` 如实声明，绝不用 UpdateTime 假装已解决。
- 未来若需支持重述：上游需保存 `revision_available_at` / source publication timestamp /
  immutable vintage raw snapshot，然后 `revision_available_at <= decision_time`。

其他限制：
- 方案 B（真实 event visibility transaction，R14 遗留）不在本轮范围。
- DA `_read_factor_matrix` audit 的 generation 读取后重算 race（P1）留后续小修。
- `read_uri` 在 research 库内仍允许（受 PathAuthorizer 白名单约束）；production/strict 默认拒绝。
