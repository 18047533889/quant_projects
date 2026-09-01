# quant_platform — 平台集成/合同层

量化研究平台**薄集成层**：纯 stdlib DTO 合同（23 个模块，`__all__` 135 项）、auth/RBAC、
outbox/worker/orchestrator 骨架、存储适配器。它是域包的**平台侧边界**，域包不得 import 它。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/quant_platform (私有)

> 状态：**DRAFT / WIP** —— Control 与 Metadata 平面未建（无 Postgres、无 workflow runtime、
> 无统一 registry）。合同 **未 V1_FROZEN**，待跨包 reconcile。

## 拥有什么

- **`app/contracts/`** — 纯 stdlib frozen dataclasses + `typing.Protocol`
  （PURE-DTO 规则：禁止 fastapi/sqlalchemy/pydantic）。模块：`artifact_ref`、
  `event_envelope`、`jobs`、`workflow`、`candidate`、`lifecycle`（23 态 + health）、
  `identities`、`cluster_library`、`feature_set`、`rbac`/`permission`/`principal`、
  `security`、`storage`、`backtest`、`model_version`、`daily_snapshot`、
  `factor_library`、`timing`、`versioning`、`admission`、`cluster`、`_contenthash`。
- **`app/db/schema.py`** — 33 张 Postgres 兼容表（principals、roles、artifacts、jobs、
  workflow_runs、outbox/inbox、factor_candidates、cluster_*...）。
- **`app/security/`** — session-token auth（HMAC-SHA256 cookie）、密码哈希（scrypt）、
  RBAC（`resolve_principal_permissions`、`effective_permissions`）。
- **`app/storage/`** — COS artifact publisher/materializer、本地 artifact cache（校验和）、
  access guard、registry。
- **`app/adapters/data_access_storage.py`** — DataAccessStorageAdapter（guarded import）。
- **`app/worker/`** — job runner、outbox worker、publish pump（in-memory + sqlite）。
- **`app/orchestrator.py`** — candidate → evaluation → admission 流水线骨架。
- **`app/observability/`** — health checks、metrics registry、collectors。
- **`app/api/app.py`** — FastAPI：`/auth/login`、`/auth/logout`、`/auth/me`、`/health`。

## 设计规则（硬性）

- 平台**禁止重算量化逻辑**：不算 RankIC、不落第二套因子湖、不做第二套元数据注册表。
  域权威留在域包。
- `quant_platform.app.contracts` 零第三方依赖可 import（`tests/test_smoke.py` 断言）。
- 平台 ↔ 域翻译发生在 worker/adapters：`Platform DTO ↔ Domain Native Contract`
  （如 `FactorCandidateManifest` ↔ FA treatment artifact、`BacktestRequest` ↔
  vectorbt_qs `BacktestRequest`、model DTO ↔ modeling artifact）。

## 已知缺口（见 `docs/GAP_ANALYSIS.md` / `docs/CURRENT_ARCHITECTURE.md`）

- Control Plane（workflow runtime、生产 outbox、retry/promotion）— 未建。
- Metadata Plane（Postgres 统一 registry、factor catalog）— 未建。
- `research_platform` 旧双 artifact authority 待迁移。

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（`app/adapters/data_access_storage.py`、
  `data_access.read.object_store` 等，guarded import，缺装可降级）。
- **被谁调用**：`platform_web`（前端类型即 contracts DTO）。

## 安装与测试

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/quant_platform.git
cd quant_platform
pip install -e .
python -m pytest tests/ -q    # 31 个测试文件
```

## 相关仓库

- **所有域包** — 本层是它们的平台侧边界
- **platform_web** — 前端 types 镜像 `app/contracts/*`
- **data_access** — 存储适配器来源
