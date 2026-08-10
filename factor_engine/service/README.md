# Factor Engine HTTP 服务（完整使用说明）

仓库：https://github.com/HKUST-QUANT-SOCIETY/factor_engine  
源码目录：本仓库 `service/`（`app.py` 适配层；计算内核仍是 `runtime.engine.FactorEngine` / `pipeline.py`）  
CLI：`factor-engine-serve`（`pip install 'factor-engine[service]'`）

这是一层 **薄 FastAPI 适配**：不改编译/执行内核，只把校验、计算、物化挂成 HTTP。

---

## 1. 安装与启动

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git
cd factor_engine
pip install -e ".[service]"

# 私有仓无 clone 时（<TOKEN> = 有 repo 权限的 GitHub PAT）：
# pip install "factor-engine[service] @ git+https://<TOKEN>@github.com/HKUST-QUANT-SOCIETY/factor_engine.git"

# 必须从仓库根启动，保证 `api` / `runtime` 可 import
PYTHONPATH=. factor-engine-serve --host 0.0.0.0 --port 8088
# 等价：
# PYTHONPATH=. python -m service.app --host 0.0.0.0 --port 8088
```

读数仍走 `data_access` 时，同机再装：

```bash
pip install "data-access @ git+https://<TOKEN>@github.com/HKUST-QUANT-SOCIETY/data_access.git"
```

仓库：https://github.com/HKUST-QUANT-SOCIETY/data_access

| 环境变量 | 含义 | 默认 |
|----------|------|------|
| `FACTOR_ENGINE_SERVICE_ROOT` | job manifest 落盘根 | `./.factor_engine_service` |
| `FACTOR_ENGINE_SERVICE_SYNC` | `1` 时默认同步跑完再返回 | 关（后台线程） |
| `FACTOR_ENGINE_OPERATOR_BACKEND` | inline compute 未指定 backend 时 | `auto` |
| `FACTOR_ENGINE_SERVICE_API_KEY` | 服务 API key（配置后所有路由需认证） | 无 |
| `FACTOR_ENGINE_SERVICE_API_KEY_MAPPING` | JSON `{key: {identity, roles}}` 主体映射（R21-015） | `{}` |
| `FACTOR_ENGINE_SERVICE_ALLOW_OPEN` | 允许 research 路由匿名（不影响 production 路由，R21-013） | 关 |
| `FACTOR_ENGINE_SERVICE_DURABLE_STORE` | `sqlite` 启用多 worker 安全的持久化 job store（R21-076..079） | JSON manifests |
| `FACTOR_ENGINE_SERVICE_MAX_QUEUE` / `MAX_WORKERS` | 有界队列 / 并发 worker（R21-048..051） | `64` / `4` |
| `FACTOR_ENGINE_SERVICE_JOB_TIMEOUT` | 任务超时秒（R21-055） | `300` |
| `FACTOR_ENGINE_SERVICE_SOURCE_PROFILES` | JSON 已批准数据源 profile（R21-020） | `{}` |
| `FACTOR_ENGINE_SERVICE_MAX_FORMULA_BYTES` | 公式字节上限（R21-038） | `65536` |

Manifest 路径：`$FACTOR_ENGINE_SERVICE_ROOT/manifests/{run_id}.json`

---

## 2. 接口一览（R21）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活：`{"status":"ok","service":"factor_engine"}` |
| GET | `/livez` | 仅进程存活（R21-083） |
| GET | `/readyz` | 就绪：blocker / disk / queue / 证据一致性（R21-084..086） |
| GET | `/metrics` | 核心 metrics + 队列快照 |
| GET | `/factor-engine/operators` | DSL 白名单算子列表 |
| POST | `/factor-engine/validate-spec` | 校验 DSL / spec（**不跑数**） |
| POST | `/factor-engine/research/compute` | research 计算（允许 sync） |
| POST | `/factor-engine/production/compute` | production 计算（**恒异步**，需 API key） |
| POST | `/factor-engine/production/materialize` | production 物化（需 MATERIALIZE 权限） |
| POST | `/factor-engine/jobs/compute` | 通用计算（与专用路由同一校验/admission，R21-225） |
| POST | `/factor-engine/jobs/materialize` | 通用物化 |
| GET | `/factor-engine/jobs/{run_id}` | 任务状态（owner-or-admin，R21-017） |
| GET | `/factor-engine/jobs/{run_id}/artifacts` | 产物（owner-or-admin） |
| POST | `/factor-engine/jobs/{run_id}/cancel` | 取消（R21-057） |
| POST | `/factor-engine/jobs/{run_id}/retry` | 重试（R21-167..170） |

OpenAPI：`http://主机:8088/docs`

### 2.1 认证 / 授权（R21-011..018）

- `FACTOR_ENGINE_SERVICE_API_KEY` 配置后，所有路由都要求有效 `X-API-Key`；
  `FACTOR_ENGINE_SERVICE_ALLOW_OPEN=1` 只放行 research 路由。
- production 路由**无条件**要求认证，不受 ambient `QUANT_PRODUCTION_MODE` 影响。
- 主体来自 API-key→principal 映射 / JWT / mTLS / 反向代理，**不信任
  `X-Request-Identity` 头**（R21-014）。
- `GET /jobs/{id}` 与 `/artifacts` 默认 owner-or-admin（R21-017）。

### 2.2 production 政策不可降级（R21-001..005）

production endpoint 调 config 时强制 production：config 中 research /
`pit_enforce=false` / `dq.strict=false` / 直接本地写 target 一律拒绝
（`PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT`），不会静默按 research 跑。

---

## 3. 请求 / 响应约定

### 3.1 `POST /factor-engine/validate-spec`

```bash
curl -s -X POST http://127.0.0.1:8088/factor-engine/validate-spec \
  -H 'content-type: application/json' \
  -d '{"formula":"rank(ts_mean(close, 5))","surface":"compat"}'
```

也接受 `dsl` 或 `factor.expr` 字段。返回：`ok` / `errors` / `warnings` / `checked`。

### 3.2 `POST /factor-engine/jobs/compute`

两种模式（二选一）：

**A. 配置文件**（服务机本地路径）：

```json
{
  "config_path": "/path/to/factor.yaml",
  "requested_by": "alice",
  "sync": true
}
```

**B. 内联 DSL**（需带 `data_source`）：

```json
{
  "formula": "rank(ts_mean(close, 5))",
  "name": "mom5_rank",
  "data_source": {
    "type": "data_access",
    "dataset": "ashare_stock_daily",
    "fields": {"close": "Close"},
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  },
  "backend": {"type": "auto"},
  "sync": true
}
```

立即返回：

```json
{"run_id":"...","status":"submitted|running|succeeded|failed","submitted_at":"...","job_type":"compute"}
```

`sync: true`（或环境变量 `FACTOR_ENGINE_SERVICE_SYNC=1`）时在同一 HTTP 周期内跑完；否则后台线程，用 `GET .../jobs/{run_id}` 轮询。

### 3.3 `POST /factor-engine/jobs/materialize`

```json
{
  "config_path": "/path/to/factor.yaml",
  "lake_root": "/data/factors/lake",
  "factor_id": "mom5_rank",
  "author": "alice",
  "sync": true
}
```

`config_path` **必填**。可选透传：`lake_root` / `factor_id` / `author` / `frequency` / `description` / `expression`。

### 3.4 查询任务

```bash
curl -s http://127.0.0.1:8088/factor-engine/jobs/<run_id>
curl -s http://127.0.0.1:8088/factor-engine/jobs/<run_id>/artifacts
```

公共字段：`run_id`、`service`、`job_type`、`status`、`submitted_at`、`started_at`、`finished_at`、`error`、`result_summary`、`artifacts`。

---

## 4. 最小端到端示例

```bash
BASE=http://127.0.0.1:8088

curl -s "$BASE/health"
curl -s "$BASE/factor-engine/operators" | head

curl -s -X POST "$BASE/factor-engine/validate-spec" \
  -H 'content-type: application/json' \
  -d '{"formula":"rank(ts_mean(close, 5))"}'

# 同步计算（按你机器上真实 data_source 改）
curl -s -X POST "$BASE/factor-engine/jobs/compute" \
  -H 'content-type: application/json' \
  -d '{
    "formula":"rank(ts_mean(close, 5))",
    "name":"mom5_rank",
    "data_source":{
      "type":"data_access",
      "dataset":"ashare_stock_daily",
      "fields":{"close":"Close"},
      "start_date":"2024-01-01",
      "end_date":"2024-01-31"
    },
    "sync": true
  }'
```

---

## 5. 相关文档

| 文档 | 用途 |
|------|------|
| [`../README.md`](../README.md) | 库 + 服务入口 |
| [`../docs/FactorEngine完全指南.md`](../docs/FactorEngine完全指南.md) | 模块总指南（含 HTTP 章节） |
| [`../IT_HANDOFF.md`](../IT_HANDOFF.md) | IT 对接（§8–9 服务化） |
| [`../INTERFACE.md`](../INTERFACE.md) | 模块级接口清单 |
| https://github.com/HKUST-QUANT-SOCIETY/data_access | 读数层（`data_source.type: data_access`） |
