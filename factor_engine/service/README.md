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

Manifest 路径：`$FACTOR_ENGINE_SERVICE_ROOT/manifests/{run_id}.json`

---

## 2. 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活：`{"status":"ok","service":"factor_engine"}` |
| GET | `/factor-engine/operators` | DSL 白名单算子列表 |
| POST | `/factor-engine/validate-spec` | 校验 DSL / spec（**不跑数**） |
| POST | `/factor-engine/jobs/compute` | 提交计算任务 |
| POST | `/factor-engine/jobs/materialize` | 提交物化任务 |
| GET | `/factor-engine/jobs/{run_id}` | 任务状态 + `result_summary` |
| GET | `/factor-engine/jobs/{run_id}/artifacts` | 产物路径 / preview |

OpenAPI：`http://主机:8088/docs`

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
