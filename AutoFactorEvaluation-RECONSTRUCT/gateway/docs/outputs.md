# Gateway 模块产物结构

## 一、输出目录约定

根据路由结果，输出到不同的静态库目录：

| 路由 | 输出目录 | 包含文件 |
|------|---------|---------|
| Pass | `{gateway_pass_base}/{candidate_id}/` | afv.json, manifest.json |
| Duplicated | `{duplicated_base}/{candidate_id}/` | afv.json, manifest.json |
| Rejected | `{anti_sample_base}/gateway/{candidate_id}/` | afv.json, manifest.json |

---

## 二、afv.json 格式

Gateway 阶段统一使用 `afv.json`（Alpha Factor Verification）作为元数据文件，替代旧版 `candidate.json`。文件包含上游原始字段以及本模块追加的 `Gateway` 段落。

### Gateway 段落（由本模块追加）

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "cand_20260528_a1b2c3d4",
  "campaign_id": "campaign_alpha_001",
  "formula": "ts_mean(close, 10) / ts_std(close, 10)",
  "gateway_version": "1.0.0",
  "Gateway": {
    "Label": "Pass",
    "Reason": "",
    "run_id": "gateway_20260528_083000_a1b2c3d4",
    "checked_at": "2026-05-28T08:30:00.123456+00:00",
    "recommendation": "pass",
    "checks": {
      "schema_valid": true,
      "operator_allowed": true,
      "has_future": false,
      "complexity_score": 3.5,
      "is_within_budget": true,
      "is_duplicate": false
    }
  }
}
```

### Gateway 段字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `Label` | string | 是 | 路由决策：`Pass` / `Duplicated` / `Rejected` |
| `Reason` | string | 是 | 路由原因描述 |
| `run_id` | string | 是 | 网关运行唯一标识 |
| `checked_at` | string | 是 | ISO 8601 时间戳 |
| `recommendation` | string | 是 | 推荐标签（小写）：`pass` / `duplicated` / `rejected` |
| `checks` | dict | 是 | 各项检查结果 |
| `checks.schema_valid` | bool | 是 | Schema 校验结果 |
| `checks.operator_allowed` | bool | 是 | 算子白名单校验结果 |
| `checks.has_future` | bool | 是 | 是否检测到未来函数 |
| `checks.complexity_score` | float | 是 | 复杂度评分 |
| `checks.is_within_budget` | bool | 是 | 是否在复杂度预算内 |
| `checks.is_duplicate` | bool | 是 | 是否重复因子 |
| `checks.data_quality` | dict | 否 | 数据质量检测详情（启用时） |
| `historical_report_id` | string | 否 | 重复因子的历史报告 ID（Duplicated 时） |
| `error_message` | string | 否 | 详细错误信息（Rejected 时提供） |

---

## 三、manifest.json 格式

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "cand_20260528_a1b2c3d4",
  "campaign_id": "campaign_alpha_001",
  "formula": "ts_mean(close, 10) / ts_std(close, 10)",
  "domain_root": "price_volume",
  "frequency_bucket": "daily",
  "universe_id": "A_SHARE_LQTP",
  "depth": 6,
  "parent_local_hashes": [],
  "metrics": { ... }
}
```

### manifest.json 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `schema_version` | string | 是 | Schema 版本，固定为 `disk.v1` |
| `candidate_id` | string | 是 | 候选因子 ID |
| `campaign_id` | string | 否 | 所属 campaign |
| `formula` | string | 是 | DSL 表达式 |
| `domain_root` | string | 是 | 域根 |
| `frequency_bucket` | string | 是 | 频率桶 |
| `universe_id` | string | 否 | 选股域标识 |
| `depth` | int | 否 | 表达式深度 |
| `parent_local_hashes` | list | 否 | 父因子哈希列表（用于谱系追踪） |
| `metrics` | dict | 否 | 元指标（如基准因子 IC/ICIR） |
