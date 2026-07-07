# factor_evaluation 模块接口规范 (INTERFACE.md)

本文档定义了 `factor_evaluation` 模块作为独立服务节点时的标准交互契约。

## 1. 模块定位
负责对因子资产进行标准化评估，产出 IC、收益风险、分组回测等指标，并生成 Job Manifest。

## 2. Query API (同步查询)

### `GET /catalog/evaluation-templates`
返回当前支持的评估模板及其参数要求。

**响应示例**:
```json
{
  "templates": [
    {
      "name": "us_daily_cross_section",
      "description": "美股日频横截面标准评估",
      "required_params": ["factor_id", "sample_window"],
      "metrics": ["ic", "rank_ic", "long_short_returns", "monotonicity"]
    }
  ]
}
```

## 3. Job API (异步任务)

### `POST /evaluation-jobs`
提交一个新的评估任务。

**请求参数**:
- `factor_id` (string, required): 目标因子 ID。
- `template_name` (string, optional): 默认为 `default`。
- `sample_start` (string, required): 格式 `YYYY-MM-DD`。
- `sample_end` (string, required): 格式 `YYYY-MM-DD`。
- `universe` (string, optional): 股票池，如 `nasdaq100`。

**响应示例**:
```json
{
  "run_id": "eval_20260502_001",
  "status": "queued",
  "submitted_at": "2026-05-02T14:00:00Z"
}
```

### `GET /evaluation-jobs/{run_id}`
查询任务状态及结果索引。

**响应示例**:
```json
{
  "run_id": "eval_20260502_001",
  "status": "success",
  "manifest": {
    "artifacts": {
      "summary": "results/internal/eval_20260502_001/summary.json",
      "metrics": "results/internal/eval_20260502_001/metrics.csv",
      "plot_ic": "results/public/eval_20260502_001/ic_chart.png"
    },
    "visibility": "internal"
  }
}
```

## 4. Event (事件通知)

### `factor_evaluated`
任务成功完成后发出。

**Payload**:
```json
{
  "event_type": "factor_evaluated",
  "run_id": "eval_20260502_001",
  "factor_id": "f_vol_std_20",
  "status": "success",
  "summary": {
    "ic_mean": 0.05,
    "sharpe": 1.2
  }
}
```

## 5. 结果权限分层 (Manifest Visibility)
- **Private**: 包含逐日回测明细数据、底层计算 Trace。
- **Internal**: 包含汇总指标 (Summary)、IC 序列。
- **Public**: 包含脱敏后的评估结论 (Pass/Fail) 和概览图表。

## 6. 核心方法映射 (内部实现)
- `evaluate_factor()`: 执行主评估逻辑。
- `save_results()`: 生成并持久化 Artifacts。
- `create_manifest()`: 构造标准 Manifest JSON。
