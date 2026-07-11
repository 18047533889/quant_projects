# Label 子模块功能概述（4.3）

## 一、模块目的

依据 Indicator 产出的评估摘要对因子进行贴标，并给出入库推荐路由信息。

## 二、核心函数

### run_label_pipeline

**签名**: `run_label_pipeline(source_factor_dir, evaluation_summary_path, config, operator, deepseek_client, materialize) → LabelRunResult`

**入参**:
- `source_factor_dir`: PureFactor 因子目录
- `evaluation_summary_path`: 评估摘要 JSON 路径
- `materialize`: 是否物化入库（默认 True；pipeline 模式传 False）

**出参 LabelRunResult**:
| 字段 | 说明 |
|------|------|
| `tag_package` | 标签包（规则标签 + DeepSeek 语义标签） |
| `admission_decision` | 准入决策（目标 tier） |
| `route_record` | 路由记录 |
| `lifecycle_event` | 生命周期事件 |
