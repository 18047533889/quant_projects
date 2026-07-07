# Factor Layer

这一层负责因子相关的核心能力，包括因子定义与计算、自动化 YAML 生成、单因子评估、准入决策，以及特化场景下的分钟级因子项目。

## 目录结构

| 目录 | 作用 | 推荐入口 |
| --- | --- | --- |
| [factor_engine/README.md](factor_engine/README.md) | 因子表达式、DSL、编译、执行与 factor lake | `runtime/engine.py` |
| [factor_agent/README.md](factor_agent/README.md) | 研报 PDF → YAML → 评分修订循环 | `main.py` |
| [factor_evaluation/README.md](factor_evaluation/README.md) | 因子值和市场数据的评估流水线 | `run_from_config.py` |
| [factor_admission/README.md](factor_admission/README.md) | 因子评估结果的准入与 catalog 写入 | `run_from_config.py` |
| [factor_indicators_lysj/README.md](factor_indicators_lysj/README.md) | NQ 分钟级因子与评估框架 | `factor_vwap_reversion.py` |
| [factor_pool/README.md](factor_pool/README.md) | 预留的因子池目录 | 目前以说明文档为主 |

## 典型流程

### 1. 因子生产主链

`factor_engine` 负责把表达式或 YAML 配置转成因子值，并物化到 factor lake。

常见入口：

- `FactorEngine.run(...)`
- `FactorEngine.run_from_config(...)`
- `FactorEngine.materialize_from_config(...)`

### 2. 因子评估与准入

`factor_evaluation` 读取因子值与市场价格，输出 IC、RankIC、分层收益和 long-short 指标；`factor_admission` 再根据规则或人工配置做 approve/reject 决策，并写入 catalog。

### 3. Agent 自动化生成配置

`factor_agent` 负责把研报内容转成 YAML 配置草稿，并通过评分反馈循环持续修订。

### 4. 特化分钟级项目

`factor_indicators_lysj` 是一条独立的小型项目线，重点做 NQ 分钟级 VWAP Reversion 因子和对应评估。

## 建议阅读顺序

1. [factor_engine/README.md](factor_engine/README.md)
2. [factor_evaluation/README.md](factor_evaluation/README.md)
3. [factor_admission/README.md](factor_admission/README.md)
4. [factor_agent/README.md](factor_agent/README.md)
5. [factor_indicators_lysj/README.md](factor_indicators_lysj/README.md)

## 与上下游的关系

- 上游数据来自 [../raw_data_layer/README.md](../raw_data_layer/README.md)
- 下游策略层消费 factor lake，见 [../strategy_layer/README.md](../strategy_layer/README.md)
- 端到端示例见 [../demo/README.md](../demo/README.md)