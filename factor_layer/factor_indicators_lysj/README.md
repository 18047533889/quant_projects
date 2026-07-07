# factor_indicators_lysj

这是一个特化的分钟级期货因子项目，当前重点是 NQ 的 VWAP Reversion 因子，以及配套的评估框架和结果导出。

## 当前包含的能力

| 文件 | 作用 |
| --- | --- |
| `factor_vwap_reversion.py` | 生成 VWAP Reversion 因子 |
| `factor_evaluation_framework.py` | horizon 评估、分层回测、多空持有回测 |
| `evaluation/` | 评估相关模块和输出 |
| `PROJECT_HANDOFF_AI.md` | 更细的项目交接和实现说明 |

## 运行路径

### 1. 生成因子

```bash
python factor_layer/factor_indicators_lysj/factor_vwap_reversion.py
```

### 2. 跑评估框架

```bash
python factor_layer/factor_indicators_lysj/factor_evaluation_framework.py
```

## 当前数据文件

- `NQ.parquet`
- `NQ_vwap.parquet`
- `factor_output.parquet`
- `evaluation_output/`

## 使用前注意

- 这个目录是独立小项目，不走 factor lake 主链
- 交接文档指出 `factor_vwap_reversion.py` 中历史 `DATA_DIR` 仍可能指向已不存在的分钟级分区目录，运行前需要确认路径
- 如果你要了解指标定义、状态标签、回测口径和扩展点，应直接阅读 [PROJECT_HANDOFF_AI.md](PROJECT_HANDOFF_AI.md)

## 适用场景

- 你在做分钟级期货因子研究
- 你需要比日频 factor_evaluation 更细粒度的 horizon / 状态分析
- 你希望保留一条和主 factor lake 链路相对独立的实验环境