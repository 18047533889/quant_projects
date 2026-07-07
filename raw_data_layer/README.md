# Raw Data Layer

这一层负责市场数据的获取、校验、清洗和样例数据管理，是因子层、策略层和回测层的上游数据入口。

## 目录结构

| 目录 | 作用 |
| --- | --- |
| [raw_data_fetching/README.md](raw_data_fetching/README.md) | 下载原始数据、检查 parquet 完整性 |
| [raw_data_cleaning/README.md](raw_data_cleaning/README.md) | 按 source 清洗 Massive parquet |
| `sample_data/` | 小规模样例数据或临时研究数据 |

## 推荐使用顺序

1. 先在 `raw_data_fetching/` 下载或补齐原始数据
2. 用 `validate_parquet.py` 做完整性检查
3. 用 `raw_data_cleaning/` 生成更适合下游使用的 cleaned 数据
4. 再将数据交给 factor、strategy 或 backtest 层

## 何时进入这一层

- 你需要新增或刷新原始数据
- 你怀疑上游 parquet 有损坏、缺失、日期错位或质量问题
- 你要把 Massive 数据规范成更稳定的下游输入

## 与下游关系

- 因子计算与 factor lake：见 [../factor_layer/README.md](../factor_layer/README.md)
- 策略与回测：见 [../strategy_layer/README.md](../strategy_layer/README.md) 和 [../backtest_layer/README.md](../backtest_layer/README.md)