# factor_evaluation

这个模块负责把因子值和市场价格对齐后，产出单因子评估指标与标准化评估文件，供研究、筛选和 `factor_admission` 消费。

## 模块定位

它解决的是“这个因子有没有研究价值”这个问题，当前主要覆盖：

- IC / RankIC
- horizon 级别的评估摘要
- quantile backtest
- long-short returns
- summary CSV / JSON

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `run_from_config.py` | CLI 入口 |
| `config_runner.py` | 配置驱动入口 |
| `pipeline.py` | 评估主流程 |
| `io.py` | 因子值和市场数据读入、标准化 |
| `config.py` | 配置 dataclass |
| `tests/` | runtime 与指标输出测试 |

## 运行方式

```bash
python factor_layer/factor_evaluation/run_from_config.py path/to/factor_evaluation.yaml
```

CLI 成功后会打印：

- `factor_id`
- `run_id`
- `output_dir`

## 当前已确认的评估产物

结合运行入口和测试，可确认输出目录会生成至少这些文件：

- `summary.csv`
- `summary.json`
- `daily_ic.parquet`
- `quantile_backtest.parquet`
- `long_short_returns.parquet`

## 配置关注点

当前配置里常见的关键项包括：

- 因子源路径和因子列
- 市场价格列，默认 `open`
- horizon 列表
- 最小资产数
- 年化因子
- 因子方向 `direction`

## 与上下游的关系

- 上游通常来自 [../factor_engine/README.md](../factor_engine/README.md) 产出的 factor lake 或因子值文件
- 下游可直接接 [../factor_admission/README.md](../factor_admission/README.md)
- 组合与单资产 demo 都会在需要时消费这里的评估结果

## 何时使用

- 你需要验证一个因子的方向、稳定性和分层表现
- 你想把评估结果标准化落盘，供 admission 或 notebook 使用
- 你要把评估流程放入配置驱动的批处理链路