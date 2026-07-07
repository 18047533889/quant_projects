# factor_admission

这个模块负责把 `factor_evaluation` 的评估结果转成明确的准入决策，并把运行记录写入 catalog，方便后续查询、审计和策略层消费。

## 模块定位

它不负责重新计算因子，也不负责生成评估指标，而是消费已经存在的评估产物，完成以下事情：

1. 校验评估 run 是否存在且和配置一致
2. 读取 `summary.json` 等评估结果
3. 根据规则阈值或人工指定配置生成决策
4. 将评估摘要和 admission 决策写入 catalog
5. 按配置选择是否额外写出决策文件

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `run_from_config.py` | CLI 入口 |
| `config_runner.py` | 配置加载和运行组装 |
| `admission.py` | 决策主逻辑 |
| `catalog.py` | catalog 与 SQLite 写入逻辑 |
| `config.py` | 配置 dataclass 和校验 |
| `tests/` | 配置和 admission 逻辑测试 |

## 运行方式

```bash
python factor_layer/factor_admission/run_from_config.py path/to/factor_admission.yaml
```

CLI 正常执行后会打印：

- `factor_id`
- `run_id`
- `decision`

## 依赖上游

这个模块依赖 `factor_evaluation` 先产出评估目录，尤其是运行目录下的 `summary.json`。

推荐配合阅读：

- [../factor_evaluation/README.md](../factor_evaluation/README.md)

## 典型产物

从当前实现可以确认，catalog 会记录：

- `factor_evaluation_runs`
- `factor_evaluation_summary`
- `factor_admission_decisions`

也就是说，这个模块更像“准入登记与决策层”，而不是单纯输出一个 approve/reject 文本。

## 何时使用

- 你已经有 factor lake 和评估结果
- 你需要按规则批量做准入
- 你需要把评估 run 和 admission 结果持久化到 catalog

如果你还没有评估结果，应先跑 [../factor_evaluation/README.md](../factor_evaluation/README.md)。