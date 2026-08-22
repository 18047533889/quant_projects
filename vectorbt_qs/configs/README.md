# 配置模板

本目录只保存可提交、可复用的 YAML 模板。个人路径、临时参数和包含凭据的配置请使用
`*.local.yaml` 命名；该类文件已由 Git 忽略。

| 文件 | 用途 | 命令 |
|---|---|---|
| `config.example.yaml` | 通用单次回测 | `python -m vectorbt_qs run configs/config.example.yaml` |
| `config.factor_fast.example.yaml` | Fast 因子研究 | `python -m vectorbt_qs run configs/config.factor_fast.example.yaml` |
| `config.batch.example.yaml` | 通用参数网格 Batch | `python -m vectorbt_qs batch configs/config.batch.example.yaml` |
| `config.accurate_batch.example.yaml` | Accurate V2 最简三路径配置 | `python -m vectorbt_qs batch configs/config.accurate_batch.example.yaml` |
| `configs_module.yaml` | Accurate V2 完整显式参数模板 | `python -m vectorbt_qs batch configs/configs_module.yaml` |

YAML 内的输入数据、Barra 等路径相对配置文件所在目录解析。通用 `run`/`batch` 的
输出路径仍按项目根目录解析；冻结 Accurate Batch 的 `output.root` 相对配置文件解析。

