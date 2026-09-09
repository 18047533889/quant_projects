# Riskfolio + VectorBT 通用网格调参 Pipeline

入口脚本：

```text
riskfolio_qs/scripts/riskfolio_vectorbt_grid.py
```

MeanVar Barra 示例配置：

```text
riskfolio_qs/examples/pipeline/meanvar_barra_vectorbt_grid.yaml
```

基础 Riskfolio YAML 中通过 `run.optimizer_name` 指定要调优的优化器；
pipeline YAML 只负责固定覆盖、网格、回测、并行和排名。可用于
Barra预计算、历史协方差、绝对收益、MinVar、MeanVar以及TopN等显式优化器，
但搜索参数必须位于所选参数文件的 `overrides_whitelist` 中。

Pipeline 对每个参数组合执行以下流程：

1. 基于 `meanvar_barra_h10_base.yaml` 生成独立 trial 配置。
2. 通过 `riskfolio_qs.cli.run_from_config` 生成标准优化产物。
3. 将 `target_positions` 传入 VectorBT 的
   `standard_accurate_benchmark_v1` 冻结单配置。
4. 汇总 `search_results.csv`，按配置指标生成 `ranked_results.csv`。
5. 通过正式 `accurate-benchmark` CLI 为前 N 名生成完整报告。

## 运行

在工作区根目录执行：

```powershell
$py = "C:\Users\user\.conda\envs\whh_local_workspace\python.exe"
$env:DATA_ACCESS_SKIP_COS_MIRROR = "1"

# 只检查路径、配置和60个网格组合，不运行优化。
& $py .\riskfolio_qs\scripts\riskfolio_vectorbt_grid.py `
  -c .\riskfolio_qs\examples\pipeline\meanvar_barra_vectorbt_grid.yaml `
  --dry-run

# 正式运行。
& $py .\riskfolio_qs\scripts\riskfolio_vectorbt_grid.py `
  -c .\riskfolio_qs\examples\pipeline\meanvar_barra_vectorbt_grid.yaml
```

`execution.resume: true` 时，脚本会复用已经完成的 trial 统计和优化产物。
中断后执行同一条命令即可续跑。

## 多进程

Pipeline 支持 trial 级多进程：

```yaml
execution:
  workers: 2
```

每个 worker 独立执行一次 Riskfolio 优化和一次冻结的 VectorBT 单配置回测。
worker 内部的 VectorBT 保持 `workers=1`，避免嵌套多进程。

当前MeanVar Barra示例的两年、约5200只股票、33个因子配置，建议按每个worker峰值
约6～10 GB内存估算：

- 16 GB内存：`workers: 1`；
- 32 GB内存：`workers: 2`；
- 64 GB内存：建议从`workers: 2`开始，确认峰值后再增加。

## 主要产物

输出目录默认为：

```text
outputs/meanvar_barra_h10_grid/
```

其中：

- `trial_configs/`：每个 trial 的完整 Riskfolio 配置；
- `riskfolio_runs/`：标准 Riskfolio 优化产物；
- `trial_stats/`：单 trial 回测指标；
- `search_results.csv`：全部成功和失败记录；
- `ranked_results.csv`：按选择指标排序的成功记录；
- `best_trials.yaml`：入选组合；
- `final_reports/`：前 N 名的 VectorBT 完整准确回测报告。

## 配置边界

`backtest.profile` 当前只接受：

```yaml
profile: standard_accurate_benchmark_v1
```

这保证不同优化器之间的回测口径保持冻结。Riskfolio 的固定参数与搜索参数分别写在：

```yaml
riskfolio:
  fixed_overrides: {}
  grid: {}
```

同一个参数不能同时出现在两处。

基础配置必须显式指定优化器，例如：

```yaml
run:
  optimizer_name: meanvar_enhance_barra_precomputed
```

切换优化器时只需更换 `riskfolio.base_config`，并调整
`fixed_overrides` 与 `grid`。当 `reports.enabled: false` 时，
`backtest.barra_root` 可以省略；完整 Accurate 报告包含风险暴露分析，
因此启用报告时仍要求提供一个Barra目录。

例如TopN优化器可以使用：

```yaml
riskfolio:
  fixed_overrides: {}
  grid:
    top_n: [20, 50, 100, 200]
```

脚本会拒绝所选优化器未定义的参数，避免出现“网格成功运行但参数实际不生效”
的空维度。
