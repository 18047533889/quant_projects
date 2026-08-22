# 从零跑通 accurate-batch

本文只说明一件事：在 Windows 本机从空 Python 环境开始，使用现有
`lqtp_data`、目标仓位 parquet 和 Barra-lite 风险模型，跑通
`standard_accurate_v2` 的 24 组准确回测。

以下命令均假设当前目录是工作区根目录。工作区名称和所在盘符可以任意设置。

## 1. 准备目录

推荐把代码和数据放成下面的同级结构：

```text
workspace/
├── vectorbt_qs/                  # 回测代码
├── data_access/                  # 本地数据读取组件
├── lqtp_data/                    # A 股行情与指数数据
│   ├── Calendar/
│   │   └── full.parquet
│   ├── StockDailyBar/
│   │   ├── 2024-01-02.parquet
│   │   └── ...
│   ├── StockDividend/
│   │   ├── 2024-01-02.parquet
│   │   └── ...
│   ├── IndexDailyBar/
│   │   ├── 2024-01-02.parquet
│   │   └── ...
│   └── IndexConstituent/
│       ├── 2024-01-02.parquet
│       └── ...
├── v2_sbi_mvl_fullA/             # Barra-lite 风险模型
│   ├── manifest.yaml
│   ├── exposure.parquet
│   ├── factor_returns.parquet
│   ├── factor_cov.parquet
│   └── specific_risk.parquet
├── strategy_data/
│   └── target_positions.parquet  # 用户的目标仓位
└── backtest_results/             # 回测输出；程序可自动创建
```

其中：

- `lqtp_data` 必须覆盖回测区间，并至少包含结束信号日之后的下一个交易日；
- `StockDailyBar` 提供原始价、复权因子、停牌、涨跌停和成交量；准确模式使用原始
  `open`/`vwap`、原始 High/Low 和原始涨跌停价执行并按真实股数估值；
- `StockDividend` 提供除权除息日的现金分红、送股与转增股，准确模式会将它们分别
  计入现金和真实股数，不能缺失；
- `IndexDailyBar` 提供沪深300、中证500和中证1000的基准净值；
- `IndexConstituent` 提供基准成分权重，用于计算基准及主动风险暴露；
- Barra 目录最少必须存在 `manifest.yaml`、`exposure.parquet` 和
  `factor_returns.parquet`。当前 `v2_sbi_mvl_fullA` 的风险暴露日期为
  2024-01-02 至 2025-12-31。

按推荐结构摆放时，不需要手动配置行情路径。`accurate-batch` 会自动使用同级的
`lqtp_data`，并让 `data_access` 跳过 COS 镜像。

如果 `lqtp_data` 必须放在其他位置，运行命令前设置：

```powershell
$env:VECTORBT_QS_ASHARE_DATA_ROOT = "C:\path\to\lqtp_data"
```

## 2. 创建并安装环境

支持 Python 3.11 和 3.12，推荐新环境使用 Python 3.12。首次安装执行：

```powershell
conda create -n vectorbt_qs python=3.12 -y
conda activate vectorbt_qs

python -m pip install --upgrade pip
python -m pip install -e .\data_access
python -m pip install -e .\vectorbt_qs
```

`data_access` 是推荐安装项：它会用 DuckDB 做列裁剪和日期、股票过滤。本地行情
仍有兼容读取路径，但全量回测会明显更慢。

确认命令入口可用：

```powershell
python -m vectorbt_qs --help
python -m vectorbt_qs accurate-batch --help
```

如果使用已经配置好的本机环境，只需：

```powershell
conda activate vectorbt_qs
```

## 3. 准备目标仓位

`target_positions.parquet` 必须是 pandas 宽表：

| 位置 | 要求 |
|---|---|
| index | 无时区、无日内时间、升序且不重复的 `DatetimeIndex` |
| columns | 唯一且非空的 A 股代码，例如 `000001.SZ`、`600519.SH` |
| values | 数值型目标权重；A 股不允许负权重 |
| 每行权重和 | 不超过 `1.0` |
| `0.0` | 该股票目标仓位为零 |
| `NaN` | 不对该股票下达新目标，保持实际持仓 |

因为冻结配置同时产出日频和周频结果，推荐输入每个交易日一行完整目标权重；未选中
股票明确写 `0.0`。冻结口径使用 `execution_lag=1`，即信号日的目标仓位在下一个
交易日尝试成交。

运行前可以做一次只读检查：

```powershell
@'
import pandas as pd

path = r"strategy_data\target_positions.parquet"
w = pd.read_parquet(path)

assert isinstance(w.index, pd.DatetimeIndex), "index 必须是 DatetimeIndex"
assert w.index.is_monotonic_increasing and w.index.is_unique, "日期必须升序且唯一"
assert w.index.tz is None and w.index.equals(w.index.normalize()), "日期不能带时区或日内时间"
assert w.columns.is_unique and all(isinstance(x, str) and x.strip() for x in w.columns)
w = w.astype(float)
assert not (w.fillna(0.0) < 0.0).any(axis=None), "A 股不能有负权重"
assert not (w.fillna(0.0).sum(axis=1) > 1.0 + 1e-8).any(), "单日权重和不能超过 1"

print(f"OK: {w.shape[0]} 个日期 × {w.shape[1]} 只股票")
print(f"日期范围: {w.index.min().date()} ~ {w.index.max().date()}")
'@ | python -
```

## 4. 检查三类输入

仍在工作区根目录执行：

```powershell
$required = @(
    "strategy_data\target_positions.parquet",
    "lqtp_data\Calendar\full.parquet",
    "lqtp_data\StockDailyBar",
    "lqtp_data\IndexDailyBar",
    "lqtp_data\IndexConstituent",
    "v2_sbi_mvl_fullA\manifest.yaml",
    "v2_sbi_mvl_fullA\exposure.parquet",
    "v2_sbi_mvl_fullA\factor_returns.parquet"
)

foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "缺少输入: $path"
    }
}

Write-Host "输入路径检查通过"
```

## 5. 先跑短区间

先用少量交易日确认环境、路径和输出全部打通。即使是短区间，也会运行冻结参数网格
中的全部 24 组。下面的日期必须落在仓位和风险模型共同覆盖的区间内，可按实际
仓位日期替换：

```powershell
python -m vectorbt_qs accurate-batch `
  --positions="strategy_data\target_positions.parquet" `
  --barra-root="v2_sbi_mvl_fullA" `
  --output-root="backtest_results\smoke" `
  --start="2024-01-02" `
  --end="2024-01-12" `
  --workers=2
```

正常启动时应依次看到：

```text
[profile] standard_accurate_v2
[profile] 固定生成 24 组准确回测
[profile] 回测进程数: 2
[ashare] 加载日频行情: ...
[ashare] 开始准确回测: ...
```

首次运行需要编译和缓存 Numba 函数，会比之后的运行慢。`--workers=1` 为串行；
本机正式使用建议从 `--workers=2` 开始，内存充足后再提高到 4。

## 6. 跑正式区间

短测成功后换一个新的结果根目录：

```powershell
python -m vectorbt_qs accurate-batch `
  --positions="strategy_data\target_positions.parquet" `
  --barra-root="v2_sbi_mvl_fullA" `
  --output-root="backtest_results\standard_accurate_v2" `
  --start="2024-01-01" `
  --end="2025-12-31" `
  --workers=4
```

`--start` 不写时默认是 `2024-01-01`；`--end` 不写时默认使用仓位数据的最后
日期。`--workers=4` 和 `--workers 4` 两种写法等价。

这条命令固定运行：

- 基准：沪深300、中证500、中证1000；
- 成交价格：原始 `open`、原始 `vwap`；
- 佣金：`0.002`、`0.001`；
- 调仓频率：`1D`、`1W`。

总计生成 `3 × 2 × 2 × 2 = 24` 份报告，执行模式固定为 `accurate`。benchmark
不影响交易，因此底层只计算 `2 × 2 × 2 = 8` 组交易，再为每组挂接三个基准生成
相对绩效和暴露报告。冻结口径还将单股每次实际成交限制在当日 `Volume` 的 10%；
容量不足产生 `partial_volume` 日志，剩余目标会在后续调仓日重新计算。

如果更习惯 YAML，仓库中的示例已经按本文目录结构填写了相对路径。修改其中三个
路径后可执行：

```powershell
python -m vectorbt_qs batch .\vectorbt_qs\configs\config.accurate_batch.example.yaml
```

YAML 中同样支持 `backtest.start`、`backtest.end` 和 `batch.workers`。

## 7. 检查结果

以上正式命令会生成：

```text
backtest_results/standard_accurate_v2/
└── target_positions/
    ├── batch_parameters.csv
    ├── effective_configs.json
    ├── performance_comparison.csv
    ├── nav_curves.parquet
    ├── batch_nav_curves.png
    ├── batch_nav_curves.html
    ├── benchmark_index_000300.SH_price_type_open_cost.commission_0.002_freq_1D/
    │   ├── performance_stats.csv
    │   ├── effective_config.json
    │   ├── nav_curves.parquet
    │   ├── equity_curve.png
    │   ├── annual_returns.png
    │   ├── portfolio_dashboard.html
    │   └── exposure/
    └── ...其余 23 个参数目录
```

建议先检查：

1. `batch_parameters.csv` 中是否有 24 条 `complete` 记录；若日期超出风险模型范围，
   应明确显示 `complete_with_exposure_truncation`；
2. `performance_comparison.csv` 是否有 24 组绩效；
3. `nav_curves.parquet` 是否有 24 条组合 NAV；
4. 任意参数目录中的 `nav_curves.parquet` 是否同时包含策略、基准和超额 NAV；
5. `exposure/coverage.csv` 是否满足组合 98%、基准 99% 的冻结覆盖率门槛。

PNG 图可以离线查看。Plotly HTML 当前从 CDN 加载前端库，打开交互式报告时需要能
访问互联网。

## 8. 常见失败

### 找不到 `vectorbt_qs`

确认已激活正确环境，并重新安装：

```powershell
conda activate vectorbt_qs
python -m pip install -e .\vectorbt_qs
```

### 找不到 `lqtp_data`

优先按本文目录结构把它放到 `vectorbt_qs` 同级；否则设置：

```powershell
$env:VECTORBT_QS_ASHARE_DATA_ROOT = "C:\path\to\lqtp_data"
```

### 缺少 `StockDividend`

`standard_accurate_v2` 会在交易前检查公司行为目录及
`CashDividend`、`StockDividend`、`StockTransfer` 等字段。不能用空表或复权因子
代替；请补齐现有 `lqtp_data/StockDividend`。

### `data_access` 读取失败并回退

确认安装的是当前工作区版本：

```powershell
python -m pip install -e .\data_access
python -c "import data_access; print(data_access.__file__)"
```

### 风险暴露日期或覆盖率报错

先检查 `manifest.yaml` 中的 `quality.date_min/date_max` 是否覆盖回测分析区间，再查看
对应结果目录中的 `exposure/coverage.csv`。不要通过移动风险模型日期来绕过覆盖
问题。

### 多进程占用内存过高

将 `--workers` 降为 2 或 1。并行只改变运行时间，不改变每组回测的配置和结果口径。
