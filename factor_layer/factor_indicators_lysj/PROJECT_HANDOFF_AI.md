# Factor Indicators 项目交接文档（AI-Friendly）

## 1. 项目目标
本项目用于评估分钟级期货因子（当前为 NQ 的 VWAP Reversion 因子），提供：
- 因子有效性统计：IC、RankIC、ICIR、RankICIR、IC 胜率
- 分层回测：10 组分层收益（可按 N 期转成单期收益）
- 多空持有回测：严格仓位状态机（-1/0/+1）
- 交易成本回测：支持单边手续费（当前默认 0.00002）

文档面向下一个开发者，强调可维护性、数据契约、关键公式和扩展点。

## 2. 当前目录结构
- .venv/: Python 虚拟环境
- NQ.parquet: 汇总行情（data, open, high, low, close, volume, average, barCount, Contract, trade_month）
- NQ_vwap.parquet: 回测用 VWAP 序列（timestamp, vwap）
- factor_output.parquet: 因子输出（timestamp, factor）
- factor_vwap_reversion.py: 因子生成脚本
- factor_evaluation_framework.py: 核心评估框架（含市场状态标签 & 分状态分析器）
- evaluation_output/: 评估结果导出目录（包含历史版本输出）

说明：
- evaluation_output 内可见旧 horizon 文件（如 h15/h30），当前框架默认 horizon 为 1/5/10/20。
- 原始分钟级分区数据目录 `NQ/` 已不存在；factor_vwap_reversion.py 中 `DATA_DIR` 仍指向该路径，运行前需重建该目录或修改路径。

## 3. 运行入口
### 3.1 因子生成
执行 factor_vwap_reversion.py：
- 读取 NQ 原始分钟数据（从 `NQ/` 分区目录，当前该目录已不存在，需重建或修改 `DATA_DIR`）
- 计算日内累计 VWAP 偏离因子
- 输出 factor_output.parquet

### 3.2 批量评估与导出
执行 factor_evaluation_framework.py：
- 加载 factor_output.parquet 和 NQ_vwap.parquet
- 对每个 horizon 计算完整指标
- 输出 evaluation_output 下的 CSV

## 4. 数据契约
### 4.1 输入数据
- 因子文件：factor_output.parquet
  - 必需列：timestamp, factor
- VWAP 文件：NQ_vwap.parquet
  - 必需列：timestamp, vwap

### 4.2 时间对齐规则
在 factor_evaluation_framework.py 中：
- 先按时间戳去重（保留最后一条）
- 取 factor 与 vwap 的重叠时间区间
- 用并集索引重建时间轴后再对齐

## 5. 核心模块说明
## 5.1 因子生成模块（factor_vwap_reversion.py）
- 因子定义：-(close - cum_vwap) / cum_vwap
- 逻辑含义：价格高于 VWAP 越多，因子越偏负（预期回归）
- 暖启动处理：每天前 5 条置 NaN
- 自带日度 Spearman IC 报告（独立于评估框架）

## 5.2 评估框架模块（factor_evaluation_framework.py）

框架包含以下核心类与模块：

| 类 / 函数 | 职责 |
|---|---|
| `EvalConfig` | 全局评估参数（horizons, z-score 窗口, 分层数等） |
| `MarketStatusLabel` | 市场状态标签生成器（波动率 & 日内时段） |
| `Predictivity` | IC / RankIC 计算 |
| `Backtest` | 分层回测 + 多空持有回测 |
| `StatusAnalyzer` | 分状态因子表现分析（集成上述所有组件） |
| `evaluate_horizon` / `run_evaluation` | 跨 horizon 编排入口 |

### 5.2.1 EvalConfig
主要配置由 EvalConfig 控制：
- horizons: 默认 (1, 5, 10, 20)
- zscore_window: 默认 200
- winsor_quantile: 默认 1%
- n_quantiles: 默认 10
- min_obs_per_day: 默认 30
- holding_fee_rate: 默认 0.00002（单边手续费）

### 5.2.2 MarketStatusLabel
用于为每一条分钟 bar 生成市场状态分类标签，所有计算严格使用 **t 及之前的数据**，不引入未来函数。

**初始化输入**：NQ.parquet（需包含 data, close, volume 列）

**两种标签方法**：

#### volatility_regime — 波动率状态
- 标签: `LV`（低波）/ `MV`（中波）/ `HV`（高波）
- 计算步骤：
  1. 计算 1-bar 对数收益率 `ln(close_t / close_{t-1})`
  2. 滚动标准差（窗口=120 bars，约 2 小时）→ 实现波动率
  3. 滚动百分位排名（窗口=7200 bars，约 1 周）
  4. 百分位 < 1/3 → LV；1/3 ≤ 百分位 < 2/3 → MV；百分位 ≥ 2/3 → HV
- 窗口不足时标签为 `pd.NA`

#### intraday_session — 日内时段
- 标签: `OPEN` / `MIDDAY` / `CLOSE` / `OVERNIGHT`
- 基于 NQ 期货 CME Globex 常规交易时间（ET）：
  - OPEN:       09:30–10:00（开盘前 30 分钟）
  - MIDDAY:     10:00–15:30
  - CLOSE:      15:30–16:00（收盘前 30 分钟）
  - OVERNIGHT:  其余时段（盘前 / 盘后 / 夜盘）
- 纯时间标签，零计算，无未来函数风险

**使用方式**：
```python
labeler = MarketStatusLabel.from_parquet(Path('NQ.parquet'))
vol_labels  = labeler.volatility_regime()   # pd.Series[str]
sess_labels = labeler.intraday_session()    # pd.Series[str]
```

### 5.2.3 StatusAnalyzer
用于分析在**不同市场状态**下某一因子的表现。

**预处理流程**（与主 `evaluate_horizon` 类似，但有以下差异：始终使用单期 horizon=1；不应用日尾不足 horizon 的剔除掩码 `_eligibility_mask_by_day`）：
1. 时间对齐（`align_on_overlap_reindex`）
2. 滚动 z-score 标准化
3. 单期前瞻收益：factor_t → `vwap[t+2]/vwap[t+1] − 1`（即 t 时刻因子对应 t+1→t+2 的收益）
4. 日内 winsorize 去极值（仅对因子 z-score，与前瞻收益计算互不依赖）

**分状态分析**：按状态标签分组后分别计算：

| 指标类别 | 具体指标 |
|---|---|
| IC 分析 | ic_mean, ic_std, icir, ic_win_rate, ic_n_days |
| Rank IC 分析 | rank_ic_mean, rank_ic_std, rank_icir, rank_ic_win_rate, rank_ic_n_days |
| 多空回测（无费） | holding_sharpe, holding_max_drawdown, holding_win_rate, holding_total_return |
| 多空回测（含费） | 同上 _with_cost 后缀 |

**使用方式**：
```python
sa = StatusAnalyzer.from_parquet(
    factor_path, vwap_path, market_path,
    status_method='volatility_regime',  # 或 'intraday_session'
)
results = sa.analyze()
StatusAnalyzer.print_summary(results)
df = sa.summary_dataframe(results)
```

也可传入自定义标签序列：
```python
sa = StatusAnalyzer(factor_series, vwap_series, custom_label_series)
```

### 5.2.4 主评估流程
关键流程：
1. rolling_zscore 预处理因子
2. 构建前瞻收益 target_ret_n
3. 剔除每个交易日尾部不足 horizon 的样本
4. 日内 winsorize 去极值
5. 计算 daily IC / daily RankIC
6. 分层回测（N 期收益折算单期）
7. 持有回测（无手续费）
8. 持有回测（有手续费）
9. 汇总 summary 与各类序列结果

## 6. 关键计算口径
### 6.1 前瞻收益口径（因子 t 时刻）
target_ret_n 使用 VWAP：
- 计算区间为 [t+1, t+1+N]
- 形式为 vwap.shift(-(N+1)) / vwap.shift(-1) - 1

### 6.2 分层回测口径
- 因子按**滚动 lookback 窗口**（默认 1000 bars）内的分位数分 10 组（非全样本横截面），即每个 bar 仅参考其前 1000 条数据的分布进行分组
- 先计算 N 期均值收益
- 再折算为单期收益，便于不同 N 横向比较

### 6.3 多空持有状态机
- 触发：Group10 开多，Group1 开空
- 入场延迟：信号 t 触发，t+1 开始持有
- 仓位仅允许 -1/0/+1
- 每次持有严格 horizon 条 bar
- 持有窗口内忽略新信号
- 同时触发时多头优先

### 6.4 交易成本口径
`Backtest.apply_transaction_cost` 中：
- 按仓位绝对变化收单边手续费
- 单边费率：holding_fee_rate（默认 0.00002）
- 每 bar 成本 = abs(delta_position) * fee_rate
- 开仓或平仓计 1 次
- 反手（例如 +1 到 -1）计 2 次

### 6.5 累计净值口径
- cum_pnl 为权益曲线（初始值基于 1）
- 累计总收益 = 最终权益 - 1

### 6.6 持有回测统计指标
`Backtest.holding_stats` 返回：
- holding_sharpe
- holding_max_drawdown
- holding_annual_return
- holding_turnover
- holding_avg_daily_trades
- holding_win_rate
- holding_profit_loss_ratio

其中：
- holding_turnover = sum(abs(delta_position)) / sum(abs(position))
- holding_avg_daily_trades = 按日汇总的交易笔数均值
  - 开/平各记 1
  - 反手记 2

## 7. 返回结果结构（开发接口约定）

### 7.1 run_evaluation(cfg)
返回 Dict[horizon, payload]，payload 包含：
- summary: 汇总字典
- daily_ic: 日度 IC 序列
- daily_rank_ic: 日度 RankIC 序列
- layered_single_period: 分层单期收益
- holding_pnl: 无手续费持有序列
- holding_stats: 无手续费统计
- holding_pnl_with_cost: 有手续费持有序列
- holding_stats_with_cost: 有手续费统计

summary 额外包含：
- holding_total_return
- holding_total_return_with_cost
- 以及手续费统计的 with_cost 后缀字段

### 7.2 StatusAnalyzer.analyze()
返回 Dict[status_value, payload]，payload 包含：
- n_bars: 该状态下的 bar 数量
- ic_summary: IC 汇总字典（mean, std, ir, win_rate, n_days）
- rank_ic_summary: Rank IC 汇总字典
- daily_ic: 日度 IC 序列
- daily_rank_ic: 日度 Rank IC 序列
- holding_pnl: 无手续费持有序列
- holding_stats: 无手续费统计
- holding_pnl_with_cost: 有手续费持有序列
- holding_stats_with_cost: 有手续费统计
- holding_total_return / holding_total_return_with_cost

`summary_dataframe()` 方法可将上述结果转为单张 pd.DataFrame，便于导出和对比。

## 8. 已知问题与注意事项
- evaluation_output 历史文件混有旧 horizon，分析时注意版本口径
- 交易成本非常敏感：高频换手会显著侵蚀收益

## 9. 推荐的后续开发任务
- 扩展 MarketStatusLabel 增加更多市场状态维度：
  - 趋势/方向状态（基于滚动窗口收益率的符号或均线关系）
  - 成交量状态（基于滚动成交量相对于同时段历史均值的偏离）
  - 跳空/Gap 状态（基于当日开盘价与前日收盘价的跳空幅度）
  - 日内已实现动量（基于从当日开盘到当前 bar 的累计收益方向与幅度）
- 将 save_results 扩展为同时导出有手续费版本持有曲线
- 增加单元测试：
  - 状态机仓位边界测试
  - 交易成本计数规则测试（开平/反手）
  - 统计口径回归测试
- 增加配置文件化（yaml 或 toml），替代硬编码参数
- 增加异常监控与日志（输入列缺失、时间戳异常、NaN 比例）

## 10. 快速交接清单
- 确认 Python 环境与依赖可用（pandas, numpy, pyarrow）
- 运行 framework 脚本，确认 CSV 输出
- 验证无手续费与有手续费两套统计均包含 holding_avg_daily_trades

---
文档生成时间：2026-03-27
最后更新时间：2026-04-01（全量审查：修正目录结构、函数名称、分层回测口径、StatusAnalyzer 预处理差异说明）
