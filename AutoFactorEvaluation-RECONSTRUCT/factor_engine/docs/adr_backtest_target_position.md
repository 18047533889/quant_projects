# ADR: Backtrader 单标的 `target_position` 接口冻结

## 状态

Accepted（2026-04-03）

## 背景

单标的择时路径中，研究员 C 负责信号与目标仓位逻辑，研究员 D 负责消费目标仓位并执行 Backtrader 回测。两方交接点必须冻结在数据接口而非策略代码细节。

## 决策

### 1) `target_position` 语义

- 字段：`target_position`
- 语义：**目标仓位权重**，范围 `[-1, 1]`
  - `1.0` = 满仓多头
  - `0.0` = 空仓
  - `-1.0` = 满仓空头

### 2) 时间索引约束

- 输入必须包含可解析为时间的 `timestamp`（列或 DatetimeIndex）。
- 标准化后使用无时区 `DatetimeIndex`。
- 时间索引不得重复；会按时间升序处理。

### 3) 缺失值处理

- `target_position` 缺失按 **forward-fill** 处理。
- 序列开头缺失按 `0.0`（空仓）填充。

### 4) 越界处理

- 默认严格模式下，若 `target_position` 超出 `[-1,1]`，直接报错。
- 非严格模式可裁剪到 `[-1,1]`。

### 5) 回测输出协议

统一返回三段结构：

- `returns`
  - `equity_curve`
  - `period_return`
  - `realized_position`
- `metrics`
  - 至少包含：`total_return`、`annual_return`、`volatility`、`sharpe`、`max_drawdown`、`turnover`、`trades`、`commission_paid`
- `summary`
  - 至少包含：`schema_version`、`start`、`end`、`bars`、`initial_cash`、`final_equity`

### 6) 策略注册与版本追踪（D-3）

- 执行入口支持：`strategy_name`、`strategy_version`、`strategy_params`。
- 内置策略通过策略库注册，并使用 `name@version` 管理。
- 若未显式指定 `strategy_version`，默认选择该策略的最新版本。

### 7) 回测输出扩展字段兼容策略

在不破坏核心冻结协议（`returns/metrics/summary` 必需字段）的前提下，允许 `summary` 增量扩展：

- `strategy_name`
- `strategy_version`
- `strategy_params`
- `strategy_instance_id`

这些字段用于回测结果追溯，不影响既有消费者对核心字段的读取。

### 8) 指标分层配置

- `BacktestConfig.metrics_profile` 支持：`core` / `standard` / `industrial`。
- `core` 保持最小核心指标集（与冻结协议一致）。
- `standard` / `industrial` 在核心指标基础上按层扩展（如 `sortino`、`calmar`、`var_95`、`cvar_95`、回撤持续时长等）。

### 9) 真实数据强约束（工业回测路径）

- `BacktestConfig.strict_real_data=True` 时，执行入口禁止传入 inline `ohlcv`。
- 该模式下必须通过 `data_root + symbol + frequency` 自动发现并加载真实文件。
- 找不到数据直接失败，不允许 synthetic/fallback 兜底。
- 当前黄金路径兼容 IBKR 等外部抓取产物（如 `~/quant_projects/data/ibkr/gold/XAU_*.parquet`）与频率别名匹配（如 `1h -> 1_hour`）。

### 10) 工业指标矩阵（增量，不破坏冻结必需键）

在 `metrics_profile=industrial` 下，除核心必需键外，支持以下扩展：

- 收益/风险路径：`ulcer_index`、`mar_ratio`、`sterling_ratio`、`burke_ratio`、`tail_ratio`、`omega_ratio`、`recovery_factor`、`hurst_exponent`、`skew`、`kurtosis`、`var_95`、`cvar_95`、`max_drawdown_duration_bars`。
- 基准依赖：`alpha`、`beta`、`information_ratio`、`tracking_error`、`treynor`、`up_market_capture`、`down_market_capture`、`r_squared`。
- 交易微观：`win_rate_trade`、`profit_factor`、`max_consecutive_losses`、`avg_holding_period_bars`、`expectancy`、`kelly_fraction`、`avg_mfe`、`avg_mae`、`time_in_market`、`exposure`。

### 11) 交易账本产物

- 通过 `notify_order` / `notify_trade` 静默采集交易生命周期。
- 可选输出 `artifacts.trade_ledger`，包含 order 完成事件与 trade closed 事件。
- 在 `metrics_profile=industrial` 下，账本会自动用于交易微观指标计算。

### 12) 可复现与审计元数据（single/multi 通用）

在不破坏冻结必需键的前提下，`summary` 增量包含：

- `run_id`：单次运行唯一标识。
- `data_fingerprint`：输入数据稳定指纹（同输入应稳定）。
- `dependency_versions`：运行时依赖版本（至少 `python/pandas/numpy/backtrader`）。
- `git_sha`：仓库提交（best-effort，可空）。
- `mode`：`single` 或 `multi`。
- `signal_timestamp`：信号时间语义标注（当前为 `bar_close_t`）。
- `decision_timestamp`：决策时间语义标注（当前为 `bar_close_t`）。
- `execution_effective_lag_bars`：收益归因使用的有效滞后 bar 数。
- `return_attribution`：收益归因公式字符串（例如 `weights(t-1) * returns(t)`）。

### 13) 信号时间语义与避免前视（未来函数）

引擎**无法**自动知道每条 `target_position` / `target_weight` 在业务上依赖哪些信息（例如是否用了同根 K 线的收盘价）。以下约定把「可证明的无前视」拆成**流水线责任**与**引擎可选滞后**两部分。

**单资产（`run_single_asset_backtest`）**

- 策略在每一根 bar 的 `next()` 中读取**当前 bar 时间戳**对应的目标仓位；Backtrader 默认不在同一根 K 线用「未来价」成交，但**信号本身若误用当日收盘后才可得的信息**，仍属研究侧前视，引擎不会替你检测。
- `BacktestConfig.target_lag_bars`（默认 `0`）：在与行情索引对齐并 ffill 之后，再对目标序列做 **`shift(target_lag_bars)`**，空缺填 `0`。设为 `1` 时，第 `t` 根 bar 使用的是原序列在 `t-1` 的值，适用于明确约定「信号在上一根收盘才确定」、且上游仍按 `t` 标注行但希望执行上强制晚一根的情形。**若上游已在因子层完成滞后，应保持为 `0`，避免双重滞后。**

**多资产（`run_multi_asset_backtest`）**

- 先由 **`weight_matrix`（对齐后的目标权重）** 与 OHLCV 经 **逐 bar 执行层**（最小调仓、可选 ADV 上限、成本模型）得到 **`executed_weights`**；再 **`realized_weights = executed_weights.shift(portfolio_weight_lag_bars)`**（默认 `1`）。组合 **毛收益** 为 **`sum(realized_weights * asset_return)`**，其中 `asset_return` 为收盘价逐 bar 收益率；**净收益** 再减去执行层给出的 **每 bar 成本收益率**。滞后作用在 **执行后权重** 上，避免把「仅对目标矩阵 shift」误当成真实可成交仓位。
- `portfolio_weight_lag_bars` 必须 **≥ 1**；若设为 `0` 会报错。
- 若 YAML 中 **`portfolio_execution_engine` 为 `python`**，引擎会将请求替换为 **`FACTOR_BACKTEST_EXECUTION_ENGINE`**（见 `runtime/perf_config.py`），以便与显式配置 **`numpy`/`numba`/`auto`** 区分：后者不经环境变量覆盖。

**基准与指标**

- `benchmark_return` 与策略 `period_return` 在指标层按索引 **inner join** 对齐；基准序列若包含未收盘数据或错误对齐，会产生错误归因，需在上游保证时间与收益口径一致。

**ffill**

- 目标仓位向前填充传播的是**历史已产生**的信号，不引入未来信息；但会延长旧信号有效期，属于建模假设而非前视。

### 14) 多资产执行层与可复现指纹（实现约定）

- **执行输出**：`executed_weights`（每 bar 各标的可执行权重）、`turnover`、`cost_return`（每 bar 成本收益率）、`participation`；**`data_fingerprint`** 中多资产 **权重列统计** 针对 **`executed_weights`**，而非原始目标矩阵。
- **审计**：`summary` 可含 **`execution_engine_requested`** / **`execution_engine_resolved`**；当配置为 `python` 时，`requested` 反映环境变量解析后的内核，而非 YAML 字面量 `python`。

## 影响

- 研究员 C 与 D 的接口变更需走 ADR 更新。
- 回测层可以替换实现细节，但对外协议保持稳定。
