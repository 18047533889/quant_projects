# vectorbt 调研报告

> 撰写日期：2026-06-28  
> 调研对象：[vectorbt](https://github.com/polakowo/vectorbt)（开源社区版）  
> 调研目的：评估 vectorbt 作为 A 股/美股量化回测引擎的适配性、可扩展性及与现有数据架构的集成方案
>
> **归档说明：** 本文记录早期调研结论，不代表当前 Accurate V2 的最终实现。
> 当前功能与口径请以仓库 README、`docs/backtest_modes.md` 和版本化 profile 为准。

---

## 目录

1. [底层机制与主要模式](#1-底层机制与主要模式)
2. [from_orders 完整生命周期](#2-from_orders-完整生命周期)
3. [A 股/美股交易规则在回测中的实现](#3-a-股美股交易规则在回测中的实现)
4. [当前数据适配方案](#4-当前数据适配方案)
5. [后续功能扩展难度评估](#5-后续功能扩展难度评估)
6. [总结与建议](#6-总结与建议)

---

## 1. 底层机制与主要模式

### 1.1 三层架构

vectorbt 采用严格的 **三层分离架构**：

```
┌─────────────────────────────────────────┐
│  第 1 层：Python/pandas 信号生成层        │
│  - 向量化指标计算（MA、RSI、布林带等）     │
│  - 信号交叉检测（crossed_above/below）     │
│  - 参数广播（标量 → 矩阵对齐）             │
├─────────────────────────────────────────┤
│  第 2 层：Python 调度层                   │
│  - Portfolio.from_signals / from_orders  │
│  - 引擎选择（Numba / Rust）               │
│  - 结果对象构建（Orders/Trades/Positions） │
├─────────────────────────────────────────┤
│  第 3 层：Numba/Rust 仿真引擎层            │
│  - JIT 编译的双重 for 循环                │
│  - 逐 cell 状态机推进（现金/持仓/负债）     │
│  - 返回结构化订单记录数组                  │
└─────────────────────────────────────────┘
```

**核心设计哲学**：把能向量化的（信号生成、参数广播）放在 Python/pandas 层一次性完成；把必须顺序执行的（持仓/现金状态推进）放在 Numba 层编译为机器码运行。

### 1.2 两种主要回测模式

| 模式 | 入口 | 输入 | 适用场景 |
|------|------|------|---------|
| **信号模式** | `Portfolio.from_signals` | 布尔信号矩阵（entries/exits） | MA 交叉、突破策略、技术指标触发 |
| **订单模式** | `Portfolio.from_orders` | 可直接传目标权重/目标金额/目标股数 | 因子模型、每日再平衡、组合优化 |

两种模式最终都汇聚到同一个 Numba 引擎，区别在于信号模式多了一层 "信号 → 订单参数" 的转换管道（`signals_to_size_nb`），而订单模式直接将参数传入。

### 1.3 广播机制

vectorbt 的核心抽象是 **广播（Broadcasting）**：所有参数支持三种粒度，内部自动对齐到 `(n_days, n_assets)` 矩阵。

| 粒度 | 示例 | 含义 |
|------|------|------|
| 标量 | `fees=0.001` | 所有资产、所有时间同一费率 |
| 向量 (n_assets,) | `fees=[0.001, 0.002]` | 每个资产不同费率 |
| 矩阵 (n_days, n_assets) | `fees=时变矩阵` | 每个资产每天不同费率 |

---

## 2. from_orders 完整生命周期

### 2.1 调用全景图

```
Portfolio.from_orders(close, size=target_weights, size_type='targetpercent', ...)
│
├─ ① 参数解析与广播
│     size_type='targetpercent' → SizeType.TargetPercent (整数 5)
│     target_weights → broadcast 到 (n_days, n_assets) 矩阵
│     close → broadcast 到 (n_days, n_assets) 矩阵
│
├─ ② 构建模拟上下文
│     cash_sharing=True → group_lens=[n_assets]，所有资产共享资金池
│     call_seq='auto' → 按订单金额自动排序（卖先买后）
│     init_cash → 广播到每个 group
│
├─ ③ dispatch.simulate_from_orders()
│     路由到 Numba 或 Rust 引擎
│
├─ ④ simulate_from_orders_nb()  ← 主循环（Numba JIT 编译为机器码）
│     │
│     │  for group in groups:
│     │    for i in range(n_days):                  ← 逐日
│     │      │
│     │      ├─ 第 1 遍：计算每列的 order_price 和 val_price
│     │      ├─ 第 2 遍（cash_sharing 时）：按订单金额排序 call_seq
│     │      └─ 第 3 遍：按 call_seq 顺序处理每列
│     │            │
│     │            col = call_seq[i, k]
│     │            _size = size[i, col]              ← 如 0.05
│     │            _size_type = size_type[i, col]    ← 如 TargetPercent=5
│     │            │
│     │            order = order_nb(size=_size, size_type=_size_type, ...)
│     │            │
│     │            └─ process_order_nb(order)
│     │                  │
│     │                  └─ execute_order_nb(state, order)
│     │                        │
│     │                        ├─ TargetPercent → TargetValue
│     │                        │    size = 0.05 × $1,000,000 = $50,000
│     │                        │
│     │                        ├─ TargetValue → TargetAmount
│     │                        │    size = $50,000 / $100 = 500 股
│     │                        │
│     │                        ├─ TargetAmount → Amount (delta)
│     │                        │    size = 500 - 当前持仓300 = +200
│     │                        │
│     │                        └─ Amount → buy_nb/sell_nb
│     │                              adj_price = price × (1 + slippage)
│     │                              fees_paid = req_cash × fees + fixed_fees
│     │                              cash -= cost, position += delta
│     │
│     return order_records, log_records
│
├─ ⑤ Portfolio 对象构建
│     order_records → Orders → Trades → Positions → Drawdowns
│     懒加载：访问时才计算
│
└─ ⑥ 结果访问
      pf.stats()          → 几十个绩效指标
      pf.plot_value()      → 净值曲线
      pf.trades.records_readable → 每笔交易详情
```

### 2.2 SizeType 链式转换（核心机制）

`execute_order_nb` 支持 6 种 `SizeType`，按优先级链式降级，最终全部落到 `Amount`（delta 股数）：

```
TargetPercent  →  TargetValue  →  TargetAmount  →  Amount  →  buy_nb/sell_nb
Value          →  Amount       →  buy_nb/sell_nb
Percent        →  Amount       →  buy_nb/sell_nb
Amount         →  buy_nb/sell_nb
```

| SizeType | 语义 | 降级为 | 转换公式 |
|----------|------|--------|---------|
| TargetPercent | 目标占组合净值的百分比 | TargetValue | `size × value_now` |
| TargetValue | 目标持仓金额 | TargetAmount | `size / val_price_now` |
| TargetAmount | 目标持仓股数 | Amount | `size - position_now` |
| Value | 买入金额 | Amount | `size / val_price_now` |
| Percent | 可用现金/持仓的百分比 | Amount | 保留原始语义给 buy_nb |
| Amount | 买入/卖出股数 | — | 直接执行 |

### 2.3 cash_sharing + call_seq='auto' 的资金管理

当 `cash_sharing=True` 时：

- 所有资产共享一个现金池（`group_lens=[n_assets]`）
- `call_seq='auto'` 自动按订单值排序：**卖单优先，释放资金后执行买单**
- 同一组内后执行的订单只能用前面订单释放的现金
- 这是实现真实组合再平衡的关键机制

---

## 3. A 股/美股交易规则在回测中的实现

vectorbt 的定制化能力分为 **5 个层次**，越往下侵入性越强：

```
══════════════════════════════════════════════════
  层次          改动量       侵入 vbt 源码
══════════════════════════════════════════════════
  ① 输入矩阵层   0 行         零侵入
  ② 参数广播层   0 行         零侵入
  ③ 回调钩子层   ~30 行       函数注入，不改源码
  ④ 自定义订单层  ~80 行       函数替换，不改源码
  ⑤ 引擎源码层   数百行        直接改 nb.py / enums.py
══════════════════════════════════════════════════
```

### 3.1 A 股交易规则实现

| 规则 | 实现层级 | 实现方式 | 改动量 |
|------|:---:|------|:---:|
| **T+1 交易制度** | ⑤ | 需新增 `locked_shares` 状态变量跟踪当日买入不可卖 | 高 |
| | ① 折中 | 外部跟踪持仓 ≥ 1 日的可卖部分，构造约束矩阵 | 中 |
| **涨跌停 ±10%** | ① | `price = close.clip(lower=low_limit, upper=high_limit)` | 低 |
| **停牌** | ① | `target_weights = target_weights.where(~is_suspend)` | 低 |
| **印花税（卖出 0.05%）** | ② | 构造分方向的 `fees` 矩阵，卖出日加印花税 | 低 |
| **佣金（约 0.025%）** | ② | `fees=0.00025` | 低 |
| **过户费（¥0.001/股）** | ② | `fixed_fees` 参数 | 低 |
| **最小 100 股（1 手）** | ② | `size_granularity=100` | 低 |
| **融券费率（年化 ~8.6%）** | ③ | `post_segment_func_nb` 每日扣除持有成本 | 中 |
| | ③ 折中 | 后处理从 `pf.positions` 中扣除 | 低 |

#### T+1 的两种实现对比

**折中方案（① 层）**：外部维护一个 "锁定股数" DataFrame，每次调仓时锁定当日买入部分，次日释放。复杂度中等，回测精度可接受（忽略日内的 T+1 约束）。

**精确方案（⑤ 层）**：在 `SimulationContext` 中新增 `locked_shares` 数组，在 `buy_nb` 成交后将对应股数加到 `locked_shares`，在 `sell_nb` 中检查 `position - locked_shares` 可卖量。需要改动 `nb.py` 和 `enums.py` 约 50 行。

### 3.2 美股交易规则实现

| 规则 | 实现层级 | 实现方式 | 改动量 |
|------|:---:|------|:---:|
| **SEC Section 31 Fee** | ② | `fees` 矩阵 +$8/$1M | 低 |
| **FINRA TAF** | ② | `fixed_fees` 或合并入 `fees` | 低 |
| **佣金（IBKR ~$0.0035/股）** | ② | `fees` / `fixed_fees` | 低 |
| **SSR 限制（跌 10% 禁做空）** | ① | `target_weights[ssr_active & (w < 0)] = 0` | 低 |
| **做空费率（0.25%~200%+）** | ③ | `post_segment_func_nb` 每日扣除 | 中 |
| **Hard-to-Borrow 标记** | ① | HTB 标的的 `target_weights` 负值 → 0 | 低 |
| **退市收益** | ① | 退市日 `close` 替换为退市后终值 | 低 |
| **ticker 更名** | ① | 使用 `DimTickerMap` 做前后衔接 | 低 |
| **ADR 去重** | ① | 使用 `is_adr_asset` 标记过滤 | 低 |

### 3.3 各层级典型代码示例

#### 第 ① 层：停牌预处理

```python
target_weights = target_weights.where(~is_suspend)  # 停牌日不调仓
```

#### 第 ② 层：分方向费率

```python
fees = pd.DataFrame(0.00025, index=dates, columns=tickers)   # 佣金
fees[delta < 0] += 0.0005  # 卖出加印花税
pf = vbt.Portfolio.from_orders(close, size=target_weights, fees=fees)
```

#### 第 ③ 层：每日融券成本扣除

```python
from numba import njit
from vectorbt.portfolio.nb import SegmentContext

@njit
def deduct_borrow_fee_nb(c: SegmentContext, borrow_fee_daily: np.ndarray):
    for col in range(c.from_col, c.to_col):
        if c.last_position[col] < 0:
            cost = abs(c.last_position[col]) * c.last_val_price[col] * borrow_fee_daily[c.i, col]
            c.last_cash[col] -= cost
    return ()

pf = vbt.Portfolio.from_order_func(
    close, order_func_nb=...,
    post_segment_func_nb=deduct_borrow_fee_nb,
    post_segment_args=(borrow_fee_daily.values,),
)
```

#### 第 ④ 层：涨跌停挂单到次日

```python
@njit
def order_with_price_limit_nb(c: OrderContext, target_weights, price_limits, pending):
    # 涨停买不进 → 挂到次日
    # 跌停卖不出 → 挂到次日
    ...
```

---

## 4. 当前数据适配方案

### 4.1 数据现状

当前数据存储在腾讯云 COS（`cos://qs-cold/`），以 `YYYY-MM-DD.parquet` 格式按日分区。

**A 股**：19 张表，覆盖 2016-2026（财务表最早到 2003 年）
**美股**：7 张表，覆盖 2003-2026

### 4.2 需要的数据转换

#### 转换 1：parquet → Pandas 宽表（核心步骤）

```python
import pandas as pd
from glob import glob

# 从按日分区的 parquet 拼接为 (date, ticker) 宽表
def load_close_from_parquet(parquet_dir, date_col, symbol_col, value_col, adj_factor_col=None):
    """通用加载函数：parquet 分片 → pivot 宽表"""
    files = sorted(glob(f"{parquet_dir}/*.parquet"))
    frames = []
    for f in files[start_idx:end_idx]:
        df = pd.read_parquet(f)
        if adj_factor_col:
            df[value_col] = df[value_col] * df[adj_factor_col]
        frames.append(df[[date_col, symbol_col, value_col]])
    
    raw = pd.concat(frames)
    result = raw.pivot(index=date_col, columns=symbol_col, values=value_col)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()
```

| 市场 | 数据源 | date_col | symbol_col | value_col | adj_factor |
|------|--------|----------|------------|-----------|------------|
| A 股 | StockDailyBar | TradeDate | Symbol | Close | Factor |
| 美股 | PanelDaily | TradeDate | Ticker | Close | —（已有复权价） |
| 美股 | StockDailyBar | TradeDate | Ticker | Close | AdjFactor |

#### 转换 2：通用加载函数封装

```python
def load_ashare_data(start, end):
    """加载 A 股完整数据"""
    close = load_close_from_parquet(
        "cos://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/",
        'TradeDate', 'Symbol', 'Close', 'Factor'
    )
    # 加载辅助数据
    is_suspend  = load_pivot("StockDailyBar", 'IsSuspend')    # 停牌
    high_limit  = load_pivot("StockDailyBar", 'HighLimit')     # 涨停价
    low_limit   = load_pivot("StockDailyBar", 'LowLimit')      # 跌停价
    return close, is_suspend, high_limit, low_limit

def load_us_data(start, end):
    """加载美股完整数据"""
    close = load_close_from_parquet(
        "cos://qs-cold/clean_data/us_stock/massive_data/PanelDaily/",
        'TradeDate', 'Ticker', 'Close'
    )
    # PanelDaily 还包含收益率、基本面 PIT 数据（2010 年起）
    returns = load_pivot("PanelDaily", 'RetPrice')
    return close, returns
```

#### 转换 3：目标权重矩阵

无论你的策略是因子打分、均值方差优化还是机器学习预测，最终都需要输出一个 `(n_days, n_assets)` 的权重矩阵：

```python
# 因子打分 → 目标权重
scores = compute_factor_scores(factor_data)          # (n_days, n_assets)
target_weights = scores.rank(axis=1, pct=True)       # 横截面排名
target_weights = (target_weights - 0.5) * 2          # [-1, 1] 范围
target_weights = target_weights / target_weights.abs().sum(axis=1)  # 归一化
```

### 4.3 当前数据可直接覆盖的要素

| 要素 | 是否可直接使用 | 备注 |
|------|:---:|------|
| OHLCV 行情 | ✅ | A 股复权价乘 Factor，美股 PanelDaily 已含复权 |
| 停牌标记 | ✅ | `IsSuspend` 字段 |
| 涨跌停价 | ✅ | `HighLimit` / `LowLimit` 字段 |
| 复权因子 | ✅ | A 股 `Factor`，美股 `AdjFactor` |
| 交易日历 | ✅ | Calendar / DimCalendar |
| 市值 | ✅ | `StockValuationDaily.MarketCap` / PanelDaily 隐含 |
| ETF 列表 | ✅ | 用于可交易标的过滤 |
| ticker 更名 | ✅ | DimTickerMap 做前后衔接 |

### 4.4 当前数据缺口

| 缺口 | 影响 | 优先级 | 临时方案 |
|------|------|:---:|------|
| **退市收益（美股）** | 回测收益率严重虚高 | P0 | 暂不做退市股 |
| **美股融券费率** | 做空回测不可信 | P1 | 使用常量估算 |
| **2010 前流通股** | 流动性过滤不完整 | P2 | 用总股本代替 |
| **A 股融券池** | 做空标的筛选 | P2 | 假设所有可融 |

---

## 5. 后续功能扩展难度评估

### 5.1 扩展难度矩阵

| 扩展需求 | 实现层级 | 工作量 | 风险 | 是否需改 vbt 源码 |
|----------|:---:|:---:|:---:|:---:|
| 新增交易规则（如新的费率模型） | ①-② | 0.5 人天 | 低 | ❌ |
| 分买卖方向不同费率 | ② | 0.5 人天 | 低 | ❌ |
| 每日持有成本扣除 | ③ | 1 人天 | 低 | ❌ |
| 涨跌停挂单到次日 | ④ | 2 人天 | 中 | ❌ |
| T+1 精确锁定（新增状态变量） | ⑤ | 3-5 人天 | 高 | ✅ |
| 多账户/多策略并行 | ⑤ | 10+ 人天 | 高 | ✅ |
| 分钟级/日内回测 | — | 0 人天 | — | ❌ 无需改，直接传分钟数据 |
| 杠杆/保证金账户 | ⑤ | 3 人天 | 中 | ✅ |
| 期权/期货回测 | ⑤ | 10+ 人天 | 高 | ✅ |
| 自定义资产类型 | ⑤ | 5 人天 | 高 | ✅ |

### 5.2 前 4 层典型扩展模板

#### 模板 A：新增一个类似滑点的交易规则（② 层）

```python
# 步骤：
# 1. 将你的规则转化为一个 (n_days, n_assets) 矩阵
# 2. 传入对应的参数位置

# 例如：按市值分档的冲击成本
impact_cost = compute_impact_cost(close, volume, market_cap)  # 自己的函数
pf = vbt.Portfolio.from_orders(close, size=target_weights, slippage=impact_cost)
```

#### 模板 B：新增一个类似融券费的持有成本（③ 层）

```python
# 步骤：
# 1. 写一个 @njit 函数，接收 SegmentContext
# 2. 在函数内直接修改 c.last_cash 或 c.last_val_price
# 3. 通过 from_order_func 的 post_segment_func_nb 参数注入

@njit
def my_holding_cost_nb(c: SegmentContext, cost_matrix: np.ndarray):
    for col in range(c.from_col, c.to_col):
        c.last_cash[col] -= cost_matrix[c.i, col]
    return ()

pf = vbt.Portfolio.from_order_func(
    close, order_func_nb=target_weight_order_nb,
    order_args=(target_weights.values,),
    post_segment_func_nb=my_holding_cost_nb,
    post_segment_args=(cost_matrix.values,),
)
```

### 5.3 不推荐修改源码的场景

以下场景 **不需要改源码**，有更好的替代路径：

| 场景 | 错误做法 | 正确做法 |
|------|---------|---------|
| 按市值分档滑点 | 改 `buy_nb` 加判断 | 构造 `slippage` 矩阵传入（② 层） |
| 日内交易规则 | 改日频引擎 | 直接传分钟级数据即可 |
| 新的信号逻辑 | 改 `signals_to_size_nb` | 用 `from_order_func` 自定义（④ 层） |
| 多因子模型 | 改引擎支持新 size_type | 先算好目标权重再传入（① 层） |

### 5.4 必须改源码的指标

只有当需要 **引擎不存在的状态变量** 或 **新的跨组通信机制** 时，才需要改动 `nb.py` 和 `enums.py`：

- 新增 `locked_shares`（T+1 精确实现）
- 新增 `margin_used`（杠杆账户）
- 新增跨组约束（全组合杠杆上限）
- 新增订单类型（如冰山订单、TWAP 拆单）

---

## 6. 总结与建议

### 6.1 vectorbt 能力评估

| 维度 | 评分 | 说明 |
|------|:---:|------|
| 回测速度 | ⭐⭐⭐⭐⭐ | Numba JIT + 列优先遍历，百万级 cell/s |
| 灵活性 | ⭐⭐⭐⭐ | 5 层定制架构，95% 需求不改源码 |
| 数据接入 | ⭐⭐⭐⭐⭐ | pandas 原生，与当前 COS parquet 无缝对接 |
| 组合回测 | ⭐⭐⭐⭐ | cash_sharing + call_seq='auto' 天然支持 |
| A 股适配 | ⭐⭐⭐ | T+1 精确实现需改动源码（折中方案可用） |
| 美股适配 | ⭐⭐⭐ | 做空费率需 hook 注入，退市数据需外部补齐 |
| 文档与生态 | ⭐⭐⭐⭐ | 详尽文档 + Jupyter 示例 + 活跃社区 |
| 源码可维护性 | ⭐⭐⭐⭐ | 结构清晰，Numba 代码可读性好 |

### 6.2 推荐实施路径

```
Phase 1: 基础回测（1-2 周）
  ├── 编写 COS parquet → pandas 宽表的加载模组
  ├── A 股做多策略回测（from_orders + TargetPercent）
  ├── 接入停牌/涨跌停/手续费（①② 层，0 行 vbt 改动）
  └── 验证结果的 stats/trades/positions

Phase 2: 约束完善（1 周）
  ├── A 股 T+1 折中实现（① 层）
  ├── 融券费率后处理（③ 层折中方案）
  └── 美股做多回测（不加退市收益，仅限大市值）

Phase 3: 精确化（2 周）
  ├── 补退市收益数据
  ├── 补融券费率数据
  ├── 融券费率精确实现（③ 层 post_segment_func_nb）
  └── 美股做空 + 全约束回测

Phase 4: 高级功能（按需）
  ├── 涨跌停挂单（④ 层）
  ├── T+1 精确实现（⑤ 层，如需）
  └── 杠杆/多账户（⑤ 层，如需）
```

### 6.3 关键风险

1. **退市收益缺失**：美股多空回测结果不可信，必须优先补齐
2. **T+1 精确实现**：如果业务要求严格 T+1，需要改源码约 50 行
3. **Numba 调试困难**：③④⑤ 层的 Numba 代码无法用 pdb 调试，需要 print 调试法
4. **版本依赖**：vectorbt 开源版更新活跃，后续升级时需关注 API 兼容性

---

> **附录**：本文档中涉及的关键源码文件路径（基于 vectorbt 开源版）
> - 主引擎：`vectorbt/portfolio/nb.py` — `simulate_from_orders_nb`、`execute_order_nb`、`buy_nb`/`sell_nb`
> - Python 入口：`vectorbt/portfolio/base.py` — `Portfolio.from_orders`、`Portfolio.from_signals`
> - 枚举定义：`vectorbt/portfolio/enums.py` — `SizeType`、`Direction`、`OrderContext`、`SegmentContext`
> - 调度层：`vectorbt/portfolio/dispatch.py` — `simulate_from_orders`
