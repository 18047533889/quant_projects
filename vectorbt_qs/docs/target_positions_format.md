# 目标持仓数据格式规范

> 本文档描述 vectorbt_qs 回测系统的核心输入——目标权重矩阵的数据格式、语义、约束和生成方式。

---

## 一、数据结构

### 1.1 基本形态

```
目标权重矩阵 = pandas.DataFrame
  - index: 日期 (datetime64, 每个交易日一行)
  - columns: 标的代码 (string, A股=Symbol, 美股=Ticker)
  - values: 目标权重 (float64)
  - dtype: 全部 float64
```

### 1.2 维度

| 维度 | 含义 | 示例 |
|------|------|------|
| `n_days` | 回测交易日数 | 2500 天 |
| `n_assets` | 候选标的数 | 500 只 |

---

## 二、值的语义

### 2.1 权重含义

每个 cell 的数值表示 **该标的目标占组合净值的百分比**：

| 值 | 含义 | 等价描述 |
|:---:|------|------|
| `0.05` | 做多组合净值的 5% | 用 5% 资金买入该标的 |
| `0.00` | 不持有 | 如果已有持仓，则全部卖出 |
| `-0.03` | 做空组合净值的 3% | 融券卖出等值 3% 的该标的 |
| `NaN` | 跳过，不调仓 | 保持当前持仓不变 |

### 2.2 权重之和

```
∑|target_weight| ≤ 1.0  (多空各自不超过 100%)
```

- **纯做多**：`∑(正值) ≤ 1.0`，通常 `∑ = 0.95`（留 5% 现金缓冲）
- **多空**：`∑(正值) ≤ 1.0` 且 `∑|负值| ≤ 1.0`，如做多 80% + 做空 50% = 130% 总敞口
- **不限制**：可以超过 1.0（模拟杠杆），vectorbt 的 `cash_sharing=True` 会按现金实际可用量自动限制

### 2.3 数值示例

```python
# 某一天的 target_weights 截面 (pd.Series)
AAPL    0.06    # 做多 6%
MSFT    0.04    # 做多 4%
GOOGL   0.03    # 做多 3%
TSLA   -0.02    # 做空 2%
AMZN    0.00    # 不持（有持仓会全平）
META     NaN    # 跳过，不调仓（如果有持仓则保留）
其他      0.00    # 不持
────────────────
∑long   0.13    # 多头总敞口 13%
∑short  0.02    # 空头总敞口 2%
       现金 87%  # 闲置
```

---

## 三、完整示例

### 3.1 纯做多（A 股典型场景）

```python
import pandas as pd
import numpy as np

# 10 个交易日 × 4 只股票
dates = pd.date_range("2024-01-02", "2024-01-15", freq="B")
tickers = ["000001.SZ", "000002.SZ", "600519.SH", "300750.SZ"]

target_weights = pd.DataFrame(
    [
        # 平安银行  万科A    贵州茅台   宁德时代
        [ 0.05,     0.05,    0.05,     0.05],  # 2024-01-02：各 5%
        [ 0.05,     0.05,    0.05,     0.05],  # 2024-01-03：不变
        [ 0.00,     0.05,    0.10,     0.05],  # 2024-01-04：卖平安，加茅台
        [ 0.00,     0.00,    0.15,     0.05],  # 2024-01-05：卖万科，继续加茅台
        [ 0.00,     0.00,    0.15,     0.00],  # 2024-01-08：卖宁德
        [ 0.03,     0.00,    0.12,     0.00],  # 2024-01-09：重新买入平安
        [ 0.03,     0.00,    0.12,     0.00],  # ...
        [ 0.03,     0.00,    0.12,     0.00],
        [ 0.03,     0.00,    0.12,     0.00],
        [ 0.00,     0.00,    0.00,     0.00],  # 2024-01-15：全部清仓
    ],
    index=dates[:10],
    columns=tickers,
    dtype=np.float64,
)

print(target_weights)
```

```
            000001.SZ  000002.SZ  600519.SH  300750.SZ
2024-01-02       0.05       0.05       0.05       0.05
2024-01-03       0.05       0.05       0.05       0.05
2024-01-04       0.00       0.05       0.10       0.05
2024-01-05       0.00       0.00       0.15       0.05
2024-01-08       0.00       0.00       0.15       0.00
2024-01-09       0.03       0.00       0.12       0.00
2024-01-10       0.03       0.00       0.12       0.00
2024-01-11       0.03       0.00       0.12       0.00
2024-01-12       0.03       0.00       0.12       0.00
2024-01-15       0.00       0.00       0.00       0.00
```

### 3.2 多空（美股典型场景）

```python
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META"]

target_weights = pd.DataFrame(
    [
        # 做多前 3 只，做空后 2 只
        [ 0.05,  0.05,  0.04, -0.03, -0.02,  0.00],
        [ 0.06,  0.04,  0.04, -0.03, -0.02,  0.00],
        [ 0.06,  0.04,  0.00, -0.03, -0.02,  0.03],
        [ 0.07,  0.00,  0.00, -0.04, -0.03,  0.03],
        [ 0.07,  0.00,  0.00, -0.04, -0.03,  0.03],
    ],
    index=pd.date_range("2024-01-02", periods=5, freq="B"),
    columns=tickers,
    dtype=np.float64,
)
```

### 3.3 部分调仓（NaN 跳过）

```python
# 只在周一调仓，其余交易日保持不动
target_weights = pd.DataFrame(0.0, index=all_dates, columns=tickers)

# 每周一设目标权重
mondays = all_dates[all_dates.dayofweek == 0]
target_weights.loc[mondays] = weekly_strategy_output

# 非周一全部设为 NaN → 保持持仓不变
non_mondays = all_dates[all_dates.dayofweek != 0]
target_weights.loc[non_mondays] = np.nan
```

---

## 四、常见来源

### 4.1 因子打分 → 目标权重

```python
def factor_to_weights(factor_scores: pd.DataFrame, 
                       long_quantile: float = 0.8,
                       short_quantile: float = 0.2,
                       weight_per_stock: float = 0.05) -> pd.DataFrame:
    """
    因子打分 → 目标权重
    
    factor_scores: (n_days, n_assets) 因子值（越高越好）
    """
    rank = factor_scores.rank(axis=1, pct=True)
    
    weights = pd.DataFrame(0.0, index=factor_scores.index, columns=factor_scores.columns)
    weights[rank >= long_quantile] = weight_per_stock     # 前 20% 做多
    weights[rank <= short_quantile] = -weight_per_stock   # 后 20% 做空
    
    return weights
```

### 4.2 均线交叉 → 目标权重

```python
def ma_cross_to_weights(close: pd.DataFrame,
                         fast_window: int = 10,
                         slow_window: int = 30,
                         weight: float = 0.05) -> pd.DataFrame:
    """均线交叉 → 做多/平仓"""
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights[fast_ma > slow_ma] = weight  # 快线在上 → 做多
    
    return weights
```

### 4.3 等权再平衡

```python
def equal_weight(universe: pd.DataFrame, rebalance_freq: str = 'M') -> pd.DataFrame:
    """
    等权再平衡
    
    universe: (n_days, n_assets) 可交易标记 (True/False)
    """
    weights = pd.DataFrame(np.nan, index=universe.index, columns=universe.columns)
    
    for date in universe.index:
        # 只在再平衡日调仓
        if is_rebalance_day(date, rebalance_freq):
            tradable = universe.loc[date][universe.loc[date]].index
            n = len(tradable)
            if n > 0:
                weights.loc[date, tradable] = 1.0 / n  # 等分
    
    return weights
```

### 4.4 组合优化输出

```python
# PyPortfolioOpt / cvxpy 等优化器的输出
# 格式完全一致：(n_days, n_assets) 权重矩阵

from pypfopt import EfficientFrontier

def optimize_portfolio(returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    ef = EfficientFrontier(returns, cov_matrix)
    weights = ef.max_sharpe()
    return pd.Series(ef.clean_weights())  # 已是 {ticker: weight} 格式
```

---

## 五、约束与校验

### 5.1 传入 runner 前应满足的条件

| 条件 | 检查 | 不满足后果 |
|------|------|-----------|
| index 为 DatetimeIndex | `isinstance(df.index, pd.DatetimeIndex)` | vbt 无法识别频率 |
| columns 与 close 一致 | `set(df.columns) == set(close.columns)` | 找不到标的不报错但跳过 |
| dtype 为 float64 | `df.dtypes.unique() == [np.float64]` | Numba 引擎可能类型报错 |
| 无 inf 值 | `~np.isinf(df).any().any()` | 引擎行为不确定 |
| 日期单调递增 | `df.index.is_monotonic_increasing` | 回测顺序错乱 |

### 5.2 快速校验函数

```python
def validate_target_weights(target_weights: pd.DataFrame, close: pd.DataFrame) -> bool:
    """校验目标权重矩阵"""
    errors = []
    
    if not isinstance(target_weights.index, pd.DatetimeIndex):
        errors.append("index 必须是 DatetimeIndex")
    if not target_weights.index.is_monotonic_increasing:
        errors.append("日期必须单调递增")
    if target_weights.isna().all(axis=None):
        errors.append("全部为 NaN，无可调仓日")
    if np.isinf(target_weights.values).any():
        errors.append("包含 inf 值")
    
    # 检查标的交集
    common = set(target_weights.columns) & set(close.columns)
    if len(common) == 0:
        errors.append("target_weights 和 close 无共同标的")
    elif len(common) < len(target_weights.columns):
        missing = set(target_weights.columns) - set(close.columns)
        errors.append(f"{len(missing)} 个标的不在 close 中: {list(missing)[:5]}...")
    
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False
    
    print("✅ 目标权重矩阵校验通过")
    return True
```

---

## 六、与向量化回测的对应关系

```
target_weights.iloc[i, j] = 0.05
          │
          ▼
vectorbt 内部:
  Day i, Asset j: TargetPercent(0.05)
    → 0.05 × 当日组合净值 $1,000,000 = $50,000
    → $50,000 / 当日价格 $100 = 500 股 (TargetAmount)
    → 500 - 当前持仓 300 = +200 股 (Amount delta)
    → buy(200 股)
```

**关键认知**：
- 你不需要自己算 delta（买卖多少股）
- 你只需要告诉 vectorbt **"目标是什么"**（目标权重）
- vectorbt 自动比较目标 vs 当前持仓，算出买卖量
