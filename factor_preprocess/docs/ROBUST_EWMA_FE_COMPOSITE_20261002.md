# 原生 Polars 稳健 EWMA

你通过 FP 注册表调用 `robust_ewma` 时，库使用 `FE_COMPOSITE:long_robust_ewm.lagged_robust_ewma:v1`。FE 在 `native_long_robust_ewma.py` 中用 Polars 表达式执行滞后、滚动统计、截尾和递推。pandas 包装层仅转换输入边界和恢复原索引。

## 公式与缺失值

对每个资产，按输入中该资产的行序定义滞后值：

$$\ell_t=x_{t-1}.$$

在 $\ell_{t-9},\ldots,\ell_t$ 中，仅对有限值计算均值 $\mu_t$ 和样本标准差 $s_t$（分母为 $n_t-1$），至少需要两个有限值。标准差下限为 $10^{-12}$。截尾宽度为 $c=\mathtt{winsor\_std}$；直接数值接口对非正宽度沿用 $c=10^{-12}$ 的兼容行为。

$$u_t=\operatorname{clip}(\ell_t,\mu_t-cs_t,\mu_t+cs_t),\qquad \alpha=1-2^{-1/h}.$$

NaN 统计边界不施加约束，因此暖机阶段的有限滞后值仍进入 EWMA。原始无穷值不进入滚动统计，但它可先被有限边界截断；截尾后仍非有限的值作为缺失输入。

EWMA 使用 `adjust=False, ignore_nulls=False`。相邻有限输入满足：

$$z_t=(1-\alpha)z_{t-1}+\alpha u_t.$$

首次有限输入作为状态初值。若两次有限输入之间有 $m$ 个缺失输入，则下一次更新满足：

$$z_{\mathrm{new}}=\frac{(1-\alpha)^{m+1}z_{\mathrm{old}}+\alpha u_{\mathrm{new}}}{(1-\alpha)^{m+1}+\alpha}.$$

库按资产携带已建立的输出，保留缺失输入的递推权重；累计有限输入数达到 `min_periods` 前仍输出缺失。它与遇到缺口重置状态的 IIR 口径不同。

## 调用与边界

```python
from factor_preprocess.registry.transforms import get_default_registry
executor = get_default_registry().get_execution("robust_ewma")
result = executor(frame, halflife=10.0, winsor_std=4.0, min_periods=1)
```

`frame` 默认包含 `asset_id/date/value`。同一资产的时间必须非空且单调不降；资产可交错排列，同日重复行保留输入顺序。库保留重复 pandas 索引，缺失资产键的行输出 NaN。FE 原生 API 拒绝非数值列；包装层校验参数后再处理空输入。

FE 不可用时，生产入口拒绝执行。你可用 `get_execution("robust_ewma", allow_research=True)` 明确选择研究回退。FP 直接函数仍保留历史实现作为参考。

## 验证与性能范围

回归测试覆盖暖机、缺失与无穷值、交错分组、重复索引、前缀因果性，以及大偏移小波动的统计与截尾判断。240,000 行（600 日 × 400 资产）的交替 A/B 中，历史 FP 中位数约 0.186 秒，FE 包装接口约 0.049 秒；NaN 掩码一致，最大绝对误差约 $1.8\times10^{-15}$。详见 `benchmarks/native_robust_ewma_ab_20261002.json`。

这次测量使用合成面板，未证明 COS 全流程的同等加速。该 recipe 支持长面板执行，尚未准入 FE DSL 或因子导出。FE 既有 `ts_robust_ema` 使用不同的稳健构造，不能替代此 recipe。
