# FE 原生 Polars 截面 rank 候选适配器

`factor_preprocess.adapters.fe_rank.cs_rank_fe_native` 是显式 opt-in 候选。
它调用 `factor_engine.backend.rank_spec.polars_cs_rank_expr`，不重写排名内核，
不改变 `cs_rank`、FP registry、recipe 或 optimizer 的默认执行路径，也不宣称 FE
算子已获得 FP 生产准入。`execution_identity()` 标记 `production_admitted=False`，
并分别记录候选 adapter 与 FE rank helper 模块的有界源码 SHA-256 和 NumPy/Polars 版本。

```python
import numpy as np
from factor_preprocess.adapters.fe_rank import cs_rank_fe_native, execution_identity

# 每行是一个时间截面，每列是一只资产。
values = np.array([[2.0, 2.0, 5.0], [4.0, np.nan, 8.0]])
ranked = cs_rank_fe_native(values)
assert ranked.shape == values.shape
assert execution_identity()["production_admitted"] is False
```

## 候选契约与精确子集

公开签名为：

```python
cs_rank_fe_native(
    values, *, method="average", pct=True, axis=-1,
    max_chunk_cells=1_000_000, max_result_bytes=256 * 1024**2,
)
```

本候选仅支持 `method="average"`、严格布尔 `pct=True` 和最后一轴截面。
对 FP `cs_rank` 来说，这对应日期 × 资产面板沿资产轴排名；调用者需保证每个完整
截面位于一行。其他 tie 方法、`pct=False`、对非最后轴排名、复数/object 输入及
高于 Float64 精度的扩展浮点输入均明确拒绝，不能当作等价候选。实整数保持原始
整数类型传给 Polars，避免 Float64 将 `2**53` 以上相邻整数合并；布尔与低于
Float64 的浮点类型只作安全数值传输转换。非空数值输出是同形状 Float64；空输入
沿用 FP 函数的早退规则并返回同 dtype 的副本。0-D 非空数组与 FP 一样因轴越界报错。
常量有限截面及单个有效值输出 0.5；无有效值的截面输出 NaN。

有限有效值的平均名次为 $r_i$，有效样本数为 $n$，rank 为：

$$
p_i = \frac{r_i - 1}{n - 1}, \qquad n > 1.
$$

唯一有效值采用明确定义的 $p_i=0.5$。FP 的 `np.isfinite` 与 FE `rank` 的
`RankSpec` 一致：NaN、正无穷和负无穷均不参与排名/分母，且这些输入位置输出
缺失。原生实现复用 FE 的 `RankSpec(method="average", pct_formula=...
rank_minus1_over_nminus1, singleton_value=0.5, inf_policy="exclude")` 及
`polars_cs_rank_expr`，由 Polars 完成排名和归一化。

`axis=-1` 或等价的非负最后轴索引受支持；其他轴以及 Boolean/非整数 axis 明确
拒绝。此候选不包含时间窗口，也不自行检查时间索引或 train/test 切分。对于因子
面板，日期应保持在前置分组维、资产保持在最后一维；不能把多日时间序列放入同一
截面排名，也不能跨时间或样本边界排名。跨日期因果性与数据选择由上游面板构造
负责，ndarray 候选本身没有时间元数据可以证明这些约束。

## 传输与内存界限

每个末轴完整截面都作为一个 FE rank group，不能为了满足预算拆分一个截面。
`max_chunk_cells` 限制每次送入 Polars 的总单元数；单个截面宽于该上限时在调用 FE
前报错。其余维度按组分块，从原数组按坐标提取 bounded tile，因此不会为完整
非连续输入先构造转置/连续副本。NumPy 只负责有限大小的数据载体、group/order ID、
索引和最终输出缓冲区。Float64 结果大小在分配前按 `values.size * 8` 检查
`max_result_bytes`；传 `None` 可显式取消输出上限。chunk 预算约束 FE carrier
单元数，不是对进程峰值 RSS 的硬保证，Polars 与运行时仍需要工作内存。
`bool`、NaN、非正数和非整数预算 fail closed。

## 验证与选择边界

定向测试使用独立 Python 排序/tie-run oracle、FP 实际 `cs_rank` 和真正的 FE
`polars_cs_rank_expr`；覆盖 ties、NaN/Inf、singleton/constant、空/0-D、只读非连续
view、多组分块、扩展预算拒绝、预算类型、禁止 FP fallback、实整数精度和 identity。
这只证明候选窄语义下的数值和 adapter contract，不证明完整 FE registry/bootstrap
调用链已通过，也不等于全因子 universe 上的吞吐/峰值内存优势。当前未运行全量 FE
startup 门禁；相关权威门禁不得因该候选而放宽。

尽管已有 FE 的 native-Polars 实现，本候选也不宣称它“最快”。要考虑默认切换或
推广前，仍需在相同数据、排序、输入搬运、warm/cold 状态及输出边界下做 FE/FP
A/B，报告实际端到端延迟与峰值内存，并继续走独立的 admission/governance 审核。
