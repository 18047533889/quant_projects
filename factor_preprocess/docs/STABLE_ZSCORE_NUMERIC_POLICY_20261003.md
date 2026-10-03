# 横截面 z-score：稳定数值合同 v2

## 为什么修复

旧实现直接计算大数的均值，再做减法。例如 Float64 输入
`1e16 + [0, 2, 4, 6]`，总体标准差口径 `ddof=0` 的数学结果应为
`[-3, -1, 1, 3] / sqrt(5)`。旧 NumPy 路径输出约
`[-0.816497, 0, 0.816497, 1.632993]`，不是正确的中心化结果。
旧 Polars 直接 mean/std 路径在此输入上也有明显数值误差。
两条后端相互一致或与某条旧代码一致，不能替代独立数学 oracle。

## 数学口径

每个截面将输入显式转换为 Float64，对有限且非缺失的成员计算：

$$a=\frac{\min(x)}2+\frac{\max(x)}2,\quad
d_i=x_i-a,\quad s=\max_i |d_i|.$$

当 $s>0$，定义 $u_i=d_i/s$；常数截面的 $u_i=0$。然后：

$$\bar u=\frac1n\sum_i u_i,\quad
\sigma_u=\sqrt{\frac{\sum_i(u_i-\bar u)^2}{n-\mathrm{ddof}}},\quad
z_i=\frac{u_i-\bar u}{\sigma_u}.$$

这与精确实数域中的 $(x_i-\bar x)/\sigma_x$ 等价，但避免直接计算
大偏移量均值、巨大数平方和以及微小数平方导致的精度损失/溢出/下溢。
Float64 转换会舍入超出精确表示范围的整数；**这不是整数精确 rank**。
rank 的精确整数排序合同没有因此改变。

## 边界行为和使用方式

- 默认 `numeric_policy="finite_anchor_centered_v2"`。
- NumPy 历史复现：显式 `numeric_policy="legacy_numpy_v1"`。
- Polars 历史复现：显式 `numeric_policy="legacy_polars_v1"`。
- NaN（以及 Polars null）仍为缺失输出，不参与有限截面的统计。
- 零标准差或有限成员数不足以满足 ddof 时，非缺失成员输出
  `constant_value`；默认是 0。不能把 singleton 自动改成另一个库的 NaN 口径。
- 含正负 Inf 的截面保留 FP 历史的传播/常数填充合同，不能静默删除 Inf
  后将剩余样本作为正常有限截面计算。v2 的稳定性承诺针对有限 Float64 输入。
- `axis`、`ddof`、`constant_value` 等原有参数继续有效。
- 非法 numeric policy 必须拒绝，包括空输入；不能因为 early return 绕过校验。

```python
from factor_preprocess.transforms.cross_sectional import cs_zscore
z = cs_zscore(values, axis=-1, ddof=1)
old = cs_zscore(values, ddof=1, numeric_policy="legacy_numpy_v1")
```

原生 Polars 路径使用表达式的分组 min/max、差分、缩放、mean/std，
不通过 pandas/NumPy 内核做统计，也没有 Python group UDF。
pandas 输入/输出转换仅是现有适配边界；保留原始行顺序及重复 index。

## 版本、训练和 FE 路由

注册 `cs_zscore` 为 `2.0.0`，semantic id 为
`CROSS_SECTIONAL_ZSCORE:cs`，numeric policy 为
`finite_anchor_centered_v2`。改变默认数值算法必须进入实现和数值政策身份，
不能沿用 v1 的训练/缓存身份声称完全相同。
保留原语义家族名称，避免新增后缀绕过横截面重复标准化等 lineage guard；
数值版本由 transform version、numeric policy 和 implementation hash 区分。

目前 FE 旧 mean/std 路径及 singleton 合同不能被冒称与 FP v2 等价。
因此本阶段不登记旧 FE 等价身份；FE v2 算子接入和生产准入仍需独立完成。
本修复不证明全平台所有算子无 bug，也不证明 v2 比所有其他后端更快。

## 本轮验证范围

2026-10-03 最终针对性合跑：111 passed / 1.24s。包含
27 项独立 Decimal/极端数值/非连续轴/分组顺序测试、7 项默认及历史口径身份测试、
37 项原生 Polars 数值合同测试、24 项 lineage 防重复标准化和 16 项 registry 身份测试。
此计数是该次合跑结果，不是整个库的全套结果。

独立参考使用 `Decimal.from_float`，验证**实际二进制 Float64 值**，
而不是先舍入为十进制字符串。精度设为 1200 位，覆盖最小 subnormal 的
精确十进制展开；100 位精度可能给重复 subnormal 捏造非零方差，不能当 oracle。
普通、大偏移、正负最大 Float64、subnormal、常数、NaN/Inf、singleton、
ddof 0/1/2、axis 0/-1、非连续数组、重复 pandas index 和 null 分组均有反例测试。

尚未完成：所有输入 dtype 的全面准入、大批量峰值内存及速度测量、
FE v2 生产路由，以及当前最终源码上的完整 FP 全套。不要把这些写成已认证。
空 NumPy 输入仍保持历史的同 dtype 副本返回，但必须先校验 numeric policy。
