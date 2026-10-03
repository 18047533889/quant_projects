# 稳定截面 z-score v2：FE 原生候选与复用边界

## 当前状态

新增独立 canonical `cs_zscore_finite_anchor_v2`，不替换历史 `zscore`。
模块是 `cleaned_operators/polars_native/cs_zscore_finite_anchor_v2.py`。
目前仅支持显式注册的研究候选：未加入正式 bootstrap loader、Polars
production-safe 名单或 FP 默认执行路由。导入模块不自动注册；显式注册要求
registry 为 BUILDING，不能重置或解冻已初始化的生产 registry。
不要把本文件或测试通过解释成所有参数、生产落值与最快后端认证。

两种实现是相互独立的 FE backend：`pandas_numpy` 为宽面板数值参考，
`polars` 使用真正的 Polars horizontal expressions / `with_columns`。
Polars 数学路径不调用 NumPy/Pandas 面板计算，NumPy 仅用于有限标量参数检查。
这里的“原生”不等于“流式”：当前实现 eager、物化完整宽面板，
临时列内存随行数和股票数增长；大批量适用性仍需单独性能及峰值内存测试。

## 计算公式

对一行截面中的有限 Float64 值集合 $F$，令 $n=|F|$，自由度参数为 $d$。
先按 Float64 转换输入，整数超过 $2^{53}$ 的舍入与 Float64 契约一致。
稳定中心和缩放量是：

$$
a=\frac{\min(F)}{2}+\frac{\max(F)}{2},\qquad
s=\max_{x\in F}|x-a|.
$$

当 $s>0$ 时，定义：

$$
u_i=\frac{x_i-a}{s},\quad
\bar u=\frac1n\sum_{i\in F}u_i,\quad
c_i=u_i-\bar u,\quad
v=\frac{\sum_{i\in F}c_i^2}{n-d},\quad
z_i=\frac{c_i}{\sqrt v}.
$$

它与正常有限输入的 $(x_i-\bar x)/\mathrm{std}_d(x)$ 含义一致，
但避免直接求和/平方溢出和大偏移下的消减。分阶段给 anchor、scale、mean、
center、std 命名，末阶段引用列别名；不把整棵标准差表达式复制到每一列。

缺失与退化策略：

- 输入 NaN 和 Polars null 保持缺失；不参与有限样本的计数和矩。
- 任意非缺失值为正负 Inf 时，该截面所有非缺失输出为 `constant_value`。
- 有限样本数不大于 `ddof`、常数截面或零中心平方和时，非缺失输出为
  `constant_value`（默认 0）。
- `ddof` 必须是非负整数、不能是 bool；`constant_value` 必须能转换为
  有限实数 Float64。极大的 `ddof` 会安全短路，不发生整型减法下溢。
- 布尔、复数和非数值资产列拒绝，不静默丢弃虚部或把字符串转换成 null。
  Polars 全 null 列接受并输出 Float64 缺失列。

时间/身份列使用模块中明确列出的名称，不参与截面；其他列都是资产数值列。
例如 `__csa_count` 若作为调用者资产列出现，也是有效资产值，不会被内部临时
列覆盖。只有身份列或没有任何资产列的面板会拒绝。调用只支持宽 DataFrame，
不把 Series 擅自解释成另一种截面轴。

## 验证与尚未闭合的部分

候选测试在 `factor_preprocess/tests/test_fe_stable_zscore_v2_candidate_oct03.py`。
当前 36 项通过，包含 1200 位精度的 `Decimal.from_float` 参考、Float64
极大值、次正规值、大偏移、Inf、常数、缺失、参数拒绝、空面板、临时列碰撞，
以及隔离子进程中真实注册/调用两种 backend。两点截面 `ddof=1` 的值是
$[-1/\sqrt2,+1/\sqrt2]$；三点 $[1,2,3]$ 则是 $[-1,0,1]$。

FP 独立函数允许的参数域与本候选并非全部相同，尤其不能把 fractional
`ddof`、legacy numeric policy 或任意轴用这份证据宣称整体等价。

## 已提供的 FE 原生长表接口

`backend/long_stable_zscore.py` 已新增分组长表数学接口，使用真正的 Polars
分组表达式，不 pivot。输入/返回都是 eager Polars DataFrame，保留原始行序、
列序和非值列，仅替换 value 列：

```python
from factor_engine.backend.long_stable_zscore import finite_anchor_centered_zscore_long

result = finite_anchor_centered_zscore_long(
    frame, value_col="value", group_col="date", ddof=1, constant_value=0.0,
)
```

`group_col` 是截面分组列，不是资产列。数学公式与上文一致，但这个长表接口
将输入 null 和 NaN 都输出为 NaN，以对齐 FP 现有长表 v2 缺失策略。
参数域仍是非负整数 `ddof` / 有限实数常数，布尔值列拒绝。
独立测试 14 项通过，包含 3 天 × 5461 股票、极端值、稀疏/Inf、打乱行序、
临时列冲突和 Decimal 参考。它是可直接调用的 FE 数学接口，不是正式算子
注册或 FP 默认路由；尚未完成真实多年面板性能 A/B 和端到端内存准入。
接 FP 默认之前还须验证完整调用契约和真实宽宇宙性能，不只改 metadata 声称复用。
