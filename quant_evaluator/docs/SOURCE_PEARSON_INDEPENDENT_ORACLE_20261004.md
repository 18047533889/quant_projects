# 批量 Pearson 链的独立参考

`scripts.source_pearson_oracle.reference_source_pearson_chain` 为真实 source
测速准备独立数值参考，不调用 QE 的相关系数、均值、标准差或 IR 内核。
它只能验证 Pearson 四指标，不能据此认证其他指标、所有后端或最快路线。

对每个日期、因子，取因子和标签同时有限且 validity=True 的资产集合 $V$。
有效资产数小于 20，或因子/标签为常数截面时，日相关值为 NaN。
其余情况独立计算：

$$
r_t = \frac{\sum_{i\in V}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i\in V}(x_i-\bar{x})^2\sum_{i\in V}(y_i-\bar{y})^2}}.
$$

普通截面由 SciPy `pearsonr` 在 Float64 缩放后计算；近常数截面使用
`Decimal.from_float` 和 80 位精度的中心矩，避免巨大偏置导致消减。
Float32 存储会先升级 Float64 参考运算；更宽浮点、整数及 object 不在此参考
支持范围内，明确拒绝。每个时间 × 因子位置都计算，不使用抽样验收。

`pearson_ic_series` 返回全部 $r_t$。令有限日集合为 $D$，$m=|D|$：

$$
\mu = \frac{1}{m}\sum_{t\in D}r_t,\qquad
s = \sqrt{\frac{1}{m-1}\sum_{t\in D}(r_t-\mu)^2},\qquad
IR = \frac{\mu}{s}.
$$

`pearson_ic` 在至少 1 个有限日时返回 $\mu$；`pearson_ic_std` 和
`pearson_ic_ir` 至少需要 20 个有限日。常数序列的 IR 为 NaN，标准差为 0。
所有指标的公开 observation_counts 都是 $m$，不是资产数；即使不足 20 日、
标准差/IR 不可计算，仍保留实际有限日数。完整值、NaN/Inf 类别及这些计数都
需要与每次待认证的运行比较，CPU/GPU 相互一致不能替代此步骤。

传入新的顺序读取 source；函数不重置 source，也不负责 close，调用者须在
finally 中关闭。读取宽度受到显式 max_tile_size 和 source 实际预算同时限制。
默认只读 1 个因子 tile，不构建完整因子立方体，只保存 T×F 日序列和小型汇总。
默认 max_result_bytes=64 MiB 只约束返回数组，不是峰值 RSS 上限；读取、SciPy
临时空间与 Decimal 列表仍需要内存。真实 COS 运行继续遵守 ≥32 GiB 可用 RAM
门禁，不能用此较小输出预算替代总体资源检查。

本模块是 benchmark/reference 工具，不用于默认生产计算。必须放在 API 计时
之外；真实性能资格还需要当前源码/运行环境一致、完整实际 tile 回执、零 OOM、
相反运行顺序都出现同一严格赢家。本模块自身不构成 COS 性能资格。
