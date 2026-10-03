# Event decay 与 freshness 的时序和计算合同

## event_decay

每个资产按时间递增处理，输入先延迟一期：$z_t=x_{t-1}$。
有限、正的半衰期参数 $H$ 对应 $\alpha=\min(\ln(2)/H,1)$。
这是本库既有 alpha 定义，不等同于 $1-2^{-1/H}$，不能仅凭算子同名换后端。

每个连续有限 lagged 输入段的第一个值播种 $s_t=z_t$；随后每个有限值都更新：

$$s_t=(1-\alpha)s_{t-1}+\alpha z_t.$$

令 $c_t$ 为当前连续有限段的 lagged 观察数，$m$ 为 min_periods：

$$y_t=\begin{cases}s_t,&c_t\ge m\\\mathrm{NaN},&c_t<m.\end{cases}$$

暖启动仅隐藏输出，不暂停 EMA 状态更新。NaN、正负 Inf 都重置状态和连续计数；
之后的第一个有限值重新播种。第一行因不存在 lagged 输入而输出 NaN。
H 必须有限且为正；m 必须是非 bool 的正整数。

本次修复的是 m>=3 时暖启动提前输出；m=1、2 的既有有限输入递推口径保持不变。
独立 scalar oracle 覆盖暖启动、重置、资产隔离、索引保持和前缀因果性。

## freshness_aware_fill

本算子是当期可用（current-inclusive），不是 shift(1)：当期有限输入直接输出，
刷新最近值 $v$ 并把年龄 $a$ 重置为 0。资产内随后每个缺失或 Inf 行增加年龄。

$$y_t=v\exp(-\ln(2)\,a/H)=v\,2^{-a/H}.$$

没有历史有限值时输出 NaN。max_lag=L 表示最多携带 L 个连续缺失行；
第 L+1 个起输出 NaN，新的有限输入当期重新刷新。None 表示无界携带。
H 必须有限且为正；L 必须为非 bool 正整数或 None。
使用时必须确保当期原始观察在决策时刻已真实可得；本算子不会自行处理公告发布时间。

## FE 接入边界

目前这两个算子仍走 FP_NATIVE；注册中的语义描述不是 FE 执行绑定。
FE event_decay_asof 使用当期输入及不同的指数衰减定义，不能直接替代上述 lagged seeded EMA。
未来原生 Polars 接入必须先证明延迟、缺失重置、播种、暖启动和衰减参数完全一致。
