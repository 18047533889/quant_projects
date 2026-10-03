# FP event decay 与 FactorEngine 复用审计（2026-10-04）

## 结论

`event_decay` 可复用 FE 的原生 `lagged_iir_lowpass` 数值内核，但需将 FP 的半衰期参数换算为 FP 原有的直接 alpha，并在 `min_periods > 1` 时补上有限值预热遮罩。短向量对照覆盖了 H=0.5/2、min_periods=1/3、NaN/Inf、双资产与前缀；四组均通过。未经遮罩时，min_periods=3 的所选输入有 8 个有限性不匹配位置。

`freshness_aware_fill` 目前没有单个语义完全相同的 FE 算子。FE 的有限长 forward fill 只提供携带上限，不做随年龄指数衰减。`EventDecayAsOf` 是事件贡献累计，不是普通数值序列 EMA；Polars `EventDecayAsofNative` 目前仍是 TODO 占位实现。

本审计只记录内核语义和短向量证据；没有端到端性能或生产链路性能证据。

## 语义对照

FP 定义见 `factor_preprocess/transforms/event_decay.py`：

- 严格使用 `shift(1)`，输出 t 不读当前值。
- `alpha = min(ln(2) / halflife, 1)`；第一个有限 lagged 值直接 seed，之后按 `(1-alpha)*y + alpha*x` 更新。
- lagged 值为 NaN 或 ±Inf 时输出 NaN 并重置 EMA 与连续有限计数；后续有限 lagged 值重新 seed。
- `min_periods` 隐藏预热输出，但期间状态仍更新；缺失后计数从零开始。

递推数值候选为 `factor_engine/backend/long_ewm.py::lagged_iir_lowpass`：Polars 原生 `ewm_mean(adjust=False)` 位于 `factor_engine/backend/native_long_ewm.py::collect_lagged_ewma`。但当前 FE 包装层以 Pandas/NumPy 做资产分组、排序与 run-code 准备，并把输入输出转换为 NumPy；这是原生 Polars EWMA 内核，不是全流程原生 Polars 管线。包装层分段用于处理 shifted EMA 中的缺口；原始输入 x[k] 非有限时，第 k 行仍消费 x[k-1] 的有限值，至第 k+1 行 lagged 输入才为空并重置，随后有限 lagged 值重新 seed。该内核固定 `min_periods=1`。直接 alpha `min(ln(2)/H,1)` 可匹配 FP 状态递推；H=0.5 时不能以 FE `lagged_ewma(halflife=H)` 替代，因为其标准 alpha `1-exp(-ln(2)/H)` 不同。

`freshness_aware_fill` 定义见 `factor_preprocess/transforms/treatment_variants.py`：有限值在当前行原样发出并重置 age；NaN 和 ±Inf 都是缺失，继续按 `last_value * exp(-ln(2)/H * age)` 衰减携带，最多输出 L 个连续缺失行，超限后输出 NaN。一个短实测为 `[2, NaN, +Inf, NaN, 8]`、H=2、L=3，输出 `[2, 1.4142, 1, 0.7071, 8]`。Inf 不会重置旧值或 age。

## 注册与分发现状

FP registry 在 `factor_preprocess/registry/transforms.py` 为 `event_decay`、`freshness_aware_fill` 填入 `fe_equivalent_semantics` 描述，但未设置 FE_OPERATOR 或 FE_COMPOSITE 的实际 origin 绑定。因此当前解析仍是 FP_NATIVE；“equivalent semantics”元数据不表示运行时复用已接通。正式接入需另行改 adapter/registry，本文件没有做这些修改。

已存在的 FE composite `one_sided_iir_lowpass` 在 `factor_preprocess/adapters/fe_smoothing.py` 中指向 `long_ewm.lagged_iir_lowpass`，但 FP event_decay 尚无参数映射与 origin 绑定。Freshness 的 FE `ffill_limit`（`factor_engine/cleaned_operators/safe_ops.py`）只做有界 forward fill，没有衰减权重，因此它仅是可复用构件，不是语义等价算子。

相似命名不可作为复用证据：`factor_engine/cleaned_operators/state_event.py::EventDecayAsOf` 是累计事件贡献，包含当前观测并实现 carry/break 策略；`factor_engine/cleaned_operators/common/polars_event.py::EventDecayAsofNative` 的计算体标有 TODO，目前只将 NaN 转 null。

## 已执行的有界数值验证

环境：server-c `/home/sunhaiwei/quant_projects/.venv/bin/python`；输入使用两个资产、每个 8 行以及重复 index 标签。正式 FP 测试 `factor_preprocess/tests/test_event_freshness_numeric_contract_oct04.py` 中的 `_event_decay_reference` 作为独立标量 oracle；FE 候选直接调用 `lagged_iir_lowpass`。数据为：

```text
A = [2, 4, 8, NaN, 3, 9, +Inf, 5]
B = [11, 12, NaN, 5, 1, 2, 3, 4]
```

验证 H ∈ {0.5, 2}、min_periods ∈ {1, 3}。对 FE 输出按每个资产的严格过去有限 lag 计数遮罩后，与 FP oracle 和 FP 实现逐值对照，`rtol=0, atol=1e-12`：4 组均通过；有限性逐位相同；输出 Series index 与输入完全一致。最大有限绝对差：H=0.5 为 0，H=2 为 `4.44e-16`。全长与前 6 行截断面板的 FP 和 FE 输出均通过 prefix 对照（相同容差；FE 在 m=3 时加同一预热遮罩）。

对于同一输入，FE 原始 `min_periods=1` 输出与 FP `min_periods=3` 的有限性在 8 个位置不符：拼接面板位置 `[1, 2, 5, 6, 9, 10, 12, 13]`。这是需要预热遮罩的直接反例，不是 EMA 数值递推差异。

### 纯 Polars 表达式微型验证

另用 Polars 表达式直接做有限值清洗、按资产 shift、生成 run id，并按 `(asset, run)` 的时间序列计算 `ewm_mean`；没有调用 `long_ewm` 包装层的 `pd.factorize`、`np.argsort` 或 NumPy run-code 构造。源 gap 与 shifted gap 要区分：原始 `x[k]` 非有限时，输出第 k 行仍消费有限 `x[k-1]`；第 k+1 行才消费到 shifted null、输出 NaN，并开始新 segment；后续有限 lagged 值从新状态 seed。

核心表达式（`asset`、`ts`、`x` 为长表列；保存 `_pos` 用于结果按原输入次序恢复）：

```python
import math
import polars as pl
alpha = min(math.log(2.0) / halflife, 1.0)
finite = pl.col("x").is_finite().fill_null(False)
base = frame.with_columns(
    pl.when(finite).then(pl.col("x").cast(pl.Float64))
    .otherwise(None).alias("_x")
).sort(["asset", "ts"])
base = base.with_columns(
    pl.col("_x").shift(1).over("asset", order_by="ts").alias("_lag"),
    pl.col("_x").is_null().cast(pl.UInt32).cum_sum()
    .shift(1).over("asset", order_by="ts").fill_null(0).alias("_run"),
)
base = base.with_columns(
    pl.col("_lag").ewm_mean(
        alpha=alpha, adjust=False, min_samples=min_periods,
        ignore_nulls=False,
    ).over(["asset", "_run"], order_by="ts").alias("_out")
)
result = base.sort("_pos").get_column("_out")
```

微型对照输入及四组参数与上节相同；在两个资产各 8 行、NaN/Inf、重复 index 标签上，表达式输出和 FP oracle/实现的有限性完全相同，`rtol=0, atol=1e-12` 下最大误差为 H=0.5 时 0、H=2 时 `4.44e-16`。H={0.5,2}、m={1,3} 的全长对照及前 6 行 prefix 对照均通过；Polars `min_samples=m` 直接处理预热，无需额外 mask。该结果证明此微型表达式组合与这些输入契约相符，不代表 FE 已提供此 API、其生产长表/排序/索引集成已经完成，亦不提供全量性能结论。

### 极值与正负分流

单路 `ewm_mean` 表达式在 `x=[1e308,-1e308,1e308,-1e308,NaN,3]` 上不稳定。对 alpha `0.34657359027997264`、`0.9`、`1` 均观察到输出 `[NaN, 1e308, -Inf, NaN, NaN, NaN]`，而 FP oracle 在位置 1–4 都是有限值；除首个更新外无可用重叠有限值，因此重叠点上的相对误差没有判别意义。截到前四行的纯表达式与全长输出一致，但两者都保留上述错误，故 prefix 通过不等于数值正确。

FE `factor_engine/backend/native_long_ewm.py::collect_lagged_ewma` 已对大于 `float64.max/2` 的危险组走正负分开 EWMA 再相加的稳定分支。内存验证中，`factor_engine/backend/long_ewm.py::lagged_iir_lowpass` 在这三档 alpha 下与 FP oracle 的有限性完全相同，尺度归一化最大误差分别为约 `9.98e-17`、`9.98e-17`、0；前四行 prefix 均通过。

另对纯 Polars 试验把 lagged 值拆为同长度正负部分：正支为正值、其余有限值记 0；负支为负值、其余有限值记 0；缺失位置两支都保留 null。两支各自按相同 `(asset, run)` 做 `ewm_mean(adjust=False, min_samples=m, ignore_nulls=False)` 后求和。这个分流表达式对上节常规双资产矩阵的四组 H/m 全部通过，也对本节 `1e308` 正负交替输入在 alpha `0.34657359027997264`、`0.9`、`1` 下数值、有限性及前缀均通过。它是可继续实现的全表达式候选；只验证了小样本，不证明生产端到端性能。若直接采用单支 Polars `ewm_mean`，上述极值会破坏 finite mask；实现必须保留 FE 现有的稳定分流语义，或显式验证等价的安全算法。

复核命令要点（在仓库根目录执行；下列只展示复算核心，不代表完整测试套件）：

```python
import runpy
import numpy as np
import pandas as pd
from factor_engine.backend.long_ewm import lagged_iir_lowpass
from factor_preprocess.transforms.event_decay import event_decay
oracle = runpy.run_path(
    "factor_preprocess/tests/test_event_freshness_numeric_contract_oct04.py"
)["_event_decay_reference"]
alpha = min(np.log(2.0) / halflife, 1.0)
fe = lagged_iir_lowpass(frame, alpha=alpha)
fp = event_decay(frame, halflife=halflife, min_periods=min_periods)
# oracle(raw_asset, halflife, min_periods) 独立核对数值；
# FE 对照需先按每资产 shift(1) 的连续有限计数隐藏 warmup。
```

## 后续 adapter 回归建议

- `event_decay`: 参数域含 H<1 clipping；H=2；min_periods=1 与 >1；首行 NaN、严格 shift、有限值 seed/update、NaN/±Inf reset/reseed、连续缺失重置 warmup、资产隔离、重复 index 保序、prefix invariance。与 oracle 对数值及有效性；容差可用 `rtol=0, atol=1e-12`。
- `freshness_aware_fill`: leading missing；当前有限值立即发出；NaN/±Inf 均延续 age 与衰减；第 L 个缺失仍输出、第 L+1 个缺失为空；超限后有限值重新启动；资产隔离、index 保序、prefix invariance。需覆盖 max_lag=None 与有限 L。
- identity 应绑定实际 FE backend/kernel 与参数映射；不应把 `fe_equivalent_semantics` 标签当作执行证据。完成接线后再跑现有 FE adapter parity / identity 合约。
- 以上只建议语义验证。没有全链路吞吐、内存或生产性能测量，不能据此声称性能提升。

## 2026-10-04 FE 原生接入记录

本节补记本轮已经完成的 FP registry 接线；前文表达式和 IIR 对照是本次实现前的候选与历史证据，不应读作已完成的 API 接入。当前 recipe 为 `FE_COMPOSITE:long_ewm.event_decay_native:v1`，registry production execution 经 `factor_preprocess.adapters.fe_smoothing.execute_event_decay` 调用 `factor_engine.backend.long_ewm.event_decay_native`，再由 `factor_engine.backend.native_event_decay.collect_event_decay` 执行数值递推。

语义保持 FP 的直接 alpha 定义：`alpha = min(log(2)/halflife, 1)`；每个输出严格消费 `shift(1)`；首个有限 lagged 值 seed，之后用 `(1-alpha)*state + alpha*lagged` 更新。非有限 lagged 值输出空并重置 state 和连续有限 warmup 计数；`min_periods` 只遮住输出，不暂停 warmup 内的更新。

collector 用 Polars 表达式完成有限值清理、按资产生成 lag、构造 gap run id、EWM 与 min-sample warmup。先以 Polars 前缀表达式检测每个 asset/run 的历史 `absmax` 是否超过 `float64.max/2`：整帧无危险值时只构造单路 EWM；若整帧存在危险前缀，则构造普通 EWM 与正负分流两路 EWM，并按每行 hazard 选择结果，使极值到达前的前缀仍取普通分支。含危险值时多路表达式会额外计算，尚无性能证据。wrapper 仍使用 pandas 做列和日期验证、资产 ID 编码、NumPy 长表输入/输出与原行位置恢复，因此这是 Polars 原生数值 kernel 加轻量桥接层，不是全流程纯 Polars frontend。

identity 把 registry recipe、`execute_event_decay`、`long_ewm.event_decay_native`、`collect_event_decay`、运行时版本和参数策略写入 scoped digest；新增回归会改写选中 kernel 后检查 identity digest 随之变化。registry 仍保留原有治理/候选域 `halflife=(1,5)`；底层 FE wrapper/direct adapter 接受可转为有限正数的半衰期、拒绝 bool 及非有限或非正值（短于 1 时 alpha 截为 1）。因此 H=0.5 和更长半衰期的底层语义测试直接调用 FE adapter；registry 路由测试遵守原来的 1 到 5 域。

本次没有给 `event_decay_native` 增加 FE DSL/operator/expression export。FE 现有 `event_decay_asof` / `EventDecayAsOf` 计算的是事件贡献 as-of，不是此普通数值序列 EMA，不能作为别名。此次也未部署或发布生产因子；小型数值测试不构成大批量性能、吞吐或内存证据。
