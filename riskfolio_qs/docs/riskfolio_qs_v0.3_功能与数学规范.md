# riskfolio_qs v0.3 功能与数学规范

## 1. 文档地位

本文是 `riskfolio_qs 0.3.0` 的统一功能与数学规范，覆盖当前已经落地的：

1. 优化器路由；
2. Alpha 处理；
3. 风险模型；
4. 目标函数；
5. 硬约束与成本项；
6. 求解、fallback 和冷启动；
7. CLI 数据装配；
8. 优化产物；
9. P0/P1 仓位分析；
10. 数据时点、provenance 和真实能力边界。

当本文与旧 v0.2 设计稿不一致时，以当前代码、冻结配置和本文为准。

当前代码版本关系：

| 项目 | 版本 |
|---|---|
| Python package | `0.3.0` |
| optimizer mapping | `v0.2.4` |
| frozen parameters | `v0.2.4` |
| position analysis schema | `1` |

本文只将“已经实现并测试”的功能标为可用。设计存在但未稳定交付的功能会明确标记
为 draft、Python-only 或未实现。

## 2. 系统边界

### 2.1 riskfolio_qs 负责

```text
Alpha
  -> 输入装配
  -> 风险模型
  -> 目标仓位求解
  -> 目标交易权重
  -> 风险/约束诊断
  -> 目标仓位 P0/P1 分析
```

### 2.2 riskfolio_qs 不负责

```text
目标仓位
  -> 订单生成
  -> 撮合成交
  -> 实际持仓
  -> 实际成本
  -> 收益、净值、回撤和绩效归因
```

以上执行与绩效链路属于 `vectorbt_qs` 或实盘交易系统。

`target_positions.parquet` 是目标仓位，不是实际持仓。仓位分析模块不能从目标权重
推断真实成交。

## 3. 记号

设决策日为 $t$，资产数为 $N$，因子数为 $K$。

| 记号 | 含义 |
|---|---|
| $w_t\in\mathbb{R}^N$ | 本期目标仓位 |
| $w_{t-1}\in\mathbb{R}^N$ | 优化前仓位；批量回测中是上一期求解结果 |
| $\Delta w_t=w_t-w_{t-1}$ | 目标交易权重 |
| $b_t\in\mathbb{R}^N$ | 基准权重 |
| $a_t=w_t-b_t$ | 主动仓位 |
| $\alpha_t\in\mathbb{R}^N$ | 原始 Alpha |
| $\mu_t\in\mathbb{R}^N$ | 转换到目标期限的预期收益向量 |
| $\Sigma_t\in\mathbb{R}^{N\times N}$ | 目标期限资产协方差 |
| $B_t\in\mathbb{R}^{N\times K}$ | Barra 因子暴露 |
| $F_t\in\mathbb{R}^{K\times K}$ | 因子协方差 |
| $D_t=\operatorname{diag}(d_t)$ | 特异方差对角阵 |
| $A$ | 年化因子，默认 252 |
| $H_\alpha$ | Alpha 预测期限，单位为交易期 |
| $H_r$ | 风险输入声明的期限 |
| $\epsilon$ | 数值容差 |

除非特别说明，组合权重、主动权重和交易权重都使用小数而非百分数。

## 4. 总体运行流程

一次优化运行的实际顺序是：

1. 解析 optimizer name 或 scenario；
2. 按所选优化器装配必要数据；
3. 校验输入 schema、日期、资产顺序和风险单位；
4. 计算 Alpha top-N 换手并按配置执行 EMA；
5. 检查强依赖和弱依赖；
6. 合并参数：

   ```text
   global -> risk -> solver -> optimizer -> runtime override
   ```

7. 逐决策日构造 point-in-time 风险模型；
8. 逐决策日求解；
9. 后处理并重新检查全部已启用硬约束；
10. 串联上一日目标仓位；
11. 生成 `OutputBundle` 和 CLI 审计产物。

仓位分析是独立的后处理流程，不调用优化器私有函数。

## 5. 当前优化器总表

### 5.1 正式支持矩阵

| 优化器 | 目标 | 风险模型 | CLI | Python | 状态 |
|---|---|---|---:|---:|---|
| `meanvar_enhance_barra_precomputed` | 主动 Mean-Variance | 预计算 Barra B/F/D | 是 | 是 | 正式 |
| `minvar_enhance_barra_precomputed` | 最小主动风险 | 预计算 Barra B/F/D | 是 | 是 | 正式 |
| `meanvar_enhance_hist` | 主动 Mean-Variance | 历史协方差 | 是 | 是 | 正式 |
| `minvar_enhance_hist` | 最小主动风险 | 历史协方差 | 是 | 是 | 正式 |
| `meanvar_absolute_return` | 绝对 Mean-Variance | 历史协方差 | 是 | 是 | 正式 |
| `topn_long_only_equal_weight` | Alpha 排序等权 | 无 | 是 | 是 | 正式 |
| `meanvar_enhance_index` | 主动 Mean-Variance | 原始 Barra 收益估计 | 见下文 | 是 | Python-only |
| `minvar_enhance_index` | 最小主动风险 | 原始 Barra 收益估计 | 见下文 | 是 | Python-only |
| `topn_long_short_equal_weight` | Alpha 多空排序 | 无 | 技术可调用 | 技术可调用 | draft |

### 5.2 CLI legacy alias

CLI 对以下名称做兼容转发：

```text
meanvar_enhance_index
    -> meanvar_enhance_barra_precomputed

minvar_enhance_index
    -> minvar_enhance_barra_precomputed
```

因此：

- CLI 中的 `meanvar_enhance_index` 不会运行原始 `barra_factor`；
- CLI 中的 `minvar_enhance_index` 不会运行原始 `barra_factor`；
- 若要真正使用 `F + F_ret + F_spec`，必须通过 Python `InputBundle` 调用。

### 5.3 默认场景路由

冻结 mapping 的默认路由：

| scenario | 默认优化器 |
|---|---|
| `index_enhancement` | `meanvar_enhance_barra_precomputed` |
| `conservative_index_enhancement` | `minvar_enhance_barra_precomputed` |
| `absolute_return` | `meanvar_absolute_return` |
| `fallback` | `topn_long_only_equal_weight` |

Python `OptimizationPipeline` 在未显式指定优化器时还可根据已有输入动态选择：

```text
B + factor_cov + specific_var -> precomputed Barra
F + F_ret + F_spec           -> raw Barra
market                       -> historical covariance
otherwise                    -> fail
```

CLI 为了先装配输入，会先解析目标优化器。要运行历史模式，应显式选择
`*_enhance_hist` 或绝对收益优化器。

## 6. Alpha 数学定义

### 6.1 `expected_return`

当：

```yaml
adapter:
  alpha_input_type: expected_return
```

时：

$$
\mu_{t,i} =
\begin{cases}
\alpha_{t,i}, & \alpha_{t,i}\text{ 有限}\\
0, & \text{否则}
\end{cases}
$$

调用方必须保证 Alpha 已经是 `alpha_horizon_days` 对应期限的小数收益。

### 6.2 `score`

当 `alpha_input_type=score` 时，先做当日横截面标准化：

$$
z_{t,i} =
\frac{\alpha_{t,i}-\bar\alpha_t}{s_t}
$$

代码使用总体标准差：

$$
s_t =
\sqrt{\frac{1}{M_t}\sum_{i\in V_t}
(\alpha_{t,i}-\bar\alpha_t)^2}
$$

其中 $V_t$ 是当日有限 Alpha 集合，$M_t=|V_t|$。

随后：

$$
\mu_{t,i}
=z_{t,i}\times
\text{score\_return\_scale\_bps}\times 10^{-4}
$$

默认 `score_return_scale_bps=10`。

若横截面标准差不大于 $10^{-12}$，全部 $\mu_{t,i}=0$。
原始缺失值对应的 $\mu_{t,i}=0$，但求解前仍会按原始 Alpha 检查覆盖率。

### 6.3 Alpha 覆盖率

$$
\operatorname{coverage}^{\alpha}_t
=\frac{\#\{i:\alpha_{t,i}\text{ 非缺失}\}}{N}
$$

默认要求：

$$
\operatorname{coverage}^{\alpha}_t\ge 80\%
$$

覆盖率基于未平滑的原始 Alpha，EMA 不能掩盖当日缺失。

## 7. 信号平滑

### 7.1 Top-N 信号换手

令 $S_t$ 为当日 Alpha 最大的前 $n$ 个资产集合：

$$
\operatorname{signal\_turnover}_t
=1-
\frac{|S_t\cap S_{t-1}|}
{\max(1,\min(n,|S_t|,|S_{t-1}|))}
$$

第一期定义为 0。

默认参数：

```text
topn_n = 50
turnover_window = 5
turnover_threshold = 0.30
ema_span = 5
```

### 7.2 EMA

Pandas `ewm(span=s, adjust=False)` 对应递推：

$$
\widetilde\alpha_t
=\lambda\alpha_t+(1-\lambda)\widetilde\alpha_{t-1}
$$

其中：

$$
\lambda=\frac{2}{s+1}
$$

### 7.3 模式

| mode | 行为 |
|---|---|
| `never` | 始终使用原始 Alpha |
| `always` | 始终使用 EMA Alpha |
| `auto` | 截至当日的滚动平均信号换手超过阈值时使用 EMA |

`auto` 是逐日因果判断，不使用样本末尾状态回填历史。

## 8. 风险期限换算

所有凸优化风险项最终换算到 Alpha 期限。

定义源风险期限：

$$
H_s =
\begin{cases}
1, & \text{daily\_variance}\\
H_r, & \text{horizon\_variance}\\
A, & \text{annual\_variance}
\end{cases}
$$

风险缩放：

$$
\kappa=\frac{H_\alpha}{H_s}
$$

协方差和特异方差均乘以 $\kappa$。

该换算假设方差随时间线性缩放。

## 9. 预计算 Barra 风险模型

### 9.1 基础模型

$$
\Sigma_t=B_tF_tB_t^\top+D_t
$$

优化器不展开 $N\times N$ 稠密矩阵。对任意风险向量 $v$：

$$
\operatorname{Var}(v)
=(B_t^\top v)^\top F_t(B_t^\top v)
+\sum_i d_{t,i}v_i^2
$$

边际风险：

$$
\Sigma_tv
=B_tF_tB_t^\top v+D_tv
$$

### 9.2 Point-in-time 选择

对决策日 $t$ 和 lag $L$，在所有不晚于 $t$ 的可用截面中选择倒数第
$L+1$ 个：

$$
t^*=
\operatorname{eligible}(t)[-(L+1)]
$$

暴露和风险允许使用不同 lag：

- `exposure_data_lag_periods`：选择 $B$；
- `risk_data_lag_periods`：选择 $F$ 和 $D$。

### 9.3 数值处理

因子协方差先检查：

$$
\max_{i,j}|F_{ij}-F_{ji}|
\le \text{precomputed\_symmetry\_tolerance}
$$

默认容差 $10^{-10}$。

换算到 Alpha 期限后，对称化并做 PSD 投影：

$$
F=Q\Lambda Q^\top
$$

$$
\widetilde\Lambda_{kk}
=\max(\Lambda_{kk},10^{-12})
$$

$$
\widetilde F=Q\widetilde\Lambda Q^\top
$$

特异方差：

$$
\widetilde d_i
=\max(\kappa d_i,10^{-8})
$$

### 9.4 Barra adapter 的数据修复

当前 adapter 会执行：

1. 风格因子缺失填 0；
2. 特异方差缺失按当日横截面中位数填充；
3. 行业 dummy 缺失或不满足 one-hot 时，将所有行业置 0，再把第一个行业置 1。

第 3 条是兼容性修复，会掩盖原始行业缺失；分析和审计时必须结合数据质量报告理解。

### 9.5 当前 specific-risk quality gate 限制

adapter 计算年化特异波动率后先执行：

$$
\sigma^{spec,ann}_i
\leftarrow
\max(\sigma^{spec,ann}_i,\sigma_{\min})
$$

然后才检查是否低于 $\sigma_{\min}$。因此当前
`minimum_annual_specific_volatility` 不会拒绝过低原始值，也不会回写
`specific_var`；真正进入求解的下限仍是
`precomputed_specific_variance_floor`。

这是当前实现边界，不应把该参数理解为有效的原始数据质量闸门。

## 10. 原始 Barra 收益估计

该模式仅通过 Python `InputBundle` 正式表达。

### 10.1 因子协方差

使用最近 $L_f$ 期因子收益：

$$
\widehat F_t
=\operatorname{Cov}
\left(f_{t-L_f+1},\ldots,f_t\right)
$$

默认 $L_f=252$，Pandas 样本协方差使用 $ddof=1$。

若可用因子收益少于 2 期：

$$
\widehat F_t=10^{-4}I
$$

### 10.2 系统性协方差

$$
\Sigma_{sys,t}=B_t\widehat F_tB_t^\top
$$

### 10.3 特异方差

使用最近 $L_s=252$ 期特异收益：

$$
\widehat d_{t,i}
=\operatorname{Var}
\left(\epsilon_{t-L_s+1,i},\ldots,\epsilon_{t,i}\right)
$$

少于 2 期时使用 $10^{-4}$。有限值最低为 $10^{-8}$，非有限回退为
$10^{-6}$。

### 10.4 收缩与 PSD

先构造：

$$
\Sigma_t^{raw}
=\Sigma_{sys,t}+\operatorname{diag}(\widehat d_t)
$$

再向等方差对角阵收缩：

$$
\bar d_t
=\max\left(\frac{\operatorname{tr}(\Sigma_t^{raw})}{N},10^{-6}\right)
$$

$$
\Sigma_t^{shrunk}
=(1-\rho)\Sigma_t^{raw}+\rho\bar d_tI
$$

默认 $\rho=0.05$，之后将特征值截断到至少 $10^{-8}$，再乘风险期限系数
$\kappa$。

## 11. 历史协方差

### 11.1 收益输入

CLI 从：

```text
ashare_stock_daily.Return
```

读取收益，并执行：

$$
r_{t,i}=\frac{\text{Return}_{t,i}}{10000}
$$

该字段已经是收益，不能再次 `pct_change`。

Python 输入若声明 `market_input_type=price`，才执行：

$$
r_{t,i}=\frac{P_{t,i}}{P_{t-1,i}}-1
$$

### 11.2 样本窗口

默认最近 60 个可用交易期，并按 `market_data_lag_periods` 滞后。

### 11.3 Sample covariance

当 `hist_cov_method=sample` 时：

$$
\widehat\Sigma_t
=\operatorname{Cov}_{sample}(r)
$$

每个协方差元素按 pairwise 可用样本计算。

### 11.4 EWMA covariance

默认 `hist_cov_method=ewma`，half-life 为 $h=20$。

对窗口内由旧到新的位置 $s$，年龄为 $a_s$：

$$
q_s\propto
\exp\left(\frac{\log(0.5)a_s}{h}\right)
$$

并归一化使：

$$
\sum_s q_s=1
$$

对资产 $i$：

$$
\bar r_i
=\frac{\sum_s q_sI_{s,i}r_{s,i}}
{\sum_s q_sI_{s,i}}
$$

pairwise 协方差：

$$
\widehat\Sigma_{ij}
=
\frac{
\sum_s q_sI_{s,i}I_{s,j}
(r_{s,i}-\bar r_i)(r_{s,j}-\bar r_j)
}{
\sum_s q_sI_{s,i}I_{s,j}
}
$$

其中 $I_{s,i}$ 表示该收益是否有效。

### 11.5 缺失回退

资产 $i$ 可靠的条件为：

1. 有效样本不少于 `effective_min`；
2. 对角方差有限；
3. 对角方差大于 `covariance_variance_floor`。

不可靠资产与其他资产的协方差置 0，其自身方差取可靠资产方差中位数；若没有可靠
资产，则取：

$$
\max(10^{-6},10^{-4})=10^{-4}
$$

窗口少于 2 期时，整个协方差直接使用 $10^{-4}I$。

### 11.6 收缩、PSD 和期限

$$
\bar d_t
=\max\left(
\frac{\operatorname{tr}(\widehat\Sigma_t)}{N},
10^{-6}
\right)
$$

$$
\Sigma_t^{shrunk}
=(1-\rho)\widehat\Sigma_t+\rho\bar d_tI
$$

默认 $\rho=0.05$。随后特征值截断到至少 $10^{-8}$，最终：

$$
\Sigma_t=\kappa\Sigma_t^{PSD}
$$

## 12. 交易成本

### 12.1 交易权重

$$
\Delta w_t=w_t-w_{t-1}
$$

### 12.2 单边换手

$$
T_t=\frac12\sum_i|\Delta w_{t,i}|
$$

### 12.3 线性成本

输入单位为 bps，转换为小数：

$$
c_{t,i}=\text{linear\_cost\_bps}_{t,i}\times10^{-4}
$$

线性成本项：

$$
C_t^{linear}
=\lambda_c\sum_i c_{t,i}|\Delta w_{t,i}|
$$

CLI 当前固定：

```text
linear_cost_bps = 5
linear_cost_penalty = 1
```

即单边每单位权重成本为 5 bps。

### 12.4 换手软惩罚

$$
C_t^{turnover}
=s_c\lambda_T\frac12\|\Delta w_t\|_1
$$

其中：

```text
soft_cost_scale = 0.05
turnover_penalty = 0
```

当前默认换手软惩罚为 0。

必须注意：`soft_cost_scale` 只乘换手软惩罚，不乘线性成本和冲击成本。

### 12.5 二次冲击成本

启用时：

$$
C_t^{impact}
=\lambda_q\sum_i q_{t,i}(\Delta w_{t,i})^2
$$

当前 production CLI：

- 不接受 impact cost 文件；
- 拒绝通过 runtime override 打开 impact cost；
- 默认 `enable_impact_cost=false`。

因此二次冲击成本当前是 Python API 能力，不是 production CLI 能力。

### 12.6 总软成本

$$
C_t
=C_t^{turnover}
+C_t^{linear}
+\mathbf 1_{\text{impact enabled}}C_t^{impact}
$$

## 13. 凸优化目标函数

### 13.1 最小主动风险

`OBJ_MINVAR_ACTIVE`：

$$
\min_{w_t}
\quad
(w_t-b_t)^\top\Sigma_t(w_t-b_t)
+C_t
$$

该目标不使用 `risk_weight`。

### 13.2 主动 Mean-Variance

`OBJ_MEANVAR_ACTIVE`：

$$
\min_{w_t}
\quad
\lambda_r(w_t-b_t)^\top\Sigma_t(w_t-b_t)
-\lambda_\alpha\mu_t^\top w_t
+C_t
$$

默认：

```text
risk_weight = 1
alpha_weight = 1
```

### 13.3 绝对 Mean-Variance

`OBJ_MEANVAR_ABS`：

$$
\min_{w_t}
\quad
\lambda_rw_t^\top\Sigma_tw_t
-\lambda_\alpha\mu_t^\top w_t
+C_t
$$

### 13.4 未实现目标

当前没有正式实现：

- Max Sharpe；
- CVaR；
- Risk Parity；
- 最大分散化；
- 非凸 cardinality；
- 凸优化 long-short；
- 多期联合优化。

## 14. 硬约束

### 14.1 预算

$$
\sum_iw_{t,i}=B
$$

默认 $B=1$。

### 14.2 Long-only

$$
w_{t,i}\ge0
$$

当前凸优化器若 `long_only=false` 会直接报错，因为没有正式定义下界。

### 14.3 个股上限

$$
w_{t,i}\le u_i
$$

当前使用统一上限：

- 指数增强默认 3%；
- 绝对收益默认 5%；
- TopN 配置上限 5%。

### 14.4 总敞口

$$
\|w_t\|_1\le G
$$

默认 $G=1$。在 long-only 且预算为 1 时，该约束与预算约束等价。

### 14.5 主动权重

$$
-u^a\le w_{t,i}-b_{t,i}\le u^a
$$

指数增强默认：

$$
u^a=2\%
$$

### 14.6 换手上限

$$
\frac12\|w_t-w_{t-1}\|_1\le T_{\max}
$$

指数增强默认 20%，绝对收益默认 25%。

### 14.7 不可交易资产冻结

若资产 $i$ 不可交易：

$$
w_{t,i}=w_{t-1,i}
$$

CLI 的可交易定义：

$$
\operatorname{tradable}_{t,i}
=
(\operatorname{Volume}_{t,i}>0)
\land
(\operatorname{Amount}_{t,i}>0)
$$

缺失行情行视为不可交易。

### 14.8 行业主动暴露

对行业 $g$，令 $I_{i,g}\in\{0,1\}$：

$$
e_{t,g}^{industry}
=\sum_iI_{i,g}(w_{t,i}-b_{t,i})
$$

`strict`：

$$
e_{t,g}^{industry}=0
$$

`band`：

$$
-\delta_g\le e_{t,g}^{industry}\le\delta_g
$$

预计算 Barra 指增默认 strict。

### 14.9 风格主动暴露

对风格因子 $k$：

$$
e_{t,k}^{style}
=\sum_iB_{t,i,k}(w_{t,i}-b_{t,i})
$$

默认 band：

$$
|e_{t,k}^{style}|\le 0.10
$$

### 14.10 市值主动暴露

原始市值 $m_i>0$ 先取对数并横截面标准化：

$$
x_i=
\frac{\log m_i-\operatorname{mean}(\log m)}
{\operatorname{std}(\log m)}
$$

缺失或非正市值用有效市值中位数填充。

主动暴露：

$$
e_t^{mcap}=x^\top(w_t-b_t)
$$

原始 Barra 模式默认 band 0.10；预计算 Barra adapter 当前不提供 `F_mcap`，
所以预计算模式冻结参数将市值中性设为 off。

### 14.11 Tracking Error 上限

目标期限主动方差：

$$
V_t^a=(w_t-b_t)^\top\Sigma_t(w_t-b_t)
$$

年化 Tracking Error：

$$
TE_t^{ann}
=\sqrt{V_t^a\frac{A}{H_\alpha}}
$$

若年化上限为 $c_{TE}$，求解器内约束写为：

$$
V_t^a
\le
c_{TE}^2\frac{H_\alpha}{A}
$$

指数增强默认：

$$
c_{TE}=3\%
$$

该约束是二次约束，OSQP 不支持；应使用 CLARABEL 或 SCS。

## 15. 约束预检查与后检查

### 15.1 求解前检查

代码综合：

- long-only；
- 个股上限；
- 主动权重上下界；
- 不可交易资产冻结；

形成每个资产的可行区间 $[l_i,u_i]$，并检查：

$$
l_i\le u_i
$$

$$
\sum_il_i\le B\le\sum_iu_i
$$

### 15.2 求解后处理

求解器返回 $w^{raw}$ 后：

1. 非有限值替换为 0；
2. 小幅负值截断为 0；
3. 检查预算误差；
4. 重新归一化：

   $$
   w=w^{raw}\frac{B}{\sum_iw_i^{raw}}
   $$

5. 对归一化后的最终权重重新计算全部已启用约束；
6. 最大 violation 超过：

   $$
   \max(10^{-5},10\times\text{solver\_tol},10^{-6})
   $$

   时，视为求解失败。

## 16. 冷启动与持仓串联

### 16.1 批量回测

只从外部提供首期初始仓位，之后严格串联：

$$
w_{t-1}^{input}=
\begin{cases}
w_{initial}, & t=t_0\\
w_{t-1}^{solved}, & t>t_0
\end{cases}
$$

### 16.2 CLI 初始化

| 场景 | 首期仓位 |
|---|---|
| 指数增强 | 当日基准 |
| 历史绝对收益 | 全现金，即股票权重为 0 |
| Alpha-only TopN | 无显式 previous，输出适配时按 0 |

### 16.3 冷启动换手例外

仅当：

```text
position == first period
and sum(previous_weights) == 0
```

才不施加换手硬上限。

该例外只关闭换手硬约束，不关闭目标函数中的线性成本或换手软惩罚。

## 17. 求解器

正式默认求解器为 CLARABEL。

| solver | 参数映射 |
|---|---|
| CLARABEL | `max_iter`, `tol_gap_abs`, `tol_gap_rel`, `tol_feas` |
| SCS | `max_iters`, `eps` |
| OSQP | `max_iter`, `eps_abs`, `eps_rel` |
| ECOS | `max_iters`, `abstol`, `reltol`, `feastol` |

有效状态：

```text
optimal
optimal_inaccurate
```

未知求解器、异常、infeasible、unbounded、无权重或后验约束失败都进入失败路径。

## 18. Fallback

### 18.1 路由级 fallback

- 原始 Barra 优化器缺强输入时，可降级到对应历史协方差优化器；
- 预计算 Barra 和历史优化器默认 fail-fast；
- TopN 规则模式作为 default fallback 场景。

### 18.2 单日求解 fallback

`on_solve_failure` 支持：

| 值 | 行为 |
|---|---|
| `fail_fast` | 直接失败 |
| `carry_forward` | 有非零上期持仓时延续上期仓位 |
| `equal_weight` | 仅允许绝对收益优化器 |

Fallback 后仍重新计算全部硬约束；不满足约束仍失败。

CLI 默认 `allow_fallback=false`。即使得到可行 fallback 权重，整次 CLI 仍返回失败，
除非显式允许。

## 19. 规则型优化器

### 19.1 TopN long-only equal weight

对当日有限 Alpha 取最大 $n$ 个资产集合 $S_t$：

$$
w_{t,i}=
\begin{cases}
\frac{B}{|S_t|}, & i\in S_t\\
0, & i\notin S_t
\end{cases}
$$

默认：

```text
n = 50
B = 1
```

求解后只额外检查最大等权是否超过 `single_name_max`。

该规则后端不使用：

- 风险模型；
- benchmark；
- tradable；
- 换手约束；
- 交易成本；
- cvxpy。

### 19.2 TopN long-short equal weight

该模式当前为 draft。

从 Alpha 最大端选择 $L$，从排除多头后的最小端选择 $S$：

$$
w_i=
\begin{cases}
w_L, & i\in L\\
w_S, & i\in S\\
0, & \text{otherwise}
\end{cases}
$$

默认：

```text
long_n = 10
short_n = 10
long_weight = 0.05
short_weight = -0.05
```

当前不会自动归一化 gross/net exposure，也没有完整约束后验检查，因此不能视为正式
生产能力。

## 20. 冻结参数摘要

### 20.1 通用参数

| 参数 | 默认值 |
|---|---:|
| `budget` | 1.0 |
| `long_only` | true |
| `gross_exposure_cap` | 1.0 |
| `min_alpha_coverage` | 0.80 |
| `min_benchmark_coverage` | 0.95 |
| `min_exposure_coverage` | 0.95 |
| `min_prev_positions_coverage` | 1.00 |
| `default_linear_cost_bps` | 5.0 |
| `soft_cost_scale` | 0.05 |
| `score_return_scale_bps` | 10.0 |
| `solver_name` | CLARABEL |
| `max_iters` | 5000 |
| `solver_tol` | $10^{-6}$ |

指数增强优化器覆盖为 `max_iters=10000`、`solver_tol=1e-7`。

### 20.2 优化器差异

| 优化器族 | single max | active max | turnover cap | TE cap | 行业 | 风格 | 市值 |
|---|---:|---:|---:|---:|---|---|---|
| raw Barra index | 3% | 2% | 20% | 3% | strict | band 0.10 | band 0.10 |
| precomputed Barra index | 3% | 2% | 20% | 3% | strict | band 0.10 | off |
| historical index | 3% | 2% | 20% | 3% | off | off | off |
| historical absolute | 5% | 不适用 | 25% | 不适用 | off | off | off |
| TopN long-only | 5% | 不适用 | 无 | 无 | 无 | 无 | 无 |

## 21. CLI 输入装配

### 21.1 用户输入

Production optimize CLI 只接受外部 Alpha 文件：

```yaml
inputs:
  alpha: ...
```

不接受用户提供：

- benchmark 文件；
- previous positions 文件；
- tradable 文件；
- linear-cost 文件；
- impact-cost 文件。

### 21.2 data_access 数据

| 用途 | dataset / field |
|---|---|
| benchmark | `ashare_index_constituent.Weight` |
| tradable | `ashare_stock_daily.Volume/Amount` |
| 历史收益 | `ashare_stock_daily.Return / 10000` |
| 交易日历 | `ashare_calendar.IsTradeDay` |
| P1 行业 | `ashare_stock_industry`, `IndustrySource=sw_l1` |
| P1 市值 | `ashare_stock_valuation_daily.MarketCap` |
| P1 ADV | `ashare_stock_daily.Amount` |

### 21.3 Benchmark universe coverage

设完整指数权重和为 $W_t^{full}$，Alpha universe 中保留权重为
$W_t^{selected}$：

$$
\operatorname{coverage}^{benchmark}_t
=\frac{W_t^{selected}}{W_t^{full}}
$$

要求不低于 95%。通过后，仅在所选 universe 内重新归一化：

$$
\widetilde b_{t,i}
=\frac{b_{t,i}}{\sum_{j\in U_\alpha}b_{t,j}}
$$

因此求解中的 benchmark 是 Alpha universe 内的归一化基准。

## 22. Barra risk package 契约

manifest 必须至少声明：

```yaml
product: barra_lite
version: ...
provider: ...
factors:
  - id: ...
    type: continuous
  - id: ...
    type: dummy
estimation:
  units: daily_variance
files:
  exposure: exposure.parquet
  factor_cov: factor_cov.parquet
  specific_risk: specific_risk.parquet
```

合法风险单位：

```text
daily_variance
horizon_variance
annual_variance
```

必须至少有一个 continuous factor 和一个 dummy factor。

## 23. 优化输出

### 23.1 `target_positions`

宽表：

```text
index = date
columns = asset
value = target weight
```

### 23.2 `trades`

$$
\Delta w_t=w_t-w_{t-1}
$$

存储为：

```text
MultiIndex(date, asset)
column = delta_weight
```

首期使用输入初始仓位，之后使用上一日实际求解出的目标仓位。

### 23.3 基础 summary

$$
\operatorname{gross}_t=\sum_i|w_{t,i}|
$$

$$
\operatorname{net}_t=\sum_iw_{t,i}
$$

$$
\operatorname{turnover}_t
=\frac12\sum_i|\Delta w_{t,i}|
$$

### 23.4 风险诊断

绝对组合波动率：

$$
\sigma_t^p=\sqrt{w_t^\top\Sigma_tw_t}
$$

目标期限 Tracking Error：

$$
TE_t^H
=\sqrt{a_t^\top\Sigma_ta_t}
$$

年化 Tracking Error：

$$
TE_t^{ann}
=TE_t^H\sqrt{\frac{A}{H_\alpha}}
$$

目标函数风险：

$$
\sigma_t^{obj}
=
\begin{cases}
\sqrt{a_t^\top\Sigma_ta_t}, & \text{active objective}\\
\sqrt{w_t^\top\Sigma_tw_t}, & \text{absolute objective}
\end{cases}
$$

### 23.5 预期收益和成本诊断

$$
\operatorname{expected\_return}_t
=\mu_t^\top w_t
$$

$$
\operatorname{expected\_linear\_cost}_t
=\lambda_c\sum_ic_{t,i}|\Delta w_{t,i}|
$$

$$
\operatorname{expected\_impact\_cost}_t
=\lambda_q\sum_iq_{t,i}(\Delta w_{t,i})^2
$$

### 23.6 Active share

$$
\operatorname{active\_share}_t
=\frac12\sum_i|w_{t,i}-b_{t,i}|
$$

### 23.7 风险贡献诊断

对目标函数风险向量 $v_t$：

$$
m_t=\Sigma_tv_t
$$

$$
RC_{t,i}=v_{t,i}m_{t,i}
$$

输出：

$$
\operatorname{risk\_contribution\_sum}_t
=\sum_iRC_{t,i}
=v_t^\top\Sigma_tv_t
$$

以及：

$$
\operatorname{largest\_abs\_risk\_contribution}_t
=\max_i|RC_{t,i}|
$$

## 24. CLI 运行状态

优化 CLI 成功要求：

1. 每日 `feasible_flag=true`；
2. solve status 可接受；
3. 最大约束 violation 不超过容差；
4. 若使用 fallback，必须由 `allow_fallback=true` 授权。

输出目录包含：

```text
target_positions.parquet
trades.parquet
summary.parquet
metadata.json
run_manifest.yaml
resolved_cli_config.yaml
resolved_mapping.yaml
resolved_params.yaml
```

## 25. 仓位分析 P0

P0 只依赖优化产物，不访问外部数据。

### 25.1 首期仓位重建

$$
\widehat w_{t_0-1}=w_{t_0}-\Delta w_{t_0}
$$

第二期以后验证：

$$
\|w_t-w_{t-1}-\Delta w_t\|_\infty
\le\epsilon_{reconstruction}
$$

首期没有独立初始仓位文件时，交易与目标仓位只能反推出初始仓位，不能自证。
若 manifest 声明从现金或基准初始化，P0/P1 会进一步验证该假设。

### 25.2 仓位统计

$$
\operatorname{long\_count}_t
=\#\{i:w_{t,i}>\epsilon_w\}
$$

$$
\operatorname{short\_count}_t
=\#\{i:w_{t,i}<-\epsilon_w\}
$$

$$
\operatorname{unallocated}_t
=B-\sum_iw_{t,i}
$$

通用输出使用 `unallocated_weight`，不把 long-short 组合的剩余预算错误命名为现金。

### 25.3 集中度

定义绝对权重份额：

$$
p_{t,i}
=\frac{|w_{t,i}|}{\sum_j|w_{t,j}|}
$$

HHI：

$$
HHI_t=\sum_ip_{t,i}^2
$$

有效持仓数：

$$
N_t^{eff}=\frac1{HHI_t}
$$

Top-k 绝对权重占比：

$$
\operatorname{topk\_share}_t
=\sum_{i\in TopK(|w_t|)}p_{t,i}
$$

### 25.4 交易统计

总交易权重：

$$
\operatorname{gross\_traded\_weight}_t
=\sum_i|\Delta w_{t,i}|
$$

单边换手：

$$
\operatorname{one\_way\_turnover}_t
=\frac12\operatorname{gross\_traded\_weight}_t
$$

买入和卖出：

$$
\operatorname{buy\_weight}_t
=\sum_i\max(\Delta w_{t,i},0)
$$

$$
\operatorname{sell\_weight}_t
=\sum_i\max(-\Delta w_{t,i},0)
$$

### 25.5 持仓延续性

令 $S_t=\{i:|w_{t,i}|>\epsilon_w\}$：

$$
\operatorname{holding\_overlap}_t
=\frac{|S_t\cap S_{t-1}|}{|S_t\cup S_{t-1}|}
$$

权重余弦相似度：

$$
\operatorname{cosine}_t
=\frac{w_t^\top w_{t-1}}
{\|w_t\|_2\|w_{t-1}\|_2}
$$

### 25.6 独立复核

P0 独立重算：

- gross exposure；
- net exposure；
- turnover。

同时保留：

```text
reported_*
recomputed_*
*_difference
```

不会用重算值覆盖优化器原始上报值。

## 26. 仓位分析 P1：基准和暴露

### 26.1 主动仓位

$$
a_t=w_t-b_t
$$

### 26.2 Active share

$$
AS_t=\frac12\sum_i|a_{t,i}|
$$

### 26.3 一般因子暴露

$$
e_t^p=B_t^\top w_t
$$

$$
e_t^b=B_t^\top b_t
$$

$$
e_t^a=B_t^\top(w_t-b_t)
$$

P1 同时输出：

- Barra industry dummy 暴露；
- Barra style 暴露；
- data_access 申万一级行业分类权重；
- 市值暴露。

### 26.4 市值分析

对有效 `MarketCap`：

$$
e_t^{mcap,p}
=\sum_iw_{t,i}\log(m_{t,i})
$$

$$
e_t^{mcap,b}
=\sum_ib_{t,i}\log(m_{t,i})
$$

$$
e_t^{mcap,a}
=e_t^{mcap,p}-e_t^{mcap,b}
$$

并同时报告资产覆盖率和绝对权重覆盖率。

## 27. 仓位分析 P1：Barra 风险分解

风险基准：

```text
absolute: v = w
active:   v = w - b
```

### 27.1 总方差

$$
V(v)=v^\top\Sigma v
$$

### 27.2 系统性与特异方差

$$
V_{factor}(v)
=(B^\top v)^\top F(B^\top v)
$$

$$
V_{specific}(v)
=v^\top Dv
=\sum_id_iv_i^2
$$

$$
V(v)=V_{factor}(v)+V_{specific}(v)
$$

### 27.3 资产风险贡献

$$
MRC_i=(\Sigma v)_i
$$

$$
RC_i=v_i(\Sigma v)_i
$$

$$
\sum_iRC_i=V(v)
$$

long-short 或主动风险下，单个资产贡献可以为负，分析器不会强制取绝对值。

### 27.4 因子风险贡献

令：

$$
z=B^\top v
$$

则：

$$
FRC_k=z_k(Fz)_k
$$

$$
\sum_kFRC_k=V_{factor}(v)
$$

### 27.5 特异风险贡献

$$
SRC_i=d_iv_i^2
$$

$$
\sum_iSRC_i=V_{specific}(v)
$$

分析器会验证资产贡献、因子贡献和特异贡献的加总关系。

## 28. 仓位分析 P1：流动性

只有提供正数 `portfolio_notional` 才计算金额指标。

交易金额：

$$
Q_{t,i}
=|\Delta w_{t,i}|\times NAV
$$

决策日前 $L$ 个可用交易日平均成交额：

$$
ADV_{t,i}
=\operatorname{mean}
\left(
\operatorname{Amount}_{s,i}
\right),
\quad s<t
$$

参与率：

$$
P_{t,i}=\frac{Q_{t,i}}{ADV_{t,i}}
$$

在最大允许参与率 $p_{\max}$ 下的预计交易天数：

$$
D_{t,i}
=\frac{Q_{t,i}}{ADV_{t,i}p_{\max}}
$$

当前默认：

```text
adv_window_days = 20
maximum_adv_participation = 0.10
```

ADV 不使用决策日当日或未来数据。

## 29. 仓位分析约束审计

分析器区分：

1. 优化器上报 violation/slack；
2. 分析器独立重算值；
3. 无法恢复输入时的 unavailable。

上界约束：

$$
\operatorname{slack}=u-x
$$

下界约束：

$$
\operatorname{slack}=x-l
$$

违反量：

$$
\operatorname{violation}=\max(0,-\operatorname{slack})
$$

上界利用率：

$$
\operatorname{utilization}=\frac{x}{u}
$$

当前可复核：

- budget；
- long-only；
- single-name；
- gross exposure；
- turnover；
- active weight；
- tradable；
- industry；
- style；
- tracking error。

## 30. 仓位分析输出

```text
analysis/
├── position_summary.parquet
├── holdings_detail.parquet
├── turnover_detail.parquet
├── exposure_summary.parquet
├── risk_contribution.parquet
├── constraint_summary.parquet
├── quality_checks.parquet
├── resolved_analysis_config.yaml
├── analysis_manifest.yaml
└── position_report.html
```

状态：

| 状态 | 含义 |
|---|---|
| `passed` | 核心检查和启用的 enrichment 通过 |
| `partial` | 结构化结果可用，但存在 snapshot、数据缺失或重算差异 |
| `failed` | 核心契约、required enrichment 或风险加总失败 |

CLI 退出码：

| 情况 | 退出码 |
|---|---:|
| passed | 0 |
| partial 且允许 | 2 |
| partial 且不允许 | 1 |
| failed | 1 |

## 31. 数据时点和 provenance

### 31.1 不允许未来数据

风险、暴露、历史收益和 ADV 都只能使用决策时可得数据。

### 31.2 data_access snapshot

优化和分析记录：

```text
snapshot_id
registry_hash
schema_hash
file_manifest_hash
```

分析时重新读取后比较 snapshot：

- `strict`：不一致时终止对应 enrichment；
- `warn`：允许继续，但结果为 partial；
- 不会将新 snapshot 冒充旧 snapshot。

### 31.3 当前回放限制

`data_access` 当前能生成 snapshot identity，但不能按历史 `snapshot_id` 直接回读。
如果旧文件已经被替换，只能检测漂移，不能自动恢复旧版本。

### 31.4 Barra hash

新分析 manifest 会记录当前 Barra risk package 的 SHA-256。

若旧优化 manifest 只保存路径，没有保存原始 hash：

- 可以重算；
- 可以发现重算值与 summary 不一致；
- 不能证明差异来自 Barra 文件变化还是历史代码变化。

## 32. 当前真实能力边界

### 32.1 已经可靠可用

1. Precomputed Barra long-only 指数增强；
2. 历史协方差 long-only 指数增强；
3. 历史协方差 long-only 绝对收益；
4. TopN long-only 等权；
5. 预算、个股、主动权重、换手、tradable、行业、风格和 TE 硬约束；
6. 线性成本进入目标函数；
7. 配置驱动 optimize CLI；
8. P0 全优化器产物审计；
9. P1 benchmark、行业、市值、Barra 风险和流动性分析；
10. Parquet、HTML 和 provenance/hash 输出。

### 32.2 Python-only

1. 原始 Barra `F + F_ret + F_spec`；
2. impact cost；
3. 自定义 `InputBundle`；
4. 自定义 mapping 和 parameter 文件；
5. draft long-short TopN。

### 32.3 尚不具备

1. 正式凸优化 long-short；
2. 实际成交和实际持仓；
3. 收益、净值、回撤和绩效归因；
4. 多期联合优化；
5. 非凸持仓数约束；
6. Max Sharpe、CVaR、Risk Parity 等目标；
7. 历史协方差运行的事后完整风险贡献回放；
8. data_access 历史 snapshot pinning；
9. production CLI 外部 benchmark/tradable/cost 文件兼容；
10. production CLI impact cost。

## 33. 已知逻辑和数据风险

### 33.1 行业缺失粗暴归类

预计算 Barra adapter 将异常行业行归入第一个行业。这可以保证 one-hot 和求解，
但可能掩盖上游行业数据问题。

### 33.2 特异波动率质量闸门无效

`minimum_annual_specific_volatility` 当前先 floor 后校验，不能拒绝原始低值，也不改变
进入求解的 `specific_var`。

### 33.3 旧 Barra 运行不可完全复现

旧 optimize manifest 未记录 Barra 内容 hash。路径相同不等于内容相同。

### 33.4 历史风险未持久化

历史协方差矩阵没有随优化结果落盘，因此 P1 不能精确重建历史优化运行的资产风险贡献。

### 33.5 TopN long-short 未受完整约束治理

draft long-short 规则后端不会自动校验预算、gross/net exposure、tradable 和换手。

### 33.6 名称与实际路由差异

CLI legacy `*_enhance_index` 名称实际转发到 precomputed Barra，调用者不能仅凭名字
判断风险输入。

## 34. 代码与配置来源

本文对应以下实现：

- `src/riskfolio_qs/configs/optimizer/mapping.v0.2.0.yaml`
- `src/riskfolio_qs/configs/optimizer/parameters.v0.2.0.yaml`
- `src/riskfolio_qs/optimizers/portfolio_optimizer.py`
- `src/riskfolio_qs/adapters/barra_precomputed_adapter.py`
- `src/riskfolio_qs/adapters/data_access_inputs.py`
- `src/riskfolio_qs/smoothers/signal_smoother.py`
- `src/riskfolio_qs/adapters/output_adapter.py`
- `src/riskfolio_qs/analysis/`
- `src/riskfolio_qs/cli.py`

相关使用文档：

- [优化 CLI 使用指南](CLI使用指南.md)
- [仓位分析 CLI 使用指南](仓位分析CLI使用指南.md)
- [仓位分析模块设计](design/riskfolio_qs_v0.2_仓位分析模块设计.md)

## 35. 结论

当前 `riskfolio_qs` 的核心是：

$$
\boxed{
\text{point-in-time inputs}
\rightarrow
\text{long-only convex target weights}
\rightarrow
\text{independent position/risk audit}
}
$$

其中生产主路径是：

$$
\boxed{
\text{Alpha}
+\text{benchmark}
+\text{tradable}
+\text{Barra B/F/D or historical returns}
\rightarrow
\text{target portfolio}
}
$$

仓位分析主路径是：

$$
\boxed{
\text{optimization artifacts}
+\text{manifest-driven enrichment}
\rightarrow
\text{position/exposure/risk/constraint report}
}
$$

任何超出本文“已经可靠可用”范围的调用，都不应被描述为当前生产能力。
