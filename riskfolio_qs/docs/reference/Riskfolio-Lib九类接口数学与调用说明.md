# Riskfolio-Lib 九类接口：数学表达与调用说明

## 1. 文档目标与适用范围

本文面向量化研究与工程落地，系统梳理 Riskfolio-Lib 中常用的 9 类接口（接口簇），给出：

- 数学优化表达式（目标函数与典型约束）
- 最小可运行调用路径
- 关键参数解释与实践建议

说明：Riskfolio-Lib 基于 CVXPY 建模。默认以连续权重优化为主；若启用基数等整数约束，会进入混合整数优化。

---

## 2. 全局统一记号

- 资产数：$N$，样本期数：$T$
- 权重向量：$w \in \mathbb{R}^N$
- 期望收益：$\mu \in \mathbb{R}^N$
- 协方差矩阵：$\Sigma \in \mathbb{S}_{+}^{N}$
- 风险度量：$\phi(w)$（可为 MV/CVaR/CDaR/EVaR 等）
- 无风险利率：$r_f$
- 风险厌恶系数：$\lambda > 0$
- 线性约束矩阵：$Aw \le b$

常见目标（Riskfolio 的 obj）：

- MinRisk：$\min_w \phi(w)$
- MaxRet：$\max_w \mu^\top w$
- Utility：$\max_w \mu^\top w - \lambda\phi(w)$
- Sharpe（广义风险调整收益）：$\max_w \frac{\mu^\top w-r_f}{\phi(w)}$（实现上常做等价变换）

---

## 3. 接口 1：Classic 均值-风险优化

### 3.1 数学形式

$$
\begin{aligned}
\underset{w}{\text{opt/min}}\;&F(w)\\
\text{s.t.}\;&Aw\le b,\\
&\mathbf{1}^\top w = 1\;\text{(或预算约束)},\\
&l\le w\le u,\\
&\phi_i(w)\le c_i\;\text{(可选)}.
\end{aligned}
$$

其中 $F(w)$ 由 obj 决定，风险函数由 rm 决定。

### 3.2 调用路径

1) 初始化组合对象  
2) 估计参数（均值/协方差）  
3) 调用 optimization(model="Classic", ...)

### 3.3 最小示例

```python
import riskfolio as rp

port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="ledoit")

w = port.optimization(
    model="Classic",
    rm="MV",          # 如 MV, CVaR, CDaR, EVaR ...
    obj="Sharpe",     # MinRisk, MaxRet, Utility, Sharpe
    rf=0,
    l=2,
    hist=True,
)
```

---

## 4. 接口 2：BL（Black-Litterman）

### 4.1 数学形式

BL 的核心是后验参数更新：

$$
\mu_{BL}=\left[(\tau\Sigma)^{-1}+P^T\Omega^{-1}P\right]^{-1}
\left[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q\right]
$$

$$
\Sigma_{BL}=\Sigma + \left[(\tau\Sigma)^{-1}+P^T\Omega^{-1}P\right]^{-1}
$$

随后将 $\mu_{BL},\Sigma_{BL}$ 代入与 Classic 同型的优化问题。

### 4.2 调用路径

1) 先生成历史统计（常见做法）  
2) blacklitterman_stats(P, Q, ...)  
3) optimization(model="BL", ...)

### 4.3 最小示例

```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="hist")

port.blacklitterman_stats(
    P=P, Q=Q,
    rf=0,
    eq=True,
    method_mu="hist",
    method_cov="hist",
)

w_bl = port.optimization(model="BL", rm="MV", obj="Sharpe", rf=0)
```

---

## 5. 接口 3：FM（风险因子模型）

### 5.1 数学形式

常见线性因子结构：

$$
r_t = B f_t + \varepsilon_t
$$

$$
\mu = B\mu_f + \mu_\varepsilon,\qquad
\Sigma \approx B\Sigma_f B^T + \Sigma_\varepsilon
$$

将因子驱动得到的 $\mu,\Sigma$ 输入优化目标。

### 5.2 调用路径

1) 提供 factors（以及可选 B）  
2) factors_stats(...)  
3) optimization(model="FM", ...)

### 5.3 最小示例

```python
port = rp.Portfolio(returns=returns, factors=factors)

port.factors_stats(
    method_mu="hist",
    method_cov="hist",
    B=None,            # None 时可由库估计
    const=True,
)

w_fm = port.optimization(model="FM", rm="MV", obj="MinRisk", hist=False)
```

---

## 6. 接口 4：BLFM（BL + 因子模型）

### 6.1 数学形式

BLFM 在因子结构下融合资产观点与因子观点，形成后验 $\mu,\Sigma$：

$$
(\mu,\Sigma) \leftarrow \text{BLFM}(B, P,Q, P_f,Q_f, \text{先验})
$$

然后进入标准组合优化问题。

### 6.2 调用路径

1) 设置 returns + factors  
2) blfactors_stats(...)  
3) optimization(model="BLFM", ...)

### 6.3 最小示例

```python
port = rp.Portfolio(returns=returns, factors=factors)

port.blfactors_stats(
    flavor="BLB",     # 或 "ABL"
    B=B,
    P=P, Q=Q,
    P_f=P_f, Q_f=Q_f,
    eq=True,
)

w_blfm = port.optimization(model="BLFM", rm="MV", obj="Sharpe", hist=False)
```

---

## 7. 接口 5：EP（Entropy Pooling）

### 7.1 数学形式

通过最小化相对熵，重定价情景概率：

$$
\min_{p}\sum_{t=1}^{T} p_t\log\frac{p_t}{q_t}
$$

$$
\text{s.t. } P_{eq}p=Q_{eq},\; P_{in}p\le Q_{in},\; p\ge0,\;\mathbf{1}^Tp=1
$$

得到后验分布后计算 $\mu_{EP},\Sigma_{EP}$ 并用于优化。

### 7.2 调用路径

1) entropy_pooling_stats(...)  
2) optimization(model="EP", ...)

### 7.3 最小示例

```python
port = rp.Portfolio(returns=returns)

port.entropy_pooling_stats(
    P_eq=P_eq, Q_eq=Q_eq,
    P_in=P_in, Q_in=Q_in,
    higher_comoments=False,
    solver="CLARABEL",
)

w_ep = port.optimization(model="EP", rm="MV", obj="Utility", l=2)
```

---

## 8. 接口 6：风险平价/风险预算（RP）

### 8.1 数学形式

Riskfolio 的风险预算类接口可概括为：

$$
\min_w \phi(w)
$$

$$
\text{s.t. } b^T\log(w)\ge c,\; Aw\le b,\; \mu^Tw\ge \bar\mu,\; w\ge0
$$

并可设置目标风险贡献向量。

### 8.2 调用路径

1) assets_stats 或 factors_stats  
2) rp_optimization(model=..., rm=..., b=...)

### 8.3 最小示例

```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="hist")

w_rp = port.rp_optimization(
    model="Classic",
    rm="MV",
    rf=0,
    b=None,      # None 表示等风险预算
    hist=True,
)
```

---

## 9. 接口 7：层次聚类优化（HRP/HERC/HERC2/NCO）

### 9.1 数学形式

这类方法的结构是“聚类 + 组内分配 + 组间分配”：

1) 基于相关性构造距离：
$$
D_{ij}=\sqrt{\frac{1-\rho_{ij}}{2}}\quad(\text{例如 pearson})
$$

2) 层次聚类得到树结构  
3) 在簇内和簇间做递归权重分配（NCO 会在子问题中调用优化器）

### 9.2 调用路径

1) 初始化 HCPortfolio  
2) optimization(model="HRP"/"HERC"/"HERC2"/"NCO", ...)

### 9.3 最小示例

```python
hc = rp.HCPortfolio(returns=returns)

w_hrp = hc.optimization(
    model="HRP",
    codependence="pearson",
    rm="MV",
    linkage="single",
    max_k=10,
    leaf_order=True,
)
```

---

## 10. 接口 8：有效前沿（Frontier）

### 10.1 数学形式

通过在不同收益或风险水平下重复求解，得到前沿离散点：

$$
\{w^{(1)},\dots,w^{(K)}\}
$$

对应风险-收益点集：

$$
\left(\phi(w^{(k)}),\mu^Tw^{(k)}\right),\;k=1,\dots,K
$$

### 10.2 调用路径

1) 先完成参数估计与约束设置  
2) frontier_limits(...)（可选但推荐）  
3) efficient_frontier(...)

### 10.3 最小示例

```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="ledoit")

limits = port.frontier_limits(model="Classic", rm="MV", rf=0)

w_frontier = port.efficient_frontier(
    model="Classic",
    rm="MV",
    points=30,
    rf=0,
    hist=True,
)
```

---

## 11. 接口 9：扩展凸优化接口（RRP / OWA / WC）

这一类可视作进阶接口簇，覆盖三种常见扩展。

### 11.1 RRP（Relaxed Risk Parity）

目标示意（库内文档给出锥约束形式）：

$$
\min_w\;\psi-\gamma
$$

配合 SOC 约束、预算约束与线性约束，实现“放松的风险平价”。

调用：

```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="hist")

w_rrp = port.rrp_optimization(
    model="Classic",
    version="A",   # A/B/C
    l=1,
    b=None,
    hist=True,
)
```

### 11.2 OWA（Ordered Weighted Averaging）

将情景收益排序后做加权聚合风险度量：

$$
\phi_{OWA}(w)=\sum_{i=1}^{T}\omega_i\,x_{(i)}(w)
$$

调用：

```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="hist")

w_owa = port.owa_optimization(
    obj="Sharpe",
    owa_w=owa_w,     # 用户提供 OWA 权重
    kelly=None,
    rf=0,
    l=2,
)
```

### 11.3 WC（Worst-Case Mean-Variance）

先估计不确定集，再做最坏情形优化：

$$
\max_w\min_{\mu\in\mathcal U_\mu,\Sigma\in\mathcal U_\Sigma}
\left(\mu^Tw-\lambda\,\text{Risk}_\Sigma(w)\right)
$$

调用：

```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="hist")

port.wc_stats(
    box="s",
    ellip="s",
    q=0.05,
    n_sim=3000,
)

w_wc = port.wc_optimization(
    obj="Sharpe",
    rf=0,
    l=2,
    Umu="box",      # box / ellip / None
    Ucov="box",     # box / ellip / None
)
```

---

## 12. 统一约束注入方式（实务重点）

Riskfolio 的“硬约束”主要通过以下方式注入：

- 直接属性约束：上下限、预算、是否做空、TE、TO、lowerret 等
- 矩阵线性约束：$Aw\le b$
- 风险贡献/因子风险贡献约束（对应特定模型）
- 可选整数约束（启用后变为 MIP，不再是纯连续凸问题）

典型流程：

1) 构造约束表或矩阵  
2) 写入 Portfolio 对象对应属性  
3) 调用 optimization 或其扩展接口

---

## 13. 选择建议（快速决策）

- 追求稳健与解释性：Classic + MV/CVaR + 行业约束
- 有明确主观观点：BL 或 EP
- 强因子框架：FM 或 BLFM
- 强调分散与结构稳健：HRP/HERC/NCO
- 强调风险预算：RP 或 RRP
- 强调分布稳健：WC

---

## 14. 最小端到端模板（可复制）

```python
import riskfolio as rp

# 1) 初始化
port = rp.Portfolio(returns=returns)

# 2) 参数估计
port.assets_stats(method_mu="hist", method_cov="ledoit")

# 3) 可选：设置约束
# port.upperlng = 0.10
# port.lowerlng = 0.00
# port.lowerret = 0.0003
# port.allowTO = True
# port.turnover = 0.05

# 4) 优化
w = port.optimization(model="Classic", rm="CVaR", obj="Sharpe", rf=0)

# 5) 前沿
w_frontier = port.efficient_frontier(model="Classic", rm="CVaR", points=20)
```

---

## 15. 备注

- 不同 rm 对应不同锥规划类型（LP/SOCP/SDP/EXP/POW），建议根据问题规模与数值难度选择求解器。  
- 若需“完全自定义的新目标函数”，建议直接在 CVXPY 层单独建模；Riskfolio 更适合在其内置范式下快速落地。
