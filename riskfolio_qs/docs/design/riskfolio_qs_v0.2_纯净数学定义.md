# riskfolio_qs 纯净数学定义 v0.2（草案）

> 本文为历史设计稿。当前已实现功能、精确公式与能力边界统一以
> [riskfolio_qs v0.3 功能与数学规范](../riskfolio_qs_v0.3_功能与数学规范.md)
> 为准。

## 1. 记号

给定资产集合大小 $N$。单期优化时点 $t$：

- $w \in \mathbb{R}^N$：目标权重
- $w^{-} \in \mathbb{R}^N$：上期权重
- $w^{b} \in \mathbb{R}^N$：基准权重
- $\mu \in \mathbb{R}^N$：预期收益（或 alpha 映射收益）
- $\Sigma \in \mathbb{S}_{+}^{N}$：协方差矩阵
- $z := w - w^{-}$：调仓向量
- $a := w - w^{b}$：主动权重
- $F \in \mathbb{R}^{N \times K}$：风格/因子暴露矩阵
- $G \in \mathbb{R}^{N \times J}$：行业 one-hot 暴露矩阵
- $F_{\mathrm{mcap}} \in \mathbb{R}^{N}$：市值暴露向量（可取对数市值后标准化）
- $F_{\mathrm{ret}} \in \mathbb{R}^{K}$：Barra 因子收益率向量（外采数据）
- $F_{\mathrm{spec}} \in \mathbb{R}^{N}$：个股特异性收益率向量（外采数据）

定义：

- 换手：$\mathrm{TO}(w) := \frac{1}{2}\|w - w^{-}\|_{1}$
- 线性交易成本：$C_{\text{lin}}(w) := c^\top |w-w^{-}|,\; c \ge 0$
- 二次冲击成本：$C_{\text{quad}}(w) := (w-w^{-})^\top \Lambda (w-w^{-}),\; \Lambda \succeq 0$
- 主动风险：$\mathrm{TE}^2(w) := (w-w^b)^\top \Sigma (w-w^b)$

---

## 2. 专题：什么是组合暴露

本专题统一定义组合暴露口径，并明确 v0.2 的默认使用方式。

### 2.1 资产层暴露输入

给定风格/因子暴露矩阵 $F \in \mathbb{R}^{N \times K}$，其中第 $i$ 行第 $k$ 列表示资产 $i$ 在风格 $k$ 上的暴露。

### 2.2 组合绝对暴露

组合在各风格维度上的绝对暴露定义为：

$$
e = F^\top w
$$

其中 $e \in \mathbb{R}^{K}$。

### 2.3 基准暴露与主动暴露

基准暴露定义为：

$$
e^b = F^\top w^b
$$

主动暴露定义为：

$$
\Delta e = e - e^b = F^\top (w-w^b)
$$

### 2.4 v0.2 默认口径

在指增与相对收益场景中，v0.2 默认以主动暴露 $\Delta e$ 作为约束与风控主口径。

常见约束写法如下：

- 严格中性：$\Delta e = 0$
- 带宽中性：$-\epsilon \le \Delta e \le \epsilon$

### 2.5 行业与市值暴露的同构定义

若将 $F$ 替换为行业 one-hot 矩阵 $G$，可得行业暴露与行业主动暴露：

$$
e_{\text{ind}} = G^\top w, \quad \Delta e_{\text{ind}} = G^\top (w-w^b)
$$

若将 $F$ 替换为市值暴露向量 $F_{\mathrm{mcap}}$（视为单因子矩阵），可得市值主动暴露：

$$
\Delta e_{\text{mcap}} = F_{\mathrm{mcap}}^\top (w-w^b)
$$

### 2.6 Barra 因子暴露简介

在 v0.2 中，Barra 口径按“外采数据”处理，不在 riskfolio_qs 内部重算 Barra 因子。标准输入包含三类矩阵/向量：

1. 个股 Barra 因子暴露矩阵：$F \in \mathbb{R}^{N \times K}$。
2. Barra 因子收益率向量：$F_{\mathrm{ret}} \in \mathbb{R}^{K}$。
3. 个股特异性收益率向量：$F_{\mathrm{spec}} \in \mathbb{R}^{N}$。

收益分解口径写为：

$$
r = F F_{\mathrm{ret}} + F_{\mathrm{spec}}
$$

其中 $r \in \mathbb{R}^{N}$ 为个股收益向量。

因此，在 v0.2 的相对收益口径下，Barra 主动暴露统一定义为：

$$
\Delta e_{\text{barra}} = F^\top (w-w^b)
$$

### 2.7 常见中性控制口径（实务）

实务中通常不采用“全部风格因子严格中性”，而采用分层控制：

1. 严格中性子集 $S$：

$$
F_S^\top (w-w^b) = 0
$$

2. 带宽中性子集 $R$：

$$
-\epsilon_R \le F_R^\top (w-w^b) \le \epsilon_R
$$

常见默认做法是：

1. 行业因子严格中性或窄带宽中性。
2. 市场 beta 与 size 因子较严格中性。
3. 其余 Barra 风格因子采用带宽中性。

该分层方式在可行性、风险控制与 alpha 保留之间更易取得平衡。

---

## 3. 专题：风险的定义方式

本专题将风险项分为“可明确定义”和“暂不能明确定义”两类。

### 3.1 可明确定义的风险项

以下风险项在 v0.2 中可给出明确数学定义，并可直接进入优化问题。

1. 总体方差风险（Mean-Variance / Min-Variance）：

$$
\phi_{\mathrm{MV}}(w) = w^\top \Sigma w
$$

2. 主动风险（Tracking Error 的平方）：

$$
\phi_{\mathrm{TE}}(w,w^b) = (w-w^b)^\top \Sigma (w-w^b)
$$

3. 因子模型风险（Barra 外采口径）：

$$
\Sigma = F\Sigma_f F^\top + D,\quad D \succeq 0
$$

$$
\phi_{\mathrm{FM}}(w) = w^\top (F\Sigma_f F^\top + D) w
$$

对应主动风险形式为：

$$
\phi_{\mathrm{FM,act}}(w,w^b) = (w-w^b)^\top (F\Sigma_f F^\top + D) (w-w^b)
$$

其中：

- $F$ 为个股 Barra 因子暴露矩阵（外采）。
- $\Sigma_f = \mathrm{Cov}(F_{\mathrm{ret}})$ 为因子收益率协方差。
- $D = \mathrm{Diag}(\mathrm{Var}(F_{\mathrm{spec},i}))$ 为特异风险对角阵。

### 3.2 可明确定义但暂不作为 v0.2 必选项

以下风险项可明确写出数学形式，但默认不作为 v0.2 第一批必选：

1. 半方差风险（Downside Semivariance）。
2. 条件在险价值（CVaR）。
3. 回撤类风险（如 MDD/CDaR）作为直接优化目标。

这些风险项在研究上常见，但在生产化上对数据窗口、分布估计、计算稳定性依赖更强。

### 3.3 暂不能明确定义的风险项（严格注明）

以下内容在 v0.2 阶段暂不能给出统一、可复现的“单一定义”，必须依赖外部口径冻结后才能进入正式契约：

1. “黑盒风险评分”类输入：
- 严格说明：若上游仅提供风险分数而不公开可复现计算公式，则不能在本规范中定义为标准风险项。

2. 动态状态切换风险（Regime-Switching）的一体化参数：
- 严格说明：若状态识别模型、切换规则、概率平滑口径未冻结，则不能形成统一数学定义。

3. 事件冲击风险（公告、政策、突发事件）的一体化风险函数：
- 严格说明：若事件分类和冲击映射函数未统一，当前仅能作为外部附加约束信号，不能作为标准风险项。

4. 团队特定自定义风险因子但无固定构建流程：
- 严格说明：未固定数据源、清洗规则、标准化流程、版本管理前，不纳入本规范。

### 3.4 v0.2 默认风险项选型

v0.2 默认采用：

1. 以外采 Barra 数据构建的因子模型风险作为主风险项。
2. 以主动风险（TE）作为相对收益场景的核心风险约束或惩罚。
3. 保留总体方差风险作为兼容口径。

默认不强制采用 CVaR、回撤类风险作为第一批生产必选项。

---

## 4. 专题：交易成本约束的定义方式

本专题将交易成本约束分为“可明确定义”和“暂不能明确定义”两类，并给出 v0.2 默认处理方式。

### 4.1 可明确定义的交易成本项

以下成本项在 v0.2 中可给出明确数学定义，并可直接进入优化问题。

1. 线性交易成本：

$$
C_{\mathrm{lin}}(w) = c^\top |w-w^-|,\quad c \ge 0
$$

2. 二次冲击成本：

$$
C_{\mathrm{quad}}(w) = (w-w^-)^\top \Lambda (w-w^-),\quad \Lambda \succeq 0
$$

3. 换手代理成本：

$$
\mathrm{TO}(w) = \frac{1}{2}\|w-w^-\|_1
$$

其中：

- $w^-$ 为上期权重。
- $c$ 为线性成本系数向量。
- $\Lambda$ 为冲击成本系数矩阵（常取对角或分块对角）。

### 4.2 交易成本进入优化的两种方式

1. 软约束（惩罚项）方式：

$$
\min_w f_0(w)
+ \lambda_{\mathrm{tc}} C_{\mathrm{lin}}(w)
+ \lambda_{\mathrm{imp}} C_{\mathrm{quad}}(w)
$$

2. 硬约束（预算上限）方式：

$$
C_{\mathrm{lin}}(w) \le B_{\mathrm{tc}}
$$

以及常见换手上限：

$$
\mathrm{TO}(w) \le \tau_{\max}
$$

### 4.3 可明确定义但暂不作为 v0.2 必选项

以下项可定义，但默认不作为 v0.2 第一批必选：

1. 分买卖不对称冲击模型（买入冲击与卖出冲击不同）。
2. 分时段交易成本曲线（开盘/收盘/盘中差异）。
3. 非凸成本函数（如分段非凸冲击）。

这些项在实盘中常见，但会显著提高参数治理和求解复杂度。

### 4.4 暂不能明确定义的交易成本项（严格注明）

以下内容在 v0.2 阶段暂不能给出统一、可复现的“单一定义”，必须依赖外部口径冻结后才能进入正式契约：

1. 黑盒成交仿真输出的单值成本参数：
- 严格说明：若仅给出结果数值而无可复现模型与参数，则不能作为标准成本函数。

2. 券商专有执行引擎返回的隐式冲击参数：
- 严格说明：若缺少统一字段定义与版本管理，当前不能纳入统一数学契约。

3. 事件驱动流动性突变的一体化成本函数：
- 严格说明：若事件分类、冲击映射和时效规则未冻结，不纳入 v0.2 标准定义。

### 4.5 v0.2 默认成本约束选型

v0.2 默认采用：

1. 线性交易成本惩罚项（软约束）作为默认配置。
2. 换手上限约束（硬约束）作为默认可选开关。
3. 二次冲击成本作为可选增强项。

默认不强制要求成本硬预算上限与复杂非线性冲击模型。

---

## 5. 基础可行域（硬约束）

### 4.1 预算与敞口

- 全投资：$\mathbf{1}^\top w = 1$
- 净敞口区间：$n_{\min} \le \mathbf{1}^\top w \le n_{\max}$
- 总敞口上限：$\|w\|_1 \le g_{\max}$

### 4.2 个股权重上下限

- 逐资产约束：$l_i \le w_i \le u_i,\; i=1,\dots,N$

### 4.3 主动权重带宽（相对基准）

- $d_i^{-} \le w_i - w_i^b \le d_i^{+}$

### 4.4 换手硬约束

- $\frac{1}{2}\|w-w^{-}\|_1 \le \tau_{\max}$

### 4.5 中性约束

- 市场/风格中性：$F^\top (w-w^b) = 0$
- 行业中性：$G^\top (w-w^b) = 0$
- 市值中性：$F_{\mathrm{mcap}}^\top (w-w^b) = 0$

可放松为带宽形式：

- $-\epsilon_f \le F^\top (w-w^b) \le \epsilon_f$
- $-\epsilon_g \le G^\top (w-w^b) \le \epsilon_g$
- $-\epsilon_{\mathrm{mcap}} \le F_{\mathrm{mcap}}^\top (w-w^b) \le \epsilon_{\mathrm{mcap}}$

---

## 6. 目标函数标准定义

### 5.1 Min Variance

$$
\min_{w} \quad w^\top \Sigma w
$$

s.t. 基础可行域约束。

### 5.2 Mean-Variance

$$
\min_{w} \quad \frac{1}{2}w^\top\Sigma w - \lambda_{\mu}\mu^\top w
$$

s.t. 基础可行域约束。

### 5.3 Index Enhancement（主动组合）

$$
\min_{w} \quad
-\lambda_{\alpha}\mu^\top (w-w^b)
+\lambda_{\text{te}}(w-w^b)^\top\Sigma(w-w^b)
+\lambda_{\text{to}}\cdot \frac{1}{2}\|w-w^{-}\|_1
+\lambda_{\text{tc}} C_{\text{lin}}(w)
+\lambda_{\text{imp}} C_{\text{quad}}(w)
$$

s.t. 基础可行域约束。

### 5.4 Max Sharpe 的工程等价形式

v0.2 采用可凸求解近似：使用 mean-variance 形式替代分式目标，

$$
\max_w \frac{\mu^\top w-r_f}{\sqrt{w^\top \Sigma w}}
\;\;\Longrightarrow\;\;
\min_w \frac{1}{2}w^\top\Sigma w - \lambda_{\mu}\mu^\top w
$$

其中 $\lambda_{\mu}$ 由配置给定或通过网格选择。

---

## 7. 软约束统一写法

将软约束作为惩罚项进入目标：

$$
\min_w f_0(w)
+ \rho_{\text{to}}\cdot \frac{1}{2}\|w-w^{-}\|_1
+ \rho_{\text{tc}} C_{\text{lin}}(w)
+ \rho_{\text{imp}} C_{\text{quad}}(w)
+ \rho_f \|F^\top(w-w^b)\|_1
+ \rho_g \|G^\top(w-w^b)\|_1
$$

其中 $f_0(w)$ 为主目标（如 min_variance 或 mean_variance）。

---

## 8. 场景合约参数化（数学接口）

定义合约参数集：

$$
\Theta =
\{
\lambda_{\mu},\lambda_{\alpha},\lambda_{\text{te}},
\lambda_{\text{to}},\lambda_{\text{tc}},\lambda_{\text{imp}},
l,u,\tau_{\max},g_{\max},
\epsilon_f,\epsilon_g,\epsilon_{\mathrm{mcap}},d^{-},d^{+}
\}
$$

- 合约模式：$\Theta = \Theta_{\text{preset}}[\text{name}]$
- 目标函数模式：$\Theta$ 由配置直接给定
- 两种模式最终映射到同一优化问题实例 $\mathcal{P}(\Theta)$

---

## 9. 数据需求映射（最小集合）

- $\mu$：alpha 信号或收益预测
- $\Sigma$：行情收益估计或风险模型输出
- $w^b$：基准权重
- $w^{-}$：上期权重
- $F,G,F_{\mathrm{mcap}}$：风格、行业、市值暴露
- $F_{\mathrm{ret}},F_{\mathrm{spec}}$：Barra 因子收益率与个股特异性收益率（外采）
- $c,\Lambda$：交易成本参数（可由上游成交数据估计）

若某项数据缺失：

- 对应约束/惩罚项不可启用，或触发配置校验失败（由运行模式决定）。
