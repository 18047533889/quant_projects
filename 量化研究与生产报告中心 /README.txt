量化研究与生产报告中心

目录
====
index.html
  报告首页

标准中心
--------
methodology.html
  评价框架：时间、Return、Profile、方向、Utility、Hard Gate、Streaming、权限

scorecard.html
  评分标准：明确 Predictive / Stratification / Portfolio / Stability / Robustness /
  Novelty / Tradability / StatisticalReliability 的具体 Metric 构成、权重、Grade Floor

metric-dictionary.html
  181 项算法 / 因子 / 模型 / 组合优化指标，包含公式、方向、参考标准、适用 Profile

tag-dictionary.html
  42 个自动标签，包含触发规则与 Severity

chart-dictionary.html
  40 张标准图及其必备范围和诊断目的

机器可读标准
------------
standards-policy.json
metric-registry.json
tag-registry.json
chart-registry.json

18 份核心报告
-------------
FAR-RES.html
FAR-TST.html
FAR-DEP.html
FAR-EMG.html

FOP-DLY-NEXP.html
FOP-WKL-NEXP.html
FOP-WKL-EXP.html

SML-BMK.html
SML-TST.html
SML-DEP.html
SML-EMG.html

FML-DLY-NSTR.html
FML-WKL-NSTR.html
FML-WKL-STR.html

PFO-BMK.html
PFO-TST.html
PFO-DEP.html
PFO-EMG.html

独立详情目录
------------
factor_details/nexp/
  单因子完整评估，不带表达式

factor_details/exp/
  单因子完整评估，带表达式与内部 Lineage

model_details/
  单模型完整评估

optimizer_details/
  单组合优化器完整评估

核心规则
========
1. Grade = 长期资产质量；Health = 当前状态。
2. Search Fitness != 正式 Grade。
3. Factor Score：
   25% Predictive
 + 15% Stratification
 + 15% Portfolio
 + 15% Stability
 + 10% Robustness
 + 10% Novelty
 +  5% Tradability
 +  5% StatisticalReliability

4. 每个维度内部的 Metric 与权重见 scorecard.html 和 standards-policy.json。
5. Grade 从 S+ 向下判定，必须同时满足 Total Score、Dimension Floors、Hard Gates、Evidence Caps。
6. Continuous / Binary / Ternary / Event / Rule Intensity / Regime / Time-Series 采用不同 Profile。
7. Direction 只允许 Train / Calibration 冻结。
8. 报告只渲染 EvaluationBundle，不重新计算指标。
9. NEXP / EXP 使用同一底层 Snapshot，只改变字段可见性。
10. 单因子必备图包括：
    - Daily / Rolling IC
    - Yearly IC / ICIR
    - Quantile Return / NAV（适用时）
    - Long / Short / Long-Short NAV
    - Drawdown / Underwater
    - Horizon Decay
    - Execution Delay Sensitivity
    - Correlation / Novelty
    - Style Exposure
    - Cost / Capacity
    - Streaming Health

指标完整性
==========
本目录不宣称“数学上穷尽所有未来可能发明的指标”，但当前 Registry 已覆盖：
- 自动挖掘算法效率与搜索健康
- 因子数据/语义、预测、排序、非线性、分层、Tail、组合、收益集中、稳定、衰减
- 交易成本、容量、A股可交易性、风格/行业/微盘/Universe/Regime 鲁棒性
- 参数/公式/Return Definition 鲁棒性、Novelty/Residual 信息
- Newey-West / Bootstrap / FDR / DSR / PBO / Leakage
- Binary / Event / Rule Intensity / Regime 专属评价
- 模型误差、排序、泛化、Seed、Ablation、Feature、Drift、Runtime、神经网络专项
- 组合优化器 Objective、Risk、Constraint、Cost、Capacity、Solver、Sensitivity、Live Parity

后续新增指标必须先登记到 metric-registry.json，再由报告引用。
