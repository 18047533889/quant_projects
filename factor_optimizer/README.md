# factor_optimizer — 证据引导的因子寻优（sealed-test 纪律）

变异语法 + 搜索编排（train/validation/test sealed 切分）+ desirability/Pareto 验收。
本库只通过 adapter protocol 协调 **factor_engine**（计算/校验）与 **quant_evaluator**
（证据），自己**不执行因子、不算指标、不做最终入库决策**。

**定位:** 企业级 A 股横截面日频多因子量化项目的**因子寻优层** —— 拿到候选因子后，
在本库做变异搜索、分层评估、赢家选择，把入选 treatment 交给 factor_assets 入库。
组合参数寻优（Optuna/LGBM）是组合链专属，不在本库。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer (私有)

---

## 它是什么 / 不是什么

**做什么：** 候选变异生成（6 种变异算子 × 语法校验）、SearchRunner 编排、
sealed-test 一次性封存评估、desirability/Pareto/不确定性感知三种赢家选择、
TreatmentDecisionPolicy 八步处理决策、TrialLedger 失败分类、预算管控。

**不做什么：** 不算因子（FE adapter 转）、不算指标（QE adapter 转）、
不入库（factor_assets 的 PromotionGate 决定）。

## 功能全清单

### 搜索编排（search/runner.py）
- `SearchRunner` / `SearchSession` / `SearchConfig`：
  - `require_evaluation_protocol=True` —— **fail-closed**：没有评估协议直接拒绝开跑
  - `max_concurrency=1` —— 默认串行，保证证据可复现
  - `SearchBudget` —— 预算管控（默认 100 trials / 50 保存 / 1000 上限）
- `tiered_evaluation` —— 分层评估：便宜指标先筛，贵指标只留给幸存者
- `multifidelity` —— 保真度档位（低频窗粗筛 → 全窗确认）
- `plateau` —— 平台期检测（连续无改进提前收束）

### Sealed-test 纪律（防泄露核心）
- `freeze_for_sealed_test` / `consume_sealed_test` —— test 集**一次性封存消费**，
  用过即作废，杜绝"反复偷看 test 调参"
- `TestAuthorityBroker` —— test 集访问权集中管控
- `SealedTestExecutor` —— 强制 train/validation/test 隔离（purge overlap + embargo）
- `TreatmentIntegrityEvidence` —— treatment 前后证据 digest 绑定，
  evidence schema 版本不一致直接判定失效（防"结果对不上当时的证据"）

### 赢家选择（三套，可组合）
1. **`select_winner`（RobustBalancedUtility）**：
   `U = α·min(跨窗) + β·geomean + γ·robustness − λ·complexity`
   —— 稳健均衡效用，惩罚过拟合复杂度
2. **`UncertaintyAwareWinnerSelector`**：bootstrap 置信区间**下界**排序 +
   等价区域（equivalence region）判定 —— 差异在噪声内的两个候选不乱选
3. **desirability + Pareto**：`Desirability`（含 catastrophic floor，灾难值直接出局）+
   `ParetoPoint/Frontier/Archive` 非支配前沿

### 变异语法（grammar/）
- **6 种变异**：`parameter_tune` / `window_adjust` / `operator_swap` /
  `decay_adjust` / `linear_combination` / `threshold_adjust`
- **8 种 ParameterKind × 9 种 ParameterRole**（ParamRole 守卫：inactive 参数禁止变异）
- `mutation_spec` / `registry` / `validation` —— 变异合法性校验，
  非法变异在生成层就被拒（不浪费评估预算）

### 处理决策（policy/treatment_decision.py）
`TreatmentDecisionPolicy` **八步决策**：证据完整性 → 诊断 → 修复建议
（LLM repair mapper）→ 预期收益 → 风险 → 决策（ACCEPT/RETRY/REJECT）→ 理由 → 血缘。
`policy/` 另含准入标准/判定、诊断、治理。

### 台账与契约（contracts/）
- `Trial` / `TrialLedger` —— 每次试验全记录，**失败分类**（评估失败/语法非法/预算耗尽…）
- `SearchBudget` / `TreatmentIntegrity` / `CandidateMutation` / `Objective` /
  `Splits` / `Validator`

### 其他
- `llm/` —— LLM 辅助变异的 prompts + proposal（**默认 mock-model**，报告体系禁跑模型）
- `seen/` —— 候选身份去重（与 factor_assets SeenIndex 联动）
- `complexity/` —— 复杂度 profile/预算（进赢家效用的 λ 项）
- `data_capabilities.py` / `data_providers.py` —— train/validation/test 数据作用域声明
- `lineage` —— 搜索过程血缘（哪次试验产生了哪个 treatment）

## 安装

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer.git
cd factor_optimizer
pip install -e .                     # core
pip install -e ".[factor_engine,quant_evaluator]"   # 可选适配器
```

```python
from factor_optimizer.search.runner import SearchRunner, SearchConfig, SearchSession
from factor_optimizer.search.winner_selector import select_winner
from factor_optimizer.search.uncertainty_winner import UncertaintyAwareWinnerSelector

session = SearchSession(SearchConfig(search_budget=...))
runner = SearchRunner(session, fe_adapter=..., qe_adapter=...)
# runner.run(...) -> trial ledger / selection / acceptance
```

## 核心流程

```
候选因子 → 变异语法生成合法变异 → SearchRunner 逐 trial 评估
  （tiered/multifidelity 分层省钱）→ TrialLedger 记录（含失败分类）
→ sealed test 一次性封存消费 → 三套赢家选择（效用/不确定性/Pareto）
→ TreatmentDecisionPolicy 八步决策 → 入选 treatment → factor_assets PromotionGate
```

## 目录

```
grammar/       mutation_spec / registry / validation（6 变异 × 8 Kind × 9 Role）
search/        runner / session / desirability / pareto / uncertainty_winner /
               tiered_evaluation / multifidelity / plateau / treatment_decision / lineage
policy/        准入标准/判定、诊断 + LLM repair mapper、治理
contracts/     Trial / TrialLedger / SearchBudget / TreatmentIntegrity / CandidateMutation
adapters/      factor_engine（canonical hash、validate_mutation、complexity、算子元数据）
               quant_evaluator（证据）
llm/           prompts + proposal（默认 mock）
seen/          候选身份
complexity/    复杂度 profile / 预算
```

## 硬性规则

- `require_evaluation_protocol=True` fail-closed —— 没有协议不跑
- sealed test 一次性消费 —— 绝不重复偷看
- LLM 默认 mock-model —— 因子报告/落值/评估链不跑真实模型（组合链专属）
- 所有评估证据必须 vwap-to-vwap 口径
- 本库不执行因子不算指标 —— 一切经 FE/QE adapter

## 依赖与接口（谁 import 谁）

- **依赖（适配器）**：`factor_engine`（canonical hash / validate_mutation / 复杂度）、
  `quant_evaluator`（评估证据）。
- **被谁调用**：`quant_platform` 候选流水线（candidate → treatment → search）、
  `alphaprobe`（挖掘候选进入寻优）。
- **输出给**：`factor_assets`（入选 treatment 提交 PromotionGate）。

## 相关仓库

- **factor_engine / quant_evaluator** — 计算与证据（adapter protocol）
- **factor_assets** — 下游入库治理
- **quant_platform** — 上游候选流水线编排
