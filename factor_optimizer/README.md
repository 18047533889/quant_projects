# factor_optimizer — 证据引导的因子寻优

变异语法 + 搜索编排（train/validation/test sealed 切分）+ desirability/Pareto 验收。
本库只通过 adapter protocol 协调 **factor_engine**（计算/校验）与 **quant_evaluator**（证据），
自己**不执行因子、不算指标、不做最终入库决策**。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer (私有)

## 安装

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer.git
cd factor_optimizer
pip install -e .                     # core
pip install -e ".[factor_engine,quant_evaluator]"   # 可选适配器
```

## 核心 API

```python
from factor_optimizer.search.runner import SearchRunner, SearchConfig, SearchSession
from factor_optimizer.search.winner_selector import select_winner
from factor_optimizer.search.uncertainty_winner import UncertaintyAwareWinnerSelector

session = SearchSession(SearchConfig(search_budget=...))
runner = SearchRunner(session, fe_adapter=..., qe_adapter=...)
# runner.run(...) -> trial ledger / selection / acceptance
```

- `grammar/` — 变异语法：`mutation_spec` / `registry` / `validation`。
- `search/` — `SearchRunner`、`SearchSession`、`desirability`（Desirability + catastrophic floor）、
  `pareto`（ParetoPoint/Frontier/Archive）、`uncertainty_winner`（不确定性感知赢家选择）、
  `tiered_evaluation`（分层评估）、`multifidelity`（保真度档位）、`plateau`（平台期检测）、
  `treatment_decision`、`lineage`。
- `policy/` — 准入标准/判定、诊断 + 修复（LLM repair mapper）、治理。
- `contracts/` — `Trial` / `TrialLedger` / `SearchBudget` / `TreatmentIntegrity`
  （evidence schema version）/ `CandidateMutation` / `Objective` / `Splits` / `Validator`。
- `adapters/` — `factor_engine.py`（canonical hash、validate mutation、estimate complexity、
  operator metadata）、`quant_evaluator.py`（证据）。
- `llm/` — LLM 辅助变异的 prompts + proposal。
- `seen/` — 候选身份；`complexity/` — 复杂度 profile/预算。
- `data_capabilities.py` / `data_providers.py` — train/validation/test 数据作用域。

## Sealed-test 纪律

`SealedTestExecutor` 强制 train/validation/test 隔离（purge/embargo）；搜索预算限制总评估次数；
`TreatmentIntegrityCheck` 把证据 schema 版本绑定到 treatment 结果。

## 依赖与接口（谁 import 谁）

- **依赖（适配器）**：`factor_engine`（校验/哈希/复杂度）、`quant_evaluator`（评估证据）。
- **被谁调用**：`quant_platform` 候选流水线（candidate → treatment → search）。
- **输出给**：`factor_assets`（入选 treatment 提交入库）。

## 相关仓库

- **factor_engine / quant_evaluator / factor_assets / quant_platform** — 见上
