# 05 — FactorOptimizer Development Specification

## 5.1 定位

`factor_optimizer` = Evidence-Guided Factor Evolution Engine。

输入：FactorDefinition + FactorDiagnosis/EvaluationBundle + search constraints。
输出：CandidateMutation / optimized descendants / search result。

它不是第二个 FactorEngine，也不是第二个 Evaluator。

## 5.2 Legacy 来源

优先利用：

- `toolkit/registry.py`, `alpha_tools/registry.py` 的 registry/parameter metadata/callable hash 思想
- Gateway ComplexityReport / complexity budget 思想
- Gateway Candidate 的 origin/parent/campaign metadata
- `factor_agent` 的 verifier/structured tool loop 思想

不迁：

- FE 已有的 EMA/rank/ATR/rolling 等 kernel
- 旧 future scanner/operator validator
- 旧 Data IO

## 5.3 Core Protocols

FO core 依赖抽象：

```python
class FactorExecutorProtocol:
    def validate(definition): ...
    def compute(definitions, request): ...
    def complexity(definition): ...

class EvaluatorProtocol:
    def evaluate(values, labels, metrics, context): ...
```

官方 adapters：FactorEngine / QuantEvaluator。

因此 FO 可以单独安装做 plan/proposal，也可以装 extras 后执行。

## 5.4 MutationSpec

至少：

- name/version/family
- allowed input semantic types
- output semantic type
- required FE operators
- parameters/bounds/defaults
- causal flag
- requires_fit
- allowed data domains
- complexity increment estimate
- production tier
- LLM-exposed flag
- description/mechanism intent

Mutation registry 与 FE operator registry 要有 consistency audit；FO 不复制 operator 语义。

## 5.5 Diagnosis -> Repair Policy

Production rules examples：

- TOP_TAIL -> percentile/tail score/smooth threshold
- BOTTOM_TAIL -> symmetric bottom-tail transforms
- U_SHAPE -> absolute-distance/two-sided score
- OUTLIER_SENSITIVE -> rank/winsor/robust transform
- HIGH_TURNOVER -> EMA/hysteresis/persistent component
- SIZE/INDUSTRY/LIQUIDITY_DEPENDENT -> residual/channel/soft-neutralization candidate（不要默认全中性）
- FAST_DECAY -> execution-delay-aware smoothing or discard if unimplementable
- LONG_ONLY / SHORT_ONLY -> asymmetric representation candidate
- STALE_FUNDAMENTAL -> freshness/announcement-age conditioning

映射只是 candidate generation prior，不是自动“治疗成功”。所有 child 都必须重新 QE。

## 5.6 Mutation Families

结合当前 A 股数据：

- temporal: lag/window/EMA/fast-slow/persistent-transitory
- scaling: vol/liquidity/market-cap/free-float scaling
- shape: rank, winsor, sigmoid, tail, piecewise
- interaction: value×quality, momentum×liquidity, fundamentals×price reaction, ownership×liquidity
- intraday-to-daily: FE 已有 minute operators/aggregations 的组合定义
- fundamental: level/change/YoY/QoQ/single-quarter/TTM/freshness
- ownership: concentration/HHI/change/pledge/freeze

禁止生成当前数据不支持的 Level2、订单簿、分析师一致预期、北向等生产核心候选。

## 5.7 Search

必须支持分层成本：

- L0：静态合法性/type/availability/complexity
- L1：短窗口/子股票池 T0 快评
- L2：完整历史 T1
- L3：walk-forward / robust T2
- L4：简单组合/成本/实现性

可实现：successive halving / Hyperband-like / local evolutionary / budgeted Pareto。不要第一版做复杂分布式 BO 平台。

## 5.8 Parameter Plateau

不只找 peak。对参数邻域记录：

- plateau width
- local sensitivity
- curvature
- neighbor survival
- worst-neighbor performance

child admission 需要稳定平台，而不是孤立尖峰。

## 5.9 Complexity

不要继续用 regex 括号计数作为真值。

优先从 FE canonical AST/IR adapter 获取：

- AST depth
- operator count/weights
- lookback
- stateful ops
- CS ops
- nonlinear ops
- interaction count
- data domains/source count
- estimated/actual compute latency

Utility 不只 RankIC：

`ΔOOS - λ_complexity - λ_turnover - λ_fragility - λ_data_cost`

多目标用 Pareto，不强制固定万能线性总分。

## 5.10 LLM Researcher

LLM 只：

- 读取 Diagnosis
- 提机制假设
- 选择/参数化允许的 MutationSpec
- 解释实验结果
- 建议下一批

LLM 输出必须是结构化 `CandidateMutation`，不能直接提交任意 Python production code。

记录：model, prompt version, hypothesis, expected signatures, trial ID。

机制假设必须可 falsify：例如“低流动性 reversal”要由 QE 检验 liquidity slices/horizon，而不是用语言证明。

## 5.11 Search History

FO 不自己建设大 Research DB；通过 `ResearchLedgerProtocol` 写中央 research_control。

每个 trial：parent, mutation, config, FE validation, QE result ref, decision, budget consumed。

## 5.12 Tests

- grammar legality
- invalid operator/data domain rejection
- deterministic mutations
- parent/child lineage
- plateau synthetic tests
- search budget never exceeded
- future-poison test for fitted/search selection
- evaluator/executor mocked protocol tests
- optional integration with FE/QE
- no duplicate metric/operator implementations
