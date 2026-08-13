# 22 — API Examples & Pseudocode

这些是目标用户体验，不是强制逐字 API；Contract Freeze 时可微调名称，但保持简洁。

## QuantEvaluator

```python
import quant_evaluator as qe

bundle = qe.evaluate(
    factor_values,
    labels=forward_returns,
    metrics="factor_core",
    context=ctx,
)

rank_ic_2026 = qe.evaluate(
    factor_values,
    labels=forward_returns,
    metrics=["ic.rank.mean"],
    where={"year": 2026},
)
```

## FactorOptimizer

```python
from factor_optimizer import optimize

result = optimize(
    definition=parent,
    diagnosis=evaluation.diagnosis,
    executor=fe_adapter,
    evaluator=qe_adapter,
    search_budget={"fast": 200, "full": 30, "robust": 5},
)
```

内部伪代码：

```python
for proposal in policy.propose(parent, diagnosis):
    if not grammar.legal(proposal):
        continue
    if budget.allow("L1"):
        child_values = executor.compute(proposal)
        fast = evaluator.evaluate(child_values, preset="optimizer_feedback_fast")
        search.observe(proposal, fast)
# successive promotion -> full/robust
```

## FactorAssets

```python
asset_id = assets.register(candidate, evidence_ref=bundle.ref)

decision = assets.assess(asset_id, policy="production_daily_equity")

neighbors = assets.neighbors(asset_id, k=100)

factor_set = assets.assemble(
    universe="active_core",
    strategy="family_robust",
)
```

FA 不因 `neighbors()` 直接计算全历史 correlation；ANN shortlist 后通过 QE adapter 请求 exact stats。

## FactorPreprocess

```python
policy = fp.Policy.linear_ready(
    keep_raw=True,
    neutralization="soft",
    alpha=0.7,
)

state = fp.fit(policy, train_values, context=train_ctx)
train_features = fp.transform(train_values, state, train_ctx)
test_features = fp.transform(test_values, state, test_ctx)
```

Stateless-only policy 可以没有 fit state。

## Research Ledger

```python
with ledger.campaign("alphaprobe_2026_08", algorithm="AlphaProbe") as c:
    trial = c.start_trial(parent_id=..., mutation=...)
    trial.attach_evaluation(bundle.ref)
    trial.finish(decision="REJECTED", reason="low_residual_information")
```

Ledger API 保持轻，不成为 worker execution framework。
