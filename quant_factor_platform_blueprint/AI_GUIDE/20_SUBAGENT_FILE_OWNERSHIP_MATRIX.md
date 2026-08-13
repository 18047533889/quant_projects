# 20 — Subagent File Ownership Matrix

并行开发的关键不是 Agent 数量，而是共享文件最少。

## Shared / Lead-only

仅 `lead-architect` / `chief-integrator`：

- `/schemas/**`
- root-level build/workspace config
- 各 package 顶层 `pyproject.toml`（worker 可提 patch，不直接抢写）
- 各 package `__init__.py` 最终 public exports
- cross-package integration fixtures
- version bumps

## QuantEvaluator

- `qe-runtime-agent`: `quant_evaluator/api/**`, `contracts/**`, `registry/**`, `planner/**`, `runtime/**`
- `qe-core-metrics-agent`: `metrics/quality.py`, `distribution.py`, `ic.py`, `quantile.py`, `turnover.py`, matching reference/tests
- `qe-temporal-robustness-agent`: `metrics/temporal.py`, `exposure.py`, `robustness.py`, `ashare.py`, `diagnosis/**`
- `qe-performance-agent`: `kernels/**`, `benchmarks/**`; 不改 MetricSpec semantics

## FactorPreprocess

- `fp-transform-agent`: `transforms/**`, reference tests
- `fp-neutralization-agent`: `neutralization/**`
- `fp-representation-agent`: `representation/**`, `policies/**`, FeatureBundle contract implementation
- `fp-performance-agent`: `kernels/**`, benchmarks

## FactorOptimizer

- `fo-grammar-agent`: `grammar/**`, legality/type rules
- `fo-search-agent`: `search/**`, `complexity/**`
- `fo-llm-agent`: `llm/**`, structured proposal adapter

## FactorAssets

- `fa-registry-identity-agent`: `registry/**`, `identity/**`, `seen_index/**`, lifecycle base
- `fa-novelty-agent`: `novelty/**`, QE/provider adapter related code
- `fa-graph-cluster-agent`: `similarity/**`, `graph/**`, `clustering/**`
- `fa-aggregation-agent`: `selection/**`, `aggregation/**`

## Read-only Auditors

不得 Write/Edit：math, leakage, performance, boundary, simplification, extraction, license, corpus regression, platform boundary。

## Collision Rule

如果 worker 发现必须修改 shared contract：

1. 停止修改。
2. 提交 `CONTRACT_CHANGE_REQUEST`：理由、兼容性、受影响包、migration。
3. Lead 决定。
4. Contract 修改后相关 worker rebase/重新读 schema。
