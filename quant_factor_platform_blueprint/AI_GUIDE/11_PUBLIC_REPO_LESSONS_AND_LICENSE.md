# 11 — Public Repository Lessons & Third-party License Policy

> 本页是架构/研究参考，不是允许复制源码的清单。开发时再次检查当前 license/version。

## 11.1 QuantSkills

已研究的公开项目包括：

- `quantskills/skill-factor-review`
- `quantskills/skill-quant-factor-volume-stat-alpha`
- `quantskills/skill-quant-factor-directional-alpha`
- `quantskills/skill-quant-factor-risk-pattern-alpha`
- `quantskills/skill-quant-factor-skill-factory`
- `quantskills/skill-factor-mining-pandaai`
- `quantskills/skill-risk-model`

可借鉴：

- FactorCard/FactorIndex metadata
- library-level review
- factor factory/ledger
- real-validation artifact
- factor folder completeness/portable skill design

不照搬：

- 一个因子一个 Skill（10万因子不可扩展）
- runnable/pass = alpha research admission
- 简单 O(K²) 相关性库级 review
- 外部平台语义当作我们 PIT/执行真值

QuantSkills 多个仓明确为 GPL-3.0/GPL-3.0-only。专有私募 runtime 默认 clean-room reimplementation；外部 formula/metadata 做 corpus 也要保存 source/license provenance。

## 11.2 Microsoft Qlib

借鉴：

- loose-coupled components
- workflow/Experiment/Recorder/Task
- benchmark/corpus 组织
- online/offline 分离思想

不引入整个 Qlib 作为依赖；我们已有 DA/FE。

Repo: https://github.com/microsoft/qlib

## 11.3 Microsoft RD-Agent

借鉴：

- proposal -> experiment -> feedback/developer 的 R&D loop
- factor experiment workspace/template
- LLM research workflow 与真正计算环境分离

不把 RD-Agent 变成本项目 runtime 必需依赖。

Repo: https://github.com/microsoft/RD-Agent

## 11.4 Alphalens Reloaded

借鉴：Evaluator 保持窄；IC/returns/turnover/grouped analysis 的稳定用户心智。

我们的 QE 指标更广、规模更大，但 API 应保持同样清晰。

Repo: https://github.com/stefan-jansen/alphalens-reloaded

## 11.5 VectorBT

借鉴：matrix/broadcasting + NumPy/Numba/Rust hot path，批量 configuration 而不是 Python loop。

QE/FP 的 fast kernel 可以学习这种思路，但不复制整个 backtest stack。

Repo: https://github.com/polakowo/vectorbt

## 11.6 RQAlpha / QuantConnect Lean

主要作为后续 Portfolio/Execution 模块化参考；当前不扩展到执行系统。

Repos:

- https://github.com/ricequant/rqalpha
- https://github.com/QuantConnect/Lean

## 11.7 Copy Policy

每次引用第三方：

1. 记录 repo/commit/license。
2. `IDEA_ONLY` / `CORPUS` / `DIRECT_DEPENDENCY` / `SOURCE_COPY` 分类。
3. GPL/AGPL source 默认禁止复制进 proprietary runtime。
4. 算法/论文思想可根据公开说明独立重写，但保留 attribution 和验证来源。
5. 第三方库作为 optional dependency 时，license 也必须审计部署影响。
