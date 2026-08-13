# 07 — Research Control, Factor Review Skills & Corpora

## 7.1 Research Control 是 Ledger，不是第五个平台

中央记录：

- campaign
- algorithm/researcher
- trial
- parent/child
- mutation/search config
- code version
- data snapshot/ref
- QE evaluation ref
- FA decision ref
- prompt/model version（LLM 场景）
- failures/crashes
- search/multiple-testing budget

目标：研究自由度可追溯、防止“只留下赢家”。

第一版 SQLite/JSONL 均可，但必须有 append-only event 语义与稳定 ID。不要先上 Kafka/微服务。

## 7.2 Library-level Review

吸收公开 `skill-factor-review` 的思路，但底层全部使用我们的 ledger/QE/FA：

三层输出：

1. Quantitative inventory：试验数、接受/拒绝/Crash、突破轨迹、边际发现率。
2. Structural analysis：family 覆盖、graph/cluster、重复率、机制空白、搜索空间拥挤度。
3. Research recommendation：下一轮算力/机制/参数范围分配、应暂停的搜索方向。

不要使用“接受率 25–50%”等固定经验值作为硬标准；只作为描述性诊断。

## 7.3 Corpora

### Internal

- GTJA185
- Week2 PV factors
- 现有已认证 production factors（去敏后）
- synthetic edge cases

### External

- QuantSkills structured factor collections
- Qlib Alpha158/相关公开 formula corpus
- 其他许可允许的公开公式/metadata

用途：

- FE formula/AST 回归
- QE batch/performance benchmark
- FO mutation legality/search benchmark
- FA duplicate/family/graph benchmark
- FP transform regression

**Corpus 不等于 production factor pool。**

## 7.4 Public QuantSkills Lessons

可借鉴：FactorCard/FactorIndex、每个因子的 metadata、real-validation artifact、factor library review/factory workflow。

不直接照搬：

- 一个因子一个 Skill（10万因子不可扩展）
- `status=pass` 等同研究通过
- 小规模 O(K²) correlation review
- GPL runtime source 直接并入专有代码

## 7.5 Corpus Adapters

外部 formula -> adapter -> FE-compatible definition。

Adapter 应记录：source repo, source ID, license, original formula, translation notes, unsupported semantics。

不要静默修正第三方公式；翻译/近似必须标记。
