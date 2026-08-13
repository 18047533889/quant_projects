# 02 — Server-First Repository Audit & Migration Plan

## 2.1 GitHub 不是本次改代码的事实源

开发前必须在服务器运行：

```bash
git status --short
git diff --stat
git log -1 --oneline
find . -maxdepth 2 -type d | sort
rg -n "sys\.path|from factor_layer|AutoFactorEvaluation|toolkit|AlphaPurifier|FactorAnalyzer|Exposures|factor_admission" .
```

如服务器不是 Git 仓，则跳过 Git 命令，但必须记录目录 snapshot/hash。

输出 `LOCAL_REPO_SNAPSHOT.md`，至少包含：

- 当前 commit/branch（若有）
- 未提交修改
- 实际目录
- 当前 `dataaccess` / `factor_engine` 版本与 public entrypoints
- legacy 目录实际位置
- Python packaging 情况
- 现有 tests / CI / benchmark

## 2.2 Migration Matrix 必须函数级

示例：

| Source | Symbol/Area | Decision | Target | Reason | Required Tests |
|---|---|---|---|---|---|
| `factor_evaluation/pipeline.py` | RankIC semantics | REFERENCE_ONLY/REWRITE | QE reference/ic | 简单易审计但慢 | golden + ties + NaN |
| `AFE/.../batch_metrics.py` | row corr / HAC | REWRITE | QE kernels/statistics | 有向量化价值 | parity + perf |
| `FactorAnalyzer.py` | turnover | REWRITE | QE metrics/turnover | 功能有用、旧实现需审计 | golden |
| `AlphaPurifier.py` | robust zscore | REUSE_AFTER_TEST | FP transforms | 语义成熟 | parity |
| `APr_utils.py` | OLS neutralize | REWRITE | FP neutralization | 需统一权重/NaN/fit contract | regression |
| `toolkit/cross_sectional.py` | cs_rank | REUSE_AFTER_TEST | FP reference | 纯函数、边界清楚 | metamorphic |
| `factor_admission/catalog.py` | decision ledger | REWRITE | FA registry | schema 思路有价值 | migration test |
| gateway future scanner | whole | DISCARD | - | FE 已负责 | no legacy import |
| gateway physical plan | whole | DISCARD | - | FE 已负责 | no duplicate capability |
| GTJA185 | formulas/manifests | CORPUS_ONLY | corpora/internal | 真实 workload | regression |

## 2.3 已知迁移方向

### 到 QuantEvaluator

重点审计：

- `factor_layer/factor_evaluation/pipeline.py`
- `AutoFactorEvaluation-RECONSTRUCT/evaluation/batch_metrics.py`
- `FactorAnalyzer.py`
- `Exposures.py` 中纯 evaluation/exposure/attribution 部分

只迁数学语义、reference、可复用 kernel；删除所有：

- 数据路径读取
- 自建 PIT/universe
- parquet/COS wrapper
- 自建 factor materialization
- report/pipeline 大壳

### 到 FactorPreprocess

重点审计：

- `AlphaPurifier.py`
- `APr_utils.py`
- `toolkit/cross_sectional.py`
- `Exposures.py` 中 neutralization/decomposition 可复用部分

迁移前必须把所有 transform 分类为：

- stateless
- rolling causal
- fitted unsupervised
- fitted supervised
- research-only

### 到 FactorOptimizer

重点审计：

- `toolkit/registry.py`
- `alpha_tools/registry.py`
- Gateway ComplexityEvaluator / ComplexityReport
- Gateway Candidate contract 的 origin/lineage/campaign fields
- `factor_agent` 的 verifier/agent-loop/tool-registry 思想

不迁：

- `generated_library.py` 中 FE 已存在的 operator/factor kernels
- 旧 future scanner/operator validator
- 旧 data loader

### 到 FactorAssets

重点审计：

- `factor_admission/catalog.py`
- Gateway Candidate/decision segments
- exact hash / seen cache 思想
- Assetization 的 FactorCandidate/FactorAsset 元数据模型
- factor-pool-standard 的 domain taxonomy（若仍有独特信息）

不迁：

- 固定 IC/Sharpe 阈值作为核心 admission
- PhysicalPlan
- 大 factor value storage
- 第二套 evaluation summary 真值

## 2.4 Legacy Bug Policy

不得因为“旧代码已经跑过”就认为正确。

迁移时：

1. 先为旧行为写 characterization test。
2. 对公式单独审计。
3. 如果旧行为有 bug，建立 `legacy_bug_regression`，明确新结果为什么不同。
4. 不为了兼容 bug 保留错误行为。

## 2.5 Legacy Archive

完成迁移后，旧目录只有两种状态：

- 仍独立有业务价值：明确保留并重新定位。
- 已被新库替代：移动/标记为 archive，不得再被 production import。

必须有 `audit_no_legacy_imports.py` 防回流。
