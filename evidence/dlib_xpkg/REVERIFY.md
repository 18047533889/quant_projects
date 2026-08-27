# XPKG-REVERIFY — wheel 重建复核（新字段/新模块入轮）B 轮

状态: **PASS** (2026-08-28)
工作树 HEAD 无改动（纯验证，未动任何域源码）。

## 结果表（每域 build/install/import/e2e）
| 域 | wheel build | 全新venv install | import smoke | e2e | 备注 |
|---|---|---|---|---|---|
| factor_assets (FA) | PASS | PASS | PASS (6/6 新符号) | PASS | 新模块 _frozen / incremental / promotion_gate 全部进轮 |
| factor_optimizer (FO) | PASS | PASS | PASS (1/1) | PASS | contracts/library_snapshot_ref.py 进轮 |
| factor_preprocess (FP) | PASS | PASS | PASS (1/1) | PASS | contracts/treatment_recipe.py 进轮 |
| quant_evaluator (QE) | PASS | PASS | PASS (2/2) | PASS | adapters/recipe_refs.py 进轮 |
| quant_platform (QP) | PASS | PASS | PASS (1/1) | PASS | 无新改动，复验通过 |

所有 wheel 均从当前工作树重建，一次成功（无 retry 需要），输出 `/tmp/dlib_wheels/reverify/`。

## 新符号 import 清单（全部 PASS，均从 site-packages 加载，非源树）
- FA `factor_assets.contracts._frozen.FrozenMapping` — `site-packages/factor_assets/contracts/_frozen.py` (ABCMeta)
- FA `factor_assets.clustering.incremental.incremental_assign` — `site-packages/factor_assets/clustering/incremental.py` (function)
- FA `factor_assets.library.promotion_gate.PromotionGate` — `site-packages/factor_assets/library/promotion_gate.py` (type)
- FA `factor_assets.contracts.factor_set.FactorSetSpec.treatment_optimization_ref` 字段存在 (Optional[str | dict], PURE-DTO)
- FA `factor_assets.contracts.library_governance.FactorLibraryMembership.treatment_optimization_ref` 字段存在 (Optional[str | dict])
- FO `factor_optimizer.contracts.library_snapshot_ref.LibrarySnapshotRef` — 含 content_hash / to_dict 全绑定
- QE `quant_evaluator.adapters.recipe_refs.factor_batch_for_recipe` (function)
- QE `quant_evaluator.adapters.recipe_refs.recipe_evidence_evaluation` (function)
- FP `factor_preprocess.contracts.treatment_recipe.TreatmentRecipe` (dataclass, 含 RecipeStep/FitBoundary)
- QP `quant_platform` (site-packages/quant_platform/__init__.py)
smoke 脚本: /tmp/dlib_run/xpkg_smoke.py —— 10/10 PASS（含中立 cwd + 源树路径断言）。

## PURE-DTO 复核
- **FA 不 import FO**: 全树 grep 无 `from factor_optimizer` / `import factor_optimizer` 编程 import（仅一条注释/文档字符串提及 factor_optimizer）。`treatment_optimization_ref` 为可选裸 `str | dict` PURE-DTO（FO `LibrarySnapshotRef.to_dict()` 形态），FA 经 `_canonical_ref` / `_canonical_optimization_ref` 规范化（content_hash 即字符串传输形态），双 dataclass 字段都存在。
- **QE 只在 adapters/recipe_refs.py 内部延迟 import FP**: 唯一 FP import 在 recipe_refs.py:348 的函数体作用域内（`_load_fp_transform`，`from factor_preprocess.transforms import ...`）。QF core 无 FP import。
- **import-independence 实测**（全新 venv, unit cwd）: `import quant_evaluator.adapters.recipe_refs` 后 sys.modules 无 `factor_preprocess`；FA 三项模块 import 后 sys.modules 无 `factor_optimizer`。
- 结论: PURE-DTO 结构成立——跨包只传裸 dict/str 引用，不传正式联系人对象。

## e2e 结果（最小 3 步: FP recipe → QE ref → FO winner + FA 一致性）
脚本 /tmp/dlib_run/xpkg_e2e.py，全新 venv 中立 cwd 全 PASS:
1. FP 构建 TreatmentRecipe（CS_RANK:pct 一步，content_hash 自动派生 dba2490b**…**）
2. QE `factor_batch_for_recipe` → FactorBatch shape (6,4,1)；`recipe_evidence_evaluation` → 证据 bundle（coverage=1.0 valid；rank_ic/pearson_ic computed valid=False——面板仅 4 assets 且 IC 系列统计自由度不足，属预期，metrics 键齐备）；`recipe_to_factor_value_ref` → factor_value:treated 引用
3. FO `LibrarySnapshotRef` + `TreatmentOptimizationResultArtifact(library_snapshot_ref=snap, require=True)` + `select_winner` → winner trial_0001
4. FA `FactorSetSpec`/`FactorLibraryMembership` 以 `{"library_snapshot_ref": snap.to_dict(), "content_hash": ...}` 裸 dict 消费 → 各自规范化为 PURE-DTO 字符串 `== snap.content_hash`
5. 一致性断言 PASS: `winner.library_snapshot_ref.content_hash == FA treatment_optimization_ref`（规范字符串形态）

winner 无正式 library_snapshot_ref 字段（ParetoPoint 只有 trial_id/objectives/metadata），任务级绑定在 FO `TreatmentOptimizationResultArtifact` 上（已含 library_snapshot_ref），e2e 用 artifact 绑定 + FA 消费断言链路闭合。

## BLOCKED 项
无。

## 环境与命令记录
- 构建: project venv `/home/sunhaiwei/quant_projects/.venv/bin/python` (Python 3.12.3, build 1.5.0) `cd <包根> && python -m build --wheel --outdir /tmp/dlib_wheels/reverify`
- 包根: FA `/home/sunhaiwei/quant_projects/factor_assets`, FO `/home/sunhaiwei/quant_projects/factor_optimizer`（pyproject packages=["factor_optimizer"] 直接映射嵌套因子包）, FP `/home/sunhaiwei/quant_projects/factor_preprocess`, QE `/home/sunhaiwei/quant_projects/quant_evaluator`, QP `/home/sunhaiwei/quant_projects/quant_platform`
- 全新 venv: `/tmp/dlib_venv/reverify`（`--without-pip` + 从 project venv 拷 pip 26.2.1）；`pip install --no-deps` 5 wheel + `--only-binary :all:` 装运行时依赖（numpy scipy pandas psutil pyyaml pywavelets statsmodels typing-extensions dataclasses-json）→ `pip check` 无 broken
- wheel 内容核验: zipfile 检查新模块文件全部在 wheel 内（_frozen.py / incremental.py / promotion_gate.py / library_snapshot_ref.py / treatment_recipe.py / recipe_refs.py）
- neutral cwd: /tmp/dlib_run, PYTHONPATH 清空
- 产物: /tmp/dlib_wheels/reverify/*.whl (5)、/tmp/dlib_venv/reverify、/tmp/dlib_run/xpkg_smoke.py、/tmp/dlib_run/xpkg_e2e.py