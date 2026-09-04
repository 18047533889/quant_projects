# P0_PROGRESS_LOG (任务代号 P0-A)

## 2026-09-04 — P0-E RoundManager._update_memory 三修
- 范围：src/alphaprobe/continuous/round_manager.py _update_memory（§74/§28/§77）
  1. pool_snapshot[:50] 截断删除 → 全量遍历兜底对账（实时写在 pipeline.py
     _remember_attempt/_remember_evaluation 内完成，此处兜底写被淘汰成员）
  2. 本地 dedup regex identity → build_identity_view（FactorIdentityAuthorityError
     → print 跳过该成员不阻塞；绝不回落文本 regex）
  3. round 事件不再写 market_regime_events → GlobalMemoryStore 新增
     add_search_run_event/search_run_event_exists + search_run_events 表；
     两处 add_regime_event 全改 add_search_run_event
  4. §77 LineagePatienceTracker round 级 dummy 加注释说明现状与 P1 目标（不改行为）
- memory/__init__.py：新增 search_run_events 建表（含 round 索引）+ 2 个方法
  （照抄 add_regime_event INSERT OR REPLACE 幂等风格）；import datetime/timezone
- 附带修复（factor_assets_adapter.py 既有 in-flight 改动区）：
  get_or_build_index 缓存判定 bug —— keys（coerce 首位 dim 占位 0）与索引
  version_key（首位真实 embedding_dim）恒不等 → 缓存永不命中；改为比较后 4 位。
- 测试：新建 tests/test_v3_p0_round_manager.py 4 tests（事件幂等+regime 零污染、
  两表分离、60 成员全量写、authority error 跳过）。全套 tests/（排除 3 个 torch
  import 模块：test_pool_transition/test_funnel_admission/test_pipeline 中 2 个用例）
  = 423 passed + 1 skipped；2 个 torch 失败 pre-existing（stash 干净验证 ModuleNotFoundError）


## 2026-09-04 — P0-A 启动：SearchPipeline/runner P0 主链修复
- 任务范围：
  1. 新建 src/alphaprobe/llm_client.py（ExecutionMode / LLMClientProtocol / DeterministicStubLLMClient / resolve_llm_fn）
  2. pipeline.py SearchPipeline 改造（data_search_valid / data_audit_valid / mode / llm_client / Train-Valid 分离 / fallback fail-closed / structured_generation 默认 True）
  3. runner.py parents 不再 [:5]
  4. 新增 tests/test_v3_p0_pipeline.py


## [P0-B] retrieval/搜索调度三个语义 bug 修复（2026-09-04）
### Bug 1 retrieval_count 冒充 — 已修
- GlobalMemoryStore 新增 retrieval_events(factor_id,ts,action) 轻量表 + idx_retrieval_events_factor 索引，新增公共 API `record_retrieval(factor_id, action=None)` / `retrieval_count_of(factor_id)`（src/alphaprobe/memory/__init__.py）。
- BayesianRetriever._retrieval_count_of 不再用 lineage_parents(「几个孩子」)；优先查 retrieval_count_of，退化内存 dict。新增公共 `record_retrieval(factor_id, action=None, memory=True)` 同步维护 _node_states 退化计数；select_parents 命中 top-k 即记一次真检索（disabled/空候选不记）。向后兼容：candidate 显式带 retrieval_count 的既有路径不变。
- 测试：store 闭环、孩子繁殖不增计数、retriever 优先真计数、记忆态退化、select_parents 落表/退化/disabled 不记。

### Bug 2 missing-as-best — 已修
- search_opportunity.py：cluster_rarity(None)/非法/<=0 → 中性 `CLUSTER_RARITY_NEUTRAL=0.5`（不再 1.0）。已知 singleton(1)→1/sqrt(2)≈0.707。docstring 厘清「无 cluster 上下文(None)→0.5」vs「已知 singleton(1)→0.707」；_local_cluster_size_of 返回值 1.0 保留并加注释。
- 更新 test_retrieval_v2.py 旧断言（None/0/-5 → 0.5）。

### Bug 3 reward=candidate 数 — 已修
- orchestrator.py：删除 len(candidates) reward。OrchestratorStep 加 `pending_reward: bool=True`；`step_with_llm(*, reward=None)`：reward is None → 不 update scheduler（等待 settle_reward 延迟回传）；显式 float(含 0.0) → 照常 update 并置 pending_reward=False。新增 `settle_reward(step, reward)` 回填真实 reward（公式参考注释已加）。
- 更新 test_search_arms.test_three_steps_no_throw_stats_grow 依赖旧「默认 reward=len(candidates)」的行为 → 改为断言不传 reward stats 不更新 + settle 闭环后更新。

### 顺带修（pre-existing 测试 bug，非本次语义范围但阻塞全绿）
- tests/test_factor_assets_adapter.py 两个 identity_of 测试 monkeypatch 打错位置：identity_of 在方法内 `from alphaprobe import authority`，测试 patch 了独立 import 的 authority_mod 副本——补 `monkeypatch.setattr(mod,'authority',authority_mod,raising=False)` 让包属性指向同一被 patch 模块对象。2 个测试由 FAIL 转 PASS。

### 测试数字
- tests/test_v3_p0_retrieval.py（新增 25 用例）全绿。
- tests/test_retrieval_v2.py + tests/test_search_arms.py + tests/test_memory.py 全绿（合计 96 passed）。
- 全套 tests/（除 test_pool_transition.py 因 torch 缺收集失败）：449 passed / 1 skipped(faiss) / 2 failed —— 仅剩预存 test_pipeline.py 2 个 torch ModuleNotFoundError（runner.py import torch），与本次改动无关。

## 基线确认（P0-A，2026-09-04）
- 当前工作树 tests/ 已有其它 P0 任务遗留（test_v3_p0_round_manager.py 等 4 个 untracked 测试文件）。
- 全套：419 passed + 2 个 torch pre-existing FAILED（test_pipeline.py::test_runner_import_* / ::test_record_exported_to_memory_*，root cause：runner.py 顶层 `import torch` 在无 torch venv 下必然失败——单跑即失败，非顺序污染）+ 1 skip（faiss）。
- 另 test_pool_transition.py 因顶层 `import torch` 收集即 ERROR（基线之外）。
- 待改文件 pipeline.py / runner.py 均未在工作树被占用（git diff 为空）。

## 关键观察（P0-A）
- funnel.gate("L3_search_valid", record) 对未知 level 返回 [] 不拒（funnel.py gate 分发兜底）→ 保证新 L3 bundle gate 不炸。
- pipeline._run_funnel 当前全链只在 record segment="train" 下评估；record2 (L2) 即为最终 record。
- v2_integration test_structured_generation_default_off 断言 cfg 默认 False + orchestrator False → 需要兼容层（PipelineConfig 默认 True + UNSET 哨兵保持 OFFLINE_TEST 旧行为）。

## 2026-09-04 — P0-C pareto_eviction_candidate 淘汰方向修复
### 根因
- src/alphaprobe/pool/pareto.py pareto_eviction_candidate 用 `max(cands, key=_key)`，_key=(rank, crowding, factor_id)：
  max 对 crowding 取最大值 → 同 rank 里最稀疏（最该保留）者被选淘汰；crowding=inf（边界点）反而最先淘汰，与注释语义相反。

### 修复（src/alphaprobe/pool/pareto.py）
- 改为显式比较 + min（`functools.cmp_to_key`），语义：rank 越大越该淘汰（rank0 最优 front 绝不因 min 被选）；同 rank crowding 越小（越拥挤）越该淘汰；crowding=inf（边界）视为 +inf 最不该淘汰；完全并列 → tie_break / factor_id 字典序（负=前者优先淘汰）。签名/返回类型不变。
- 边界澄清：不能用单一 max（crowding 反向）也不能用 min(rank,...)（rank0 会被误选），比较器内分域处理。

### 顺带修复 src/alphaprobe/pool/__init__.py ActivePool._eviction_candidate
- 同样存在 crowding inf 处理随 reverse=True 反向（边界点最先淘汰）。改为：排序键 `c_key = -c if c != inf else inf`，升序 sorted 首元素 = 最该淘汰者（无双重反转；淘汰消费方向 = 返回者即被 _evict 的 victim）。

### crowding_distance 边界语义加固（pareto.py，配套）
- 原实现同 rank 全部等价点（3 个完全相同目标向量）每维两端标 inf 会让「中间索引」点被标 0 → 淘汰侧绕过 tie-break 按 crowding 误淘汰某等价点。
- 加全等价组 guard：整组按 inf，淘汰退化为纯 tie-break 字典序（确定性）。
- 同 rank ≤2 点全 inf（NSGA-II 边界保护）保留。

### 测试 tests/test_pool_pareto.py
- 新增：同 rank 3 点（2 边界 inf + 1 内部拥挤）→ 淘汰内部最拥挤者；全部子集组合验证不先淘汰边界；rank 优先于 crowding；完全等价 → 字典序最小 id；自定义 tie_break 反转生效；exclude_factor_id；crowding 全等价组不产生偏置。
- 修正依赖原 max 错误方向的 2 个既有断言。

### 测试数字
- tests/test_pool_pareto.py：20 passed（修复前 13 passed + 新增/修正）。
- 相关池套件（pool_pareto + v2_integration + taskbook_units + funnel_admission + pipeline）：123 passed。
- 全套 tests/（exclude test_pool_transition.py 因 torch 收集 ERROR）：465 passed / 1 skipped(faiss) / 2 failed —— 2 个 pre-existing 为 tests/test_pipeline.py 中 runner.py 顶层 import torch 的 ModuleNotFoundError（单跑即失败，与本次改动无关）。
