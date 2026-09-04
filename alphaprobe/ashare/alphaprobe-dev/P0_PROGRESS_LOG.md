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

## 2026-09-04 — P0-D factor_assets_adapter 三项修复（identity→FE 权威 / ANN 持久索引 / cluster fallback 显式化）
### 根因
- `identity_of` 用 sha256(formula) 当 canonical_hash —— 绕过 FE identity authority（硬规矩：身份唯一来源 factor_engine.identity）。
- `nearest_neighbors` 每次 new FaissANNIndex + build + search —— 索引不持久，失去 ANN 意义。
- `cluster_assign` cluster_versions 空时静默让每个因子自成 SINGLETON —— 误导 Retriever「该方向极稀缺」。

### 修复（src/alphaprobe/factor_assets_adapter.py）
1. identity_of：删 sha256(raw formula) → 经 alphaprobe.authority.build_identity_view 取 FE canonical_ast_hash→canonical_hash、signal_equivalence_id→fe_identity_ref；FactorIdentityAuthorityError → FactorAssetsUnavailable（fail-closed 不静默回落）。返回 dict 保持 canonical_repr/canonical_hash/fe_identity_ref 三 key 兼容。模块 docstring 降级链同步更新。`_default_factor_id`/`_fingerprint` 依赖正常。
2. 新增 PersistentSimilarityIndex（embedding_dim/fingerprint_version/data_snapshot/universe_snapshot/cluster_version → 版本化 key；add/add_many 幂等/同 id 覆盖；search 前 dirty 自动 rebuild；显式 rebuild()；index_rebuild_count 计数；ANNBackend Protocol，FaissANNIndex 为默认实现，lazy faiss——不 add/build 不触碰 faiss，可注入假 backend）。adapter 加 `_sim_index` 缓存 + `persistent_index` property setter + `get_or_build_index(factor_ids, embeddings, *, version_key, _backend)`。nearest_neighbors 旧签名兼容：传入 factor_ids+embeddings 走持久索引路径（同 version_key 直接 search 不重建）；纯查询无缓存返回空（零 faiss 依赖）。
3. cluster fallback 显式化：新异常 ClusterContextUnavailable(FactorAssetsUnavailable)；构造参数 allow_singleton_fallback=False 默认；cluster_versions 空 + 未允许 → 抛（production fail-closed）；显式 True → SINGLETON 且 assignment dict 带 cluster_context="SINGLETON_FALLBACK"/"PROVIDED"（ClusterAssignResult 透传顶层）。

### 测试
- 新增 tests/test_v3_p0_factor_assets.py（14 passed）：identity→FE authority monkeypatch（canonical_hash==FE ast hash / fe_identity_ref==signal_id / 确定性 / fail-closed / _default_factor_id 链路）；PersistentSimilarityIndex 假 backend（同 version_key 二次 search 不重建、add 增量触发重建、版本变化新实例、nearest_neighbors 复用缓存、lazy faiss、persistent_index 注入）；cluster context（空版本抛 / 显式 singleton 标记 / provided 标记）。
- tests/test_factor_assets_adapter.py 更新（9 passed, 1 skipped faiss）：默认空版本抛 ClusterContextUnavailable；singleton 改显式 allow_singleton_fallback=True；incremental 路径断言 PROVIDED；identity_of 改 authority monkeypatch；nearest_neighbors faiss 不可用用例改带 factor_ids 才 fail-closed + 纯查询空结果。
- tests/test_authority_wiring.py：12 passed（回归）。
- 相关套件（v3_p0 + v2_integration + memory + retrieval + seen + taskbook_units）：155 passed。
- 全套 tests/ 目录级无法直接收集：runner.py 顶层 import torch（venv 无 torch），test_pool_transition.py import torch 收集 ERROR —— pre-existing，与本改动无关。逐文件跑：其余 21 文件全绿（test_pipeline 2 failed 为同因 runner.py import torch 的 pre-existing ModuleNotFoundError）。
### 未尽事项
- venv 无 faiss/torch：FaissANNIndex 真实路径未跑（有 faiss 环境补 nearest_neighbors 结构用例）。
- 无生产代码调用 adapter（identity_of/cluster_assign/nearest_neighbors 消费方后续接线时按新语义接入）。

## 环境澄清 + round_manager 修复（P0-A，2026-09-04）
- HEAD 已推进到 2f3abe22（R60，含 V3 P0 主链修复 stub LLM/parents截断等），且 round_manager.py 在 HEAD 里已带 torch stub（我一开始误把它当 pre-existing 改动又去掉了，现**已还原为 HEAD 内容**——stub 保留，git diff 干净）。
- 环境：项目自建 .venv 有真实 torch（2.13.0+cu130）；quant_projects 主 venv 无 torch（pytest 入口 shebang 指主 venv）→ 基线 2 个 torch 失败是「运行入口 venv 无 torch」导致，属 pre-existing，不是我该修的环境。
- 结论：用 dev .venv + PYTHONPATH=src 跑 pytest（baseline：test_pipeline 15 passed/1 failed[FE inactive-canonical 环境污染，pre-existing]；其余模块 PASS）。

## P0-A 开工基线锁定（2026-09-04，HEAD=2f3abe22 R60）
- 目标文件 pipeline.py / runner.py 工作树干净（未被其它 P0 占用）。
- 任务书指定测试命令（quant venv，无 torch）基线：
  - test_pipeline.py: 15 passed / 2 failed（runner 顶层 import torch，pre-existing）
  - test_v3_p0_round_manager.py + test_memory.py: 全绿
- test_v2_integration.py 有 10 个失败 = quant_evaluator(psutil) 形状对齐 + FE inactive-canonical 环境污染 + 我未触碰的 evaluator_adapter.py 未提交改动——全部 pre-existing，不在本任务范围。
- 我的改动策略：不动 runner.py 顶层 torch import / 不改 test_v2_integration；新测试避开 quant_evaluator/psutil/FE-inactive-canonical 污染；新增 llm_client.py + pipeline/runner P0-A 改造 + test_v3_p0_pipeline.py，跑任务指定 4 文件（容忍 2 个 torch pre-existing + 不碰 test_v2_integration 失败面）。

## P0-A 代码改造进展（2026-09-04）
- llm_client.py 已建（ExecutionMode / LLMClientProtocol / DeterministicStubLLMClient / resolve_llm_fn / LLMConfigurationError）；smoke 验证：
  - PRODUCTION None/stub → raise LLMConfigurationError（fail-closed）✓
  - OFFLINE None → stub llm_fn ✓；PRODUCTION 真 client → llm_fn wrapper ✓
- pipeline.py：UNSET 哨兵 + PipelineConfig.structured_generation 默认 UNSET；SearchPipeline 增 mode/llm_client/data_search_valid/data_audit_valid 字段；__post_init__ mode 归一 + llm fail-closed + _evaluate_fn_valid/_evaluate_fn_audit；_evaluate 生产 evaluator 失败 → EVALUATION_MISSING（不手算）；_run_funnel L3_search_valid 段（data_search_valid 非 None 时）；_structured_effective 兼容旧离线默认 False。
  - smoke：production 无 llm raise ValueError ✓；OFFLINE 默认 run_round 可跑 ✓；显式 False 尊重 ✓；PRODUCTION + 真 client orchestrator structured=True ✓
- runner.py：_run_pipeline_mining 读 llm.mode（缺省 OFFLINE_TEST）+ 传 mode/llm_client；parents 不再 [:5]（全量 seed 转 parents，上限 generate_num*4）。

## P0-A 完成（2026-09-04，HEAD=2f3abe22 R60）
- 改动文件：
  1. src/alphaprobe/llm_client.py（新建）：ExecutionMode / LLMClientProtocol /
     DeterministicStubLLMClient（复用 make_stub_llm_fn，仅 OFFLINE_TEST）/
     resolve_llm_fn（PRODUCTION fail-closed）/ LLMConfigurationError
  2. src/alphaprobe/pipeline.py：UNSET 哨兵 + PipelineConfig.structured_generation
     默认 UNSET；SearchPipeline + mode/llm_client/data_search_valid/data_audit_valid/
     _evaluate_fn_valid/_evaluate_fn_audit；__post_init__ mode 归一 + llm fail-closed；
     _evaluate 生产 evaluator 失败 → EVALUATION_MISSING（不手算）；_run_funnel
     L3_search_valid（valid 段 fn，segment=search_valid）；make_fe_evaluate_fn /
     _bundle_from_plane 标 legacy（仅 OFFLINE_TEST）
  3. src/alphaprobe/runner.py：_run_pipeline_mining 读 mining.llm.mode + 传
     mode/llm_client；parents 不再 [:5]（上限 generate_num*4）
  4. tests/test_v3_p0_pipeline.py（新建，18 tests 全绿）
- 测试数字：
  - 任务指定 4 文件：60 passed / 2 failed（torch pre-existing，runner 顶层
    import torch，运行 venv 无 torch）
  - 全套（ignore test_pool_transition.py 因顶层 import torch 收集 ERROR）：
    498 passed / 2 failed（同 pre-existing）/ 1 skipped（faiss）
- 未尽事项：无。注：test_v2_integration.py 在 quant venv 下 10 个 QE 失败是
  pre-existing 环境污染（evaluator_adapter 未提交改动 + FE inactive-canonical +
  QE psutil 形状），不在 P0-A 范围；test_pipeline 2 个 torch 失败为 pre-existing。

## 2026-09-04 — P0-F evaluator_adapter 时间轴语义 + evaluate_many 批量
- 范围：src/alphaprobe/evaluator_adapter.py（label_time_axis + evaluate_many）
  1. label_end_time 语义 bug 修复：删除 `_next_time` 自然日 +1 逻辑；新增模块级
     `label_time_axis(dates, horizon, *, calendar=None)` —— dates 序列视为交易日历
     index 前移（end[i] = dates[i+horizon]），modeling 可导入时用 LabelContract /
     ashare_decision_clock 构造即校验（entry t close → 入场 t+1 VWAP → 出场 t+H）；
     绝不 datetime+timedelta(days=1)。越界未成熟端用 maturity marker 保
     LabelBundle start<end 因果链。QE LabelBundle 构造处改用 label_time_axis。
  2. evaluate_many(factor_panels, label_panel, price_panel=None, *, factor_ids=None)
     批量接口：面板共享日期/代码轴与 LabelBundle；label_days==20 时 cohort
     next_ret/next_vwap 一次构造共享，逐因子算指标；返回 EvaluationBundle 列表与
     输入一一对应；单因子失败降级 None（不抛），批级失败（QE 不可导入/契约/形状/
     空批/缺 price）QuantEvaluatorError fail-closed。evaluate() 保留原单因子路径
     （复用 _cohort_metrics_if_applicable 语义：仅 label_days==20 走 cohort）。
  3. cohort 20 日时间对拍：小算例（4 日 × 3 票确定性价格/因子）手算 daily PnL
     与 compute_cohort_pnl / compute_metrics_from_cohort 的 pnl_net 逐项 allclose。
- 配套（pipeline.py in-flight P0-A 兼容，避免回归）：
  - SearchPipeline.mode 默认值 ExecutionMode.OFFLINE_TEST 引用在类体求值会顶层
    import llm_client → 改为 dataclass default_factory `_offline_test_mode` 惰性工厂
    （与既有「模块级不 import llm_client」纪律一致）。2 个 NameError 用例转绿。
  - tests/test_v2_integration.py::test_structured_generation_default_off：旧断言
    PipelineConfig 裸默认 False 与 P0-A UNSET 哨兵设计冲突；改为断言 OFFLINE_TEST
    SP 层 orchestrator.structured_generation 仍 False（旧行为不变，与
    test_v3_p0_pipeline 兼容断言一致）。
- 测试：新建 tests/test_v3_p0_evaluator_time.py（15 用例，QE 缺省 skip 保留纯函数
  用例全绿）。全套 tests/（dev venv + quant_projects 源树 + 根 venv site-packages
  组合 PYTHONPATH，含 test_pool_transition）：510 passed / 1 skipped（faiss）。
  QE/modeling 在根 venv（/home/sunhaiwei/quant_projects/.venv）可导入；alphaPROBE
  子目录 dev venv（.venv）无 psutil/torch → QE 源树导入依赖根 venv 补充。
- 未尽事项：
  - label_days==20 的 1/20 cohort 语义在真实长面板上的数值验收沿用既有
    tests/test_v2_integration.py TestLongShort（全绿）；evaluate_many 与逐个
    evaluate 的 rankic_valid/net_sharpe 一致性已断言（abs<=1e-9）。
  - 显式 calendar 参数（比 dates 更长的真实交易日历注入）已实现并有单测；
    生产接入点暂缺真实 calendar provider（QE/factor_engine 日历源），后续接线。

## P0-A 恢复确认（2026-09-04，多次网关中断后）
- llm_client.py 去重已完成（219 行，各 class/fn 唯一出现）；pipeline.py/runner.py
  语法 OK；18 个新测试全绿。
- round_manager.py 已还原为 HEAD（之前我误动被还原 + 恢复，git diff CLEAN）；
  test_v2_integration.py 的 diff 是 P0-B/C/D 既成落盘，保留。
- pipeline.py 当前含：UNSET 哨兵、mode field 默认工厂 _offline_test_mode、
  __post_init__ llm fail-closed、_evaluate_valid、_run_funnel L3_search_valid、
  _evaluate PRODUCTION EVALUATION_MISSING 不手算。
- runner.py 当前含：_resolve_llm_mode/_resolve_llm_client、structured_generation=True、
  parents 上限 generate_num*4（不再 [:5]）。

## P0-A 最终测试数字（2026-09-04）
- 任务指定 4 文件（test_v3_p0_pipeline + test_pipeline + test_memory + test_v2_integration）：
  60 passed / 2 failed（torch pre-existing：runner 顶层 import torch，运行 venv 无 torch）
- 全套 tests/（ignore tests/test_pool_transition.py，其顶层 import torch 收集即 ERROR）：
  498 passed / 2 failed（同 pre-existing）/ 1 skipped（faiss）
- 18 个新 P0-A 测试全绿。
