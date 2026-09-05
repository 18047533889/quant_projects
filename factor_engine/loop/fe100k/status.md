# FE 100k GO 战役 — 状态

coordinator: main (direct + subagents)
started: 2026-09-03
HEAD: 7628674b (main)
live registry: 1673 canonical / production 137 (load_all include_research=False)

## Active
- P0#1 (task #30) inventory/DirectUseMatrix rebuild — subagent a18eb66aa6d986a63 running
- P0#3-review (task #32) perf_vec_bench __vec__ fix 未开始（依赖 reviewer 模式定型）
- R13 honesty/env-cache 修复已绿（reviewer a9dc4ca83a58b9d1e in-flight）

## Queue
- P0#2 (task #31) agent allowlist fail-closed — 等 #30 matrix
- P0#4 (task #33) intraday vector coverage — ✅ 完成（见下）
- P0#5 (task #34) IntradayFeatureCompiler
- P0#6+#7 (task #35) run_many/CSE/telemetry — ✅ 完成（见下）
- P0#8 (task #36) scheduler×HybridExecutor A/B
- P0#9 (task #37) multiworker governance — ✅ 完成（见下）
- P0#10 (task #38) A-share contract gate — ✅ 完成（见下）
- P0#11+#12 (task #39) dry-run ladder ✅ (见下方「P0#11+#12」)
- P1 (task #40) family audit
- deliver (task #41) GO/NO-GO

## P0#6+#7（task #35）run_many enforcement + CSE/fusion/fallback telemetry ✅

P0#6 — run_many 批量入口强制 + 输入 canonicalization：
- `runtime/batch_service.py` 新增 `_canonicalize_batch_factors()`，`execute_run_many /
  execute_run_many_iter / execute_run_many_parallel` 三入口统一调用：结构完全相同
  （name+expr 一致）的因子去重一次；重复 name 仍 #322 硬拒绝。每批上报
  `runmany.requested / unique / duplicates / factors / contexts` counters。
- `runtime/engine.run()` 保留单因子合法用途，但批量多因子调用方在循环里逐 `run()`
  会累计 `runmany.single_run_events` counters——不静默（doc 同步 §8）。

P0#7 — CSE / fusion / fallback telemetry（新 `telemetry/execution_telemetry.py` singleton）：
- CSE（`planner/cse.py:apply_cse`）：`cse.candidates / shared_nodes / reuse_edges /
  nodes_released`。
- Native fusion（`planner/native_fusion.py:_emit_fusion_telemetry`）：`fusion.planned /
  executed / binary_split / fallback / roots_total` + `fallback.*` 显式事件。
- backend pandas fallback（`runtime/batch_service.py:_emit_pandas_fallback_telemetry`）：
  `fallback.pandas` 计数 + 首条 fallback 算子 INFO 日志。
- Snapshot API：`execution_telemetry.snapshot()` / `reset()` / `get()`。

Tests（`tests/test_run_many_telemetry.py`）9 passed：canonicalization 去重/唯一名规则、
CSE 重用触发 counters、engine run_many/run 存活且上报、fusion/fallback counters、
snapshot reset。Regression：cse_run_many + telemetry 模块 + r39 perf telemetry 76 passed。
Pre-existing（与 clean HEAD 一致）：`POLARS 生产模式算子 UNSUPPORTED` UserWarning。

## P0#11+#12（task #39）dry-run ladder + streaming sink + checkpoint + 失败分类 ✅

新增（P0#11 100/1k/5k/20k/50k/100k ladder；P0#12 streaming sink + checkpoint + §73 失败分类）：

> R61-P0 #58：原 `benchmarks/dry_run_ladder.py` 与 `scripts/dry_run_ladder.py` 已由
> **`scripts/production_factor_ladder.py`** 取代（真实 62 算子 FE-native 批量 ladder，每级一次
> `engine.run_many(factors, enable_cse=True)`，`LadderRoot` 分层、>95% 结构唯一根、强制共享子树
> `ts_mean(close,20)`/`ts_std(return,20)`、真实 `--workers`/`--per-factor-timeout`、失败六分类
> 保留真实错误、`DryRunCheckpoint` 续跑、`StreamingSink` 5000 行 shard）。两个旧文件保留为
> **thin compat shim**（DeprecationWarning + 委托），删除前请 grep 引用。

- `scripts/production_factor_ladder.py` —— FE-native 批量 ladder（见上）；状态写
  `/tmp/r61_ladder58/ladder_status.json`；CI 用 `--dry-level 100`（~1 min，20x80 面板）。
- `benchmarks/dry_run_ladder.py` / `scripts/dry_run_ladder.py` —— DEPRECATED thin shim。
- `storage/streaming_sink.py` —— 分片 pyarrow `ParquetWriter` 流式落盘，禁止 `dict[factor]=full_df` 一站式大内存；
  root done→validate→encode→partition write→release buffer；每 shard 写 sidecar（num_rows/bytes/checksum）。
- `runtime/dry_run_checkpoint.py` —— campaign manifest / completed / failed / partition completion；atomic `os.replace`；
  断点续跑只算 pending（skips completed+failed）。
- `runtime/failure_classification.py` —— GO §73 11 code（INVALID_FORMULA/FIELD_CONTRACT/PIT_VIOLATION/BACKEND_PARITY/
  NUMERIC/DATA_MISSING/OOM/TIMEOUT/IO/WRITE/INTERNAL）→ 4 bucket（contract-violation / data-degeneracy /
  resource-exhaustion / internal-bug）；禁止 `except Exception: continue` 全吞，未知裸异常归 INTERNAL。
- `tests/test_production_factor_ladder.py` —— 生成不变量/小级别端到端 batch+CSE/checkpoint resume/
  失败不丢/workers+timeout 强制；`tests/test_dry_run_ladder.py` 保留对拍 + compat。

smoke（R61-P0 #58，`--dry-level 100 --panel-stocks 20 --panel-days 80`）：
- level 100（57 roots，9 families）：43 passed / 14 failed（all-NaN semantic，intraday 算子跑日频面板，
  真实失败保留），wall 63s，shared_nodes 9，reuse_edges 75，workers 4（governance clamp），
  0 timeout / 0 OOM。pool 1000 roots 100% 结构唯一。

## Notes
- 23k FE 回归（任务 #29, pid 2326486）与战役并行跑，勿互相干扰；其失败甄别与 p0 改动可能交叉。
- 无 destructive git；evidence YAML 补在 evidence/r2/。

## 2026-09-03 下午进度（P0#1 完成 + P0#2 进行中）

- P0#1 ✅：inventory 全套 artifacts + BACKEND_COVERAGE_CURRENT_HEAD.md + operator_matrix.csv。
  production_admitted 0→86（双根因修复：certification_source 后缀匹配 + R47 64hex 严格校验回退 +
  spec_completion.py load_all 尾部回填 + SqlCapableOperator spec 缓存 + TSMeanNative spec 补齐）
- P0#2 🔄：agent_direct_allowlist.json 已产出（62 terminal admitted；86 admitted 总数）。
  发现并修复 group_winsorize NameError（fallback_policy 缺参，common/group.py:784）→ 带 group 输出 150/150 finite，nan policy 全 NaN 诚实
- 甄别结论：test_production_fastpath_tiers 3 fail 与 clean HEAD 完全一致（pre-existing，DUCKDB_REAL_SQL_VERIFIED 为空属于 evidence 漂移问题，已在 GO 风险登记）
- smoke 结果：62 terminal 中 33 直接算通、14 缺参跳过（多输入算子）、15 报错——其中 group_winsorize 已修，其余多为 smoke 脚本调用方式问题（非算子 bug），待复核

## 2026-09-03 晚进度（P0#5 完成）

- P0#5 ✅（task #34）：`IntradayFeatureCompiler.compute_many` 一次 scan 批量算分钟特征。
  - 新增 `runtime/intraday_aggregator.py` 内 `IntradayFeatureCompiler`（fields/timezone/session 构造 + `compute_many(features, dates, universe, panels|source)`）。
  - 3 个 grid-routed 算子（intra_session_mean_reversion / intra_price_delay / intra_volume_imbalance）共享一次 `_core._grid3` 物化；`_core` 的 `daily_agg{,_two,_three}` 与 3 个 vec kernel 新增 `_grid` kwarg（standalone 路径不变，向后兼容）。
  - `scan_count` 计数器证明单 scan：3 算子 compute_many → scan_count==1。
  - 测试 `tests/runtime/test_intraday_compute_many.py`（11 passed）：单 scan 等价 vs per-op（NaN 缺口+全 NaN 日，rtol/atol 1e-12）、scan 计数、空面板/缺 source/未知算子边界。
  - 回归：intraday 子集 86 passed（aggregator + perf1_equiv + vec_equiv + intraday/ + topology + lqtp_anchor）；runtime 目录后台跑。
  - 文档：`runtime/README.md` 服务层表新增 intraday_aggregator 行。

## 2026-09-03 傍晚进度（P0#9 完成）

- P0#9 ✅（task #37）：多 worker 资源治理。
  - 新增 `runtime/multiworker_governance.py`：默认单主进程内部并发（不隐式 spawn）；
    `FACTOR_ENGINE_COEXIST` 正整数=共存进程数，`FACTOR_ENGINE_RESOURCE_PROFILE`
    `solo`→31 / `shared2`→15 / `shared4`→7；每进程配额 `floor(31/coexist)`。
  - `runtime/batch_service.py` `execute_run_many_parallel` 接入：worker 被共存配额
    clamp 后重算 DuckDB/Polars 线程，保持 `workers×threads<=budget`，启动只打一次日志。
  - 新增 `tests/test_multiworker_governance.py`（7 用例）+ perf_config 4-key 缓存回归护栏。
  - 测试：`tests/test_multiworker_governance.py tests/backend/test_perf_config_env_cache.py`
    → 13 passed。
  - 甄别：`tests/runtime/test_execution_resources.py::test_memory_genuinely_limits_workers`
    1 fail 为 **pre-existing**（该测试与 subject 文件 `execution_resources.py` 均未被我改动；
    失败根因是 host 32 核 + 16GiB 内存下 `process_budget=12GiB // 1GiB peak = 12` 而非 32，
    属内存硬约束主导，与 P0#9 无关）。

## 2026-09-03 P0#10 A-share contract gate ✅

- 新模块 `factor_engine/ashare_contract_gate.py`：`validate_contract(operator_name, panel_meta) -> ContractVerdict(failures, warnings)` + `assert_contract`（fail-closed 抛 `AShareContractViolation`）。
- 四类检查（FAIL=硬 / WARN=软）：
  - (a) return-unit：return 字段必须 ratio/percent/basis_point；`dimensionless` 等非 return 单位=FAIL；带 value 时校验量级（decimal 0.05 标 bp=FAIL）；无单位=WARN。
  - (b) minute session：Asia/Shanghai 09:30-11:30/13:00-15:00；lunch-gap(12:00)/overnight/周末 bar=FAIL；非边界(09:31)=WARN。
  - (c) PIT no-lookahead：asof>decision=FAIL；null 时间戳 fail-closed=FAIL。
  - (d) membership PIT：industry/index/relation 无 PIT key（current membership）=FAIL；有 key 非 join key=WARN。
- 接线：`ir/analyzer.py` `Analyzer.lower()` 编译路径新增 `validate_ashare_return_unit_contracts(referenced_fields)` 硬门（非绕过），return 字段单位错在编译期 fail-closed。
- 测试 `tests/test_ashare_contract_gate.py`：20 passed（含 analyzer hook 2 例）。`tests/ir/` 19 passed、`tests/pit/` 15 passed 无回归。
- 验证：真实 catalog 9 个 return 字段 gate 0 误报；`Analyzer(production=True, market='ashare').lower(ts_mean(field('Return')))` 编译通过（hook 不误伤）。
- 已知 pre-existing（非本任务引入）：`tests/operator_contracts/test_manifest_sync.py` 1 fail（operator_core_specs.yaml 过期，extra=['avg2','fillna','open_close_return','ts_positive_streak']）。

## 2026-09-03 P0#8 scheduler 强制 thread vs HybridExecutor classifier 冲突 ✅

- 冲突机制：`HybridExecutor.submit(backend, fn, prefer=None)` 默认按 backend GIL 分类
  （pandas_numpy/research_python → process；duckdb_sql/polars → thread）。但 adaptive
  scheduler 在 task（`_admit_and_run`）、fusion、microbatch 三条路径都显式传
  `prefer="thread"`，无条件覆盖 classifier → Pandas CPU-bound 任务仍进 thread pool，
  GIL 下多线程扩展差（100k GO §11）。
- A/B benchmark（`tests/test_p8_ab_bench.py`，n=2e6，4 native + 4 pandas，cpu_slots=4，
  OMP=4）：
  - A: scheduler forced thread → wall=2.040s（thread=8, proc=0）
  - B: backend auto classify → wall=2.345s（thread=4, proc=4）
  - 小混合微基准下 forced-thread 略快（进程 fork/pickle 开销 > 微小 pandas 任务收益），
    但 §11 验收是**权威性**而非微基准速度：scheduler 不得无条件覆盖 executor classifier。
- 选定策略（单一权威）：默认（无 env）→ `prefer=None` → HybridExecutor classifier 权威；
  `FACTOR_ENGINE_SCHEDULER=thread|process` → scheduler 级 env 强制覆盖（运维显式权威）。
  `adaptive`（默认值）不强制。策略在 `scheduler_stats["execution_policy"]` telemetry 可见 +
  构造时日志。
- 改动：`runtime/adaptive_batch_scheduler.py` 新增 `_resolve_execution_policy()`，
  4 处 `prefer="thread"` → `prefer=self._execution_policy`，`scheduler_stats` 加
  `execution_policy` 字段。
- 测试：`tests/test_p8_scheduler_execution_policy.py`（7 passed：默认 classifier 权威 /
  env thread|process 覆盖 / adaptive 不强制 / telemetry 可见）+ `tests/test_p8_ab_bench.py`
  （1 passed）。回归：r27 shard_fusion 17 + r39 perf_scheduler 10 + r38 oom/calibration/
  read_wave/autopilot + r36 hard_gates + p011 13 + perf_config_env_cache 5 → 84 passed。
- 已知 pre-existing（非本任务引入）：`tests/r38/test_p011_resource_broker_authority.py::
  test_adaptive_scheduler_production_require_broker` 仅在**组合运行**时 fail（前序测试把
  共享 broker 留在 service ContextVar，production 下不再 raise MissingResourceBroker）；
  单独运行 13 passed，原始文件同样复现。

## 2026-09-03 P0#4 (task #33) intraday vector bind coverage ✅

- 审计：intraday 全部 daily_agg* 调用面，scalar 内核 ~15+ 处（true_gap_batch3 3 个；vwap_path 十几个 3-panel；smart_money 4 个；state_space/intra_state_space/higher_moments/pattern_recognition/time_structure 大量 1/3-panel `_kernel`）。
- 之前 bound=3（仅 true_gap_batch3），之后 bound=6。silent bind failure 已移除：`intraday/__init__.py` 去掉 `try: bind_whitelist() except Exception: pass` → 现在 import 时 fail-closed 抛错；`perf_vec_kernels.bind_whitelist()` 新增 `count_bound()` 覆盖计数 + 缺 vec 实现即抛 LookupError。
- 新增 3 个向量内核（`_core.py`，pack-finite+stable argsort 语义，harness 对拍 rtol/atol 1e-12 通过）：
  `_vec_stock_graph_features`(1-panel 对数收益 lag-1 自相关)、`_vec_common_trading_intensity`(2-panel 量峰度×集中度)、`_vec_time_above_vwap`(3-panel 累积 VWAP 占比)。
- 语义要点：pair 必须对 drop-NaN 后的连续有限值（stable argsort 打包）；lag-1 自相关是 pair-of-pair；`sw` (D,C)→`sw[:,None,:]`；price_delay 的 vol 权重按 Σvols[1:]（打包后=next-bar volumes）。
- 等价 harness `perf_vec_equiv.py` _CASES 3→6；pytest `test_perf_intra_vec_equiv.py` range 3→6；二者都加覆盖断言 `count_bound()==len(_BOUND)`，缺绑即 fail。
- 测试：`perf_vec_equiv` 6/6 PASS；pytest 7 passed（test_perf_intra_vec_equiv）；`test_smart_money` + `test_intra_state_space` + `test_true_gap_batch3` 61 passed 无回归。
- bench（4000inst×14day×240bar，OMP=31）：session_mean_reversion 1.59x / price_delay 2.45x / volume_imbalance 3.11x / stock_graph_features 2.07x / common_trading_intensity 3.58x / time_above_vwap 1.19x；TOTAL 2.02x。
- Skipped (sequential-state semantics, not byte-vectorizable without changing exact scalar behavior):
  vwap_path `_longest_streak` / `_drawdown_depth` / `_drawdown_duration` / `_drawdown_recovery_half_life` /
  `_vwap_reversion_speed` (cumulative max-tracking, consecutive-streak running count, recovery scanning) ;
  time_structure / pattern_recognition local `_kernel` closures; state_space 3-panel 2D kernels.  They keep the
  scalar path (honest) and are candidates for a dedicated sequential fast-path if 100k load shows them hot.

## 2026-09-03 P1 (task #40) operator-family correctness audit ✅

- 审计 62 terminal agent-direct admitted operators（agent_direct_allowlist.json）+ intraday/rolling families，四维：alias / DSL / hot-path / evidence。
- **alias**：270 live aliases 0 dangling / 0 self-ref / 0 collision；32 alias-bearing terminal 全部 resolve 到正确 canonical。preflight 6 处 phys-vs-DSL 差异（ts_beta/ts_topk*/ts_bottomk*/MACD_line）经 live 验证均为**增量**（额外 DSL 别名正确解析），无 impl 分歧，无需修。
- **DSL**：62/62 `resolve_canonical` OK；terminal alias 0 mismatch；sample round-trip parse 通过；62 全为 daily surface。
- **hot-path**：`perf_vec_equiv` 6/6 PASS；`count_bound()=6`（import 时 fail-closed 绑定）；`daily_agg*` 有 `__vec__` 即走 `_vec_*` 快路径。
- **evidence**：spot-check 10/72 primitive_verified 声明 0 drift；48/62 terminal 有 primitive_verified、14/62 intraday 有 intraday_minute_parity（全 present）。**发现并修复 1 处真实 drift**：committed `evidence/primitive_case_registry.json` 过期——`fillna` 有 live parity case 但缺于 3 个已含 fillna_const 的列表，`test_primitive_case_registry_matches_cases` 因此 fail。修复：仅向 polars_reference_parity/duckdb_reference_parity/duckdb_real_sql_verified 补 `fillna`（avg2/ts_positive_streak 在 pytest daily filter 下被丢弃，不可加，否则 over-declare 破坏六证交集）。
- 测试：`test_primitive_evidence_contract` 4 passed 1 skipped；alias+evidence 子集 15 passed 1 skipped；`perf_vec_equiv` 6/6 PASS。
- 产物：`artifacts/operator_audit/current_head/P1_FAMILY_AUDIT.md`；证据 `evidence/r2/P1_FAMILY_AUDIT_EVIDENCE_DRIFT.yaml`。
- 遗留：`primitive_verified.json` 未重生成（需跑完整 certify 管线，超出本审计范围）；14 intraday 依赖 intraday_minute_parity.json（evidence 绑定 present，无 drift）。

## 2026-09-03 P1 (task #40) operator-family correctness audit ✅

- 审计 62 terminal agent-direct admitted operators（agent_direct_allowlist.json）+ intraday/rolling families，四维：alias / DSL / hot-path / evidence。
- **alias**：270 live aliases 0 dangling / 0 self-ref / 0 collision；32 alias-bearing terminal 全部 resolve 到正确 canonical。preflight 6 处 phys-vs-DSL 差异（ts_beta/ts_topk*/ts_bottomk*/MACD_line）经 live 验证均为**增量**（额外 DSL 别名正确解析），无 impl 分歧，无需修。
- **DSL**：62/62 `resolve_canonical` OK；terminal alias 0 mismatch；sample round-trip parse 通过；62 全为 daily surface。
- **hot-path**：`perf_vec_equiv` 6/6 PASS；`count_bound()=6`（import 时 fail-closed 绑定）；`daily_agg*` 有 `__vec__` 即走 `_vec_*` 快路径。
- **evidence**：spot-check 10/72 primitive_verified 声明 0 drift；48/62 terminal 有 primitive_verified、14/62 intraday 有 intraday_minute_parity（全 present）。**发现并修复 1 处真实 drift**：committed `evidence/primitive_case_registry.json` 过期——`fillna` 有 live parity case 但缺于 3 个已含 fillna_const 的列表，`test_primitive_case_registry_matches_cases` 因此 fail。修复：仅向 polars_reference_parity/duckdb_reference_parity/duckdb_real_sql_verified 补 `fillna`（avg2/ts_positive_streak 在 pytest daily filter 下被丢弃，不可加，否则 over-declare 破坏六证交集）。
- 测试：`test_primitive_evidence_contract` 4 passed 1 skipped；alias+evidence 子集 15 passed 1 skipped；`perf_vec_equiv` 6/6 PASS。
- 产物：`artifacts/operator_audit/current_head/P1_FAMILY_AUDIT.md`；证据 `evidence/r2/P1_FAMILY_AUDIT_EVIDENCE_DRIFT.yaml`。
- 遗留：`primitive_verified.json` 未重生成（需跑完整 certify 管线，超出本审计范围）；14 intraday 依赖 intraday_minute_parity.json（evidence 绑定 present，无 drift）。
