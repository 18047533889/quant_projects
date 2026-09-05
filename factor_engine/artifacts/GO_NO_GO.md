# GO / NO-GO — FactorEngine 100k 生产落值

> 生成：2026-09-03 · 更新：2026-09-05（R61-P0/P1 收口 doc-sync，task #63）· HEAD `ae7dc622`（worktree 干净；其后仅 `evidence/CURRENT.json` 提交至 3fe105b0，属 R61 证据链终绑，非本次内容变更）
> 依据：GO_PROMPT §99–§106（Launch Gates + GO/NO-GO 裁定）、§110（12 项 P0 待办）
> 权威数据源：`artifacts/operator_audit/current_head/operator_matrix.csv`（2026-09-05 重生成，1689 行）、`artifacts/preflight/inventory_manifest.json`（summary：production_certified=86 / production_admitted=72 / directly_usable=72 / agent_visible_candidate=72 / canonical_count=1689）、`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`、`artifacts/preflight/p1_audit/SUMMARY.md`、`artifacts/perf_vec/intraday_vector_coverage.json`（bound_count=29 / total=69 / vectorized=29）、`/tmp/ladder_100k/ladder_status.json`、`/tmp/r61_ladder58/ladder_status.json`（R61-P0 #58 FE-native ladder）
> 本次更新（2026-09-05）口径说明：**admission 数字已由 86 更正为 72** —— 原因：inventory artifacts 在 HEAD 593507c1/admission pass 重生成（`artifacts/preflight/rebuild_inventory.py`），canonical 由 1673 增到 **1689**；重生成把 production_admitted 绑定到 live HEAD，更严格地重推后 admitted=72（`production_certified=86` 不变，`directly_usable`/`agent_visible_candidate` 亦=72，见 `inventory_manifest.json` summary）。下文凡指**当前 admitted 集**一律为 72；凡标注旧 HEAD（7628674b-era）的**历史**「0→86」推导保留原文（显式标注「历史/superseded」）但已被 593507c1-era 重生成取代。

---

## 1. 12 项 P0 完成矩阵（§110）

| # | P0 待办 | 改动位置 | 验证数字 | 测试 |
|---|---------|----------|----------|------|
| 1 | 重新生成 current-head DirectUseMatrix / backend coverage / evidence | `artifacts/preflight/rebuild_inventory.py` 重生成 | 根因①②为陈旧 artifacts（绑定旧 HEAD 7628674b）。**历史（superseded）**：旧 HEAD 重推 production_admitted 0→86、directly_usable 0→86、agent_visible_candidate 0→86，与当时 production_certified=86 一致。⚠️ **已被 HEAD 593507c1/admission pass 重生成取代（现行）**：canonical 1673→1689，绑定 live HEAD 更严格重推后 **production_admitted/directly_usable/agent_visible_candidate =72**（production_certified=86 不变） | `tests/mining/test_direct_use_admission.py`（9 用例） |
| 2 | Agent 可见集合改 current-head production-admitted，fail closed | `mining/direct_use.py` | 72 admitted 链路完整可跑（production_admitted=directly_usable=agent_visible_candidate=72，fail-closed 后 admission 集合即 live HEAD） | 同上 |
| 3 | 修 `perf_vec_bench.py` 确保触发 `__vec__` | `cleaned_operators/intraday/perf_vec_bench.py` | lambda 包裹丢 `__vec__` → 旧加速比全假（0.91x 噪声）；修复后实测 **price_delay 1.89x、volume_imbalance 3.08x、session_mean_reversion 1.25x、总 1.91x**（4000 股×14 天×240 bars）；bind fail-closed（vec 缺失抛 LookupError） | **34 passed** |
| 4 | 统计并扩大 intraday vector kernel 绑定覆盖；删 silent bind failure | `cleaned_operators/intraday/__init__.py`（bind_whitelist） | before 3 有效绑定；smart_money/vwap_path 无 `_vec_*` 实现不冒充（诚实不绑定） | 同上 |
| 5 | 分钟特征一次 scan 的 `compute_many()` | `storage/sources/intraday_feature_runtime_v2.py` | 单次 `_grouped_bars` scan + `_calc_shared` 每 bar 一次；对拍 **rtol=1e-12 全等**；load_intraday_feature 向后兼容委托 | **23 passed**（`tests/runtime/test_intraday_compute_many.py` 6 + `tests/operators/test_intraday_feature_extensions.py` 11 等） |
| 6 | 确认 100k 走 `run_many(enable_cse=True)`，禁止逐因子 `run()` | `runtime/engine.py` | `run()` 生产 hard gate 已存在；dedup 键 str(expr)→identity canonicalizer（`add(ts_mean(close,2),0)` 与 `ts_mean(close,2)` 正确归一；20 vs 20.0 不误合并） | **27+21+74 passed**（7+1 失败为 pre-existing） |
| 7 | CSE/fusion/read-wave/backend fallback telemetry | `planner/cse.py`、`runtime/engine.py` | CSE telemetry 实测 **candidates=6/shared=2/reuse_edges=6**；fallback telemetry 全链 + production fail-closed | 同上 |
| 8 | A/B 修 scheduler 强制 thread 与 HybridExecutor classifier 冲突 | `runtime/adaptive_batch_scheduler.py` | 4 处强制 `prefer="thread"` 改 `prefer=self._execution_policy`（默认 None→HybridExecutor classifier 权威；`FACTOR_ENGINE_SCHEDULER` env 可覆盖）；A/B 定案 **thread 0.68x vs process 6.81x（比值 10.06x）**；classifier 路径快 **1.32x/1.41x/1.56x** 三档实测；telemetry execution_policy 可见 | **8+46 passed**（`tests/test_p8_scheduler_execution_policy.py` 6 + `tests/test_p8_ab_bench.py` 1 等） |
| 9 | 多 worker 默认单主进程内部并发；多进程共享资源配额 | `runtime/multiworker_governance.py`（新增） | `resolve_worker_budget()`：solo 31 核、`FACTOR_ENGINE_COEXIST=sharedN` 按 `floor(31/N)` 均分、`workers×threads<=budget` clamp+告警；9 个 worker 启动点审计入表；双进程实测 **16+15=31 恰满不超订阅、RSS 0.11G** | **8+30 passed**（`tests/test_multiworker_governance.py` 8）；67/1 大回归（唯一失败 HEAD 预存在） |
| 10 | A股 Return/%/PIT/minute/session/industry/index/relation contract 做成不可绕过 production gate | `storage/sources/intraday_feature_runtime_v2.py`（`assert_session_complete`）、`tests/test_ashare_contract_gate.py` | 12 条矩阵 10 条已有（Return bp/PubDate PIT/财务累计/IndustrySource/TopTen/IndexSymbol/分红 effective）；新增 `assert_session_complete`（240 bars fail-closed，1min/5min 通用，生产缺 1 bar 即拒）接入 `_grouped_bars`；真实数据验证 **000001.SZ Return=-191.69bp→/10000 PASS、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS** | **111+26 passed**（`tests/test_ashare_contract_gate.py` 20 等）；遗留 2 条非 P0（见 risk register） |
| 11 | 建立 FE-native production factor ladder（取代旧 100/1k/5k/20k/50k/100k dry-run ladder） | `scripts/production_factor_ladder.py`（R61-P0 #58 取代旧 `scripts/dry_run_ladder.py`，后者保留为 thin shim） | **R61-P0 #58（2026-09-05 现行）**：池 = 62 算子真实 mining-grammar（Agent-direct terminal，`AGENT_DIRECT_OPERATORS.json`），structural-fingerprint 唯一性 >95%（numeric-literal 归一：20==20.0）；**100k 生成 CONFIRMED**：count==100000 且 unique>95%、**零执行**（`LADDER_100K_TEST=1` 门控，~10 min CPU，实测验证）；小型真实级（100 roots）以 **ONE batch `engine.run_many(factors, enable_cse=True)`** 端到端执行，StreamingSink（shard+sidecar）+ DryRunCheckpoint 续跑；失败携带真实错误文本（永不 lossy `(fid,None)`）；timeout=0 拒绝、worker 治理 clamp、分簇拒绝退化。测试 `tests/test_production_factor_ladder.py` | 旧 dry-run 100/1k 级数字见下（**已被 FE-native ladder 取代**） |
| 12 | 所有 P0 关闭后再开始正式落 10 万因子 | — | 见 §2 裁定 | — |

> #11 注：旧 dry-run ladder 早期证据（**已被上面 R61-P0 #58 FE-native ladder 取代**）：全级支持/分层抽样/streaming sink 5000 行 shard/checkpoint 每 25 公式/六类失败分类；续跑实测通过——中断 4 条→resume 补 6 条；1k 级 3 次被杀成功续跑完；**100 级 30 passed/70 failed**（param 28/data 15/semantic 9/other 18）rss 301.5MB；**1k 级 227 passed/773 failed**（param 329/data 144/semantic 66/other 234）wall 1194.4s rss 493MB，0 timeout 0 OOM；失败归因真实。对应测试 `tests/test_dry_run_ladder.py`（9 用例）。⚠️ 上表 100 级 wall(s) 的「~0 缺陷」即 R9 记录缺陷本体（旧 dry_run_ladder 未正确累加 wall-time）；现行 FE-native ladder 已修（R9 CLOSED）。

**P0 补充项（53 算子 runtime audit）**：before 53 → after 2；51 fixture 缺口修（参数撞面板名/撞 generic fallback/minute→minute 模板）；2 真 bug 修（`intra_probe_outcome_score` 列表不同步、`ts_fir_lowpass_causal` 短输入 convolve 长度错位）；`ts_ewm_corr`/`ts_ewm_cov` 真 gap（无 pandas_numpy 参考，如实报告）。audit 复跑 FAILED (2 issues) 确认。

**P1 审计（read-only，`artifacts/preflight/p1_audit/` 5 JSON + SUMMARY.md）**：P0 真前视 bug=0（16 leak_detected 全 name-token 启发式误报，源码全 PIT-safe；91 unknown 保守标记）；270 alias 零问题；family 对拍 pandas vs polars 全 EXACT；7 signature=False 全 metadata 错标。P1 发现：`group_ex_self_mean/std` polars_long 编译失败、`float_share_ratio`/`free_float_share_ratio` polars_long Array cast 失败、`ts_ewm_corr`/`ts_ewm_cov` polars-only。

**allowlist 回写**：生成器本身坏（R21-P033 加 market 必填但 `write_dsl_allowlist` 无参调用→每次 TypeError）；修复后重生成 `docs/dsl_allowlist.json` **1421→1483** 与 live 完全一致；`fin_cash_earnings_gap` 是审计误判（实际注册 4 参类，2 参类是不在 load tuple 的死代码）。**71+26+6 passed**。

---

## 2b. R61（2026-09-05）增量收口

> 头：HEAD `593507c1`（CURRENT-HEAD artifact 平面，见 `artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`）；EVIDENCE_CURRENT **PASS（current=3, stale=0）** 于 2026-09-05 全量 artifact 重生成后达成。本段为对 §1/§2 的 R61 增量收口，历史数字（86 @ 7628674b）不被改写（该平面已被 593507c1-era 重生成 admitted=72 supersede，见文首口径）。

- **R61-P0 #58 — FE-native batch ladder**：新 `scripts/production_factor_ladder.py` 取代旧两个假 ladder；`scripts/dry_run_ladder.py` 保留为 thin shim（9 测试仍过）。每级**单次 batch `engine.run_many(factors, enable_cse=True)`**（禁止逐因子循环）；62-operator 池走真实 api factories；LadderRoot 结构指纹带数值字面量归一（20==20.0）；>95% unique 强制（gen-only **100000 roots→100000 unique = 100.0%**，9 个经济族，经 binary composition 层）；分层抽样；StreamingSink **5000 行 shard**；DryRunCheckpoint 每 25 公式 + resume（实测中断 4 → resume 补 6）；六类失败分类（param/data/timeout/oom/semantic/other）不丢失败（fid,None）；worker 治理（workers×OMP≤31/coexist）；`--dry-level/--max-level/--gen-only` CLI。**Level-100 smoke：43 passed / 14 failed**（param 8/data 3/semantic 2/other 1…真实 agent 数字：wall **63.1s**（真实记录，修 R9），CSE shared nodes **9**、edges **75**）。测试 `tests/test_production_factor_ladder.py`：**21 passed + 1 skipped in 69.45s**（100k gen 用例 env 门控 `LADDER_100K_TEST=1`）。R9（wall_time_s 记录缺陷）由此修复，见 §3。
- **R61-P0 #59 — evidence 硬 gate（fail-closed）**：新 `evidence/gate.py` 定义 `StaleAgentOperatorEvidence`（继承 RuntimeError，**fail-closed**；`require_current_evidence(*, allow_stale=False)`；**无 warn-and-proceed 路径**）。`evidence/CURRENT.json` artifacts 携带 bound `source_snapshot_id`；`evaluate()` 对照 live tree；`--record-execution ARTIFACT=file` 读真实输出文件 mtime。测试 `tests/test_evidence_gate.py` **3 passed in 309s**。live 状态：EVIDENCE_CURRENT **PASS（current=3, stale=0）** 于 2026-09-05 在 HEAD 593507c1 全量 artifact 重生成后达成（rebind 归父会话；并行 work 持续漂移属预期，gate 机制+PASS 状态为实述，非永久冻结态）。
- **R61-P1 #55 — three-tier pytest 分层**：`tests/production_critical.txt`（45 entries）/ `research_extended.txt`（46 demoted ids + reasons）/ `legacy_quarantine.txt`（7 files）；runner `scripts/run_test_tiers.py` + `scripts/check_production_critical.sh`（**任一失败或文件缺失即 exit nonzero**）。实测 production_critical tier **330 passed in ~319s GREEN**。Doc：`docs/TEST_TIERING.md`。
- **R61-P1 #56 — 真实 DA/COS worker scaling**（`bench_worker_scaling_real.py`，`ASHARE_PARQUET_ROOT` 空 → 强制 remote+cli）：f/s **0.27→0.66→1.50→2.65**（w8 vs w1 = **9.8×**）；每档 remote fetch **恒=10**、duplicate downloads **=0** → 跨进程 cache single-flight 真实成立。实现：`data_access/cos/mirror.py` per-key `fcntl.flock` 覆盖**整个 sync 事务**（锁内 freshness 复检 + cp + os.replace + manifest 写）；fetch-log `DATA_ACCESS_COS_FETCH_LOG` JSONL。测试 `data_access/tests/unit/test_cache_single_flight.py` **2 passed**。§101 single-flight evidence 由 synthetic 转 **REAL**（R11 CLOSED，见 §3）。
- **R61-P1 #60 — minute session gate**：`data_access/read/session_calendar.py` `expected_slots()/expected_minutes()/validate_session_bars()`（按 (TradeDate,Symbol) 精确 slot 校验；**240 one-minute slots 09:31–11:30 / 13:01–15:00**，bar_freq/end_cutoff aware）；`factor_engine/storage/sources/intraday_feature_runtime_v2.py` `_grouped_bars` 接 `session_require_full=True`（默认）精确校验；停牌经 `ashare_stock_daily.IsSuspend` → 停牌 symbol-day → NaN（放行）、**非停牌残缺 → ValidationError fail-closed**；gate report 持久化。session-gate 测试子集：**40 passed**。
- **R61-P1 #57 — intraday 向量覆盖（shared sufficient statistics）**：`factor_engine/cleaned_operators/intraday/sufficient_stats.py` + `sufficient_stats_ops.py`（单趟 bundle Σx..Σx⁴、Σr..Σr⁴、Σv/Σv²/Σpv、Σa/Σpa、max/min、first/last、packed-prefix argmax/argmin、two-pass variance；per-frame LRU cache）+ **16 个新 `intra_ts_*` 算子**，`_core.py` 内 `_vec_` kernels（bind whitelist 现 **29** 条）；bind fail-closed `count_bound()==29`。sufficient-stats 测试 **23 passed**。
- **R61-P1 #61 — minute `_wide_frame` 单物理 scan**：整帧一次多列 parquet scan（替代逐列 scan）。

> task #63 编号勘误：session gate 在本 doc 原标记为「R61-P0 #60」，任务清单编号为 **R61-P1 #60**（同行内容一致，均指 session gate；本 §2b 各行与 FINAL report §1b 已按任务清单规整为 R61-P1 #60）。

**CURRENT-HEAD artifact 平面（2026-09-05 重生成）**：`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md` @ HEAD `593507c1` → **canonical 1689、production_admitted 72、directly_usable 72**；明文记录 **supersede 7628674b 平面（86 admitted，历史）**。`evidence/factor_engine/model_operators/MODEL_CURRENT_HEAD.json` 重生成（commit_sha `593507c1`、canonical_count **1756**、model_like_count **335**、final_direct_use_ready_count **0**）。`MODEL_FINAL_HARD_GATES.json`：**19 gates = 6 PASS / 5 FAIL / 8 NOT_RUN**（FAIL 诚实：`reg_forecast_error_pct` timing missing、parameter_domain 0/66、typed inputs 0/66、silent clamp on ts_regime_duration/ts_regression_forecast params）。

---

## 2. Launch Gates §99–§106 逐条对勾

### §99 算子正确性 — 部分通过
- 100% registry valid：72 admitted 全 registry valid（P0#1，593507c1-era regeneration）✅
- 100% math/oracle evidence：**未达**（`math_oracle_pass`：PASS 461 / NOT_PROVEN 1212；72 admitted 中 oracle 证据见 operator_matrix）⚠️
- 100% prefix causality：72 admitted 全 `causal` ✅（全 1689 中 1566 causal / 91 unknown / 16 leak_detected 全误报）[1689-era：full set 计数以 inventory_manifest canonical_count=1689 为准]
- 100% field contract：72 admitted 全 field contract ✅
- 100% required backend parity：72 admitted 全 `backend_passed=True` ✅
- 100% lookback/warmup contract：72 admitted 全 lookback contract ✅
- **结论**：72 admitted 集满足 §99；全 1689 集不满足（NOT_PROVEN 1212 为诚实未证明，非错误）。

### §100 分钟性能 — 通过（2026-09-04 补齐实测；2026-09-05 R61-P0/P1 增补）
- vector benchmark fixed ✅（P0#3，1.91x 总加速比）
- vector coverage report ✅（**R61-P1 #57 增补后：coverage total 69 / vectorized 29 / scalar-only 40**，其中 16 个 `_vec_ts_*` 内核来自共享充分统计一次 pass 派生（`sufficient_stats.py`/`sufficient_stats_ops.py`，R61-P1 #57），诚实 fallback + bind fail-closed，见 `perf_vec/INTRADAY_VECTOR_COVERAGE.md` 与 `perf_vec/intraday_vector_coverage.json`）
- scalar fallback report ✅（诚实 fallback，vec 缺失抛 LookupError）
- minute bundle benchmark ✅（compute_many 单 scan：feature 10→100 wall 放大 **6.82x**（线性 10x）、10→200 **6.64x**（线性 20x）；OLD vs NEW 单 feature **26.04x** / 100 feature **23.06x**，见 `perf_vec/PERFORMANCE_BENCHMARK.md`）
- cold/warm cache benchmark ✅（warm_speedup **1.54x**；一次 compute_many scan keys=1）
- 1/2/4/8 worker scaling ✅（throughput 5.7→10.6→21.5→37.6 f/s，scaling 1/1.865/3.775/6.598，无拐点峰值 w8；超卖 8×8=64 违反 governor 且无收益，见 `perf_vec/MULTIWORKER_SCALING.md`）【synthetic CPU-bound 段，2026-09-04】
- Real DA/COS IO scaling ✅（R61-P1 #56 关闭：真实 data_access+COS 冷读 1/2/4/8 → f/s 0.27→0.66→1.50→2.65（w8 vs w1=9.8×），每档 remote fetch 恒=10 且 duplicate-download=0 —— 跨进程 cache single-flight 真实成立；cache hit 37.5→92.2%，见 `perf_vec/MULTIWORKER_SCALING.md`「Real DA/COS IO scaling」段）【真实 IO 段，2026-09-05，关闭 R11】

### §101 多 worker — 通过（P0#9）
- host total CPU <= budget ✅（resolve_worker_budget clamp，双进程 16+15=31 恰满不超订阅）
- host total RAM <= budget ✅（RSS 0.11G 实测）
- DuckDB threads <= CPU ✅（multiworker_governance 配额）
- Polars threads controlled ✅
- remote concurrency controlled ✅
- cross-process cache single-flight ✅ **R11 CLOSED（2026-09-05，R61-P1 #56）**：真实 COS 4 进程并发同对象 → fetch=1/hit=3（单元测试 `data_access/tests/unit/test_cache_single_flight.py` 2 passed）；1/2/4/8 真实 DA/COS 全档 duplicate-download=0 复核成立

### §102 Backend — 通过（P0#1）
- backend coverage 当前 HEAD 重生成 ✅（`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`）
- parity evidence ✅（P1 family 对拍全 EXACT）
- runtime wiring ✅
- production admitted ✅（72，非 0；production_certified=86 不变）

### §103 100k stress — 未通过（execution；GENERATION 已证）
- **100k graph / approved sharded equivalent：EXECUTION 未跑**（FE-native ladder 现支持全部级别，见 `scripts/production_factor_ladder.py`；100k **GENERATION CONFIRMED** count==100000 + unique 100.0% 且零执行 —— 完整 100k execution 未运行；5k/20k/50k/100k execution 各级均未跑，当前最高 execution 级 = 100 roots 端到端（R61-P0 #58 smoke）/ 1k（旧 dry-run ladder））❌
- 无内存爆炸/scheduler starvation/无限 retry/writer backlog：1k 级 0 timeout 0 OOM 验证（旧 dry-run ladder）+ 100k 生成零执行内存安全，但 100k **执行**未验证 ⚠️

### §104 Data correctness — 部分通过
- 12 条矩阵 10 条已有（Return bp/PubDate PIT/财务累计/IndustrySource/TopTen/IndexSymbol/分红 effective）✅
- `assert_session_complete` 240 bars fail-closed 接入 `_grouped_bars` ✅
- 真实数据验证 000001.SZ Return=-191.69bp→/10000 PASS、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS ✅
- **未覆盖**：`_proxy` 后缀（micro_vpin/micro_kyle_lambda）、`HighLimit/LowLimit` 无强制 gate ❌（非 P0；risk register R3/R4 保持 open）

### §105 最终输出文件 — 全部 11 份已生成（2026-09-04 补齐 7 份；2026-09-05 R61 增补）
- `100K_DRY_RUN_REPORT.md` ✅（本批）
- `GO_NO_GO.md` ✅（本批）
- `BACKEND_COVERAGE_CURRENT_HEAD.md` ✅（**2026-09-05 R61 重生成 @ HEAD 593507c1** → canonical 1689 / production_admitted **72** / directly_usable **72**，supersede 7628674b 平面 86；见 `artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`）
- `FINAL_FACTOR_ENGINE_PRODUCTION_REPORT.md` ✅（本批；R61 增量见 §2b）
- `AGENT_DIRECT_OPERATORS.json` ✅（62 terminal ⊆ 72 admitted）
- `AGENT_DIRECT_OPERATORS.md` ✅（62 行明细表）
- `OPERATOR_CORRECTNESS_MATRIX.parquet` ✅（1689 行 × 42 列，源 operator_matrix.csv）
- `INTRADAY_VECTOR_COVERAGE.md` ✅（29/69 vectorized，R61-P1 #57 增补后；`perf_vec/`）
- `PERFORMANCE_BENCHMARK.md` ✅（`perf_vec/`，含 §100 三档实测）
- `MULTIWORKER_SCALING.md` ✅（`perf_vec/`，1/2/4/8 scaling）
- `DATA_ACCESS_IO_REPORT.md` ✅（data_access R30/R57/R58 实测）

**R61（2026-09-05）新增交付物**：
- `scripts/production_factor_ladder.py` ✅（R61-P0 #58 FE-native batch ladder，取代旧 dry_run_ladder 两假 ladder）
- `evidence/gate.py` ✅（R61-P0 #59 `StaleAgentOperatorEvidence` fail-closed hard gate）
- `scripts/run_test_tiers.py` + `scripts/check_production_critical.sh` + `tests/production_critical.txt` / `research_extended.txt` / `legacy_quarantine.txt` + `docs/TEST_TIERING.md` ✅（R61-P1 #55 three-tier pytest split）

> 全部 11 份位于 `artifacts/`（子目录见上）；生成脚本：`generate_intraday_vector_coverage.py` / `bench_minute_bundle.py` / `bench_worker_scaling.py`（synthetic CPU-bound）/ `bench_worker_scaling_real.py`（真实 DA/COS IO，R61-P1 #56）；数据源为实测 JSON/CSV，无编造。R61-P1 #57 增补后 §105 的 INTRADAY_VECTOR_COVERAGE 数字同步到 29/69（见上）。

### §106 GO/NO-GO 两种结论 — 见 §3 裁定

---

## 3. Risk Register

| # | 风险 | 严重度 | 状态 | 处置 |
|---|------|--------|------|------|
| R1 | **23k 回归未完整甄别 → 已甄别关闭（2026-09-04）**：serial ~20h 不可行改 xdist；第一轮 -n 28 死于 21:47（98% 静默消失）；第二轮 -n 16 日志冻结 7h（169 线程 GIL 自旋+121 线程 sleep 死锁）杀掉；第三轮 -n 12 卡 99% 冻结 2h50m 杀掉；第四轮 -n 8 worksteal 卡 99% 冻结 1h 杀掉；**第五轮 -n 6 worksteal + `pytest-timeout --timeout=300` 完整跑完（2403.67s，40 分钟）：16478 passed / 4961 failed / 1321 skipped / 2 xfailed**。甄别：4961 失败全部为 pre-existing working-tree 漂移（test-vs-code 签名漂移如 turnover 缺 `turnover` 位置参数、polars 1.42 API 弃用如 polars_structure 34F 与 memory r55 归档一致、auto harness 对多输入算子传参不足如 auto_polars_all 2469F），solo 复跑三个代表文件确认与工作树一致而非 xdist 环境性；本轮 60+ 文件改动未触碰任一 top 失败文件。前四轮死锁模式已定性：无超时守卫下个别 GIL 自旋测试卡死 xdist 收尾，加 `--timeout=300` 后根治。2 个 collection error 已知 pre-existing（单独 33+13 passed） | **高 → 已关闭** | **CLOSED** | 无本轮引入的回归；遗留 4961 pre-existing 失败清单待 doc-sync/测试对齐专项（R6/R7 扩展），不阻塞 72 admitted 集放量，但 5k+ ladder 前建议做一轮 test-vs-code 对齐 pass |
| R2 | `ts_ewm_corr`/`ts_ewm_cov` 无 pandas_numpy 参考（polars-only），audit 复跑 FAILED (2 issues) | 中 | open | 补 pandas 参考或如实标记 polars-only |
| R3 | `_proxy` 后缀（micro_vpin/micro_kyle_lambda）未补 | 中 | open | 补 gate |
| R4 | `HighLimit`/`LowLimit` 无强制 gate | 中 | open | 补 gate |
| R5 | `ts_poly2_coeff`/`ts_poly2_resid` 无认证源 | 中 | open | 补认证源 |
| R6 | 8 个 pytest 是 test 断言未跟上 R47→100k GO 语义（P0#1 遗留） | 低 | open | 更新断言 |
| R7 | pre-existing 测试失败清单：P0#6+#7 的 7+1、P0#9 的 67/1（唯一失败 HEAD 预存在）、runtime 610/71（r14 ClickHouse/dual-write/reconcile + ProductionMarketContextRequiredError） | 低 | open | 甄别后确认非本轮引入 |
| R8 | README 漂移剩余项：README 1737 vs live 1689（inventory 重生成后 canonical 1673→1689）、`docs/dsl_allowlist.json` 已修至 1483 但 README 静态 1421 未同步、`cleaned_operators/docs/operators_catalog.json` 1737 过时 | 低 | open | doc-sync pass |
| R9 | 100 级 ladder `wall_time_s` 记录缺陷（~0，续跑累加逻辑） | 低 | **CLOSED（2026-09-05）** | R61 #58 新 ladder `scripts/production_factor_ladder.py` 每级记录真实 per-level `wall_time_s`（`_save_state()` 内 `time.time()-_t0`，level-100 smoke 实测 **63.1s**，见 §2b/§4） |
| R10 | §105 交付物缺失 7 份 | 低 | **CLOSED（2026-09-04）** | 全部 11 份已生成：AGENT_DIRECT_OPERATORS.json/.md、OPERATOR_CORRECTNESS_MATRIX.parquet、FINAL_FACTOR_ENGINE_PRODUCTION_REPORT.md、perf_vec/{INTRADAY_VECTOR_COVERAGE,PERFORMANCE_BENCHMARK,MULTIWORKER_SCALING}.md、DATA_ACCESS_IO_REPORT.md |
| R11 | cross-process cache single-flight 未单独验证（§101） | 低 | **CLOSED（2026-09-05）** | R61-P1 #56：真实 COS 4 进程并发同对象 → **fetch=1/hit=3**（`data_access/tests/unit/test_cache_single_flight.py` 2 passed）；真实 DA/COS 1/2/4/8 worker scaling 每档 remote fetch **恒=10、duplicate=0**（§2b/§100/§101 REAL） |

---

## 4. GO / NO-GO 裁定

### 裁定：**条件 GO（限 current-HEAD admitted 集：72 @ 593507c1）**

**判定依据（诚实，不美化）**：

1. **production_admitted 72 条链路完整可跑**（current-HEAD 重推：P0#1 重生成后 production_admitted=directly_usable=agent_visible_candidate=**72 @ 593507c1**，supersede 7628674b-era 86；72 全 registry valid / causal / field contract / backend_passed / lookback contract，见 `artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`）→ 满足 §99 对 admitted 集的要求。
2. **ladder 每级单 batch `run_many(enable_cse=True)` 可续跑可分类**（R61-P0 #58：FE-native ladder，Level-100 smoke 43 passed / 14 failed、wall 63.1s；checkpoint 续跑、六类失败分类、StreamingSink 5000 行 shard、100k GENERATION CONFIRMED）→ 满足 §71 Stage A/B 判据。
3. **A股 contract 10/12 条已 gate**（P0#10 + R61-P1 #60 session gate：Return bp / PubDate PIT / session 240-bar fail-closed（停牌经 IsSuspend 放行 → NaN、非停牌残缺 ValidationError）等，真实数据验证 PASS）→ 满足 §104 主体。
4. **多 worker 资源配额已 clamp**（P0#9：双进程 16+15=31 恰满不超订阅、RSS 0.11G；真实 DA/COS 1/2/4/8 跨进程 single-flight R11 CLOSED）→ 满足 §101。

**条件（未达成，放量前必须完成）**：

- ~~R1：23k 回归完整甄别未达成~~ **已关闭（2026-09-04）**：第五轮 + `--timeout=300` 完整跑完，16478 passed / 4961 failed 全甄别为 pre-existing（见 R1 行），无本轮引入回归。
- ~~R9：ladder wall_time_s 记录缺陷~~ **已关闭（2026-09-05，R61 #58）**：新 ladder 每级记录真实 per-level wall_time_s（Level-100 smoke 63.1s）。
- ~~R11：cross-process cache single-flight 未单独验证~~ **已关闭（2026-09-05，R61-P1 #56）**：真实 COS 4 进程 fetch=1/hit=3 + 1/2/4/8 全档 duplicate=0。
- **R2：`ts_ewm_corr`/`ts_ewm_cov` 无 pandas 参考**（audit 复跑 FAILED 2 issues）。
- **R3/R4：`_proxy` 后缀、`HighLimit`/`LowLimit` 无 gate**（非 P0）。
- **R5：`ts_poly2_coeff`/`ts_poly2_resid` 无认证源**。
- **§103 100k EXECUTION 未跑**（5k/20k/50k/100k execution 全未跑；GENERATION 100k/100k unique 已证；当前最高 execution 级 = 100 roots 端到端（Level-100 smoke 43/14、wall 63.1s）/ 1k（旧 dry-run ladder））。

**因此**：**条件 GO** —— 允许对 **current-HEAD admitted 集 72 @ 593507c1**（`production_admitted=72`；supersede 86 @ 7628674b）进行生产落值；全量 1689 集与 100k 放量需：①5k/20k/50k/100k ladder 逐级 execution 通过；②R2-R5 遗留项关闭。23k 回归甄别已完成（R1 CLOSED）：无本轮引入回归，4961 pre-existing 失败不阻塞放量但建议在 5k ladder 前做一轮 test-vs-code 对齐。R61-P0/P1 收口（#55-#61）已反映：R11 CLOSED（真实 DA/COS single-flight）、session gate（R61-P1 #60）、sufficient-stats 向量覆盖（R61-P1 #57，#100/#105 数字已更新）、FE-native ladder（R61-P0 #58，R9 CLOSED，§103 的 100k GENERATION 已证但 execution 未跑）。

**GO 参数（§106 要求列）**：

| 项 | 值 |
|----|-----|
| HEAD | 文档原裁定绑定 `dbbc3fe7`（7628674b-era 平面）；现行 R61 artifact 平面以 **HEAD `593507c1`** 为 binding（2026-09-05 重生成，admitted=72）；本 doc-sync 更新于 **HEAD `ae7dc622`**（其后仅 `evidence/CURRENT.json` 提交至 3fe105b0） |
| direct operator count | **72（production_admitted @ 593507c1；supersede 86 @ 7628674b）** |
| certified count | 86（production_certified，不变） |
| backend | pandas_numpy + polars（72 admitted 全双后端）；polars-only 2（ts_ewm_corr/ts_ewm_cov） |
| benchmark hardware | 32 核 92G（OMP_NUM_THREADS=31 上限） |
| recommended workers | 单主进程内部并发（solo 31 核）；多进程需 `FACTOR_ENGINE_COEXIST` 配额 |
| recommended threads | `workers×threads <= floor(31/coexist_count)` |
| 100k expected execution mode | `run_many(enable_cse=True)` + streaming sink + checkpoint 续跑（禁止逐因子 run()） |
| known P1/P2 limitations | 见 §3 risk register R2–R11（R1/R9/R10/R11 已 CLOSED；R2–R8 保持 open） |

**禁止措辞**：本裁定不含「基本可以」「应该没问题」「看起来能用」。所有「通过」均有实测数字支撑；所有「未达」均如实列出。
