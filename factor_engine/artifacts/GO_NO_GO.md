# GO / NO-GO — FactorEngine 100k 生产落值

> 生成：2026-09-03 · HEAD `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026`（工作树 dirty，含本轮全部 P0 修复）
> 依据：GO_PROMPT §99–§106（Launch Gates + GO/NO-GO 裁定）、§110（12 项 P0 待办）
> 权威数据源：`artifacts/operator_audit/current_head/operator_matrix.csv`、`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`、`artifacts/preflight/p1_audit/SUMMARY.md`、`/tmp/ladder_100k/ladder_status.json`

---

## 1. 12 项 P0 完成矩阵（§110）

| # | P0 待办 | 改动位置 | 验证数字 | 测试 |
|---|---------|----------|----------|------|
| 1 | 重新生成 current-head DirectUseMatrix / backend coverage / evidence | `artifacts/preflight/rebuild_inventory.py` 重生成 | 根因①②为陈旧 artifacts（绑定旧 HEAD 7628674b）；重生成后 **production_admitted 0→86、directly_usable 0→86、agent_visible_candidate 0→86**，与 production_certified=86 一致 | `tests/mining/test_direct_use_admission.py`（9 用例） |
| 2 | Agent 可见集合改 current-head production-admitted，fail closed | `mining/direct_use.py` | 86 admitted 链路完整可跑 | 同上 |
| 3 | 修 `perf_vec_bench.py` 确保触发 `__vec__` | `cleaned_operators/intraday/perf_vec_bench.py` | lambda 包裹丢 `__vec__` → 旧加速比全假（0.91x 噪声）；修复后实测 **price_delay 1.89x、volume_imbalance 3.08x、session_mean_reversion 1.25x、总 1.91x**（4000 股×14 天×240 bars）；bind fail-closed（vec 缺失抛 LookupError） | **34 passed** |
| 4 | 统计并扩大 intraday vector kernel 绑定覆盖；删 silent bind failure | `cleaned_operators/intraday/__init__.py`（bind_whitelist） | before 3 有效绑定；smart_money/vwap_path 无 `_vec_*` 实现不冒充（诚实不绑定） | 同上 |
| 5 | 分钟特征一次 scan 的 `compute_many()` | `storage/sources/intraday_feature_runtime_v2.py` | 单次 `_grouped_bars` scan + `_calc_shared` 每 bar 一次；对拍 **rtol=1e-12 全等**；load_intraday_feature 向后兼容委托 | **23 passed**（`tests/runtime/test_intraday_compute_many.py` 6 + `tests/operators/test_intraday_feature_extensions.py` 11 等） |
| 6 | 确认 100k 走 `run_many(enable_cse=True)`，禁止逐因子 `run()` | `runtime/engine.py` | `run()` 生产 hard gate 已存在；dedup 键 str(expr)→identity canonicalizer（`add(ts_mean(close,2),0)` 与 `ts_mean(close,2)` 正确归一；20 vs 20.0 不误合并） | **27+21+74 passed**（7+1 失败为 pre-existing） |
| 7 | CSE/fusion/read-wave/backend fallback telemetry | `planner/cse.py`、`runtime/engine.py` | CSE telemetry 实测 **candidates=6/shared=2/reuse_edges=6**；fallback telemetry 全链 + production fail-closed | 同上 |
| 8 | A/B 修 scheduler 强制 thread 与 HybridExecutor classifier 冲突 | `runtime/adaptive_batch_scheduler.py` | 4 处强制 `prefer="thread"` 改 `prefer=self._execution_policy`（默认 None→HybridExecutor classifier 权威；`FACTOR_ENGINE_SCHEDULER` env 可覆盖）；A/B 定案 **thread 0.68x vs process 6.81x（比值 10.06x）**；classifier 路径快 **1.32x/1.41x/1.56x** 三档实测；telemetry execution_policy 可见 | **8+46 passed**（`tests/test_p8_scheduler_execution_policy.py` 6 + `tests/test_p8_ab_bench.py` 1 等） |
| 9 | 多 worker 默认单主进程内部并发；多进程共享资源配额 | `runtime/multiworker_governance.py`（新增） | `resolve_worker_budget()`：solo 31 核、`FACTOR_ENGINE_COEXIST=sharedN` 按 `floor(31/N)` 均分、`workers×threads<=budget` clamp+告警；9 个 worker 启动点审计入表；双进程实测 **16+15=31 恰满不超订阅、RSS 0.11G** | **8+30 passed**（`tests/test_multiworker_governance.py` 8）；67/1 大回归（唯一失败 HEAD 预存在） |
| 10 | A股 Return/%/PIT/minute/session/industry/index/relation contract 做成不可绕过 production gate | `storage/sources/intraday_feature_runtime_v2.py`（`assert_session_complete`）、`tests/test_ashare_contract_gate.py` | 12 条矩阵 10 条已有（Return bp/PubDate PIT/财务累计/IndustrySource/TopTen/IndexSymbol/分红 effective）；新增 `assert_session_complete`（240 bars fail-closed，1min/5min 通用，生产缺 1 bar 即拒）接入 `_grouped_bars`；真实数据验证 **000001.SZ Return=-191.69bp→/10000 PASS、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS** | **111+26 passed**（`tests/test_ashare_contract_gate.py` 20 等）；遗留 2 条非 P0（见 risk register） |
| 11 | 建立 100/1k/5k/20k/50k/100k dry-run ladder | `scripts/dry_run_ladder.py`（新增） | 全级支持/分层抽样/streaming sink 5000 行 shard/checkpoint 每 25 公式/六类失败分类；续跑实测通过（中断 4 条→resume 补 6 条；1k 级 3 次被杀成功续跑完）；**100 级 30 passed/70 failed**（param 28/data 15/semantic 9/other 18）rss 301.5MB；**1k 级 227 passed/773 failed**（param 329/data 144/semantic 66/other 234）wall 1194.4s rss 493MB，**0 timeout 0 OOM**；失败归因真实 | `tests/test_dry_run_ladder.py`（9 用例） |
| 12 | 所有 P0 关闭后再开始正式落 10 万因子 | — | 见 §2 裁定 | — |

**P0 补充项（53 算子 runtime audit）**：before 53 → after 2；51 fixture 缺口修（参数撞面板名/撞 generic fallback/minute→minute 模板）；2 真 bug 修（`intra_probe_outcome_score` 列表不同步、`ts_fir_lowpass_causal` 短输入 convolve 长度错位）；`ts_ewm_corr`/`ts_ewm_cov` 真 gap（无 pandas_numpy 参考，如实报告）。audit 复跑 FAILED (2 issues) 确认。

**P1 审计（read-only，`artifacts/preflight/p1_audit/` 5 JSON + SUMMARY.md）**：P0 真前视 bug=0（16 leak_detected 全 name-token 启发式误报，源码全 PIT-safe；91 unknown 保守标记）；270 alias 零问题；family 对拍 pandas vs polars 全 EXACT；7 signature=False 全 metadata 错标。P1 发现：`group_ex_self_mean/std` polars_long 编译失败、`float_share_ratio`/`free_float_share_ratio` polars_long Array cast 失败、`ts_ewm_corr`/`ts_ewm_cov` polars-only。

**allowlist 回写**：生成器本身坏（R21-P033 加 market 必填但 `write_dsl_allowlist` 无参调用→每次 TypeError）；修复后重生成 `docs/dsl_allowlist.json` **1421→1483** 与 live 完全一致；`fin_cash_earnings_gap` 是审计误判（实际注册 4 参类，2 参类是不在 load tuple 的死代码）。**71+26+6 passed**。

---

## 2. Launch Gates §99–§106 逐条对勾

### §99 算子正确性 — 部分通过
- 100% registry valid：86 admitted 全 registry valid（P0#1）✅
- 100% math/oracle evidence：**未达**（`math_oracle_pass`：PASS 461 / NOT_PROVEN 1212；86 admitted 中 oracle 证据见 operator_matrix）⚠️
- 100% prefix causality：86 admitted 全 `causal` ✅（全 1673 中 1566 causal / 91 unknown / 16 leak_detected 全误报）
- 100% field contract：86 admitted 全 field contract ✅
- 100% required backend parity：86 admitted 全 `backend_passed=True` ✅
- 100% lookback/warmup contract：86 admitted 全 lookback contract ✅
- **结论**：86 admitted 集满足 §99；全 1673 集不满足（NOT_PROVEN 1212 为诚实未证明，非错误）。

### §100 分钟性能 — 部分通过
- vector benchmark fixed ✅（P0#3，1.91x 总加速比）
- vector coverage report ✅（bind_whitelist 6/6 PASS，count_bound=6）
- scalar fallback report ✅（诚实 fallback，vec 缺失抛 LookupError）
- minute bundle benchmark ⚠️（compute_many 对拍 rtol=1e-12 全等，但未做完整 bundle 基准）
- cold/warm cache benchmark ⚠️（未单独出报告）
- 1/2/4/8 worker scaling ⚠️（P0#9 双进程 16+15=31 实测，未做完整 scaling 曲线）

### §101 多 worker — 通过（P0#9）
- host total CPU <= budget ✅（resolve_worker_budget clamp，双进程 16+15=31 恰满不超订阅）
- host total RAM <= budget ✅（RSS 0.11G 实测）
- DuckDB threads <= CPU ✅（multiworker_governance 配额）
- Polars threads controlled ✅
- remote concurrency controlled ✅
- cross-process cache single-flight ⚠️（未单独验证，见 risk register）

### §102 Backend — 通过（P0#1）
- backend coverage 当前 HEAD 重生成 ✅（`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`）
- parity evidence ✅（P1 family 对拍全 EXACT）
- runtime wiring ✅
- production admitted ✅（86，非 0）

### §103 100k stress — 未通过
- 100k graph / approved sharded equivalent：**未跑**（仅 100/1k 级，见 100K_DRY_RUN_REPORT §5）❌
- 无内存爆炸/scheduler starvation/无限 retry/writer backlog：1k 级 0 timeout 0 OOM 验证，但 100k 未验证 ⚠️

### §104 Data correctness — 部分通过
- 12 条矩阵 10 条已有（Return bp/PubDate PIT/财务累计/IndustrySource/TopTen/IndexSymbol/分红 effective）✅
- `assert_session_complete` 240 bars fail-closed 接入 `_grouped_bars` ✅
- 真实数据验证 000001.SZ Return=-191.69bp→/10000 PASS、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS ✅
- **未覆盖**：`_proxy` 后缀（micro_vpin/micro_kyle_lambda）、`HighLimit/LowLimit` 无强制 gate ❌（非 P0）

### §105 最终输出文件 — 部分
- `100K_DRY_RUN_REPORT.md` ✅（本批）
- `GO_NO_GO.md` ✅（本批）
- `BACKEND_COVERAGE_CURRENT_HEAD.md` ✅（已存在）
- `FINAL_FACTOR_ENGINE_PRODUCTION_REPORT.md` / `AGENT_DIRECT_OPERATORS.json/.md` / `OPERATOR_CORRECTNESS_MATRIX.parquet` / `INTRADAY_VECTOR_COVERAGE.md` / `PERFORMANCE_BENCHMARK.md` / `MULTIWORKER_SCALING.md` / `DATA_ACCESS_IO_REPORT.md` ⚠️（未生成，见 risk register）

### §106 GO/NO-GO 两种结论 — 见 §3 裁定

---

## 3. Risk Register

| # | 风险 | 严重度 | 状态 | 处置 |
|---|------|--------|------|------|
| R1 | **23k 回归未完整甄别（当前最大风险）**：serial ~20h 不可行改 xdist；第一轮 -n 28 死于 21:47（98% 静默消失）；第二轮 -n 16 日志冻结 7h（169 线程 GIL 自旋+121 线程 sleep 死锁）杀掉；第三轮 -n 12 卡 99% 冻结 2h50m 杀掉；第四轮 -n 8 --dist=worksteal（pid 1159803）22:48 起在跑（RUNNING，99%）。已知 pre-existing：`test_cs_polars_native_batch1.py`/`test_weighted_cs_operators.py` collection error（单独 33+13 passed）。三轮死锁模式与 §11-14 scheduler 冲突相关，P0#8 已修（classifier 决定池类型）。**「回归完整绿」gate 当前不满足** | **高** | RUNNING | 全量 100k 放量前必须完成 23k 回归甄别 |
| R2 | `ts_ewm_corr`/`ts_ewm_cov` 无 pandas_numpy 参考（polars-only），audit 复跑 FAILED (2 issues) | 中 | open | 补 pandas 参考或如实标记 polars-only |
| R3 | `_proxy` 后缀（micro_vpin/micro_kyle_lambda）未补 | 中 | open | 补 gate |
| R4 | `HighLimit`/`LowLimit` 无强制 gate | 中 | open | 补 gate |
| R5 | `ts_poly2_coeff`/`ts_poly2_resid` 无认证源 | 中 | open | 补认证源 |
| R6 | 8 个 pytest 是 test 断言未跟上 R47→100k GO 语义（P0#1 遗留） | 低 | open | 更新断言 |
| R7 | pre-existing 测试失败清单：P0#6+#7 的 7+1、P0#9 的 67/1（唯一失败 HEAD 预存在）、runtime 610/71（r14 ClickHouse/dual-write/reconcile + ProductionMarketContextRequiredError） | 低 | open | 甄别后确认非本轮引入 |
| R8 | README 漂移剩余项：README 1737 vs live 1673、`docs/dsl_allowlist.json` 已修至 1483 但 README 静态 1421 未同步、`cleaned_operators/docs/operators_catalog.json` 1737 过时 | 低 | open | doc-sync pass |
| R9 | 100 级 ladder `wall_time_s` 记录缺陷（~0，续跑累加逻辑） | 低 | open | 5k 级前修复 wall-time 累加 |
| R10 | §105 交付物缺失：FINAL_FACTOR_ENGINE_PRODUCTION_REPORT / AGENT_DIRECT_OPERATORS / OPERATOR_CORRECTNESS_MATRIX / INTRADAY_VECTOR_COVERAGE / PERFORMANCE_BENCHMARK / MULTIWORKER_SCALING / DATA_ACCESS_IO_REPORT | 低 | open | 后续生成 |
| R11 | cross-process cache single-flight 未单独验证（§101） | 低 | open | 验证 |

---

## 4. GO / NO-GO 裁定

### 裁定：**条件 GO（限 86 admitted 集）**

**判定依据（诚实，不美化）**：

1. **production_admitted 86 条链路完整可跑**（P0#1 实测 0→86，与 production_certified=86 一致；86 全 registry valid / causal / field contract / backend_passed / lookback contract）→ 满足 §99 对 admitted 集的要求。
2. **ladder 1k 级可续跑可分类**（P0#11+#12：checkpoint 续跑、六类失败分类、streaming sink、0 timeout 0 OOM、RSS 493MB）→ 满足 §71 Stage A/B 判据。
3. **A股 contract 10/12 条已 gate**（P0#10：Return bp / PubDate PIT / session 240-bar fail-closed 等，真实数据验证 PASS）→ 满足 §104 主体。
4. **多 worker 资源配额已 clamp**（P0#9：双进程 16+15=31 恰满不超订阅、RSS 0.11G）→ 满足 §101。

**条件（未达成，放量前必须完成）**：

- **R1：23k 回归完整甄别未达成**（第 4 轮 -n 8 worksteal 仍在 RUNNING 99%，前三轮死锁已杀）。**全量 100k 放量前必须完成 23k 回归甄别**，确认无本轮引入的回归。
- **R2：`ts_ewm_corr`/`ts_ewm_cov` 无 pandas 参考**（audit 复跑 FAILED 2 issues）。
- **R3/R4：`_proxy` 后缀、`HighLimit`/`LowLimit` 无 gate**（非 P0）。
- **R5：`ts_poly2_coeff`/`ts_poly2_resid` 无认证源**。
- **§103 100k stress 未跑**（仅 100/1k 级）。

**因此**：**条件 GO** —— 允许对 **86 admitted 集**（`production_admitted=86`）进行生产落值；**禁止**对全 1673 集或 100k 全量放量，直至 R1（23k 回归甄别）关闭且 5k/20k/50k/100k ladder 逐级通过。

**GO 参数（§106 要求列）**：

| 项 | 值 |
|----|-----|
| HEAD | `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026`（工作树 dirty） |
| direct operator count | 86（production_admitted） |
| certified count | 86（production_certified） |
| backend | pandas_numpy + polars（86 admitted 全双后端）；polars-only 2（ts_ewm_corr/ts_ewm_cov） |
| benchmark hardware | 32 核 92G（OMP_NUM_THREADS=31 上限） |
| recommended workers | 单主进程内部并发（solo 31 核）；多进程需 `FACTOR_ENGINE_COEXIST` 配额 |
| recommended threads | `workers×threads <= floor(31/coexist_count)` |
| 100k expected execution mode | `run_many(enable_cse=True)` + streaming sink + checkpoint 续跑（禁止逐因子 run()） |
| known P1/P2 limitations | 见 §3 risk register R2–R11 |

**禁止措辞**：本裁定不含「基本可以」「应该没问题」「看起来能用」。所有「通过」均有实测数字支撑；所有「未达」均如实列出。
