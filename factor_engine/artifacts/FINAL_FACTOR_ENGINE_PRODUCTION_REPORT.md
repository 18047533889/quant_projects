# FINAL_FACTOR_ENGINE_PRODUCTION_REPORT — 100k 生产就绪最终报告

> 生成：2026-09-04 · 更新：2026-09-05（R61-P0/P1 收口 doc-sync，task #63，见 §1b）· HEAD `ae7dc622`（worktree 干净）
> 对应：GO_PROMPT §105 交付物清单 / §99–§104 Launch Gates / §110 12 项 P0。
> 权威数据源：`artifacts/operator_audit/current_head/operator_matrix.csv`（2026-09-05 重生成 @ 593507c1：1689 行；本表列 1673-era 历史行）、`evidence/primitive_verified.json`、`evidence/factor_operator_verified.json`、`artifacts/AGENT_DIRECT_OPERATORS.json`、`/tmp/ladder_100k/ladder_status.json`、`/tmp/bench_minute_bundle.json`、`/tmp/bench_worker_scaling.json`、`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`（2026-09-05 重生成 @ 593507c1）。
> 本次更新（2026-09-05）口径说明：**admission 数字已由 86 更正为 72** —— inventory artifacts 在 HEAD 593507c1/admission pass 重生成（`artifacts/preflight/rebuild_inventory.py`），canonical 1673→**1689**；production_admitted/directly_usable = **72 @ 593507c1**（`production_certified=86` 不变），**supersede 7628674b-era 86**（见 `artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`）。下文凡指**当前 admitted 集**一律为 72；凡 1673-era 表格/历史数字保留原文（显式标注「历史/superseded」）不改写（同 GO_NO_GO 口径）。

---

## 1. 总览（一句话）

FactorEngine 已能对 **72 个 production-admitted 算子**（current-HEAD @ 593507c1，supersede 7628674b-era 86；Agent 直接可调用、五门 gate 全过、双后端）做生产落值；全量 1689 canonical 与 100k 放量仍受诚实未证实的语义/源契约 gate 与 5k+ ladder execution 未跑约束 —— 即 **条件 GO（限 current-HEAD admitted 集 72）**，明细裁定见 `GO_NO_GO.md`。

## 2. 算子正确性总账（§99，operator_matrix.csv 2026-09-05 重生成 @ 593507c1：1689 行；下表 1673-era 行保留为**历史平面**）

| 维度 | 总数 | 通过 | 未过（诚实标注） |
|------|------|------|------------------|
| canonical | 1673（1689-era：inventory_manifest canonical_count=1689） | — | — |
| surface | daily 1242 / extended 415 / unsafe 7 / research 5 / internal 3 / legacy 1 | — | — |
| lifecycle_status | — | production 137 | experimental 1532 / research 3 / deprecated 1 |
| prefix causality | 1673 | causal 1566 | unknown 91 / leak_detected 16（全 name-token 误报，源码 PIT-safe） |
| pit_safe | 1673 | 99 | 1574 |
| math_oracle | 1673 | PASS 461 | NOT_PROVEN 1212（诚实未证明） |
| backend_passed | 1673 | 86 | 1517（+70 NaN） |
| **production_admitted** | 1673 | **86**（历史平面 7628674b-era，已被 supersede；current-HEAD = **72 @ 593507c1**） | 1587 |

**86 admitted 集**（7628674b-era **历史数字，已被 supersede**；current-HEAD = **72 @ 593507c1**）满足 §99 全部要求：registry valid / prefix causal / field contract / backend parity / lookback contract。见 `AGENT_DIRECT_OPERATORS.json`（62 terminal ⊆ 72 admitted）与 `OPERATOR_CORRECTNESS_MATRIX.parquet`（全量）。

## 1b. R61（2026-09-05）增量收口

> HEAD `593507c1` CURRENT-HEAD artifact 平面；EVIDENCE_CURRENT **PASS（current=3, stale=0）**。本段为 R61 增量，历史数字不被改写（同 GO_NO_GO §2b）。

- **R61-P0 #58 — FE-native batch ladder**：`factor_engine/scripts/production_factor_ladder.py` 取代旧 `scripts/dry_run_ladder.py`（后者 thin shim，9 测试仍过）。每级**单次 batch `engine.run_many(factors, enable_cse=True)`**；62-operator 池真实 api factories；LadderRoot 结构指纹数值字面量归一（20==20.0）；>95% unique 强制（gen-only **100000 roots→100000 unique = 100.0%**，9 个经济族）；分层抽样；StreamingSink 5000 行 shard；DryRunCheckpoint 每 25 公式 + resume（中断 4→补 6）；六类失败分类不丢失败（fid,None）；worker 治理（workers×OMP≤31/coexist）；`--dry-level/--max-level/--gen-only` CLI。**Level-100 smoke：43 passed / 14 failed**（param 8/data 3/semantic 2/other 1），wall **63.1s**（真实记录，修 R9），CSE shared nodes **9**/edges **75**。测试 `tests/test_production_factor_ladder.py` **21 passed + 1 skipped in 69.45s**（100k gen 用例 env 门控 `LADDER_100K_TEST=1`）。R9（wall_time_s 记录缺陷）由此关闭。
- **R61-P0 #59 — evidence 硬 gate（fail-closed）**：新 `evidence/gate.py` —— `StaleAgentOperatorEvidence`（RuntimeError，fail-closed）、`require_current_evidence(*, allow_stale=False)`、**无 warn-and-proceed 路径**；`evidence/CURRENT.json` artifacts 携带 bound `source_snapshot_id`；`--record-execution ARTIFACT=file` 读真实输出文件 mtime。测试 `tests/test_evidence_gate.py` **3 passed in 309s**。live：EVIDENCE_CURRENT **PASS（current=3, stale=0）** @ 593507c1（rebind 归父会话；并行 work 漂移属预期，机制+PASS 为实述）。
- **R61-P1 #55 — three-tier pytest 分层**：`tests/production_critical.txt`（45）/`research_extended.txt`（46 demoted + reasons）/`legacy_quarantine.txt`（7 files）；`scripts/run_test_tiers.py` + `scripts/check_production_critical.sh`（任一失败/缺失 exit nonzero）。production_critical tier **330 passed in ~319s GREEN**。Doc：`docs/TEST_TIERING.md`。
- **R61-P1 #56 — 真实 DA/COS worker scaling + single-flight**（`bench_worker_scaling_real.py`，`ASHARE_PARQUET_ROOT` 空 → 强制 remote+cli）：f/s **0.27→0.66→1.50→2.65**（w8 vs w1 **9.8×**），每档 remote fetch **恒=10**、duplicate downloads **=0** → 跨进程 single-flight REAL。实现 `data_access/cos/mirror.py` per-key `fcntl.flock` 覆盖整个 sync 事务（锁内 freshness 复检 + cp + os.replace + manifest 写）；`DATA_ACCESS_COS_FETCH_LOG` JSONL。`data_access/tests/unit/test_cache_single_flight.py` **2 passed**（真实 COS 4 进程 fetch=1/hit=3）。R11 CLOSED。
- **R61-P1 #60 — minute session gate**：`data_access/read/session_calendar.py` `expected_slots()/expected_minutes()/validate_session_bars()`（**240 one-minute slots 09:31–11:30 / 13:01–15:00**）；`intraday_feature_runtime_v2.py` `_grouped_bars` 按 (TradeDate,Symbol) 精确 slot 校验（`session_require_full=True` 默认）；停牌经 `ashare_stock_daily.IsSuspend` DataAccessSource child → 停牌 symbol-day → NaN 放行、**非停牌残缺 → ValidationError fail-closed**；gate report 持久化。session-gate 测试子集 **40 passed**。
- **R61-P1 #57 — intraday 向量覆盖（shared sufficient statistics）**：`factor_engine/cleaned_operators/intraday/sufficient_stats.py` + `sufficient_stats_ops.py`（单趟 bundle Σx..Σx⁴、Σr..Σr⁴、Σv/Σv²/Σpv、Σa/Σpa、max/min、first/last、packed-prefix argmax/argmin、two-pass variance；per-frame LRU）+ **16 新 `intra_ts_*` 算子**（`_core.py` 内 `_vec_` kernels）；bind whitelist 现 **29**、bind fail-closed `count_bound()==29`。sufficient-stats 测试 **23 passed**。
- **R61-P1 #61 — minute `_wide_frame` 单物理 scan**：整帧一次多列 parquet scan（替代逐列 scan）。

**CURRENT-HEAD artifact 平面（2026-09-05 重生成）**：`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md` @ `593507c1` → **canonical 1689、production_admitted 72、directly_usable 72**（supersede 7628674b 86）。`evidence/factor_engine/model_operators/MODEL_CURRENT_HEAD.json`：commit_sha `593507c1`、canonical_count **1756**、model_like_count **335**、final_direct_use_ready_count **0**。`MODEL_FINAL_HARD_GATES.json`：**19 gates = 6 PASS / 5 FAIL / 8 NOT_RUN**（FAIL 诚实：`reg_forecast_error_pct` timing missing、parameter_domain 0/66、typed inputs 0/66、silent clamp on ts_regime_duration/ts_regression_forecast params）。

## 3. Evidence 现状（pinned env 实测）

| 证据 | 状态 | 数字 |
|------|------|------|
| `evidence/primitive_verified.json` | ✅ 有效（validation_errors=0） | **test-certified six-way: 79**（passed_at 存在，绑定当前 commit） |
| `evidence/factor_operator_verified.json` | ✅ 有效（validation_errors=0） | runtime audit 1667 canonical 全过，1577 Pandas-reference 生产算子；ts_ewm_cov/ts_ewm_corr 如实 polars-only |
| 根因 | **env drift** | 主 env pandas 3.0.5 破坏全部 evidence 自校验；pinned `fe_pin_venv2`（pandas 2.3.3/numpy 2.2.6/polars 1.42.1/duckdb 1.5.4 = requirements-production.lock）下 certify 全链 exit 0 |

**六门 status（诚实 fail-closed）**：daily 原语（ts_mean/add/group_sum/tanh/ts_sharpe 等 79 个 six-way certified）→ `status=production` **top-level certified**；factor-operator canonical（ts_ema/ts_quantile/ts_sma_cn）→ tier=pandas_reference 但 top-level `production_certified=False`，因 **P0-18 独立 semantic_golden / source_contract gate 无独立 payload 记录**—— 这是设计边界（需真实 math-golden 证据，不伪造），非代码 bug。

## 4. 性能 gate（§100/§52，实测数字见 PERFORMANCE_BENCHMARK.md）

- **分钟 bundle 共享 scan 生效**：feature 10→100 wall 放大 **6.82x**（线性 10x）、10→200 **6.64x**（线性 20x）；per-feature 成本单调下降（0.1116→0.0371 s）。
- **OLD 逐 feature vs NEW compute_many**：单 feature **26.04x**、100 feature **23.06x**。
- **cold vs warm cache**：1.54x（warm 复用 grouped-bars cache；一次 compute_many scan keys=1）。
- **worker 1/2/4/8**：throughput 5.7→10.6→21.5→37.6 f/s，scaling 1/1.865/3.775/6.598，无拐点（峰值 w8）；超卖 8×8=64 线程违反 governor 且无收益（36.6 vs 37.6 f/s）→ 治理规则 `workers×OMP ≤ floor(31/N)` 验证正确。【synthetic CPU-bound 段，见 MULTIWORKER_SCALING.md】
- **Real DA/COS IO scaling**（R61-P1 #56，真实 data_access+COS 冷读）：f/s 0.27→0.66→1.50→2.65（w8 vs w1=9.8×），每档 remote fetch 恒=10、duplicate-download=0 —— 跨进程 cache single-flight 真实成立；cache hit 37.5→92.2%。见 `MULTIWORKER_SCALING.md`「Real DA/COS IO scaling」段。【真实 IO 段，关闭 R11】
- **intraday vector kernel**（P0#4 + R61-P1 #57）：coverage **total 69 / vectorized 29 / scalar-only 40**（R61-P1 #57 增补 16 个 `_vec_ts_*` 内核，来自单趟 bundle 充分统计派生：`sufficient_stats.py`/`sufficient_stats_ops.py`）；旧平面 53/13/40 已被取代。lambda/`_fn` 无 `__vec__` 诚实 fallback、bind fail-closed。见 `INTRADAY_VECTOR_COVERAGE.md` 与 `intraday_vector_coverage.json`（bound_count=29）。

## 5. 多 worker 资源治理（§101，P0#9）

`multiworker_governance.resolve_worker_budget()`：solo 31 核；`FACTOR_ENGINE_COEXIST=N` 按 `floor(31/N)` 分桶；`workers×threads≤budget` clamp+告警。双进程实测 16+15=31 恰满不超订阅、RSS 0.11G。cross-process cache single-flight 已验证（R11 CLOSED，见 §1b/§9）。详细见 `MULTIWORKER_SCALING.md`。

## 6. Ladder（§103，未达满级 execution；R61-P0 #58 现行）

下表为**旧 dry-run ladder** 早期证据（100/1k 级执行数字，历史平面；**已被 R61-P0 #58 FE-native ladder 取代**，见 §1b/§8 与 `scripts/production_factor_ladder.py`）：

| 级 | 规模 | passed | failed | wall(s) | RSS(MB) | 判据（§71） |
|----|------|--------|--------|---------|---------|-------------|
| 100 | 100 | 30 | 70（param 28/data 15/semantic 9/other 18） | ~0（R9 缺陷本体） | 301.5 | 核语义 |
| 1000 | 1000 | 227 | 773（param 329/data 144/semantic 66/other 234） | 1194.4 | CSE/IO |

**0 timeout 0 OOM**；checkpoint 续跑验证（中断→resume 补跑）；streaming sink 5000 行 shard；失败六类分类真实（不吞异常）。

**R61-P0 #58（现行）**：FE-native ladder `scripts/production_factor_ladder.py`（取代旧 dry_run_ladder，后者 thin shim）。每级**单次 batch `engine.run_many(enable_cse=True)`**；池 = 62 算子真实 mining-grammar；structural-fingerprint 唯一性 >95%（numeric-literal 归一）；**100k 生成 CONFIRMED**（count==100000 + unique>95%，零执行，`LADDER_100K_TEST=1` 门控，~10 min CPU）；Level-100 smoke 43 passed/14 failed、wall 63.1s、CSE shared nodes 9/edges 75；StreamingSink shard+sidecar + DryRunCheckpoint resume（中断 4→补 6）；失败带真实 error text（永不 `(fid,None)`）；timeout=0 拒绝、worker 治理 clamp。

**5k/20k/50k/100k EXECUTION 未跑** —— 100k 级放量的硬 blocker（见 GO_NO_GO §103）。

## 7. Data correctness 契约（§104，P0#10 + R61-P1 #60）

A股 Return bp / PubDate PIT / session 240-bar / IndustrySource / IndexSymbol / 分红 effective 等 10/12 条已做不可绕过 production gate；session gate 按 (TradeDate,Symbol) 精确 slot 校验（R61-P1 #60：`session_calendar.expected_slots()/validate_session_bars()`，停牌经 `ashare_stock_daily.IsSuspend` → NaN 放行、非停牌残缺 → ValidationError fail-closed，40 tests）；真实数据验证 000001.SZ Return=-191.69bp→/10000 PASS、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS。未覆盖：`_proxy` 后缀、HighLimit/LowLimit（非 P0，risk register R3/R4 保持 open）。

## 8. 12 项 P0 完成度

P0#1–#11 全部**完成且带实测数字**（矩阵见 GO_NO_GO §1；其中 #11 的旧 dry-run ladder 已由 R61-P0 #58 FE-native ladder 取代，见 §6）；P0#12（100k 正式落值）未开启 —— 依据 §2 裁定。

## 9. 关键风险与诚实缺口

| 风险 | 级别 | 状态 |
|------|------|------|
| 全量 23k 回归甄别 | 高 | **CLOSED（2026-09-04）**：第五轮 + `--timeout=300` 完整跑完 16478 passed / 4961 failed 全 pre-existing（不阻塞 72 admitted 集放量，5k+ ladder 前建议 test-vs-code 对齐 pass） |
| R2 ts_ewm_corr/ts_ewm_cov 无 pandas 参考 | 中 | open（polars-only 如实报告） |
| R3/R4 _proxy / HighLimit-LowLimit gate | 中 | open（非 P0） |
| R5 ts_poly2_coeff/ts_poly2_resid 无认证源 | 中 | open |
| R6 8 个 pytest 断言未跟上 R47→100k 语义 | 低 | open |
| R7 pre-existing 测试失败清单 | 低 | open（甄别确认非本轮引入） |
| R8 README 漂移（1737 vs live 1689、catalog 过时） | 低 | open（doc-sync pass；注：live canonical 经 2026-09-05 inventory 重生成后为 1689，README 漂移随重生成进一步扩大，仍待 doc-sync） |
| R9 ladder wall_time_s ~0 记录缺陷 | 低 | **CLOSED（2026-09-05，R61 #58）**：新 FE-native ladder 每级记录真实 per-level wall_time_s（Level-100 smoke 63.1s） |
| R10 §105 交付物缺失 | 低 | **本批已关闭（7/7 生成，见 §10）** |
| R11 cross-process cache single-flight 未单独验证 | 低 | **CLOSED（2026-09-05，R61-P1 #56）**：真实 COS 4 进程 fetch=1/hit=3 + 1/2/4/8 全档 duplicate=0（真实 DA/COS scaling REAL） |

## 10. §105 交付物清单（最终）

| 交付物 | 位置 | 状态 |
|--------|------|------|
| FINAL_FACTOR_ENGINE_PRODUCTION_REPORT.md | `artifacts/`（本文件） | ✅（R61 增量见 §1b） |
| AGENT_DIRECT_OPERATORS.json | `artifacts/AGENT_DIRECT_OPERATORS.json` | ✅（62 terminal ⊆ 72 admitted） |
| AGENT_DIRECT_OPERATORS.md | `artifacts/AGENT_DIRECT_OPERATORS.md` | ✅ |
| OPERATOR_CORRECTNESS_MATRIX.parquet | `artifacts/OPERATOR_CORRECTNESS_MATRIX.parquet` | ✅（1689 行 × 42 列，@ 593507c1 重生成；旧 1673 行平面已被取代） |
| BACKEND_COVERAGE_CURRENT_HEAD.md | `artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md` | ✅（**2026-09-05 R61 重生成 @ 593507c1** → canonical 1689 / production_admitted **72** / directly_usable **72**，supersede 7628674b 86） |
| INTRADAY_VECTOR_COVERAGE.md | `artifacts/perf_vec/INTRADAY_VECTOR_COVERAGE.md` | ✅（29/69 vectorized，R61-P1 #57 增补后） |
| PERFORMANCE_BENCHMARK.md | `artifacts/perf_vec/PERFORMANCE_BENCHMARK.md` | ✅ |
| MULTIWORKER_SCALING.md | `artifacts/perf_vec/MULTIWORKER_SCALING.md` | ✅（含 Real DA/COS IO scaling，R61-P1 #56） |
| DATA_ACCESS_IO_REPORT.md | `artifacts/DATA_ACCESS_IO_REPORT.md` | ✅（见 data_access 实测） |
| 100K_DRY_RUN_REPORT.md | `artifacts/100K_DRY_RUN_REPORT.md` | ✅（pre-existing；dry-run ladder 已被 R61-P0 #58 FE-native ladder 取代） |
| GO_NO_GO.md | `artifacts/GO_NO_GO.md` | ✅（HEAD 数字见报告头） |

**R61（2026-09-05）新增交付物**：
- `scripts/production_factor_ladder.py` ✅（R61-P0 #58 FE-native batch ladder）
- `evidence/gate.py` ✅（R61-P0 #59 `StaleAgentOperatorEvidence` fail-closed hard gate）
- `scripts/run_test_tiers.py` + `scripts/check_production_critical.sh` + `tests/production_critical.txt` / `research_extended.txt` / `legacy_quarantine.txt` + `docs/TEST_TIERING.md` ✅（R61-P1 #55 three-tier pytest split）

## 11. 最终结论

**条件 GO（限 current-HEAD admitted 集：72 @ 593507c1，supersede 86 @ 7628674b）**。理由（§106 禁止含糊措辞）：72 条链路五门 gate 全过（current-HEAD artifact 平面重生成，`artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md`：canonical 1689、production_admitted=directly_usable=72；production_certified=86 不变）；evidence 双 artifact 在 pinned env 全链 certify 有效，且 evidence 平面现 **fail-closed** —— `evidence/gate.py` `StaleAgentOperatorEvidence`（`require_current_evidence(*, allow_stale=False)`，无 warn-and-proceed 路径），EVIDENCE_CURRENT PASS（current=3, stale=0）@ 593507c1；性能 gate 三档全实测通过（+ R61-P1 #57 向量覆盖 29 bound、#61 单 scan）；ladder 由 R61-P0 #58 FE-native 单 batch 实现取代 dry-run（Level-100 smoke 43/14、63.1s，100k GENERATION 100.0% unique）。**放量 blocker**：5k/20k/50k/100k ladder **execution** 未跑（§103，GENERATION 已证）+ R2–R5 遗留 + P0-18 独立语义 gate 无独立 payload（fail-closed 保持 experimental 是正确行为，不是缺陷）。逐条 P0 完成矩阵见 `GO_NO_GO.md` §1；每项「通过」均有实测数字支撑，每项「未达」如实列出。
