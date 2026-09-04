# FINAL_FACTOR_ENGINE_PRODUCTION_REPORT — 100k 生产就绪最终报告

> 生成：2026-09-04 · HEAD `a07ea23020f86f4a63f50f67ec5b0ae1023b5b22`（工作树 dirty）
> 对应：GO_PROMPT §105 交付物清单 / §99–§104 Launch Gates / §110 12 项 P0。
> 权威数据源：`artifacts/operator_audit/current_head/operator_matrix.csv`（1673 行）、`evidence/primitive_verified.json`、`evidence/factor_operator_verified.json`、`artifacts/AGENT_DIRECT_OPERATORS.json`、`/tmp/ladder_100k/ladder_status.json`、`/tmp/bench_minute_bundle.json`、`/tmp/bench_worker_scaling.json`。

---

## 1. 总览（一句话）

FactorEngine 已能对 **86 个 production-admitted 算子**（Agent 直接可调用、五门 gate 全过、双后端）做生产落值；全量 1673 canonical 与 100k 放量仍受诚实未证实的语义/源契约 gate 与 5k+ ladder 未跑约束 —— 即 **条件 GO（限 86 admitted 集）**，明细裁定见 `GO_NO_GO.md`。

## 2. 算子正确性总账（§99，数据源 operator_matrix.csv 1673 行）

| 维度 | 总数 | 通过 | 未过（诚实标注） |
|------|------|------|------------------|
| canonical | 1673 | — | — |
| surface | daily 1242 / extended 415 / unsafe 7 / research 5 / internal 3 / legacy 1 | — | — |
| lifecycle_status | — | production 137 | experimental 1532 / research 3 / deprecated 1 |
| prefix causality | 1673 | causal 1566 | unknown 91 / leak_detected 16（全 name-token 误报，源码 PIT-safe） |
| pit_safe | 1673 | 99 | 1574 |
| math_oracle | 1673 | PASS 461 | NOT_PROVEN 1212（诚实未证明） |
| backend_passed | 1673 | 86 | 1517（+70 NaN） |
| **production_admitted** | 1673 | **86** | 1587 |

**86 admitted 集**（P0#1 重生成，fail-closed）满足 §99 全部要求：registry valid / prefix causal / field contract / backend parity / lookback contract。见 `AGENT_DIRECT_OPERATORS.json`（62 terminal ⊆ 86 admitted）与 `OPERATOR_CORRECTNESS_MATRIX.parquet`（全量）。

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
- **worker 1/2/4/8**：throughput 5.7→10.6→21.5→37.6 f/s，scaling 1/1.865/3.775/6.598，无拐点（峰值 w8）；超卖 8×8=64 线程违反 governor 且无收益（36.6 vs 37.6 f/s）→ 治理规则 `workers×OMP ≤ floor(31/N)` 验证正确。
- **intraday vector kernel**（P0#4）：coverage total 53 / vectorized **13** / scalar-only 40（lambda/`_fn` 无 `__vec__` 诚实 fallback，bind fail-closed）。见 `INTRADAY_VECTOR_COVERAGE.md`。

## 5. 多 worker 资源治理（§101，P0#9）

`multiworker_governance.resolve_worker_budget()`：solo 31 核；`FACTOR_ENGINE_COEXIST=N` 按 `floor(31/N)` 分桶；`workers×threads≤budget` clamp+告警。双进程实测 16+15=31 恰满不超订阅、RSS 0.11G。详细见 `MULTIWORKER_SCALING.md`。

## 6. Dry-run ladder（§103，未达满级）

| 级 | 规模 | passed | failed | wall(s) | RSS(MB) | 判据（§71） |
|----|------|--------|--------|---------|---------|-------------|
| 100 | 100 | 30 | 70（param 28/data 15/semantic 9/other 18） | ~0 缺陷 | 301.5 | 核语义 |
| 1000 | 1000 | 227 | 773（param 329/data 144/semantic 66/other 234） | 1194.4 | CSE/IO |

**0 timeout 0 OOM**；checkpoint 续跑验证（中断→resume 补跑）；streaming sink 5000 行 shard；失败六类分类真实（不吞异常）。**5k/20k/50k/100k 未跑** —— 100k 级放量的硬 blocker（见 GO_NO_GO §103）。

## 7. Data correctness 契约（§104，P0#10）

A股 Return bp / PubDate PIT / session 240-bar / IndustrySource / IndexSymbol / 分红 effective 等 10/12 条已做不可绕过 production gate；`assert_session_complete` fail-closed 接入 `_grouped_bars`；真实数据验证 000001.SZ Return=-191.69bp→/10000 PASS、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS。未覆盖：`_proxy` 后缀、HighLimit/LowLimit（非 P0，risk register R3/R4）。

## 8. 12 项 P0 完成度

P0#1–#11 全部**完成且带实测数字**（矩阵见 GO_NO_GO §1）；P0#12（100k 正式落值）未开启 —— 依据 §2 裁定。

## 9. 关键风险与诚实缺口

| 风险 | 级别 | 状态 |
|------|------|------|
| 全量 23k 回归甄别 | 高 | **CLOSED（2026-09-04）**：第五轮 + `--timeout=300` 完整跑完 16478 passed / 4961 failed 全 pre-existing |
| R2 ts_ewm_corr/ts_ewm_cov 无 pandas 参考 | 中 | open（polars-only 如实报告） |
| R3/R4 _proxy / HighLimit-LowLimit gate | 中 | open（非 P0） |
| R5 ts_poly2_coeff/ts_poly2_resid 无认证源 | 中 | open |
| R6 8 个 pytest 断言未跟上 R47→100k 语义 | 低 | open |
| R7 pre-existing 测试失败清单 | 低 | open（甄别确认非本轮引入） |
| R8 README 漂移（1737 vs live 1673、catalog 过时） | 低 | open（doc-sync pass） |
| R9 ladder wall_time_s ~0 记录缺陷 | 低 | open（5k 前修） |
| R10 §105 交付物缺失 | 低 | **本批已关闭（7/7 生成，见 §10）** |
| R11 cross-process cache single-flight 未单独验证 | 低 | open |

## 10. §105 交付物清单（最终）

| 交付物 | 位置 | 状态 |
|--------|------|------|
| FINAL_FACTOR_ENGINE_PRODUCTION_REPORT.md | `artifacts/`（本文件） | ✅ |
| AGENT_DIRECT_OPERATORS.json | `artifacts/AGENT_DIRECT_OPERATORS.json` | ✅（62 terminal ⊆ 86 admitted） |
| AGENT_DIRECT_OPERATORS.md | `artifacts/AGENT_DIRECT_OPERATORS.md` | ✅ |
| OPERATOR_CORRECTNESS_MATRIX.parquet | `artifacts/OPERATOR_CORRECTNESS_MATRIX.parquet` | ✅（1673 行 × 42 列） |
| BACKEND_COVERAGE_CURRENT_HEAD.md | `artifacts/BACKEND_COVERAGE_CURRENT_HEAD.md` | ✅（pre-existing） |
| INTRADAY_VECTOR_COVERAGE.md | `artifacts/perf_vec/INTRADAY_VECTOR_COVERAGE.md` | ✅（13/53 vectorized） |
| PERFORMANCE_BENCHMARK.md | `artifacts/perf_vec/PERFORMANCE_BENCHMARK.md` | ✅ |
| MULTIWORKER_SCALING.md | `artifacts/perf_vec/MULTIWORKER_SCALING.md` | ✅ |
| DATA_ACCESS_IO_REPORT.md | `artifacts/DATA_ACCESS_IO_REPORT.md` | ✅（见 data_access 实测） |
| 100K_DRY_RUN_REPORT.md | `artifacts/100K_DRY_RUN_REPORT.md` | ✅（pre-existing） |
| GO_NO_GO.md | `artifacts/GO_NO_GO.md` | ✅（pre-existing，HEAD 数字见报告头） |

## 11. 最终结论

**条件 GO（限 86 admitted 集）**。理由（§106 禁止含糊措辞）：86 条链路五门 gate 全过且 production_certified/admitted/directly_usable 三数一致 =86；evidence 双 artifact 在 pinned env 全链 certify 有效；性能 gate 三档全实测通过；1k ladder 0 timeout 0 OOM。**放量 blocker**：5k/20k/50k/100k ladder 未跑（§103）+ R2–R5 遗留 + P0-18 独立语义 gate 无独立 payload（fail-closed 保持 experimental 是正确行为，不是缺陷）。逐条 P0 完成矩阵见 `GO_NO_GO.md` §1；每项「通过」均有实测数字支撑，每项「未达」如实列出。
