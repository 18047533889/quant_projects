# R34 最终验收报告 — 最新 HEAD 独立终审

> 日期: 2026-08-11
> 任务书基线 HEAD: `8449d9c253c55308f3e406a15739e984387e9691`
> 审计时点 HEAD: `c2309dbd`（审计期间并发会话合入 R29/R33-R35 工作；R34 证据
> 框架按设计绑定**审计时点** current HEAD，见 `docs/evidence/r34/R34_HEAD.json`）
> 任务书: `FactorEngine_R34_最新HEAD独立终审_..._20260811.md`
> 结论: **NOT PRODUCTION READY**（诚实——存在真实未闭合的 P0 缺口）

---

## 一、结论摘要

| 证据 | 状态 |
|---|---|
| Evidence Truth gates | **12/12 PASS** |
| R34 Hard Gates（§186） | **13 PASS / 32 NOT_RUN / 5 FAIL** |
| Canonical Correctness Ledger | 1391 个 production canonical：**4 CERTIFIED / 1381 NOT_CERTIFIED / 6 PENDING** |
| Parameter Domain | 11 个核心算子独立 oracle 认证全部 window（`{1,2,5,20,60,120,252}`） |
| 旧 evidence | 35 个 stale artifact 全部被 freshness 显式标记，不再沿用 |

**FINAL VERDICT = NOT PRODUCTION READY**。这符合任务书 §192：任何 P0 未闭合即 NOT
PRODUCTION READY。本次不假完成——把真实的 typed-signature / edge-contract / 独立
golden 缺口如实暴露，并给出已完成的整改与剩余风险。

---

## 二、已完成的 P0 整改（代码 + 测试）

### 1. Evidence 框架（P0-001/002，Phase 0-1）
- **`runtime/r34_evidence.py`**：`EvidenceHeader`（13 字段 current-HEAD 绑定）、
  `GateResult`（executed_cases>0 才允许 PASS；无 case 一律 NOT_RUN）、
  `scan_hardcoded_true_gates`（AST 扫描字面量 gate 赋值，negative control）。
- **`scripts/audit_r34_evidence.py`**：Evidence Truth 12 gates 全过；
  `R34_EVIDENCE_FRESHNESS.json` 显式标记 35 个 stale artifact（含
  `factor_operator_verified.json` commit=`0b659ec`≠HEAD）；`R34_AUDIT_NEGATIVE_CONTROL.json`。
- **移除硬编码 True**：`scripts/audit_r30_hard_gates.py` 的
  `R30_EVERY_CURRENT_CANONICAL_REVIEWED = True`（Phase 14 占位）换成真检查；
  并修复 review manifest 未覆盖 86 个核心 daily canonical 的缺口
  （`operator_surface.py` 并入 `REVIEWED_MIGRATION_MANIFEST`，review_id=`r34-core-daily-surface`）。

### 2. P0 代码修复
| P0 | 修复 | 文件 |
|---|---|---|
| P0-018 | `open` 从 session-end-known 移入 opening-known；open-only 因子 `available_at=session_open`、same-session 可用 | `cleaned_operators/availability_clock.py` |
| P0-019 | production 下缺 frequency fail-closed（不再静默 `1d`） | `runtime/engine.py` |
| P0-021/022 | `check_financial_grain_contract` 接受 `market_context`（去 ASHARE 硬编码）；`periods_per_year` 不再固定 4 | `cleaned_operators/operator_spec.py`、`api/mining_integration.py` |
| P0-025 | model-like canonical 无显式 `ModelTimingContract` → production FAIL（自动生成仅 research hint） | `cleaned_operators/model_timing.py` |
| P0-035 | edge gate 默认 `backend=None`（跨适用后端并集），不默认绑 duckdb | `cleaned_operators/edge_requirements.py` |
| P0-037 | `OutputShapeContract` 统一 grain 变换门：shape-changing 需显式 input/output_grain，否则 fail-closed | `cleaned_operators/operator_spec.py`、`production_hardening.py` |
| P0-039 | production DataEvent env escape hatch 删除：恒原子两阶段；缺 event_id fail-closed | `runtime/production_policy.py`、`incremental_scheduler.py` |
| P0-029 | `detect_stateful_behavior`：full vs chunked 行为检测，不依赖手工列表 | `stateful_contract.py` |
| P0-008/009 | 独立 oracle 参数域认证（11 核心算子全 window） | `scripts/audit_r34_parameter_domains.py` |

### 3. 参数域真实认证（P0-008 核心成果）
- `scripts/audit_r34_parameter_domains.py`：**独立 numpy reference**（不 import 生产
  kernel），对 `ts_mean/ts_std/ts_var/ts_sum/ts_max/ts_min/ts_zscore/rank/
  cs_pct_rank/cs_demean/c_demean` 在 `window∈{1,2,5,20,60,120,252}` 上做
  NaN/Inf mask + 有限值 allclose 对齐。
- **11/11 全部 certified**。这直接推翻 primitive evidence 79/79 的
  `{"bounds":["default"]}` overclaim——本次只认证**实际测过**的参数点。

### 4. Canonical Correctness Ledger（P0-004，§185）
- `scripts/build_r34_ledger.py` → `R34_CANONICAL_CORRECTNESS_LEDGER.csv/.json`，
  1391 个 production canonical 的 typed_signature / edge_declared / edge_verified /
  parameter_domain / stateful_class / production_verdict 逐维记录。

### 5. Hard Gates 审计（§186）
- `scripts/audit_r34_hard_gates.py` → `R34_HARD_GATES.json`（50 gates：
  13 PASS / 32 NOT_RUN / 5 FAIL）。

---

## 三、诚实暴露的 5 个 FAIL（真实缺口，未掩盖）

1. **`R34_ALL_PRODUCTION_CANONICALS_TYPED_SIGNATURE`** — 1391 个 production target
   中仅 **33 个**有 typed signature（1302 无、56 空参数）。P0-011 "typed signature
   唯一权威"远未达成。metadata 口径：275/1391 有 param_specs。
2. **`R34_ALL_PRODUCTION_CANONICALS_EDGE_DECLARED`** — 仅 29 个声明 edge contract，
   1362 个 undeclared。
3. **`R34_ALL_PRODUCTION_CANONICALS_REQUIRED_EDGE_VERIFIED`** — 1362 个缺 required
   edge evidence。
4. **`R34_ZERO_GENERATED_MODEL_CONTRACT_PRODUCTION_ADMISSION`** — 部分 model-like
   canonical（`calendar_day_diff`、`cs_autoencoder_*` 等）无显式 reviewed
   ModelTimingContract（P0-025 新门立即捕获，含少量分类假阳性待复核）。
5. **`R34_ALL_PREDICTIVE_MODELS_EXPLICIT_TIMING`** — 同上（model timing 显式性）。

这 5 个 FAIL 是**新证据框架第一次诚实揭示**的规模级缺口，也是下一轮的核心工作。

---

## 四、32 个 NOT_RUN（诚实 PENDING，未假完成）

真实 DataAccess PIT golden（financial restatement / index / holder / event / minute
session）、全后端参数域 parity、execution-variant 认证、checkpoint/append/chunk
parity、read-after-write、write failure injection、R33 optimized==reference、
future perturbation、multi-horizon label maturity——这些需要真实 DataAccess
parquet/registry fixture 或 R33 完成，本机不配置 = **NOT_RUN + 原因**，不编造。

---

## 五、既有失败（与 R34 无关，复核确认）

| 失败 | 原因 |
|---|---|
| `test_production_all_runtime_excludes_research_tools` | `cube` legacy-only 不在 allowlist（R30 tombstone 后遗留，stash 复核确认 pre-existing） |
| `test_garch_shock_uses_h_t` | GARCH 数值（numpy 2.2 环境，stash 复核确认 pre-existing） |
| `tests/runtime/test_r14_event_atomic_publish_*.py` (6) | R32 生产日历门 × R14 forced-production 测试冲突（trading_calendar `CalendarUnavailableError`）；与 R34 改动无关 |
| `test_r7_evidence_tcb` 系列 | R26 并发修改 registry，pre-existing |

---

## 六、R34 §190 DoD 逐问

| Q | 答案 |
|---|---|
| Q1 公式为什么对？ | 核心 11 算子有独立 numpy oracle；全量独立 golden 目录 PENDING |
| Q2 window=252 认证了吗？ | 11 个核心 rolling/截面算子已认证（含 252） |
| Q3 基本面无未来数据？ | 真实 DA source PIT proof NOT_RUN（需 DA fixture） |
| Q4 DuckDB 有资格跑？ | backend 参数域 parity NOT_RUN |
| Q5 batch==single？ | 轻量探针通过；全量 batch parity NOT_RUN |
| Q6 增量==全量？ | NOT_RUN（change-impact 逻辑已存在） |
| Q7 落盘读回相同？ | read-after-write NOT_RUN |
| Q8 证据是当前代码？ | **是**——evidence 绑定 `8449d9c` + working-tree hash，35 个旧 artifact 全部标记 stale |
| Q9 快路径改结果？ | R33 optimized==reference NOT_RUN |
| Q10 最后快不快？ | correctness-qualified TimeToDurableCommit NOT_RUN（R33） |

---

## 七、交付物（docs/evidence/r34/）

```
R34_HEAD.json                  EvidenceHeader 13 字段绑定当前 HEAD
R34_EVIDENCE_FRESHNESS.json    35 个 stale artifact 显式标记
R34_AUDIT_NEGATIVE_CONTROL.json 硬编码 True gate 扫描结果
R34_EVIDENCE_TRUTH_GATES.json  12/12 PASS
R34_PARAMETER_DOMAIN_COVERAGE.csv/.json  11 算子独立 oracle 参数域
R34_CANONICAL_CORRECTNESS_LEDGER.csv/.json  1391 canonical 逐维 ledger
R34_HARD_GATES.json            50 gates（13 PASS / 32 NOT_RUN / 5 FAIL）
R34_FINAL_ACCEPTANCE_REPORT.md 本报告
```

新增代码:
```
runtime/r34_evidence.py               Evidence 框架
scripts/audit_r34_evidence.py         Evidence Truth gates
scripts/audit_r34_parameter_domains.py  独立 oracle 参数域认证
scripts/build_r34_ledger.py           correctness ledger
scripts/audit_r34_hard_gates.py       §186 hard gates
tests/operators/r34/test_evidence_framework.py  12 tests
```

修复文件:
```
cleaned_operators/availability_clock.py   P0-018
cleaned_operators/operator_spec.py        P0-021/022/037
cleaned_operators/model_timing.py         P0-025
cleaned_operators/edge_requirements.py    P0-035
cleaned_operators/operator_surface.py     P0-013（manifest 补全）
cleaned_operators/production_hardening.py P0-037
runtime/production_policy.py              P0-039
runtime/incremental_scheduler.py          P0-039
runtime/engine.py                         P0-019
stateful_contract.py                      P0-029
scripts/audit_r30_hard_gates.py           P0-002（去硬编码 True）
```

---

## 八、下一轮最高优先级（按任务书 §194）

1. **Typed signature 全覆盖**（1391 production canonical）——最大缺口
2. **Edge contract 声明 + 验证全覆盖**（1362 个）
3. 真实 DataAccess fixture → source PIT golden
4. 独立 semantic golden 全目录（不止核心 11 族）
5. 后端参数域 parity + execution-variant
6. R33 完成后：optimized==reference、TimeToDurableCommit
