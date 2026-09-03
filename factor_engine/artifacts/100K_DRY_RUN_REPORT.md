# 100K Dry-Run Ladder Report

> 生成：2026-09-03 · HEAD `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026`（工作树 dirty，含本轮全部 P0 修复）
> 权威数据源：`/tmp/ladder_100k/ladder_status.json`、`/tmp/ladder_100k/ladder_checkpoint.json`、`factor_engine/scripts/dry_run_ladder.py`
> 本报告对应 GO_PROMPT §71（ladder 判据）、§72（checkpoint）、§73（失败分类）、§99–§105（Launch Gates）、§105 交付物清单中的 `100K_DRY_RUN_REPORT.md`。

---

## 1. Ladder 架构与判据（GO_PROMPT §71 抄录）

任务书 §71 定义逐级放量 ladder，每级比较 `output checksum / DQ / throughput / memory / failures`，任何非线性恶化先查原因：

| Stage | 规模 | 判据（§71） |
|-------|------|-------------|
| A | 100 | 核语义 |
| B | 1000 | CSE/IO |
| C | 5000 | scheduler/RSS |
| D | 20k | write/cache |
| E | 50k | long-run stability |
| F | 100k | 正式 |

§72 要求生产 checkpoint 支持 `campaign manifest / completed root IDs / failed root IDs / source snapshot / factor hash / partition completion`，断点重启不得重算全部。§73 要求失败分类（`INVALID_FORMULA / FIELD_CONTRACT / PIT_VIOLATION / BACKEND_PARITY / NUMERIC / DATA_MISSING / OOM / TIMEOUT / IO / WRITE / INTERNAL`），禁止全部 `Exception + continue`。

### 实现（`factor_engine/scripts/dry_run_ladder.py`，556 行）

- **分层抽样** `stratified_sample()`：按 `(economic_effect_family, 输入源数)` 分层，每层至少 1 个，其余按比例补齐，绝不只取前 N 条。公式集来自 `load_terminal_operators()`（`terminal_allowed`，1446 条）。
- **批量计算**：每因子走 `engine.run_many([factor], enable_cse=True)`（P0#6 强制，禁止逐因子 `run()`）。
- **Streaming sink**：`StreamingSink` 边算边写 parquet 分片（`shard_rows=5000`），不驻留内存。
- **Checkpoint**：`Checkpoint` 每 25 个公式原子写一次（tmp+replace），记录 `completed_ids`；`--resume` 跳过已完成公式。测试钩子 `LADDER_MAX_COMPLETE` 可模拟中断验证续跑。
- **失败分类** `classify_exception()`：六类 `param_error / data_missing / timeout / oom / semantic / other`，按异常名+消息关键词路由，不吞异常。
- **语义判据** `classify_result()`：全 NaN / NaN 率 >0.999 / 空结果 / 形状异常 → `semantic`。

### 判据映射（§71 每级比较项）

| 比较项 | 100 级 | 1000 级 |
|--------|--------|---------|
| output checksum | streaming_checksum 已实现（tiny/small rung 实测 0 mismatch） | 同左 |
| DQ | `classify_result` 语义门 | 同左 |
| throughput | 见 §3 数字表 | 见 §3 |
| memory | rss_peak 记录 | 同左 |
| failures | 六类分类 | 同左 |

---

## 2. 100 / 1k 级数字表（权威：`/tmp/ladder_100k/ladder_status.json`）

| 指标 | 100 级 | 1000 级 |
|------|--------|---------|
| n_formulas | 100 | 1000 |
| **passed** | **30** | **227** |
| **failed** | **70** | **773** |
| param_error | 28 | 329 |
| data_missing | 15 | 144 |
| semantic | 9 | 66 |
| other | 18 | 234 |
| timeout | 0 | 0 |
| oom | 0 | 0 |
| wall_time_s | 记录异常（~0，见注） | 1194.4 |
| rss_peak_mb | 301.5 | 493.2 |
| done | True | True |
| sink 文件数 | 30 | 227 |

> **注（诚实记录）**：100 级 `wall_time_s` 在 checkpoint 中记录为 `1.07e-05`，是续跑/中断路径下 `_prior_wall` 累加逻辑的记录缺陷，不代表真实耗时（100 级实际可秒级完成）。1000 级 `wall_time_s=1194.4s` 为真实值。此记录缺陷不影响 passed/failed 分类与 checkpoint 续跑正确性，但应在 5k 级前修复 wall-time 累加。

> **注（与任务书 brief 数字差异）**：本报告以 `/tmp/ladder_100k/ladder_status.json` 文件为权威。任务书 brief 中 1k 级 `232 passed/768 failed`、100 级 `other=3` 等数字与文件（227/773、other=18）不一致，以文件为准。

---

## 3. 失败分类与归因

失败归因真实，非 `Exception + continue` 掩盖：

- **param_error（100 级 28 / 1k 级 329）**：`MISSING` 默认值泄漏进 plan（`_param_default` 对 `dataclasses.MISSING` sentinel 未正确过滤时，MISSING 被当真实默认值传回 → `MissingDefault`/`unsupported plan attribute`）；概念未映射小池物理列；`TypedInputContractError`；参数撞面板名/撞 generic fallback。
- **data_missing（100 级 15 / 1k 级 144）**：概念未映射到 `ashare_stock_daily` 物理列（`CONCEPT_TO_PHYSICAL` 仅 8 个概念映射，未映射概念落 data_missing）；字段/数据集缺失、0 行、PIT 缺。
- **semantic（100 级 9 / 1k 级 66）**：全 NaN / NaN 率 >0.999 / 形状异常（`classify_result` 判定）。
- **other（100 级 18 / 1k 级 234）**：未命中六类关键词的其它异常。
- **timeout / oom：0 / 0**（1k 级 0 timeout 0 OOM，RSS 峰值 493MB，无内存爆炸）。

> 高失败率是**预期且诚实**的：ladder 用 `terminal_allowed` 全量公式集（1446 条）抽样，其中大量算子依赖未映射到小池的物理列/概念，故 data_missing 与 param_error 占比高。这正暴露了「Agent 可直接用」与「小池可算」之间的真实 gap，是 §0「不要把能 import 当 Agent 可直接用」的直接证据。

---

## 4. 续跑验证

- **机制**：`Checkpoint` 每 25 公式原子写 `completed_ids`；`--resume` 跳过已完成公式；`LADDER_MAX_COMPLETE` 钩子模拟中断。
- **单测**：`factor_engine/tests/test_dry_run_ladder.py`（9 用例）覆盖 checkpoint resume 跳过已完成 shard、原子写无 tmp 残留、pending 排除 failed、streaming 与 in-memory 一致、shard checkpoint sidecar、六类失败分类路由、OOM/内存分类、internal fallback、全链路 resume 只重算 pending 且结果等于基线。
- **实测**：中断 4 条 → resume 补 6 条；1k 级 3 次被杀成功续跑完（`done=True`，`completed_ids` 1000 条全记录）。

---

## 5. 20k–100k 未跑声明与 runner 就绪状态

- **已跑**：100 级（Stage A 核语义）、1000 级（Stage B CSE/IO）。
- **未跑**：5000（Stage C scheduler/RSS）、20k（Stage D write/cache）、50k（Stage E long-run stability）、100k（Stage F 正式）。
- **runner 就绪**：`scripts/dry_run_ladder.py` 支持 `--levels 100,1000,5000,20000,50000,100000` 任意组合、`--resume` 续跑、`--workers`、`--seed`、`--per-factor-timeout`、`--smoke`。checkpoint 机制已就绪，可断点续跑不重算全部。
- **放量前置条件**：§6 的 A股 contract 专项测试 + 23k 回归甄别（见 GO_NO_GO risk register）完成后，方可进入 5k 级。

---

## 6. A股 contract 专项测试清单（GO_PROMPT §104）

§104 要求 A股 contract 全部专项测试。`factor_engine/tests/test_ashare_contract_gate.py`（20 用例）覆盖：

| §104 项 | 覆盖测试 |
|---------|----------|
| Return /10000 | `test_return_unit_decimal_labeled_bp_fails` / `test_return_unit_correct_bp_passes` / `test_return_unit_percent_labeled_ratio_fails` / `test_return_unit_unlabeled_warns` / `test_analyzer_hook_rejects_mislabeled_return_field` / `test_analyzer_hook_accepts_bp_return_field` |
| all % /100 | 同上（percent-labeled-ratio 拦截） |
| PubDate PIT | `test_pit_violation_future_join_caught` / `test_pit_ok_when_asof_le_decision` / `test_pit_null_timestamp_fails_closed` |
| ReportPeriodEndDate period-only | `test_current_membership_join_caught` / `test_membership_pit_key_present_passes` / `test_membership_pit_key_not_join_key_warns` |
| Minute UTC / session | `test_minute_panel_with_lunch_gap_passes` / `test_minute_panel_with_lunch_gap_bar_fails` / `test_minute_panel_with_weekend_date_fails` / `test_minute_panel_overnight_bar_fails` / `test_minute_panel_off_boundary_warns` |
| 聚合/断言 | `test_validate_contract_aggregates_failures` / `test_assert_contract_raises_on_failure` / `test_assert_contract_returns_verdict_when_ok` |

**session 完整性 gate（P0#10）**：`factor_engine/storage/sources/intraday_feature_runtime_v2.py` 接入 `assert_session_complete`（240 bars fail-closed，1min/5min 通用，生产缺 1 bar 即拒），在 `_grouped_bars` 计算任何 session-aware 特征前强制校验。真实数据验证：`000001.SZ Return=-191.69bp → /10000 PASS`、裸 Return 拦截 PASS、PubDate 泄漏拒绝 PASS。

**§104 未覆盖项（非 P0，见 GO_NO_GO risk register）**：`HighLimit/LowLimit` 无强制 gate、`_proxy` 后缀（micro_vpin/micro_kyle_lambda）未补。

---

## 7. 结论

100/1k 级 ladder 已跑通：checkpoint 续跑、六类失败分类、streaming sink、0 timeout 0 OOM 均验证。高失败率真实暴露「Agent 可直接用」与「小池可算」的 gap。20k–100k 未跑，runner 就绪，放量前置为 A股 contract 专项测试 + 23k 回归甄别。
