# Review #8 执行报告（2026-08-09）

审计基线 `main@2b073f79` 之后的增量收口。本轮处理 #393 起的新增确认问题与
"静态无法完全证明、必须测试验证" 的疑点。原则：静态确认 → 直接修根因；
静态无法确认 → 先写性质测试，失败修根因，通过保留为永久回归测试。

## 一、已修复（fixed，含回归测试）

### PIT 契约（#404）
`pit_contract.py::pit_asof_join`
- **根因**：`pd.merge_asof(by=...)` 要求 asof key **全局**单调；旧实现按
  `[instrument, decision_time]` 排序，多标的时 `Jan1 Jan2 Jan1 Jan2` 非单调 →
  `ValueError: left keys must be sorted`。
- **修复**：改为 `[decision_time, instrument]` / `[available_at, instrument]` 排序 +
  `__pit_row_order__` 恢复调用方行序。
- **测试**：2 标的精确匹配、100 标的 + 乱序 + 重复行（
  `test_round6_alignment_pit_2026_08.py` 新增 2 例）。

### 物化失败态（#440/#441）
`storage/materialize/materializer.py`
- **根因**：(a) watermark 在 `MaterializePartitionError` 抛出**之前**已更新 →
  失败 run 仍推进水位线；(b) 全部分区失败时在空短路上**正常返回**
  `rows_written=0`。
- **修复**：失败判定移到最前 —— 任何 required 分区失败即抛错、不推进 watermark；
  空返回只在无失败且无写/跳过时发生。
- **测试**：部分失败 watermark 不推进、全部失败抛错（materializer 新增 2 例）。

### 检查点主键（#438）
`storage/catalog.py`
- **根因**：`PRIMARY KEY (factor_id, partition_year)`；`checkpoint_year` 把 day 分区
  20260801/20260802 都编码为 202608 → 第二天撞 PK。
- **修复**：重建表，PK 改为 `(factor_id, partition_key)`；`partition_year` 退化为
  检索列。旧库自动迁移。
- **测试**：legacy 迁移、day 分区共存、全新库 PK（3 例）。

### 物化原子性 / 并发（#442/#443 + 连带）
`storage/materialize/materializer.py`、`storage/catalog.py`
- **修复**：
  - temp 文件名带唯一 uuid 后缀，多进程不冲突；
  - 写后 `fsync` 文件 + 目录，`os.replace` 崩溃后文件要么旧要么新；
  - 每分区 `flock` 锁串行 read→merge→replace，无 lost update；
  - orphan `.tmp` reconcile 移入分区锁内 —— 修掉了"锁外清扫误删他人正在写的
    temp → os.replace 失败"的真实竞态；
  - `FactorCatalog.register` 的 check-then-insert 竞态 → `INSERT OR IGNORE` +
    post-insert hash 校验；
  - catalog 写路径加 SQLITE_BUSY 有界重试（多进程写 SQLite 单写锁突发）。
- **测试**：4 进程并发写同分区无 lost update（×5 稳定）、orphan tmp 清理、
  唯一 temp（3 例）。

### Lineage（#483/#484/#452）
`runtime/lineage.py`
- `non_null_count`：DataFrame 结果 `int(result.notna().sum())` 抛错 →
  `np.asarray(...).sum()`；
- `hash_data_source_config`：canonical 化（有限 float、ISO datetime、pydantic）、
  `allow_nan=False`（NaN/Inf 拒绝）、secret redaction（password/token/secret/…）、
  不支持对象拒绝（不再泄漏 address 的 `str()`）。
- **测试**：DataFrame/Series/empty non_null、NaN/Inf 拒绝、secret 不进入 hash、
  datetime/numpy scalar 规范化、unsupported 拒绝（7 例）。

### Registry closure hash 确定性（#453）
`cleaned_operators/registry.py`（并发会话已用 `_freeze_value` 修复）→ 本会话
**补充验证**：closure cell 分支同 payload 同 hash / 不同 payload 不同 hash /
进程内稳定（`test_r7_registry_governance.py` 新增 3 例）。

### SourceRef 标量 / 整数 / 重复参数（#468/#469/#470）
`api/source_ref.py`、`api/lqtp_compat.py`、`storage/sources/lqtp_logical_source*`
- `_scalar` 已拒 NaN/Inf（前轮）；新增：
  - `_strict_int`：拒绝 `True/1.9/"2"/NaN/Inf` 静默变成 1/2；integral float 2.0 接受；
  - `source_col` 位置 + 关键字重复参数拒绝；
  - transform 参数 allow-list（`financial_lag.quarters`、`minute_bar.period/index` 等）——
    未消费参数拒绝；`with_transform` 也过 `_scalar` + allow-list。
- 移除 resolver / compat 里的 `int()` 静默强转，统一由 `_strict_int` 裁决。
- **测试**：NaN/Inf、`_strict_int` 拒绝集、重复参数、未消费参数（5 例）。

### ParamSpec（#461/#462）
`cleaned_operators/base.py`
- #461 `MISSING` sentinel（并发会话已做）→ 验证；
- #462 `RelationalParamSpec` 的 `eval()` → **restricted predicate AST**：
  `ast.parse` + allow-list（`+ - * / // **`、`< <= > >= == !=`、`and/or/not`、
  一元正负），函数调用/属性/下标/字符串字面量/容器在构造期拒绝；运行时用受限
  interpreter 求值（NaN/缺失参数 → 视为不满足，fail 到可行组合）。
- **测试**：合法算术/比较、FloorDiv/Pow、NaN 未决、动态 Python 拒绝
  （len/`__class__`/下标/字符串/`in`/lambda/`__import__`）、whitespace 无关（7 例）。

### Composite join 语义（#474/#475/#476）
`storage/sources/composite_source.py`
- #474 join report 拆分 `key_matched/key_unmatched/value_valid/value_null`（asof
  用 sentinel 列判定 key 命中，与值 null 解耦）；`matched_rows` 保留为 value-valid
  兼容别名；
- #475 asof tolerance 拒绝负值；
- #476 asof 源 `(timestamp, instrument)` 重复行 fail-closed。
- **测试**：key 命中但值 null、key 未命中计数、负 tolerance、重复 key（4 例）。

### 列身份（#G）
`planner/rewrite_fastpath.py::_same_column`、`planner/rolling_cache.py::_column_full_ref`
- **根因**：rewrite/CSE 只按 `name` 比较列 → `sourceA.Close` 与 `sourceB.Close`
  被当作同一列，rolling 结果可能跨源错误复用。
- **修复**：列节点携带 analyzer 写出的 `source_table/field_id/source_field/
  field_registry_hash` 时，identity 参与比较；synthetic 列退回 name。
- **测试**：`_same_column` 区分同名校不同源、CSE semantic key 区分（2 例）。

### 物化 null/tombstone（#444/#445/#446）
`storage/materialize/materializer.py`
- 并发会话已将 NaN 行默认作为显式 tombstone 保留（R9-P0-027），修复了默认
  dropna 导致 valid→NaN 修订无法覆盖旧值的问题；本会话补充 `null_overwrite` /
  `deleted_keys` API 与 `tombstoned_rows` 遥测，并保留回归测试。
- **测试**：valid→NaN 覆盖旧值、`deleted_keys` tombstone、roundtrip NaN 图案（3 例）。

## 二、性质测试（test-proved-safe，永久回归）

`tests/operators/test_review8_properties_2026_08.py`（11 例）
- **B** prefix causality：`ts_mean/ts_rank/ts_std` 前缀计算 == 全量在 t 时刻值；
- **D** 轴变质：cs_rank 列置换不变；group label 重编号不变（group_mean）；
- **E** 单位/尺度变质：`cs_rank/ts_log_return/ts_zscore` 对价格×10 不变；
  `ts_mean` ×10、`ts_cov` ×10；
- **F** rewrite 等价：`ts_zscore == (x-mean)/std`、`ts_log_return == ln(x/x.shift)`
  （含 NaN / 常量 / 零 std 对抗输入）；
- **G** 列身份（见上）。

## 三、并发会话协调
本会话期间另一 Claude 会话持续编辑同一棵树（`pit_contract.py` 新增
`_market_visible_shift` R9-P0-019、`materializer.py` 默认 tombstone R9-P0-027、
`parameter_canonicalizer.py` R9-P1-038）。冲突处已协调：
- materializer 的默认 tombstone 与我的 `null_overwrite/deleted_keys` API 并存
  （48/48 通过）；
- ParameterCanonicalizer 测试与 R9-P1-038 代码矛盾（bool 不是数字）→ 测试对齐
  代码契约；
- 受其 WIP 影响的 load_all 偶发失败已随其提交消退。

## 四、回归统计
- `test_materializer.py` 48 passed（×3 稳定，含并发 race）；
- `test_composite_source.py` 8 passed；
- `test_lineage_review8_2026_08.py` 7 passed；
- `test_review8_properties_2026_08.py` 11 passed；
- `test_round8_infra_2026_08.py` 31 passed（含本会话新增 12 例）；
- `test_r7_registry_governance.py`（closure 3 例）+ `test_round6_alignment_pit_2026_08.py`（2 例）通过；
- 合并回归：**150 passed / 0 failed**。

## 五、本轮未做（需稳定树 / 属并发会话 WIP）
- P0 availability contract 全套（#393-#403）：`AvailabilityContract` /
  `DecisionContext` / `factor_visible_at <= decision_time` —— 并发会话正在做
  （已见 `_market_visible_shift`）；
- DataSource run-mode / snapshot pin（#406-#410）；
- Incremental certificate / FactorDefinition identity / dependency catalog
  （#429-#437）；
- Static OperatorSignature 收敛到单一 BoundCall（#463-#467）；
- Calendar version 进 checkpoint/cache（#477-#479）；
- 大规模差分/故障注入套件（A/C/H/I/J/K/L/M 等）—— 建议在树稳定、证据重生成后
  作为 nightly audit 加入。

## 六、建议的最终收口顺序（树稳定后）
```
python scripts/certify_primitive_evidence.py    # ~10min，全 parity + convergence
python scripts/generate_layer_manifests.py
python scripts/generate_operators_catalog.py
# 全量回归：operators / backend_parity / backend_sql / market
```
