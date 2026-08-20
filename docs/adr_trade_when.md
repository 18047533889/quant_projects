# ADR：`trade_when` 状态语义

> **第 10 版更改-shw**：与 `bucket` / `ts_step` / `hump` 实装同步归档，便于回测与挖掘对齐。

## 决策

对 MultiIndex `(timestamp, instrument)` 数据，**按每个 `instrument` 内时间升序**维护标量状态 `position`（对外输出的 alpha 值，未持仓为 NaN）。

1. **exit**（第三参数）：若当前 bar 上 `exit > 0.5` 或 `==1.0`，则 **`position ← NaN`**（强制平仓/清空），本 bar 输出为 NaN。
2. **trigger**（第一参数）：若本 bar **未**因 exit 清空，且 `trigger > 0.5` 或 `==1.0`，则 **`position ← alpha`**（第二参数）当前值（alpha 为 NaN 则 position 为 NaN）。
3. **否则**：输出 **上一 bar 的 position**（保持持仓值不变）。
4. **trigger / exit / alpha 为 NaN**：视为假；不触发更新（exit 的 NaN 不触发清空）。

## 非目标

- 未实现 WQ 可能存在的更细撮合延迟、费用、部分成交。
- 多腿 / 组合层级持仓不在此算子内表达。

## 实现

- [`backend/pandas_backend.py`](../backend/pandas_backend.py)：`PandasBackend._op_trade_when`
