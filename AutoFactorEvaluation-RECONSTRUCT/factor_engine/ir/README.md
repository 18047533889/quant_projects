# `ir` — 中间表示 IR

**IR** 位于 **`Expr`** 与 **`PlanNode`** 之间：扁平、`op` 为字符串，便于依赖分析与后端分派。

### 协作者速览

1. **入口**：`Analyzer().lower(expr)` → **`AnalysisResult`**（`ir`、`lookback`、`has_ts_op`、`has_cs_op`、`referenced_columns`）。  
2. **节点**：[`nodes.py`](nodes.py) 的 **`IRNode(op, inputs, attrs)`**。  
3. **Expr 类型**：仅处理 **`ColumnRef` / `Literal` / `CleanedCall`**（见 [`analyzer.py`](analyzer.py)）。

---

## 1. 为何需要 IR

| 目标 | 说明 |
|------|------|
| 依赖列 | 收集 `column` 节点 → 告知 `DataSource` 预拉列 |
| 统一后端 | Backend 只认 `op` 字符串 + 子树 |
| lookback | 从窗口参数推断最小历史长度 |
| CSE | Plan 层对稳定 `op`+结构做公共子式合并 |

---

## 2. 文件说明

| 文件 | 内容 |
|------|------|
| [`nodes.py`](nodes.py) | `IRNode` 定义 |
| [`analyzer.py`](analyzer.py) | `Expr` → IR；别名 canonical 化（如 `delay` → `ts_delay`） |
| [`schema.py`](schema.py) | 类型 / 形状推导辅助 |
| [`types.py`](types.py) | IR 层类型枚举 |

---

## 3. CleanedCall 降级规则

- `CleanedCall.op` 经 `OperatorRegistry._aliases` 转为 **canonical** 写入 `IRNode.op`。  
- positional 第二参数若为 `Literal` 整数，参与 **lookback** 推断。  
- `kwargs` 必须为字面量（非 `Expr`），否则 `NotImplementedError`。

---

## 4. 与占位算子的关系（2025+ cleaned 架构）

- **`api.operator_registry.STUB_IR_OPS`** 现为 **空集**（旧 `expr/*_stub` 模块已移除）。  
- 未在 `cleaned_operators` 注册的算子 **无法 parse 进 IR**。  
- 仅 catalog、无 pandas runtime 的名字 **不在 DSL 白名单**。

---

## 5. 测试

- `tests/test_planner.py`、`tests/test_cleaned_operators_comprehensive.py`（`TestIrLowering`）  
- `tests/test_end_to_end.py`

---

## 6. 延伸阅读

- [`expr/README.md`](../expr/README.md)  
- [`planner/README.md`](../planner/README.md)  
- [`runtime/engine.py`](../runtime/engine.py) — `compile()`
