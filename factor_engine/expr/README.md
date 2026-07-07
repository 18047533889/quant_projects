# `expr` — 表达式树 AST

本目录定义 **因子表达式的数据结构**。迁移后 AST **仅三种节点**：

| 类型 | 文件 | 说明 |
|------|------|------|
| `ColumnRef` | [`column.py`](column.py) | `col("close")` |
| `Literal` | [`literal.py`](literal.py) | 数值 / 布尔常量 |
| `CleanedCall` | [`cleaned_call.py`](cleaned_call.py) | 任意 cleaned 算子调用 |

**不包含** 向量化计算；求值在 **`backend`** + **`cleaned_operators`**。

> 原 `arithmetic.py` / `ts.py` / `cs.py` 等强类型节点已删除；四则运算也表达为 `CleanedCall(op="add"|"subtract"|...)`。  
> 工厂函数见 [`api/`](../api/README.md)，不直接 `import expr` 拼业务公式（除非扩展引擎）。

### 协作者速览

1. **构造**：`api` 的 `col`、`from api import rank`、或 `parse_expr` → 本目录节点。  
2. **消费**：[`ir/analyzer.py`](../ir/analyzer.py) 遍历 `Expr` → `IRNode`。  
3. **四则**：[`base.py`](base.py) 的 `Expr.__add__` 等重载 → `CleanedCall`。

---

## 1. [`base.py`](base.py)

- **`Expr`**：抽象基类；子类实现 `children()`。  
- 运算符重载 `+ - * /` 与一元 `-` → **`CleanedCall`**（`add` / `subtract` / `multiply` / `divide` / `neg`）。  
- **`ensure_expr(x)`**：标量 → `Literal`，已是 `Expr` 则原样返回。

---

## 2. 叶子与调用

### [`column.py`](column.py)

- **`ColumnRef(name: str)`**：列名字符串；执行时由 `DataSource.load_column` 拉取。

### [`literal.py`](literal.py)

- **`Literal(value)`**：常量叶子。

### [`cleaned_call.py`](cleaned_call.py)

- **`CleanedCall(op, args, kwargs)`**：  
  - `op`：canonical 或别名（IR 层会再解析）  
  - `args`：`Tuple[Expr, ...]`  
  - `kwargs`：`Tuple[Tuple[str, Any], ...]`，仅 **字面量** 可进入 IR attrs

---

## 3. 编译链

```text
Expr (ColumnRef | Literal | CleanedCall)
  → Analyzer.lower()
      → IRNode(op, inputs, attrs)
          → Lowerer → PlanNode
              → cleaned_bridge → cleaned_operators
```

阅读 `CleanedCall` 时关注：**子节点顺序** = IR `inputs` 顺序；**kwargs** = Plan `attrs`。

---

## 4. 延伸阅读

- [`api/README.md`](../api/README.md)  
- [`ir/README.md`](../ir/README.md)  
- [`docs/operators_semantics.md`](../docs/operators_semantics.md)
