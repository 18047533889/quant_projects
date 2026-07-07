# `planner` — 编译与计划层（详尽说明）

把 **`ir.IRNode`** 变为 **`PlanNode`**（逻辑计划），并做 **常量折叠、多因子 DAG、公共子表达式消除（CSE）** 等，为 **`backend`** 提供 **唯一、可哈希** 的执行树。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**IR → PlanNode**（**`lowerer`**）→ **可选优化**（**`optimizer`**）→ 多因子时 **CSE**（**`cse`**）得到 **`DAGPlan`**（共享子式字典 + 各因子根）。
2. **和 `runtime` 的关系**：**`FactorEngine.compile` / `compile_many`** 走这里；环境变量 **`FACTOR_ENGINE_DISABLE_CSE`** 关 CSE。
3. **读代码顺序**：**`logical_plan.py`**（`PlanNode`）→ **`lowerer.py`** → **`cse.py`** / **`dag.py`**。
4. **边界**：**不访问磁盘**；**不执行算子数值**，只变换计划结构。

---

## 1. 核心类型

### [`logical_plan.py`](logical_plan.py)

- **`PlanNode`**：字段通常包括 **`op`**（字符串）、**`inputs`**（子 `PlanNode` 列表）、**`attrs`**（dict）。  
- 与 IR 结构 **同构**；`Lowerer` 做 **几乎无变换** 的结构拷贝。

### [`lowerer.py`](lowerer.py)

- **`Lowerer.to_logical_plan(ir: IRNode) -> PlanNode`**：递归 **IR → PlanNode**，tuple 子节点 → list。

### [`optimizer.py`](optimizer.py)

- **`Optimizer.optimize(plan: PlanNode) -> PlanNode`**：在计划上做 **常量折叠** 等 **安全重写**；规则见 `rules.py`。

### [`physical_plan.py`](physical_plan.py)

- **占位**：`PhysicalPlan` dataclass 尚未接入编译链；当前 **`FactorEngine`** 仍直接执行 **`logical_plan.PlanNode`**。勿在此扩展直到 lowerer 接线完成。

### [`dag.py`](dag.py)

- **`FactorPlan`**：`factor_name` + **根 `PlanNode`**。  
- **`DAGPlan`**：  
  - **`roots`**：多个因子各自的 `FactorPlan`；  
  - **`shared_nodes: dict[str, PlanNode]`**：**CSE** 后抽出的 **共享子树根**，键为稳定 id（见 `plan_hash` / `cse`）。

### [`cse.py`](cse.py)

- **`apply_cse(plans: list[PlanNode]) -> (new_plans, shared_dict)`**：多因子间 **重复子树** 合并，减少 **`backend.execute`** 次数。  
- 默认 **`FactorEngine.compile_many`** 开启；可用 **`FACTOR_ENGINE_DISABLE_CSE=1`** 或 `PerfConfig` 关闭。

### [`plan_hash.py`](plan_hash.py)

- 对 **`PlanNode` 子树** 做 **结构化哈希**，供 CSE **判同** 与缓存键。

### [`rules.py`](rules.py)

- 优化 **规则** 抽象基类与注册（若使用）。

---

## 2. 多因子执行语义（与 `runtime` 对齐）

1. **`compile_many`**：每个因子 `compile()` → 多个根计划。  
2. **`apply_cse`**：生成 **`DAGPlan`**，`shared_nodes` 内子树 **只执行一次**，结果放入 **`ExecutionContext.shared_result_cache`**。  
3. **`run_many`**：先顺序执行 **`shared_nodes`**，再对每个因子根 **`execute`**。  
4. **`run_many_parallel`**：共享子式仍 **串行**；**各因子根** 可 **Joblib 线程并行**（默认 threading，避免进程 pickle 大数据）。

---

## 3. `plan_ref`（概念）

当 CSE 重写计划时，子树可能被替换为 **引用节点**（具体见 `cse.py` 与后端对 `op` 的处理）；**`run_many`** 必须走 **`FactorEngine.run_many`**，单因子 **`run`** 不处理引用。

---

## 4. 测试

- `tests/test_planner.py`  
- `tests/test_cse_run_many.py`  

---

## 5. 延伸阅读

- [`ir/README.md`](../ir/README.md)  
- [`runtime/README.md`](../runtime/README.md) — `compile` / `compile_many` / `run_many`  
- [`runtime/perf_config.py`](../runtime/perf_config.py)  
