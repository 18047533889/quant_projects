# Assetization 模块工程实现

## 一、factor_id 生成机制

### 坐标提取
```python
_COORDINATE_KEYS = ("signal_structure", "asset_class", "frequency_bucket", "domain_root")

def extract_coordinates(config):
    # 从 Config 提取四元坐标
    return {k: str(config[k]) for k in _COORDINATE_KEYS}
```

### ID 生成
```python
def generate_factor_id(coordinates, seed=None):
    # 格式: {asset_class}_{frequency_bucket}_{signal_structure}_{suffix8}
    # 示例: equity_1d_cross_sectional_224d2523
    #
    # seed=None → suffix8 为随机 UUID 前缀
    # seed=非空 → suffix8 为 seed 的稳定哈希前缀（幂等重跑）
    #
    # 实际 seed 包含 candidate_id + formula + Config 的 JSON 序列化哈希
```

## 二、公式求值引擎

```
formula 字符串
    → tokenize()               # 词法分析（正则分词）
    → build_ast()              # 语法分析（构建计算树）
    → evaluate(ast, market_df) # 自底向上求值，调用算子库
    → np.ndarray 因子值
```

使用 `assetization/scripts/compute.py` 中的轻量级词法/语法分析器，算子实现引用外部算子库（`lqtp_data_fetch/operators`）。

### FormulaEvaluator API
```python
class FormulaEvaluator:
    def evaluate(self, formula: str, market_df: pd.DataFrame) -> np.ndarray:
        """解析并计算公式值。
        
        - 公式中的字段名从 market_df 列中按名称查找
        - 支持 "Table.Field" 格式引用
        """
```

### 双后端支持（通过 compute_engine.py）

| 后端 | 实现 | 适用场景 |
|------|------|---------|
| `pandas` | pandas Series/DataFrame 操作 | 单机、小规模数据 |
| `polars` | polars 惰性/ eager API | 大规模数据、并行计算 |

`compute_engine.py` 包装 `factor_engine` 模块，对外暴露 spec 约定的四个方法：
- `parse_and_optimize`：解析 DSL 为 Expr 树 → IR → LogicalPlan → 优化 → PhysicalPlan
- `execute_plan`：在指定后端上执行 PhysicalPlan
- `asof_join`：非对齐时间轴关联
- `atomic_materialize`：原子化物化（解析 + 执行 + 写入）

## 三、输入校验

- 行情数据必需列：`TradeDate` / `datetime`, `Symbol` / `asset`
- 公式引用列校验：字段名在 market_df 列中存在的检查
- Gateway 输入兼容：从 `afv.json` 读取，校验 `Gateway.Label == "Pass"`
- 依赖上游 `manifest.json` 中的 `domain_root`, `frequency_bucket` 等元信息

## 四、输出校验

- 空值剔除：`factor_value` 列为 NaN 的行会被自动剔除
- 写入 `data/{YYYY-MM-DD}.parquet` 按日分片存储
- 写入 `afv.json`（追加 Assetization 段，保留上游信息）
- 复制 `manifest.json`（保留上游元信息，追加 `factor_id`）

## 五、路径管理

Assetization worker 不再有代码级默认路径。所有路径（`market_data_path`、`output_base`、`temp_base`）必须从外部配置传入，由 `config_manager.py` 或上层调度器负责解析 `all_configs/auto_factor_evaluation/config.yaml` 中的路径定义。
