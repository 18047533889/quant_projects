# cleaned_operators/docs — 算子文档

本目录存放 **算子说明与语义文档**，与上级目录中的 **runtime 实现代码**（`time_series.py`、`cross_sectional.py` 等）分开，便于直接找代码改逻辑。

## 文件

| 文件 | 说明 |
|------|------|
| [`算子全览.md`](算子全览.md) | 全量算子说明：含义、计算方式、LaTeX 公式、DSL 可用名 |
| [`operators_catalog.md`](operators_catalog.md) | 审计目录：canonical、别名、实现状态 |
| [`operator_doc_semantics.py`](operator_doc_semantics.py) | 算子语义与公式（生成器数据源） |
| [`_operator_doc_batch.py`](_operator_doc_batch.py) | 批量语义条目 |

## 重新生成算子全览

```bash
cd factor_engine
PYTHONPATH=. python3 scripts/generate_operators_guide.py
```

## 改算子代码

实现逻辑在上级目录，例如：

- 时序滚动 → [`../time_series.py`](../time_series.py)
- 截面变换 → [`../cross_sectional.py`](../cross_sectional.py)
- DSL 别名 → [`../_aliases.py`](../_aliases.py)

改完实现后，若语义有变，更新 `operator_doc_semantics.py` / `_operator_doc_batch.py`，再跑生成脚本。
