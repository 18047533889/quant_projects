# Exact-flat 多查询检索（2026-10-03）

FaissANNIndex.search_batch 接受 shape 为 (Q, D) 的实数 NumPy 矩阵，
返回长度为 Q 的列表，每个元素遵循标量 search 的结果语义。
整个矩阵先验证再调用一次 FAISS；任一行非法时拒绝整个 batch。
Q 为零时返回空列表；空索引对每个合法查询返回一个空结果列表。
k 与 min_similarity 沿用标量 API 的边界。batch 查询不排除自身。

精确平面索引只承诺按 cosine 分数降序。分数相同时不承诺 factor ID 顺序；
在 top-k 截断位置的相等分数也不承诺选中哪个 ID。独立 oracle 按文档
规定的 scale-safe 归一化步骤和 float32 FAISS 输入精度计算 cosine。

验证 workload 包含至少 1,000 个候选、32 个 query、全库 top-k 下的精确 tie、
正反方向、阈值过滤、零/NaN/Inf 拒绝、极大/极小/次正规幅度及输入不变性。
CPU A/B 脚本将一次 batch 调用与 32 次标量查询交错计时；它不涉及
SimilarityCache 的既有索引，也不代表真实市场数据吞吐量。

## server-c 验收结果

新增批量用例：PYTHONPATH=. .venv/bin/python -m pytest -q factor_assets/tests/similarity/test_ann_batch.py，
5 passed。导出路径 factor_assets.similarity.BatchANNIndex 与
factor_assets.similarity.ann.BatchANNIndex 已核验。

CPU A/B 使用 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1，每种配置
9 次交错计时，独立 cosine oracle 均通过：

- 1,024 candidates × 64 dimensions，32 queries，k=10：batch 2.026 ms，标量循环
  3.259 ms，1.61×。
  冻结源码复跑为 batch 2.841 ms、标量循环 3.953 ms（1.39×）；两轮独立列示。
- 4,096 candidates × 64 dimensions，64 queries，k=10：batch 5.128 ms，标量循环
  7.709 ms，1.50×。另一轮同为 4,096 × 64、64 queries 的探索结果为 1.37×；
  这是独立计时结果，不与本轮数据合并。
- 1,000 candidates × 24 dimensions，32 queries，k=全库：batch 23.166 ms，标量循环
  5.619 ms，0.24×。该全库返回规模下批处理明显变慢。


## 使用建议

这是显式的可选批量入口；不会自动替换已有的 search 调用。适合有多条 query 时显式调用：

```python
from factor_assets.similarity import FaissANNIndex

index = FaissANNIndex(embedding_dim=embeddings.shape[1])
index.build(ids, embeddings)
results_by_query = index.search_batch(queries, k=10, min_similarity=0.8)
```

不建议无条件设置 k=N 来取回全库结果：较大的返回量会显著增加构造与物化结果对象的开销；
前述全库 k 配置实测为 0.24×。请用自家数据对比实际的 Q（query 数）、D（维度）及 k，
再决定是否采用批量调用。

批量与逐条 scalar 搜索都使用 FAISS float32 计算，但不同调用路径不承诺 bitwise 一致。
靠近 min_similarity 阈值或 top-k 截断线的近似分数，应按 float32 计算精度解释；分数并列时
也不承诺稳定的 factor ID 选择或排序。

## Benchmark 复现

在项目根目录运行，使用单线程 BLAS/FAISS 设置：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m factor_assets.scripts.benchmark_ann_exact_flat_batch
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m factor_assets.scripts.benchmark_ann_exact_flat_batch \
  --factors 4096 --dimensions 64 --queries 64 --k 10
```
