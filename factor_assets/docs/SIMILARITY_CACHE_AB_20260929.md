# QEPairwiseSimilarity 缓存查询索引 A/B（2026-09-29）

本轮仅优化 `factor_assets.similarity.exact.QEPairwiseSimilarity.find_similar` 的内存缓存查询。
旧实现每次扫描全部有向缓存键；新实现保留 `_cache` 作为权威结果，
另维护按查询因子分组且保留插入顺序的键索引。覆盖写入不重复登记，
`clear()` 同时清空索引；自配对只登记一次，`count()` 正确计数。

## 正确性

`factor_assets/tests/test_qe_find_similar_index.py` 用旧扫描语义作 oracle，
比较返回的完整结果和值顺序，覆盖双向/自配对、覆写、UNKNOWN、
方法/市场/期间过滤、阈值、`max_results=0` 与清空后重建。
本轮 `factor_assets/tests` 全套为 **1513 passed**。

## 有界合成 A/B

可重复运行：

```bash
PYTHONPATH=. .venv/bin/python factor_assets/scripts/benchmark_find_similar_index.py
```

固定 seed 20260929，查询因子关联度 24，另造 1,000/10,000/30,000
个无关 pair。脚本先比较 7 类查询结果，再预热两条路径，
以扫描→索引→索引→扫描顺序交错；每次计时包含 80 次查询，
重复 3 轮后取各路径中位数。硬上限 30,000 无关 pair，不读 COS，
不写持久数据。一次复跑结果如下（微秒/查询）：

| 无关 pair | 缓存键 | 全扫描 | 邻接索引 | 倍数 |
|---:|---:|---:|---:|---:|
| 1,000 | 2,048 | 99.9 | 12.7 | 7.9× |
| 10,000 | 20,048 | 728.0 | 9.6 | 76.2× |
| 30,000 | 60,048 | 2,206.1 | 9.5 | 231.8× |

另一轮同机运行的倍数约为 15.1×/69.1×/217.9×；
微秒级计时会受负载影响。结论仅限这段缓存查询和指定合成度数，
不代表真实 QE 成对计算、COS 读取或 FA 整条流水线同比例加速。
