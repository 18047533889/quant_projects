# 09 — Testing, Performance & Acceptance

## 9.1 测试层级

每个库至少：

1. Unit
2. Golden
3. Property
4. Metamorphic
5. Reference/Fast parity（适用）
6. Batch/Chunk parity
7. Streaming/Batch parity（适用）
8. Determinism
9. Numerical stability
10. Leakage/future-poison
11. Corpus regression
12. Integration
13. Package extraction
14. Performance/memory benchmark

## 9.2 Metamorphic Examples

RankIC：

- `f' = a*f+b, a>0` -> RankIC 不变
- `f' = -f` -> RankIC 反号
- 打乱 asset row order -> 结果不变

Preprocess：

- 修改未来数据不能改变过去 stateless/causal output
- fitted state 只受 train window 影响

Assets：

- 相同 canonical ID 重复注册保持幂等
- SHADOWED 不删除 lineage/history

## 9.3 Leakage Red Team

主动注入：

- future label into factor values
- full-sample scaler/PCA
- future industry membership
- future universe membership
- future preprocessing fit state
- optimizer 用 test window 选参数
- graph/ANN fingerprint 使用未来 period 的完整行为向量

系统要么检测/拒绝，要么在 contract 上明确该方法仅 research/offline，不允许 production。

## 9.4 Performance Benchmarks

不要先拍脑袋定“1 秒”。先建立 baseline。

QE 典型规模：

- 100 factors
- 1,000 factors
- 5,000 factors
- 10,000 factors

记录：wall time, CPU time, peak RSS, throughput, copies/allocations。

FA：

- 10k/50k/100k fingerprints
- ANN build/query
- exact shortlist size 50/100/200
- sparse graph edges

FP：多 factor rank/winsor/neutralization throughput。

## 9.5 Performance Rules

- 性能优化前必须有 reference/golden。
- 不允许通过降低统计精度、静默 drop data、改变 tie/NaN 语义“提速”。
- float32 输入可接受；关键统计 accumulator 默认 float64。
- 避免不必要 DataFrame copy、重复 rank、重复 exposure design matrix。
- 不允许所谓 Polars backend 内部转 Pandas 后宣称 fast。

## 9.6 Extraction Test

每个 package：

```bash
cp -a <package> /tmp/<pkg>-standalone
python -m venv /tmp/<pkg>-venv
source /tmp/<pkg>-venv/bin/activate
pip install /tmp/<pkg>-standalone
pytest /tmp/<pkg>-standalone/tests
python -c "import <pkg>; print(<pkg>.__version__)"
```

测试前确保 `PYTHONPATH` 不包含 monorepo。

## 9.7 Definition of Done

模块完成必须同时：

- correct
- documented
- tested
- benchmarked
- isolated
- no legacy imports
- no duplicated DA/FE capability
- audit passed

开发 Agent 自己说“完成”不算验收；必须由独立 auditor 输出 PASS/FAIL + evidence。
