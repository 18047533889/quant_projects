# 13 — Architecture Decisions & Do-Not-Build List

## Confirmed Decisions

1. Server local code > GitHub baseline。
2. DA/FE 继续作为底座，不重写。
3. 新 runtime package 只有 QE/FO/FA/FP 四个。
4. Research Control 是轻量中央 ledger。
5. Root schemas 不是 runtime package。
6. QE core value-first，不要求 DA/FE。
7. FO 是 optional feedback loop。
8. FA 不保存大 factor values。
9. FP 独立，因为旧 AlphaPurifier 能力实质丰富且与资产治理职责不同。
10. GTJA/Week2/公开 factor collections = corpus，不是生产依赖。

## Explicitly Do Not Build Now

- 第二套 PIT / universe / calendar
- 第二套 Factor DSL / compiler
- 第二套 factor materialization/cache
- 第二套 data lake/storage gateway
- 通用 distributed DAG platform
- Kafka/event bus
- Neo4j graph service
- PostgreSQL 微服务集群
- 在线 Feature Store
- 完整 PortfolioOptimizer / ExecutionTCA
- 模型训练框架
- 每因子一个 Skill
- 单一万能 Factor Score
- 全因子 O(K²) correlation matrix
- 全因子 SHAP
- 自动对所有因子运行 PBO/SPA/large bootstrap
- 默认 RF/GBDT/PCA neutralization
- 默认所有因子行业+市值完全中性化
- 单一 SuperAlpha 压掉所有信息

## Reconsider Triggers

只有出现量化证据才升级复杂度，例如：

- SQLite 写并发成为真实瓶颈 -> 再评估 PostgreSQL。
- 单机 ANN 内存/延迟不够 -> 再评估服务化 vector index。
- QE 同一中间量跨任务重复占 >X% 总耗时 -> 再评估 persistent cache。
- Python/Numba 已明显成为瓶颈 -> 再评估 Rust/C++ kernels。

不要因为“以后可能用”提前建平台。
