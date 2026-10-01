# F32 coverage 默认 auto 实测：GPU 切换成本仍未闭合

2026-10-01，在 server-c 正式 main 工作树运行，代码提交为 78e0157b5。
原始回执：`real_cos_f32_coverage_cache_auto_replay_20261001.json`。

我们使用既有 COS 因子，面板为 2586 个交易日 × 5461 个标的 × 32 个因子，float64。
请求仅含 coverage。数据加载耗时 130.325 秒，以下评估耗时不含加载。
每个独立子进程执行两次，顺序为 CPU、CUDA、auto、auto、CUDA、CPU。

| 请求 | 第一次（秒） | 第二次（秒） | 实际后端 |
|---|---:|---:|---|
| CPU | 15.952 | 6.508 | CPU / CPU |
| CUDA strict | 19.287 | 5.723 | CUDA / CUDA |
| auto | 16.178 | 9.084 | CPU / CUDA |
| auto | 15.986 | 8.356 | CPU / CUDA |
| CUDA strict | 17.429 | 5.877 | CUDA / CUDA |
| CPU | 15.834 | 6.442 | CPU / CPU |

我们逐次核验了实际后端、配置身份与指标产物。同后端产物哈希一致；
CPU/CUDA 的数值在 rtol=1e-8、atol=1e-10 内一致，掩码、形状及观测计数一致。
六轮最终产物与十二次逐次回执均通过检查。

冷缓存 CPU 更快。CUDA 已完成一次评估后的第二次比 CPU 热缓存快；
但 auto 从 CPU 转到 CUDA 的第二次慢于 CPU 热缓存。
我们尚未分离 CUDA 首次启用、模块导入及资源探测的耗时，因此不能把这次
结果当作默认 auto 已接入最快路径的证明。后续需要测第三次及切换阶段耗时。
