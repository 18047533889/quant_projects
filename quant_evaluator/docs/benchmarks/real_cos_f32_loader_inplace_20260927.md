# 32 因子真实 COS 加载器复测（2026-09-27）

本次只测 `load_real_batch` 装载，不执行指标评估，也不更新 `auto` 选路。
绑定的 landing manifest SHA256 为
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
通过 DataAccess 已授权的 COS 精确对象入口读取 32 个 verified 因子；
标签沿用已登记的 `ashare_calendar` 与 `ashare_stock_daily_adj`。
临时 COS 对象由 DataAccess 清理，不保存原始面板。

运行环境：server-c 正式主树，项目 `.venv`，
`DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos`，
`DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research`，
`ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data`，BLAS 单线程。
运行前磁盘剩余约 193 GiB、系统 MemAvailable 约 52 GiB。

| 项目 | 结果 |
| --- | ---: |
| 面板 | 2586 日 × 5461 股 × 32 因子 |
| 输入精度 | float64 |
| 装载耗时 | 140.295 秒 |
| 父进程峰值 RSS | 12,770,604 KiB |
| 因子有限值比例 | 0.7661925726 |

加载器在初始因子立方体内逐行压缩选中的日期与股票轴，
仅暂存一个所选日期行；`FactorBatch` 仍执行不可变所有权拷贝。
针对日期交集、股票重排、NaN 比特位及只读所有权的回归测试已通过。
另在 300×5000×16 的有界合成面板上对比裁剪步骤：旧 `np.ix_`
为 0.0886 秒、峰值 RSS 481.4 MiB；新压缩为 0.0225 秒、
峰值 RSS 399.1 MiB，输出校验一致。合成裁剪数字不代表 COS
端到端加速，也不能据此降低整批 CPU/GPU 评估的安全内存门槛。

本次真实加载运行没有同时启动 CPU/GPU 指标 worker。
大批量 `auto` 是否应扩大仍需在资源充足时完成公开入口交错 A/B、
完整制品对拍及峰值内存测量。
