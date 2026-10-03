# 公开评估接口日分位复用 A/B（2026-10-03）

server-c，128×5461×48 合成 Float64 因子，显式因子 validity 和 NaN 标签；
seed82109，六指标：daily/full/spread/quantile_monotonicity/daily series/rate。
两臂同为 Numba 2 threads，OMP/OpenBLAS/MKL 均2。调用命令：
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 .venv/bin/python -m quant_evaluator.scripts.benchmark_daily_quantile_request_reuse_oct03 --time-count 128 --asset-count 5461 --factor-count 48 --rounds 3 --cpu-kernel numba`。

只把 baseline evaluator.py 固定为 e8fcc4b0a 的内存源码；adapters/kernels
两臂共享当前源码。因此不是完整旧仓库与新仓库比较，更不是真实 COS 或 CUDA
资格证书。三 AI 窗口串行计时，未控制其他服务器业务。

三轮 ABBA，每臂六次 warm：baseline 中位17.556239033秒；复用路径
2.845516823秒，约 **6.17×**。每次 baseline7 次 kernel、复用路径1次。
baseline 秒：17.394671,17.456432,17.545364,17.567114,18.076650,17.915709。
optimized 秒：2.852997,2.823366,2.838037,2.811880,2.946444,2.933902。
首次 baseline18.392008秒，随后首次 optimized2.860724秒，不是对称冷启动。
预热 baseline17.416890秒、optimized2.786152秒。

所有预热和计时结果均对固定首次 baseline 校验：浮点 values
rtol1e-9/atol1e-12；counts/masks/observation counts **exact**。额外六项
smoke 回归包括预热和最后 A 注入错误值的拒绝。估计峰值1,365,217,280bytes，
process-lifetime RSS峰值942,702,592bytes，不能比较独立后端峰值。
Python3.12.3/NumPy2.2.6/Numba0.67.0/llvmlite0.49.0/SciPy1.15.3。

输入SHA256：`e4f33e9cd1fb3f377e07fc6021037a0547d7d1d59dfcb9100ea2c1f32d62fa47`。
源码SHA256：
- baseline evaluator：`7f76bd873e1bd9545843c6157fb9da00d326bbf31eac44b94f269af8a80a056b`
- optimized evaluator：`f5692a3d585a3e43a563742b0036029b7e9f6b3ba554e051752bc986ae094f25`
- daily cache：`ae592cb0ba50d9e63e711d35a490874b5de701f21ce3ebfea0ed01716556d396`
- registry adapters：`4aee80ad2a010071af15333eda271302ea25c0c1d67ed111808dd99e95907af8`
- shared quantile：`8a772a4f5a7beb6eea885356af2af7927695e0332b74eebe4485f653aba35711`
