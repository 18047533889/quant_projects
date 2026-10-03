# CPU 分位内核宽批次 A/B（2026-10-03）

server-c 正式工作树实测，合成 `ties_missing` 场景：128 日期、5461 股票、
48 因子，Float64 C-contiguous；seed=81033。不是多年真实 COS 测试，
不是所有指标或 GPU 的最快资格证书。因子/标签没有显式 validity 数组；
有限值 mask 路径单独验证，公开接口的显式 validity 由另一 A/B 覆盖。

命令：`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 .venv/bin/python -m quant_evaluator.scripts.benchmark_quantile_cpu_backends_oct03 --times 128 --assets 5461 --factors 48 --rounds 3 --scenario ties_missing`。
三 AI 窗口串行计时，未停止或控制服务器其他业务；计时前 host load
约 12.75/13.50/15.15。Python 3.12.3、NumPy 2.2.6、Numba 0.67.0。

每 Q 三轮 ABBA，每后端六次 warm 样本，计时后逐次校验。

| Q | NumPy warm 中位秒 | Numba warm 中位秒 | NumPy/Numba |
|---|---:|---:|---:|
| 5 | 2.689886904 | 1.820872433 | 1.477 |
| 20 | 2.922151046 | 1.959864701 | 1.491 |

Q5 NumPy 样本秒：2.643015, 2.684911, 2.776714, 2.696490, 2.582325, 2.694863。
Q5 Numba：1.794485, 1.761752, 1.839032, 1.821914, 1.819831, 1.831345。
Q20 NumPy：2.871699, 2.933126, 2.829507, 3.009766, 2.935160, 2.911176。
Q20 Numba：1.969956, 1.944830, 2.027725, 1.949773, 1.923147, 1.981422。
首次 Q5 NumPy/Numba 2.573884/1.990219 秒；Q20 3.073286/1.963077 秒。
Numba 首次调用可能包含编译或缓存加载；Q20 复用同 specialization，不能称独立冷编译。

assignments、assignment mask、counts、bucket nonempty mask **exact**；
returns `rtol=1e-9, atol=1e-12`，均通过。估计峰值 1,878,966,272 bytes；
process-lifetime peak RSS 1,206,849,536 bytes，非各后端独立峰值。

输入 fingerprint：`d8d437defb02901bb240dbd579f049040bd2c2fe035a3848bee92918174223ed`。
源码 SHA256：
- `metrics/quantile.py`: `8a772a4f5a7beb6eea885356af2af7927695e0332b74eebe4485f653aba35711`
- `metrics/quantile_numba.py`: `ba6f0b6b1ba62d64bacaa8338dcc5e14cab74db5486f7ee2beeb75bbf7a46b57`
- `scripts/benchmark_quantile_cpu_backends_oct03.py`: `e1a631223a9c56581a39e1420029bf78f6c4a60f053ef15a1b5c4f7484e2f5b0`
