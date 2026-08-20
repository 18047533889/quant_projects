# `runtime/quality` — 数据质量门禁

因子输入/输出的 DQ 检查与 PIT（Point-in-Time）审计，可在 materialize 前后挂 gate。

| 模块 | 作用 |
|------|------|
| `runtime/dq_gates.py` | 主入口：空值率、极值、行数漂移 |
| `runtime/dq_profiles.py` | production / research 不同阈值 |
| `runtime/pit_audit.py` | 防前视：时间戳与数据可用性 |
| `runtime/input_dq.py` | 读端列完整性 |

本目录文件与 `runtime/` 根下同名模块镜像，便于按子系统 import。
