# `cleaned_operators/technical` — 技术指标算子

MACD、RSI、布林带、KDJ 等经典技术指标（多数带 `ts_` 前缀 canonical）。

| 文件 | 作用 |
|------|------|
| `signal.py` | 指标注册与 `calculate` 实现 |

参数语义（窗口 `d`、锚点等）见 [`../../docs/operators_semantics.md`](../../docs/operators_semantics.md)。
