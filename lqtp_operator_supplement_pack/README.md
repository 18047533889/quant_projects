# LQTP 算子实现补充包（给平台对接）

**这是算子代码包，不是因子。** 核心文件 `lqtp_operators_to_add.py` 里每个 `def` 都是一个 DSL 算子的可运行参考实现（pandas / 单票时序），方便你们对照语义移植到平台。

## 数量核对

| 类别 | 个数 |
|---|---:|
| NEW（对照 LQTP 已知表 **无重名**） | **115** |
| 其中 HTML/Catalog 缺口 | 27 |
| 其中 EXTRA 未来常用（非模型） | 88 |
| ALIAS（可选：`protected_div`/`clip`/`ts_delay`） | 3 |
| **合计** | **118** |

已含 **`tanh`**、`ATR_WILDER`、`ts_median`、`maximum`/`minimum`、`MACD_*`、`BB_*` 等。

## 文件（请整包发给平台）

| 文件 | 用途 |
|---|---|
| `lqtp_operators_to_add.py` | **算子实现代码**（主交付） |
| `OPERATORS_MANIFEST.md` | 全部算子签名 + 一行语义 |
| `OPERATOR_GAP_FROM_HTML_FACTOR_LIB.md` | HTML/Catalog 缺口证据 |
| `README.md` | 本说明 |

## 实现约定（请对齐）

1. 输入输出：单票 `pd.Series`（DatetimeIndex）
2. 因果 / PIT-safe：只用当前及历史
3. 除零 / 非法 → `NaN`（勿静默填 0）
4. `tanh` = 真正双曲正切（不要用 `sigmoid` 替代）
5. Python 保留字：`max_`/`min_`/`pow_` → 注册名 `max`/`min`/`pow`

## 与平台已有算子

- `NEW_OPERATORS` ∩ LQTP 已知集合 = **空**（已脚本核对）
- `ALIAS` 仅同名兼容：`protected_div→safe_div`，`clip→cap`，`ts_delay→delay`

压缩包：`/home/shw/lqtp_operator_supplement_pack.tar.gz`
