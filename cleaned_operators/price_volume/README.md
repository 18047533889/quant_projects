# `cleaned_operators/price_volume` — 价量算子

开高低收量、VWAP、收益率、量价相关等 **日频价量** 因子常用算子。

| 文件 | 作用 |
|------|------|
| `ops.py` | 主注册表 |
| `polars_price_volume.py` | Polars 实现 |

GTJA191 / week2 等因子库大量依赖本目录 + `common/time_series.py`。
