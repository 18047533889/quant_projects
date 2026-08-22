# COS 语义与 PIT 契约

## 1. 目的

物理注册表只回答“文件在哪里、有哪些物理键”。本契约额外回答：一行数据是完整日频截面、状态快照、公告事件、期间事件、稀疏占位还是维表；哪个时间代表可知时间；能否直接装载为因子面板；需要哪些强制过滤；收益率和复权因子的市场口径；COS 文件名是否可以由查询时间推断。

所有规则由代码执行，不只是说明文档。

## 2. 时间模型

| 模型 | 含义 | 面板策略 | 典型连接 |
|---|---|---|---|
| D1 | 完整日频截面 | dense | 等值连接日期与标的 |
| S1 | 截至当日有效的状态快照 | state_ready | 等值连接日期与标的 |
| E1 | 公告日事件，只含当日公告主体 | event_only | 按可知时间向后 as-of |
| E2 | 报告期或公司行动事件 | event_only | 按 filing/事件时间处理 |
| X0 | 日期状但极稀疏 | sparse | 禁止默认面板化 |
| STATIC | 维表/全量单文件 | dimension | 全量读取 |
| EMPTY | 空占位表 | forbidden | 禁止使用 |
| MINUTE | 分钟粒度 | minute | 分钟时间轴 |

## 3. 市场口径

| 项目 | A 股 | 美股 |
|---|---|---|
| 标的 | `Symbol`，带 `.SH/.SZ/.BJ` | `Ticker/ticker`，无交易所后缀 |
| 日收益 | `Return` 为 bp，进入因子前 `/10000` | `Ret` 已是小数 |
| 复权 | `Factor`，供应商前复权因子 | `AdjFactor/adj_factor`，后复权链 |
| 财报可知时间 | `PubDate` | `filing_date` |
| 报告期 | `ReportPeriodEndDate` | `period_end` |
| 股本/公司行动 | `StockCapitalDaily` 为 S1 股本状态 | `StockCapitalDaily` 为拆股事件 |
| 行业 | 多来源，必须先筛单一 `IndustrySource` | `StockIndustry` 为空表 |
| 估值 | 日频完整 D1 | `StockIndicator/ValuationDaily` 为 X0 稀疏 |

## 4. 因子面板规则

`load_factor_columns` 是 FactorEngine 推荐入口。

1. E1/E2 不得通过 `load_columns` 生成 `(date, instrument)` 伪面板；
2. X0、EMPTY、STATIC 默认拒绝；
3. A 股 `Return` 必须归一为小数；
4. `ashare_stock_industry` 必须显式指定一个行业源；
5. 原始 `read_arrow/read_frame` 保留供应商字段与单位，用于诊断，不替代因子语义入口。

## 5. 财务 PIT 算法

严格 PIT 表使用 A 股 `PubDate <= decision_timestamp` 和美股 `filing_date <= decision_timestamp`。

默认 `period_selection="latest_period"`：

1. 按标的、报告期维护截至决策时点的最新版本；
2. 在已知报告期中选择最大的报告期；
3. 返回该报告期当前最新版本。

这与“取最近一次 filing”不同。后者在旧报告期晚修订时会把因子状态错误回滚。美股财报必须过滤单一 `timeframe`，季度、年度、TTM 不能在未声明策略时竞争同一状态。

## 6. 生效日事件

A 股 `StockDividend`、美股 `StockDividend` 和美股 `StockCapitalDaily` 缺少可靠公告/可知时间，设置为 `effective_time_only`：

- 默认拒绝 PIT；
- 只有显式 `allow_effective_time=True` 才能按除权/执行日读事件；
- 禁止一行式 `read_cos_events_asof`，因为分红/拆股需要累计、窗口或调整链，选择最近一条事件会丢失语义。

## 7. COS 物理布局

- D1/S1/MINUTE：通常按日文件，可按月份 wildcard 加列谓词读取；
- A 股财报：文件日期即公告日，可按 `PubDate` 剪枝；
- 美股财报：文件名是 `period_end`，不能由决策时点或 `filing_date` 推断；
- 分红/拆股：事件文件，不能由决策时点推断；
- X0：稀疏文件，不能假定每天存在。

对无法枚举的布局：HTTPFS 读取受授权前缀 wildcard，再由 DuckDB 的可知时间谓词过滤；CLI/mirror 禁止普通查询隐式整表同步，必须设置 `DATA_ACCESS_ALLOW_FULL_COS_SYNC=1`；`auto` 只有存在完整同步 marker 时才认为本地递归布局完整。

## 8. 快照治理

Registry fingerprint 包括物理 Schema 修正、时间轴和标的键修正、COS 合同指纹，以及 PIT、单位、过滤、面板和布局规则。规则升级会产生新的 `DataSnapshot.snapshot_id`，防止历史回测在语义变化后被错误视为同一数据版本。

## 9. 已知边界

- 本层不做 winsorize、标准化、缺失值填充或因子清洗；
- 不把缺失财务字段填为 0；
- 不推断美股行业；
- 不把拆股事件转换成股本日状态，调整链需在专门流程中构建；
- 日期字符串代表午夜。需要日内严格 PIT 时，决策时间必须包含时区和具体时刻。
