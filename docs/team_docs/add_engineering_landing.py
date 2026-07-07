#!/usr/bin/env python3
"""Add §10c engineering landing: storage topology, tables, partitions, persist policy."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
ROOT = "/home/yluel/share/projects/massive_parquet"

SECTION = r'''
    <section id="sec-engineering">
      <h2>10c. 工程落地手册（存储 · 表结构 · 分区 · 读写）</h2>
      <p class="section-desc">
        <strong>读者：</strong>数据工程 / 平台 SRE / 后端落地组。读完本节应能回答：
        ① 数据落在哪一层目录；② 哪些表进 ClickHouse、哪些只留 Parquet；③ 每张表一行代表什么、主键与分区；
        ④ 跑批顺序与幂等；⑤ 日更怎么追加。可执行 DDL：
        <code>deploy/clickhouse/massive_ddl.sql</code>。
      </p>

      <div class="box-danger">
        <strong>真相源顺序：</strong>供应商 raw（Layer 0）→ cleaned（Layer 1，不改 OHLCV）→
        <strong>materialized_panel Parquet（Layer 2 真相源）</strong> → ClickHouse（热读副本）。
        因子引擎<strong>禁止</strong>直接扫 11TB cleaned 做 merge_asof。
      </div>

      <h3>10c.1 四层存储与根路径</h3>
<pre>/home/yluel/share/projects/massive_parquet/
├── raw_massive_data/              Layer 0  ~10TB  永久保留（供应商字段）
├── cleaned_massive_data/          Layer 1  ~243GB 永久保留（+ticker,+align_time）
├── materialized_panel/            Layer 2  增量增长 永久保留（物化产物，Hive 分区）
├── _staging/                      临时     跑批结束<strong>必须删除</strong>
└── (无) tick 首期不进 materialized_panel</pre>

      <h3>10c.2 保留 vs 不保留 vs 仅内存（落地红线）</h3>
      <table class="stage-table">
        <thead>
          <tr><th>数据/产物</th><th>是否落盘</th><th>落盘位置</th><th>格式</th><th>说明</th></tr>
        </thead>
        <tbody>
          <tr><td>供应商原始下载</td><td><strong>是</strong></td><td><code>raw_massive_data/</code></td><td>Parquet（按源原有布局）</td><td>只追加/重下坏分区；勿在 raw 上改 OHLCV</td></tr>
          <tr><td>Layer 1 清洗结果</td><td><strong>是</strong></td><td><code>cleaned_massive_data/</code></td><td>Parquet</td><td>覆盖式重跑 cleaning 时按数据集替换</td></tr>
          <tr><td>Polars/pandas 中间 join 宽表</td><td><strong>否</strong></td><td><code>_staging/build_panel/{batch_id}/</code></td><td>—</td><td>仅当次跑批；成功写 panel 后 <code>rm -rf</code></td></tr>
          <tr><td>截断扫描报告、单测日志</td><td>可选</td><td><code>logs/data_quality/{date}/</code></td><td>JSON/CSV</td><td>审计用，非因子读路径</td></tr>
          <tr><td>Layer 2 维表/事实表/宽表</td><td><strong>是</strong></td><td><code>materialized_panel/{表名}/</code></td><td>Parquet + Hive 分区</td><td><strong>Layer 2 真相源</strong>；CH 从此外部表导入</td></tr>
          <tr><td>ClickHouse <code>qs_massive.*</code></td><td><strong>是</strong></td><td>CH 集群</td><td>MergeTree</td><td>热数据副本；可 DROP PARTITION 重导，不以 CH 为唯一真相源</td></tr>
          <tr><td>因子暴露 / IC / 夏普</td><td><strong>是</strong></td><td><code>qs_factor.*</code></td><td>MergeTree</td><td><strong>另一库</strong>；禁止写回 <code>panel_daily</code></td></tr>
          <tr><td>截面 MAD / Barra 中性化结果</td><td><strong>否</strong>（Layer 2）</td><td>—</td><td>—</td><td>属 factor_evaluation，入 <code>qs_factor</code></td></tr>
          <tr><td>tick quotes/trades ~10TB</td><td>raw 保留；<strong>首期不进 CH</strong></td><td>raw 路径</td><td>Parquet 按日</td><td>分钟/tick 物化后再议独立表</td></tr>
        </tbody>
      </table>

      <h3>10c.3 Parquet 文件规范（全 Layer 2 统一）</h3>
      <table>
        <thead><tr><th>项</th><th>规范</th></tr></thead>
        <tbody>
          <tr><td>压缩</td><td><code>zstd</code> 或 <code>snappy</code>（全库统一一种）</td></tr>
          <tr><td>行组</td><td>建议 128MB–256MB row group；按月分区单文件 50MB–2GB 可接受</td></tr>
          <tr><td>分区目录</td><td><code>year=YYYY/month=MM/</code>（月分区为<strong>默认</strong>）；小维表可无分区单文件</td></tr>
          <tr><td>文件名</td><td><code>part-{batch_id}-{shard}.parquet</code>，例 <code>part-20260603_01-000.parquet</code></td></tr>
          <tr><td>逻辑主键</td><td>写入前断言无重复；见下表「行粒度」</td></tr>
          <tr><td>空值</td><td>缺失 = <strong>无行或 null</strong>；禁止用 0 填价格/财报</td></tr>
          <tr><td>字符串</td><td><code>ticker</code> 大写；<code>batch_id</code> 格式 <code>YYYYMMDD_nn</code></td></tr>
        </tbody>
      </table>

      <h3>10c.4 Layer 2 目录树（须实现的 Parquet 路径）</h3>
      <p>根路径：<code>__MASSIVE_ROOT__/materialized_panel/</code>。下表每一行 = 工程须交付的<strong>一张物理表</strong>（Parquet 数据集）。</p>
      <table class="stage-table">
        <thead>
          <tr>
            <th>数据集目录</th><th>行粒度（一行=）</th><th>逻辑主键</th><th>Hive 分区</th><th>任务 ID</th><th>CH 表名</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>dim_calendar/</code></td><td>1 个交易日</td><td><code>trade_date</code></td><td>无（单文件或按年）</td><td>FEAT-001</td><td><code>dim_calendar</code></td>
          </tr>
          <tr>
            <td><code>dim_security_master/</code></td><td>1 个 ticker 当前属性快照</td><td><code>ticker</code></td><td><code>snapshot_date=YYYY-MM-DD</code> 可选</td><td>PREPROC-001</td><td><code>dim_security_master</code></td>
          </tr>
          <tr>
            <td><code>dim_ticker_map/</code></td><td>1 段 ticker 有效期</td><td><code>permanent_id, valid_from, ticker</code></td><td>无</td><td>PREPROC-011</td><td><code>dim_ticker_map</code></td>
          </tr>
          <tr>
            <td><code>dim_universe_daily/</code></td><td>1 日 × 1 ticker（稀疏）</td><td><code>trade_date, ticker</code></td><td><code>year/month</code></td><td>PREPROC-002</td><td><code>dim_universe_daily</code></td>
          </tr>
          <tr>
            <td><code>fact_bars_adjusted_daily/</code></td><td>1 日 × 1 ticker 复权 OHLCV</td><td><code>trade_date, ticker</code></td><td><code>year/month</code></td><td>PREPROC-003</td><td><code>fact_bars_adjusted_daily</code></td>
          </tr>
          <tr>
            <td><code>fact_returns_daily/</code></td><td>1 日 × 1 ticker 收益</td><td><code>trade_date, ticker</code></td><td><code>year/month</code></td><td>PREPROC-004/012</td><td><code>fact_returns_daily</code></td>
          </tr>
          <tr>
            <td><code>fact_shares_out_daily/</code></td><td>1 日 × 1 ticker 股本</td><td><code>trade_date, ticker</code></td><td><code>year/month</code></td><td>PREPROC-013</td><td><code>fact_shares_out_daily</code></td>
          </tr>
          <tr>
            <td><code>fact_delisting_events/</code></td><td>1 次退市事件</td><td><code>permanent_id, last_trading_day</code></td><td>无（稀疏全量）</td><td>PREPROC-012</td><td><code>fact_delisting_events</code></td>
          </tr>
          <tr>
            <td><code>pit_fundamentals/</code></td><td>1 次财报 knowledge 版本</td><td><code>ticker, knowledge_date, period_end, timeframe</code></td><td><code>year/month</code> on <code>knowledge_date</code></td><td>PREPROC-005</td><td><code>pit_fundamentals</code></td>
          </tr>
          <tr>
            <td><code>layer2_factor_ready_panel/</code></td><td>1 日 × 1 ticker <strong>宽表</strong></td><td><code>trade_date, ticker</code></td><td><code>year/month</code></td><td>PREPROC-007</td><td><code>panel_daily</code></td>
          </tr>
        </tbody>
      </table>

      <h3>10c.5 表字段字典（Parquet 列 = CH 列，须一致）</h3>

      <h4>10c.5.1 dim_calendar</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th><th>来源</th></tr></thead>
        <tbody>
          <tr><td><code>trade_date</code></td><td>Date</td><td>美股交易日</td><td><code>day_aggs</code> 文件名 union</td></tr>
          <tr><td><code>is_trading_day</code></td><td>UInt8</td><td>1=交易日</td><td>派生</td></tr>
          <tr><td><code>source</code></td><td>String</td><td>如 <code>day_aggs_union</code></td><td>常量</td></tr>
        </tbody>
      </table>

      <h4>10c.5.2 dim_security_master</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>ticker</code></td><td>String</td><td>当前代码（大写）</td></tr>
          <tr><td><code>permanent_id</code></td><td>String</td><td>优先 <code>composite_figi</code></td></tr>
          <tr><td><code>composite_figi</code></td><td>String?</td><td>供应商 FIGI</td></tr>
          <tr><td><code>cik</code></td><td>String?</td><td>SEC CIK</td></tr>
          <tr><td><code>type</code></td><td>String</td><td>仅 <code>CS</code> 进默认 universe</td></tr>
          <tr><td><code>market, locale, primary_exchange</code></td><td>String</td><td>过滤 ETF/OTC</td></tr>
          <tr><td><code>active</code></td><td>UInt8</td><td>是否活跃</td></tr>
          <tr><td><code>list_date, delisted_date</code></td><td>Date?</td><td>上市/退市边界</td></tr>
        </tbody>
      </table>

      <h4>10c.5.3 dim_ticker_map（代码变更血缘）</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>ticker</code></td><td>String</td><td>该有效期内使用的代码</td></tr>
          <tr><td><code>permanent_id</code></td><td>String</td><td>跨 rename 不变</td></tr>
          <tr><td><code>valid_from</code></td><td>Date</td><td>含当日</td></tr>
          <tr><td><code>valid_to</code></td><td>Date?</td><td>不含当日；NULL=仍有效</td></tr>
          <tr><td><code>batch_id</code></td><td>String</td><td>物化批次</td></tr>
        </tbody>
      </table>

      <h4>10c.5.4 dim_universe_daily</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>trade_date, ticker</code></td><td>Date, String</td><td>稀疏：仅当日有 bar 的 CS</td></tr>
          <tr><td><code>permanent_id</code></td><td>String?</td><td>join map</td></tr>
          <tr><td><code>in_universe_base</code></td><td>UInt8</td><td>1=底线可交易池（因子截面 filter）</td></tr>
          <tr><td><code>universe_mask</code></td><td>UInt8</td><td>策略掩码（流动性等，默认 1）</td></tr>
          <tr><td><code>has_bar</code></td><td>UInt8</td><td>当日是否有 day_aggs 行</td></tr>
          <tr><td><code>price_source</code></td><td>String</td><td><code>sip_adj</code> | <code>rest_split_only</code></td></tr>
        </tbody>
      </table>

      <h4>10c.5.5 fact_bars_adjusted_daily</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>trade_date, ticker</code></td><td>Date, String</td><td>锚点键</td></tr>
          <tr><td><code>align_time</code></td><td>DateTime64(3) UTC</td><td>日 K 建议 04:00 UTC</td></tr>
          <tr><td><code>adj_open/high/low/close/volume</code></td><td>Float64</td><td>拆股复权后；P0 前无分红复权</td></tr>
          <tr><td><code>price_source, adj_method</code></td><td>String</td><td>审计；全库单一价源</td></tr>
          <tr><td><code>batch_id</code></td><td>String</td><td>物化批次</td></tr>
        </tbody>
      </table>

      <h4>10c.5.6 fact_returns_daily</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>ret_price</code></td><td>Float64</td><td>由 adj_close 计算</td></tr>
          <tr><td><code>ret_total</code></td><td>Float64?</td><td>含分红；dividends P0 后启用</td></tr>
          <tr><td><code>is_delisting_day</code></td><td>UInt8</td><td>1=最后交易日</td></tr>
          <tr><td><code>delisting_return</code></td><td>Float64?</td><td>惩罚收益（破产/并购分类型）</td></tr>
          <tr><td><code>delisting_policy</code></td><td>String</td><td>回测政策名，写入元数据</td></tr>
        </tbody>
      </table>

      <h4>10c.5.7 fact_shares_out_daily（动态股本）</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>shares_out</code></td><td>Float64</td><td>日频优先，否则 floats/财报 asof</td></tr>
          <tr><td><code>shares_source</code></td><td>String</td><td><code>daily_summary</code> | <code>floats_asof</code> | <code>filing_pit</code></td></tr>
          <tr><td><code>pit_market_cap</code></td><td>Float64?</td><td><code>adj_close * shares_out</code></td></tr>
        </tbody>
      </table>

      <h4>10c.5.8 pit_fundamentals（长表，Panel merge 前物化）</h4>
      <table>
        <thead><tr><th>列名</th><th>类型</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>knowledge_date</code></td><td>Date</td><td>= filing_date（禁止 period_end 直贴）</td></tr>
          <tr><td><code>knowledge_ts</code></td><td>DateTime64</td><td>filing 日 00:00 UTC</td></tr>
          <tr><td><code>period_end, timeframe</code></td><td>Date, String</td><td>Q/TTM 分组用</td></tr>
          <tr><td><code>pit_*</code></td><td>Float64?</td><td>科目 null 保持 null</td></tr>
          <tr><td><code>filing_date</code></td><td>Date</td><td>与 knowledge 对齐</td></tr>
        </tbody>
      </table>
      <p><strong>P0 前：</strong>勿写入 2024 截断财报分区，或 panel 上 2024 <code>pit_*</code> 强制 NULL。</p>

      <h4>10c.5.9 layer2_factor_ready_panel → panel_daily（因子引擎主读）</h4>
      <p>宽表 = 上表 fact + pit 列<strong>已 asof 贴到 trade_date</strong> 的 denormalized 快照。列集 = <code>fact_bars_*</code> + <code>fact_returns_*</code> + <code>dim_universe_*</code> + 核心 <code>pit_*</code> + <code>shares_out/pit_market_cap</code> + 可选 news/short。</p>
      <table>
        <thead><tr><th>列组</th><th>列（首期 MVP）</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>键</td><td><code>trade_date, ticker, permanent_id, align_time</code></td><td>CH ORDER BY (trade_date, ticker)</td></tr>
          <tr><td>审计</td><td><code>batch_id, price_source, adj_method</code></td><td>全库同一 <code>price_source</code></td></tr>
          <tr><td>掩码</td><td><code>in_universe_base, universe_mask</code></td><td>回测先 <code>WHERE in_universe_base=1</code></td></tr>
          <tr><td>价量收益</td><td><code>adj_* , ret_price, ret_total</code></td><td>已复权</td></tr>
          <tr><td>市值</td><td><code>shares_out, pit_market_cap</code></td><td>PREPROC-013</td></tr>
          <tr><td>PiT 财报</td><td><code>knowledge_ts, pit_total_assets, pit_net_income, …</code></td><td>已 asof，非 raw 报表</td></tr>
          <tr><td>退市</td><td><code>is_delisting_day, delisting_return</code></td><td>PREPROC-012</td></tr>
          <tr><td>另类/新闻</td><td><code>pit_short_* , news_* , signal_date</code></td><td>P0 后逐步加列</td></tr>
        </tbody>
      </table>

      <h3>10c.6 ClickHouse：库、分区、排序键、Projection</h3>
      <table>
        <thead><tr><th>CH 表</th><th>ENGINE</th><th>PARTITION BY</th><th>ORDER BY</th><th>Track</th></tr></thead>
        <tbody>
          <tr><td><code>dim_calendar</code></td><td>MergeTree</td><td>—</td><td>(trade_date)</td><td>维表</td></tr>
          <tr><td><code>dim_security_master</code></td><td>ReplacingMergeTree(updated_at)</td><td>—</td><td>(ticker)</td><td>维表</td></tr>
          <tr><td><code>dim_ticker_map</code></td><td>MergeTree</td><td>—</td><td>(permanent_id, valid_from, ticker)</td><td>Track B 血缘</td></tr>
          <tr><td><code>dim_universe_daily</code></td><td>MergeTree</td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>Track A</td></tr>
          <tr><td><code>fact_bars_adjusted_daily</code></td><td>MergeTree</td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>Track A</td></tr>
          <tr><td><code>fact_returns_daily</code></td><td>MergeTree</td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>Track A</td></tr>
          <tr><td><code>fact_shares_out_daily</code></td><td>MergeTree</td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>Track A</td></tr>
          <tr><td><code>pit_fundamentals</code></td><td>MergeTree</td><td>toYYYYMM(knowledge_date)</td><td>(ticker, knowledge_date)</td><td>ASOF 右表</td></tr>
          <tr><td><code>panel_daily</code></td><td>MergeTree</td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td><strong>Track A 主读</strong></td></tr>
          <tr><td><code>panel_daily</code> 投影 <code>p_ticker_timeline</code></td><td>Projection</td><td>随主表</td><td>(ticker, trade_date)</td><td><strong>Track B</strong>（CH-006）</td></tr>
          <tr><td><code>meta_load_batch</code></td><td>MergeTree</td><td>—</td><td>(started_at, table_name)</td><td>审计</td></tr>
        </tbody>
      </table>
      <p>DDL 全文：<code>quantsociety_backend_project/deploy/clickhouse/massive_ddl.sql</code>（CH-001 交付物）。</p>

      <h3>10c.7 物化跑批顺序（工程必须按此实现）</h3>
<pre>batch_id = YYYYMMDD_nn
Stage0  scan → 失败则退出

1. dim_calendar          ← day_aggs 文件名
2. dim_security_master   ← all_tickers + ticker_types（P0 全量后）
3. dim_ticker_map        ← PREPROC-011
4. fact_bars_adjusted    ← cleaned day_aggs + raw splits（按 year 分批写分区）
5. dim_universe_daily    ← bars ∩ CS（同分区 trade_date）
6. fact_returns_daily    ← bars + delisting_events
7. fact_shares_out_daily ← PREPROC-013
8. pit_fundamentals      ← cleaned 三大报表 PiT（按 knowledge_date 分区）
9. layer2_factor_ready_panel ← merge 4~8 到 (trade_date,ticker)，按 month 写出
10. meta_load_batch      ← 记录每表每分区 row_count
11. CH INSERT            ← 仅导入 9 的 panel + 可选 4~8；禁止 TRUNCATE 全表

清理: rm -rf _staging/build_panel/{batch_id}/</pre>

      <h3>10c.8 幂等与重跑（按月分区）</h3>
      <table>
        <thead><tr><th>场景</th><th>Parquet</th><th>ClickHouse</th></tr></thead>
        <tbody>
          <tr><td>首次全量</td><td>按年循环写 <code>year=YYYY/month=MM</code></td><td>按月 <code>INSERT FROM file(parquet)</code></td></tr>
          <tr><td>单月修复</td><td>覆盖该月目录下 <code>part-*.parquet</code></td><td><code>ALTER TABLE panel_daily DROP PARTITION 'YYYYMM'</code> 后重导</td></tr>
          <tr><td>T+1 日更</td><td>仅追加昨日所在月分区新文件或重写该月</td><td><code>INSERT</code> 昨日 <code>trade_date</code>；跑重复键检查</td></tr>
          <tr><td>价源切换</td><td><strong>全库重物化</strong> bars+panel，新 <code>price_source</code></td><td>全部分区重导；禁止混旧新</td></tr>
        </tbody>
      </table>

      <h3>10c.9 谁读哪张表</h3>
      <table>
        <thead><tr><th>消费者</th><th>读什么</th><th>禁止读什么</th></tr></thead>
        <tbody>
          <tr><td>factor_engine / QuantaAlpha</td><td><code>panel_daily</code>（CH 或 Parquet 宽表）</td><td>cleaned 24 源、raw tick</td></tr>
          <tr><td>PiT 调试 / 补 join</td><td><code>pit_fundamentals</code></td><td>raw 财报 period_end 直贴</td></tr>
          <tr><td>Track B 单票长序列</td><td><code>panel_daily</code> + Projection；或 Parquet 按 ticker 过滤</td><td>勿扫全市场无 ticker 条件</td></tr>
          <tr><td>factor_evaluation</td><td>写 <code>qs_factor.factor_exposures_daily</code></td><td>不写 <code>qs_massive</code></td></tr>
          <tr><td>数据质量</td><td><code>meta_load_batch</code> + 扫描脚本</td><td>—</td></tr>
        </tbody>
      </table>

      <h3>10c.10 工程脚本交付清单（待建 → 路径约定）</h3>
      <table>
        <thead><tr><th>脚本</th><th>路径</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>build_panel_daily.py</code></td><td><code>scripts/layer2/build_panel_daily.py</code></td><td>步骤 1–9 编排</td></tr>
          <tr><td><code>load_panel_to_clickhouse.py</code></td><td><code>scripts/layer2/load_panel_to_clickhouse.py</code></td><td>CH-002 按月导入</td></tr>
          <tr><td><code>massive_ddl.sql</code></td><td><code>deploy/clickhouse/massive_ddl.sql</code></td><td>CH-001 建表</td></tr>
          <tr><td><code>clickhouse_panel.py</code></td><td><code>data_access/clickhouse_panel.py</code></td><td>CH-004 只读 API</td></tr>
          <tr><td><code>tape_condition_filter.yaml</code></td><td><code>data_access/config/tape_condition_filter.yaml</code></td><td>TASK-DC-005</td></tr>
        </tbody>
      </table>

      <div class="box-info">
        <strong>相关章节：</strong>
        <a href="#sec-pipeline">§10b 管道</a> ·
        <a href="#sec-clickhouse">§11 CH 读取 SQL</a> ·
        <a href="#sec-before-factor">★入模前</a> ·
        <a href="#sec-blind-spots">§12a 补盲点</a>
      </div>
    </section>
'''


def patch_html(html: str) -> str:
    block = SECTION.replace("__MASSIVE_ROOT__", ROOT)

    if "sec-engineering" in html:
        html = re.sub(
            r'\s*<section id="sec-engineering">.*?</section>\s*',
            "\n",
            html,
            count=1,
            flags=re.DOTALL,
        )

    m = re.search(r"\n(\s*<!-- 11 ClickHouse -->\s*\n\s*<section id=\"sec-clickhouse\">)", html)
    if not m:
        raise SystemExit("sec-clickhouse anchor not found")
    html = html[: m.start()] + "\n" + block + html[m.start() :]

    nav = html.split("</nav>")[0]
    if "sec-engineering" not in nav:
        html = html.replace(
            '<li><a href="#sec-pipeline">10b. Layer 2 物化管道</a></li>\n        <li><a href="#sec-clickhouse">',
            '<li><a href="#sec-pipeline">10b. Layer 2 物化管道</a></li>\n        <li><a href="#sec-engineering"><strong>★ 10c 工程落地手册</strong></a></li>\n        <li><a href="#sec-clickhouse">',
        )

    html = html.replace(
        '平台 / ETL</td><td><a href="#sec-blind-spots">§12a</a> <a href="#sec-before-factor">★入模前</a>',
        '平台 / ETL</td><td><a href="#sec-engineering"><strong>★10c落地</strong></a> <a href="#sec-blind-spots">§12a</a> <a href="#sec-before-factor">★入模前</a>',
    )
    html = html.replace(
        '平台 / ETL</td><td><a href="#sec-before-factor">★入模前</a> <a href="#sec-pipeline">§10b</a>',
        '平台 / ETL</td><td><a href="#sec-engineering"><strong>★10c落地</strong></a> <a href="#sec-before-factor">★入模前</a> <a href="#sec-pipeline">§10b</a>',
    )

    if "§10c 工程落地手册</strong></a>。</p>\n      <h3>10b.4" not in html:
        html = html.replace(
            "<h3>10b.4 产物路径一览</h3>",
            '<p>完整目录树、列字典、分区与幂等规则见 <a href="#sec-engineering"><strong>§10c 工程落地手册</strong></a>。</p>\n      <h3>10b.4 产物路径一览</h3>',
        )

    # Expand 10b.4 table rows
    extra_rows = """
          <tr><td>日历</td><td><code>materialized_panel/dim_calendar/</code></td><td><code>qs_massive.dim_calendar</code></td></tr>
          <tr><td>证券主数据</td><td><code>.../dim_security_master/</code></td><td><code>dim_security_master</code></td></tr>
          <tr><td>代码血缘</td><td><code>.../dim_ticker_map/</code></td><td><code>dim_ticker_map</code></td></tr>
          <tr><td>复权 K 线</td><td><code>.../fact_bars_adjusted_daily/year=…/month=…</code></td><td><code>fact_bars_adjusted_daily</code></td></tr>
          <tr><td>收益</td><td><code>.../fact_returns_daily/year=…/month=…</code></td><td><code>fact_returns_daily</code></td></tr>
          <tr><td>动态股本</td><td><code>.../fact_shares_out_daily/year=…/month=…</code></td><td><code>fact_shares_out_daily</code></td></tr>
          <tr><td>退市事件</td><td><code>.../fact_delisting_events/</code></td><td><code>fact_delisting_events</code></td></tr>
"""
    if "fact_shares_out_daily" not in html:
        html = html.replace(
            "          <tr><td>Universe</td><td><code>.../universe_daily/</code></td><td><code>qs_massive.dim_universe_daily</code></td></tr>\n        </tbody>\n      </table>\n    </section>\n\n    <section id=\"sec-engineering\">",
            "          <tr><td>Universe</td><td><code>.../dim_universe_daily/year=…/month=…</code></td><td><code>qs_massive.dim_universe_daily</code></td></tr>"
            + extra_rows
            + "        </tbody>\n      </table>\n    </section>\n\n    <section id=\"sec-engineering\">",
            1,
        )

    ch_rows = """
          <tr><td><code>dim_ticker_map</code></td><td>是</td><td><code>dim_ticker_map</code></td><td>代码变更血缘</td></tr>
          <tr><td><code>fact_shares_out_daily</code></td><td>是</td><td><code>fact_shares_out_daily</code></td><td>动态股本/市值</td></tr>
          <tr><td><code>fact_delisting_events</code></td><td>是</td><td><code>fact_delisting_events</code></td><td>退市惩罚稀疏表</td></tr>
"""
    if "dim_ticker_map</code></td><td>是" not in html:
        html = html.replace(
            "          <tr><td><code>layer2_factor_ready_panel</code>（宽表）</td><td>是（主表）</td><td><code>panel_daily</code></td><td>因子引擎主读表</td></tr>",
            "          <tr><td><code>layer2_factor_ready_panel</code>（宽表）</td><td>是（主表）</td><td><code>panel_daily</code></td><td>因子引擎主读表</td></tr>"
            + ch_rows,
            1,
        )

    # CH section pointer
    html = html.replace(
        "<h3>11.3 表结构 DDL（生产可执行草案）</h3>",
        '<p>与 <a href="#sec-engineering">§10c</a> 字段字典一致；生产以仓库文件为准：<code>deploy/clickhouse/massive_ddl.sql</code>。</p>\n      <h3>11.3 表结构 DDL（生产可执行草案）</h3>',
        1,
    )

    # Architecture table layer 1.5
    html = html.replace(
        "<td><code>materialized_panel/</code></td><td>待建</td>",
        "<td><code>materialized_panel/</code></td><td>见 <a href=\"#sec-engineering\">§10c</a></td>",
        1,
    )

    if "ENG-010" not in html:
        html = html.replace(
            "          <tr><td>CH-001</td>",
            "          <tr><td>ENG-010</td><td><span class=\"badge badge-p0\">P0</span></td><td>按 §10c 实现 Layer 2 Parquet 全表 + 脚本骨架</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>\n          <tr><td>CH-001</td>",
        )

    return html


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch_html(html)
    HTML.write_text(html, encoding="utf-8")
    print("engineering section added, lines:", html.count("\n") + 1)


if __name__ == "__main__":
    main()
