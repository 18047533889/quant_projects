因子 RankIC / RankICIR 前后对比

全样本（原有列）：
- rank_ic_close_改VWAP前：close→close T+1
- rank_ic_vwap_改VWAP后：VWAP→VWAP（T+1→T+2）
- rank_ic_vwap_行业中性 / 市值中性 / 行业市值双中性：同一 VWAP 口径下的中性化 Mean RankIC
- rank_icir_* / 变化_rankicir_*：对应 RankICIR
- 变化_*：后项减前项

2026 年切分（新增列，仅 trade_date∈[20260101,20270101)）：
- rank_ic_2026_中性化前：2026 原始 VWAP RankIC
- rank_ic_2026_行业中性 / 市值中性 / 行业市值双中性：同年中性化 Mean RankIC
- 变化_2026_原始到*：中性化后减中性化前（RankIC）
- rank_icir_2026_中性化前 / 行业中性 / 市值中性 / 行业市值双中性：同年 RankICIR
- 变化_rankicir_2026_原始到*：中性化后减中性化前（RankICIR）
- rank_ic_2026_交易日数：一般为 114

重算脚本：scripts/cogalpha_lqtp/add_2026_neutral_rankic_to_csv.py
行数 228
文件 reports/factor_rankic_close_vs_vwap_vs_neutral.csv
