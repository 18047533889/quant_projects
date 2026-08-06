因子 RankIC / RankICIR 前后对比

全样本（原有列）：
- rank_ic_close_改VWAP前：close→close T+1
- rank_ic_vwap_改VWAP后：VWAP→VWAP（T+1→T+2）
- rank_ic_vwap_行业中性 / 市值中性 / 行业市值双中性：同一 VWAP 口径下的中性化 Mean RankIC
- rank_icir_* / 变化_rankicir_*：对应 RankICIR
- 变化_*：后项减前项

2026 明细（原始 / 行业 / 市值 / 双中性 + 变化）：
- rank_ic_2026_* / rank_icir_2026_* / 变化_2026_* / 变化_rankicir_2026_*
- rank_ic_2026_交易日数

各年双中性（2019–2026）：
- rank_ic_YYYY_行业市值双中性
- rank_icir_YYYY_行业市值双中性

重算脚本：
- scripts/cogalpha_lqtp/add_2026_neutral_rankic_to_csv.py
- scripts/cogalpha_lqtp/add_yearly_double_neutral_to_csv.py
行数 228
文件 reports/factor_rankic_close_vs_vwap_vs_neutral.csv
