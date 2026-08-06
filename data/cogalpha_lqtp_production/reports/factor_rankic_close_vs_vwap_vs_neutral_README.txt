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

LQTP 提交标记（汇总列，紧跟 display_name）：
- LQTP提交状态：已提交 / 未提交
- LQTP已提交：Y / N
- LQTP未提交原因码：unsupported_operator / no_formula / duplicate_formula / broken_formula 等
- LQTP未提交原因：中文短因
- LQTP问题说明：具体算子问题或平台报错（可转发）
- LQTP未知算子：阻塞算子名
- LQTP_definition_id / LQTP平台状态：已入库时的 UUID 与 validated 等
- LQTP重复于因子：因公式完全重复而未用新名入库时指向已有因子

状态来源：/home/shw/reports/lqtp_factor_submit_status.json
转发清单：/home/shw/reports/lqtp_NOT_SUBMITTED_for_forward.md
