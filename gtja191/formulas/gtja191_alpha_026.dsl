# GTJA-191 Alpha 026
# source: ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))

ts_sum(close, 7) / 7 - close + ts_corr((high + low + close) / 3, ts_delay(close, 5), 230)
