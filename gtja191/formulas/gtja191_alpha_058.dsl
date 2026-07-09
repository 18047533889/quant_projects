# GTJA-191 Alpha 058
# source: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100

ts_sum(where(close > ts_delay(close, 1), 1, 0), 20) / 20 * 100
