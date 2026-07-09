# GTJA-191 Alpha 053
# source: COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100

ts_sum(where(close > ts_delay(close, 1), 1, 0), 12) / 12 * 100
