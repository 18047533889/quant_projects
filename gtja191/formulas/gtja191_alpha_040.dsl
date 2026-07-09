# GTJA-191 Alpha 040
# source: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100

ts_sum(where(close > ts_delay(close, 1), volume, 0), 26) / (ts_sum(where(close <= ts_delay(close, 1), volume, 0), 26) + 1e-08) * 100
