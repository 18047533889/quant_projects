# GTJA-191 Alpha 003
# source: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

ts_sum(where(close == ts_delay(close, 1), 0, close - where(close > ts_delay(close, 1), flex_min(low, ts_delay(close, 1)), flex_max(high, ts_delay(close, 1)))), 6)
