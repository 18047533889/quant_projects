# GTJA-191 Alpha 003
# source: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

ts_sum(if_else(close == delay(close, 1), 0, close - if_else(close > delay(close, 1), min(low, delay(close, 1)), max(high, delay(close, 1)))), 6)
