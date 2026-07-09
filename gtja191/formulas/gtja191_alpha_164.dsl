# GTJA-191 Alpha 164
# source: SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)

ts_ema((where(close > ts_delay(close, 1), 1 / (close - ts_delay(close, 1) + 1e-08), 1) - ts_min(where(close > ts_delay(close, 1), 1 / (close - ts_delay(close, 1) + 1e-08), 1), 12)) / (high - low + 1e-08) * 100, 12)
