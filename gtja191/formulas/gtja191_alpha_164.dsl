# GTJA-191 Alpha 164
# source: SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)

EMA((if_else(close > delay(close, 1), 1 / (close - delay(close, 1) + 1e-8), 1) - ts_min(if_else(close > delay(close, 1), 1 / (close - delay(close, 1) + 1e-8), 1), 12)) / (high - low + 1e-8) * 100, 12)
