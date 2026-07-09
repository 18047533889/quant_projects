# GTJA-191 Alpha 135
# source: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)

ts_ema(ts_delay(close / ts_delay(close, 20), 1), 39)
