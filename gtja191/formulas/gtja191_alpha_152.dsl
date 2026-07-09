# GTJA-191 Alpha 152
# source: SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)

ts_ema(ts_mean(ts_delay(ts_ema(ts_delay(close / ts_delay(close, 9), 1), 17), 1), 12) - ts_mean(ts_delay(ts_ema(ts_delay(close / ts_delay(close, 9), 1), 17), 1), 26), 17)
