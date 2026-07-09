# GTJA-191 Alpha 122
# source: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)

(ts_ema(ts_ema(ts_ema(log(close), 12), 12), 12) - ts_delay(ts_ema(ts_ema(ts_ema(log(close), 12), 12), 12), 1)) / ts_delay(ts_ema(ts_ema(ts_ema(log(close), 12), 12), 12), 1)
