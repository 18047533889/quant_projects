# GTJA-191 Alpha 173
# source: 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)

3 * ts_ema(close, 12) - 2 * ts_ema(ts_ema(close, 12), 12) + ts_ema(ts_ema(ts_ema(log(close), 12), 12), 12)
