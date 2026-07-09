# GTJA-191 Alpha 089
# source: 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))

2 * (ts_ema(close, 12) - ts_ema(close, 26) - ts_ema(ts_ema(close, 12) - ts_ema(close, 26), 9))
