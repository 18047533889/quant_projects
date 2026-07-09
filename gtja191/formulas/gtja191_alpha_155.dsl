# GTJA-191 Alpha 155
# source: SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)

ts_ema(volume, 12) - ts_ema(volume, 26) - ts_ema(ts_ema(volume, 12) - ts_ema(volume, 26), 9)
