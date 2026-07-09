# GTJA-191 Alpha 109
# source: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)

ts_ema(high - low, 9) / ts_ema(ts_ema(high - low, 9), 9)
