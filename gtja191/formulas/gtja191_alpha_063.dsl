# GTJA-191 Alpha 063
# source: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100

ts_ema(flex_max(close - ts_delay(close, 1), 0), 11) / ts_ema(abs(close - ts_delay(close, 1)), 11) * 100
