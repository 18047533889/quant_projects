# GTJA-191 Alpha 079
# source: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100

ts_ema(flex_max(close - ts_delay(close, 1), 0), 23) / ts_ema(abs(close - ts_delay(close, 1)), 23) * 100
