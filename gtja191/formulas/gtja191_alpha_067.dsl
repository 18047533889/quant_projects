# GTJA-191 Alpha 067
# source: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100

ts_ema(flex_max(close - ts_delay(close, 1), 0), 47) / ts_ema(abs(close - ts_delay(close, 1)), 47) * 100
