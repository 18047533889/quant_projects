# GTJA-191 Alpha 151
# source: SMA(CLOSE-DELAY(CLOSE,20),20,1)

ts_ema(close - ts_delay(close, 20), 39)
