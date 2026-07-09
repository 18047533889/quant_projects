# GTJA-191 Alpha 024
# source: SMA(CLOSE-DELAY(CLOSE,5),5,1)

ts_ema(close - ts_delay(close, 5), 9)
