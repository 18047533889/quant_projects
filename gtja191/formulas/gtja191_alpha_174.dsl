# GTJA-191 Alpha 174
# source: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)

ts_ema(where(close > ts_delay(close, 1), ts_std(close, 20), 0), 19)
