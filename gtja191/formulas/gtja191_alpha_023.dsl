# GTJA-191 Alpha 023
# source: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100

ts_ema(where(close > ts_delay(close, 1), ts_std(close, 20), 0), 19) / (ts_ema(where(close > ts_delay(close, 1), ts_std(close, 20), 0), 19) + ts_ema(where(close <= ts_delay(close, 1), ts_std(close, 20), 0), 19) + 1e-08) * 100
