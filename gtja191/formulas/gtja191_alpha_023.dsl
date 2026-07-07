# GTJA-191 Alpha 023
# source: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100

EMA(if_else(close > delay(close, 1), ts_std(close, 20), 0), 19) / (EMA(if_else(close > delay(close, 1), ts_std(close, 20), 0), 19) + EMA(if_else(close <= delay(close, 1), ts_std(close, 20), 0), 19) + 1e-8) * 100
