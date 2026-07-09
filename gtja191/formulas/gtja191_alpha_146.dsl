# GTJA-191 Alpha 146
# source: MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60)

ts_mean((close - ts_delay(close, 1)) / (ts_delay(close, 1) + 1e-08) - ts_ema((close - ts_delay(close, 1)) / (ts_delay(close, 1) + 1e-08), 30), 20) * ((close - ts_delay(close, 1)) / (ts_delay(close, 1) + 1e-08) - ts_ema((close - ts_delay(close, 1)) / (ts_delay(close, 1) + 1e-08), 30))
