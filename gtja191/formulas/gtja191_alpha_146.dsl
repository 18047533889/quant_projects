# GTJA-191 Alpha 146
# source: MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60)

ts_mean((close - delay(close, 1)) / (delay(close, 1) + 1e-8) - EMA((close - delay(close, 1)) / (delay(close, 1) + 1e-8), 30), 20) * ((close - delay(close, 1)) / (delay(close, 1) + 1e-8) - EMA((close - delay(close, 1)) / (delay(close, 1) + 1e-8), 30))
