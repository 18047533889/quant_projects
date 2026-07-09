# GTJA-191 Alpha 144
# source: SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)

ts_sum(where(close < ts_delay(close, 1), abs(close / ts_delay(close, 1) - 1) / (amount + 1e-08), 0), 20) / (ts_sum(where(close < ts_delay(close, 1), 1, 0), 20) + 1e-08)
