# GTJA-191 Alpha 144
# source: SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)

ts_sum(if_else(close < delay(close, 1), abs(close / delay(close, 1) - 1) / (amount + 1e-8), 0), 20) / (ts_sum(if_else(close < delay(close, 1), 1, 0), 20) + 1e-8)
