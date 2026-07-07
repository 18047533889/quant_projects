# GTJA-191 Alpha 043
# source: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)

ts_sum(if_else(close > delay(close, 1), volume, if_else(close < delay(close, 1), -volume, 0)), 6)
