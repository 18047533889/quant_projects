# GTJA-191 Alpha 094
# source: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)

ts_sum(where(close > ts_delay(close, 1), volume, where(close < ts_delay(close, 1), -volume, 0)), 30)
