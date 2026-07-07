# GTJA-191 Alpha 129
# source: SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)

ts_sum(if_else(close - delay(close, 1) < 0, abs(close - delay(close, 1)), 0), 12)
