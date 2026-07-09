# GTJA-191 Alpha 129
# source: SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)

ts_sum(where(close - ts_delay(close, 1) < 0, abs(close - ts_delay(close, 1)), 0), 12)
