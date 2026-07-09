# GTJA-191 Alpha 167
# source: SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)

ts_sum(where(close - ts_delay(close, 1) > 0, close - ts_delay(close, 1), 0), 12)
