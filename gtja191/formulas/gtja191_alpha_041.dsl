# GTJA-191 Alpha 041
# source: (RANK(MAX(DELTA((VWAP), 3), 5))* -1)

rank(ts_max(ts_delta(col('vwap'), 3), 5)) * -1
