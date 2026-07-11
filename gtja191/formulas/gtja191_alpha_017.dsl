# GTJA-191 Alpha 017
# source: RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)

power(rank(col('vwap') - ts_max(col('vwap'), 15)), ts_delta(close, 5))
