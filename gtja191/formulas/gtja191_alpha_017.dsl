# GTJA-191 Alpha 017
# source: RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)

power(rank(((high + low + close) / 3) - ts_max(((high + low + close) / 3), 15)), ts_delta(close, 5))
