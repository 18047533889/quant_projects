# GTJA-191 Alpha 007
# source: ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))

(rank(ts_max((high + low + close) / 3 - close, 3)) + rank(ts_min((high + low + close) / 3 - close, 3))) * rank(ts_delta(volume, 3))
