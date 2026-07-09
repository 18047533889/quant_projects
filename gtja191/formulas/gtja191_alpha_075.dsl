# GTJA-191 Alpha 075
# source: COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)

ts_sum(where(and_(close > open, index_close < index_open), 1, 0), 50) / (ts_sum(where(index_close < index_open, 1, 0), 50) + 1e-08)
