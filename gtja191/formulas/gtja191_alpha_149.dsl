# GTJA-191 Alpha 149
# source: REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)

ts_regression(if_else(index_close < delay(index_close, 1), close / delay(close, 1) - 1, 0), if_else(index_close < delay(index_close, 1), index_close / delay(index_close, 1) - 1, 0), 252, 0, 'slope')
