# GTJA-191 Alpha 149
# source: REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)

ts_regression(where(index_close < ts_delay(index_close, 1), close / ts_delay(close, 1) - 1, 0), where(index_close < ts_delay(index_close, 1), index_close / ts_delay(index_close, 1) - 1, 0), 252, 0, 'slope')
