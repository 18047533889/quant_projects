# GTJA-191 Alpha 181
# source: SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)

ts_sum((close / delay(close, 1) - 1 - ts_mean(close / delay(close, 1) - 1, 20)) - power(index_close - ts_mean(index_close, 20), 2), 20) / (ts_sum(power(index_close - ts_mean(index_close, 20), 3), 20) + 1e-8)
