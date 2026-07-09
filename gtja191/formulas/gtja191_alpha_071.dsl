# GTJA-191 Alpha 071
# source: (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100

(close - ts_mean(close, 24)) / ts_mean(close, 24) * 100
