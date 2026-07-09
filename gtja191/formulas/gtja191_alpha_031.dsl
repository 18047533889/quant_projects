# GTJA-191 Alpha 031
# source: (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100

(close - ts_mean(close, 12)) / ts_mean(close, 12) * 100
