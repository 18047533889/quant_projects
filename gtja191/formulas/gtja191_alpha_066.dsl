# GTJA-191 Alpha 066
# source: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100

(close - ts_mean(close, 6)) / ts_mean(close, 6) * 100
