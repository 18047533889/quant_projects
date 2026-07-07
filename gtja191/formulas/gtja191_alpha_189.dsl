# GTJA-191 Alpha 189
# source: MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)

ts_mean(abs(close-ts_mean(close,6)),6)
