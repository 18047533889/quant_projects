# GTJA-191 Alpha 021
# source: REGBETA(MEAN(CLOSE,6),SEQUENCE(6))

ts_regression(ts_mean(close, 6), delay(close, 1), 6, 0, 'slope')
