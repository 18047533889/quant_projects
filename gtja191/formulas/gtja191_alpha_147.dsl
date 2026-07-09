# GTJA-191 Alpha 147
# source: REGBETA(MEAN(CLOSE,12),SEQUENCE(12))

ts_regression(ts_mean(close, 12), ts_delay(close, 1), 12, 0, 'slope')
