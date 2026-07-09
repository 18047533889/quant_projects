# GTJA-191 Alpha 116
# source: REGBETA(CLOSE,SEQUENCE,20)

ts_regression(close, ts_delay(close, 1), 20, 0, 'slope')
