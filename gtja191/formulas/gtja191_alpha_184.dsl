# GTJA-191 Alpha 184
# source: rank(ts_corr(delay(open - close, 1), close, 200)) + rank(open - close)

rank(ts_corr(delay(open - close, 1), close, 200)) + rank(open - close)
