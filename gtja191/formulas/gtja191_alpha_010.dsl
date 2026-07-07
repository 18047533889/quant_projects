# GTJA-191 Alpha 010
# source: (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))

rank(ts_max(power(if_else(ret < 0, ts_std(ret, 20), close), 2), 5))
