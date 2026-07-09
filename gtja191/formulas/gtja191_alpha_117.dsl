# GTJA-191 Alpha 117
# source: ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))

ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(ret, 32))
