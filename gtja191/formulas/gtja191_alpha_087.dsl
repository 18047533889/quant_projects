# GTJA-191 Alpha 087
# source: ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)

(rank(ts_decay_linear(ts_delta((high + low + close) / 3, 4), 7)) + ts_rank(ts_decay_linear((low * 0.9 + low * 0.1 - (high + low + close) / 3) / (open - (high + low) / 2), 11), 7)) * -1
