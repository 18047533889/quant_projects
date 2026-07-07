# GTJA-191 Alpha 090
# source: ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)

( rank(ts_corr(rank(((high + low + close) / 3)), rank(volume), 5)) * -1)
