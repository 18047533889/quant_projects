# GTJA-191 Alpha 089
# source: 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))

2*(EMA(close, 12)-EMA(close, 26)-EMA(EMA(close, 12)-EMA(close, 26), 9))
