# GTJA-191 Alpha 173
# source: 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)

3*EMA(close, 12)-2*EMA(EMA(close, 12), 12)+EMA(EMA(EMA(log(close), 12), 12), 12)
