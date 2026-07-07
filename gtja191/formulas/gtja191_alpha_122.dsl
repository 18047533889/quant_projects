# GTJA-191 Alpha 122
# source: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)

(EMA(EMA(EMA(log(close), 12), 12), 12)-delay(EMA(EMA(EMA(log(close), 12), 12), 12),1))/delay(EMA(EMA(EMA(log(close), 12), 12), 12),1)
