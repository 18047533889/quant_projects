# GTJA-191 Alpha 096
# source: SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)

EMA(EMA((close-ts_min(low,9))/(ts_max(high,9)-ts_min(low,9))*100, 5), 5)
