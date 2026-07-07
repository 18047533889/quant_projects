# GTJA-191 Alpha 072
# source: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)

EMA((ts_max(high,6)-close)/(ts_max(high,6)-ts_min(low,6))*100, 29)
