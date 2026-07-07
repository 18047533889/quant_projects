# GTJA-191 Alpha 028
# source: 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)

3*EMA((close-ts_min(low,9))/(ts_max(high,9)-ts_min(low,9))*100, 5)-2*EMA(EMA((close-ts_min(low,9))/(ts_max(high, 9)-ts_max(low,9))*100, 5), 5)
