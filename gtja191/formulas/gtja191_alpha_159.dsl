# GTJA-191 Alpha 159
# source: ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)

((close-ts_sum(min(low,delay(close,1)),6))/ts_sum(max(high,delay(close,1))-min(low,delay(close,1)),6)*12*24+(close-ts_sum(min(low,delay(close,1)),12))/ts_sum(max(high,delay(close,1))-min(low,delay(close,1)),12)*6*24+(close-ts_sum(min(low,delay(close,1)),24))/ts_sum(max(high,delay(close,1))-min(low,delay(close,1)),24)*6*24)*100/(6*12+6*24+12*24)
