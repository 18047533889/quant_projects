# GTJA-191 Alpha 076
# source: STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)

ts_std(abs((close/delay(close,1)-1))/volume,20)/ts_mean(abs((close/delay(close,1)-1))/volume,20)
