# GTJA-191 Alpha 022
# source: SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)

EMA(((close-ts_mean(close,6))/ts_mean(close,6)-delay((close-ts_mean(close,6))/ts_mean(close,6),3)), 23)
