# GTJA-191 Alpha 172
# source: MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)? HD:0,14)*100/SUM(TR,14))*100,6)

ts_mean(abs((high - low) / (close + 1e-08)), 6)
