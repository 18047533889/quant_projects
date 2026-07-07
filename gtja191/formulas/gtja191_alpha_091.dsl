# GTJA-191 Alpha 091
# source: ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)

((rank((close - ts_max(close, 5)))*rank(ts_corr((ts_mean(volume,40)), low, 5))) * -1)
