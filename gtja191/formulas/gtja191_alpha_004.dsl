# GTJA-191 Alpha 004
# source: ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))

if_else((ts_sum(close, 8) / 8 + ts_std(close, 8)) < ts_sum(close, 2) / 2, -1, if_else(ts_sum(close, 2) / 2 < ts_sum(close, 8) / 8 - ts_std(close, 8), 1, if_else(or_(1 < volume / ts_mean(volume, 20), volume / ts_mean(volume, 20) == 1), 1, -1)))
