# GTJA-191 Alpha 035
# source: (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)

flex_min(rank(ts_decay_linear(ts_delta(open, 1), 15)), rank(ts_decay_linear(ts_corr(volume, open * 0.65 + open * 0.35, 17), 7))) * -1
