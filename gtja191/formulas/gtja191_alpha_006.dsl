# GTJA-191 Alpha 006
# source: (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)

(rank(sign(ts_delta((((open * 0.85) + (high * 0.15))), 4)))* -1)
