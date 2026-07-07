# GTJA-191 Alpha 037
# source: (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))

(-1 * rank(((ts_sum(open, 5) * ts_sum(ret, 5)) - delay((ts_sum(open, 5) * ts_sum(ret, 5)), 10))))
