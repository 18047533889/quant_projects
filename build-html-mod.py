import re

with open('index.html','r') as f:
    content=f.read()

# round1 rows must all carry "本期" tag semantics. We replace the lone "新" tag in the
# round1 block with "本期" (both header + blocks appearance).
content = content.replace('本周新挖增量因子（26）','本周新挖增量因子（58）')
content = content.replace('本次增量入库 26 条，评估口径与既有池一致','本次增量入库 58 条（round1 26 + round2 32），评估口径与既有池一致')

print('replace done; header count now in content:', content.count('本周新挖增量因子（58）'))
