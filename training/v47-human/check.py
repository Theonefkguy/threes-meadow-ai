"""Rule checks for a tracked game: normal cards come in bags of 12 (four each of 1, 2, 3) and bonus
cards come exactly once per block of 21 draws. Prints the best bag and block alignment and every
violation, which points at recognition errors.  python3 check.py TRACK.json"""
import json, sys
d = json.load(open(sys.argv[1])); steps = d['steps']
cards = [s['card'] for s in steps if s['cell'] >= 0]
start = [v for row in d['startBoard'] for v in row]
opening = [start.count(1), start.count(2), start.count(3)]
# Draw i (0-based) is the card entering on move i; the opening preview was drawn before move 0.
normal = [(i, c) for i, c in enumerate(cards) if c <= 3]
# Bags: the first bag holds the opening cards plus the first (12 - opening total) normal draws.
first = 12 - sum(opening); bags = [normal[:first]] + [normal[j:j + 12] for j in range(first, len(normal), 12)]
bad = []
for b, bag in enumerate(bags):
    cnt = [sum(1 for _, c in bag if c == v) for v in (1, 2, 3)]
    if b == 0: cnt = [cnt[i] + opening[i] for i in range(3)]
    full = (b == 0 and len(bag) == first) or len(bag) == 12
    if full and cnt != [4, 4, 4] or not full and any(x > 4 for x in cnt): bad.append((b, bag[0][0], cnt))
print(f'moves {len(cards)}, normal {len(normal)}, bags {len(bags)}, bags violating 4/4/4: {len(bad)}')
for b in bad[:10]: print('  bag', b[0], 'from move', b[1], 'counts', b[2])
bonus = [i for i, c in enumerate(cards) if c > 3]
best = None
for off in range(21):  # block j covers draws [off + 21 j, off + 21 j + 21)
    blocks = {}
    for i in bonus: blocks.setdefault((i - off) // 21, []).append(i)
    n_blocks = (len(cards) - off) // 21
    viol = sum(len(v) > 1 for v in blocks.values()) + sum(1 for j in range(n_blocks) if j not in blocks and j >= 0)
    if best is None or viol < best[0]: best = (viol, off, blocks, n_blocks)
viol, off, blocks, n_blocks = best
print(f'bonus cards {len(bonus)}, best block offset {off}, violations {viol}')
print('  doubles', [v for v in blocks.values() if len(v) > 1][:8], 'empty blocks', [j for j in range(n_blocks) if j not in blocks][:8])
