"""V48 step 1: decode LogicCube's PKU 2048 Threes replays (pku-replay-v2.t) into V47-style game files.
Re-implements the site's decoder (raw/game_replay.js) and Threes move rule (raw/game_move.js) instead of
running the downloaded code, then replays every event and checks it against the rules:
spawn on an entry cell of the move, 12-card bags of 4/4/4, bonus only at highest >= 48 and within highest/8.
Bonus windows are not in the replay: each bonus step lists every 3-card window that contains the card.
  python decode.py  ->  games/<id>.json"""
import json, math
from pathlib import Path
L = Path(__file__).resolve().parent
CHARS = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ¤¾")
assert len(CHARS) == 128
TIME = [(1, 255), (3, 1021), (12, 4084), (48, 16336), (192, 65344), (768, 265966), (3072, 1050094), (12288, 4186606)]
SITE = ['up', 'down', 'left', 'right']; DIRS = ['left', 'right', 'up', 'down']

def decode_event(s):
    p = (CHARS.index(s[0]) << 14) + (CHARS.index(s[1]) << 7) + CHARS.index(s[2])
    tier, q = (p >> 10) & 7, (p >> 13) & 255
    ms = None
    if tier != 7 or q != 255:
        ms = TIME[tier][0] * q + (TIME[tier - 1][1] + TIME[tier][0] if tier > 0 else 0)
    rank = (p >> 2) & 15
    return {'move': SITE[p & 3], 'value': 1 if rank == 0 else 2 if rank == 1 else 3 * 2 ** (rank - 2), 'x': (p >> 6) & 3, 'y': (p >> 8) & 3, 'ms': ms}

def merge(a, b):
    if a <= 0 or b <= 0: return None
    if a + b == 3 and a != b: return 3
    return a * 2 if a >= 3 and a == b else None

def move(board, d):
    nxt = [r[:] for r in board]; entries = []
    for i in range(4):
        coords = [(i, j) if d == 'left' else (i, 3 - j) if d == 'right' else (j, i) if d == 'up' else (3 - j, i) for j in range(4)]
        vals = [board[y][x] for y, x in coords]; moved = False; merged = set()
        for k in range(1, 4):
            v = vals[k]
            if not v: continue
            if vals[k - 1] == 0: vals[k - 1], vals[k] = v, 0; moved = True; continue
            c = merge(vals[k - 1], v) if k - 1 not in merged else None
            if c is not None: vals[k - 1], vals[k] = c, 0; merged.add(k - 1); moved = True
        for (y, x), v in zip(coords, vals): nxt[y][x] = v
        if moved and vals[3] == 0: entries.append(coords[3])
    return nxt, nxt != board, entries

def score(b): return sum(3 ** (round(math.log2(v / 3)) + 1) for r in b for v in r if v >= 3)
def groups(h):
    vals = []; n = 6
    while n <= h / 8: vals.append(n); n *= 2
    return [vals[i:i + 3] for i in range(max(1, len(vals) - 2))] if vals else []

def run(game):
    code = game['replay']; head, body = code.split('_', 1)
    assert head == 'pku-replay-v2.t.4x4', head
    body = list(body); assert len(body) % 3 == 0
    ev = [decode_event(body[i:i + 3]) for i in range(0, len(body), 3)]
    board = [[0] * 4 for _ in range(4)]; bag = {1: 4, 2: 4, 3: 4}
    def take(v, b):
        if v <= 3:
            assert bag[v] > 0, 'bag'; bag[v] -= 1
            if not any(bag.values()): bag.update({1: 4, 2: 4, 3: 4})
        else:
            h = max(map(max, b)); assert h >= 48 and v <= h // 8 and v % 6 == 0 and (v // 6) & (v // 6 - 1) == 0, ('bonus', v, h)
    for e in ev[:9]:
        assert board[e['y']][e['x']] == 0; take(e['value'], board); board[e['y']][e['x']] = e['value']
    start = [r[:] for r in board]; steps = []; t = 0.0
    for i, e in enumerate(ev[9:]):
        take(e['value'], board)  # the site checks the card against the board it was drawn on
        nb, ok, entries = move(board, e['move']); assert ok, ('no move', i)
        assert (e['y'], e['x']) in entries, ('spawn cell', i)
        nb[e['y']][e['x']] = e['value']
        t += (e['ms'] or 0) / 1000
        s = {'i': i, 'dir': DIRS.index(e['move']), 'cell': e['y'] * 4 + e['x'], 'card': e['value'], 'board': nb, 't': round(t, 3), 'ms': e['ms'], 'multi': 1}
        if e['value'] > 3: s['windows'] = [w for w in groups(max(map(max, board))) if e['value'] in w]
        steps.append(s); board = nb
    return {'id': game['id'], 'score': int(game['score']), 'moves': int(game['moves']), 'playedAt': game['playedAt'], 'elapsedMs': int(game['elapsedMs']),
            'startBoard': start, 'steps': steps, 'finalScore': score(board), 'highest': max(map(max, board))}

if __name__ == '__main__':
    user = json.loads((L / 'raw' / 'user.json').read_text())
    (L / 'games').mkdir(exist_ok=True)
    for g in user['games']:
        if g['mode'] != 'threes' or not g.get('replay'): continue
        out = run(g)
        bonus = [s for s in out['steps'] if s['card'] > 3]
        print(g['id'], 'moves', len(out['steps']), '/', out['moves'], 'score', out['finalScore'], '/', out['score'], 'highest', out['highest'],
              'bonus', len(bonus), 'ambiguous windows', sum(len(s['windows']) > 1 for s in bonus), 'time', round(out['steps'][-1]['t']), 's /', out['elapsedMs'] // 1000)
        (L / 'games' / f"{g['id']}.json").write_text(json.dumps(out, separators=(',', ':')) + '\n')
