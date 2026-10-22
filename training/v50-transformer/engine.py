"""V50: Python port of the C++ game under the bonus-cycle rule (training/v38-progress/cgame.h,
training/v3/ntuple-fast.h, search5p.h Cycle), with the same LCG, so a seed and a move list give the
same game as the C++ tools. Boards in ranks (1, 2, 3, then 6 = 4, ..., 12288 = 15); directions
0 left, 1 right, 2 up, 3 down. Also the public state the website AI sees (card counts, cycle).
  python engine.py  -> replays V46 games from their seeds and moves and checks every board."""
FINAL_RANK = 15
LANES = [[[l * 4 + j for j in range(4)] for l in range(4)], [[l * 4 + 3 - j for j in range(4)] for l in range(4)],
         [[j * 4 + l for j in range(4)] for l in range(4)], [[(3 - j) * 4 + l for j in range(4)] for l in range(4)]]

def mergeable(a, b): return a and b and (a + b == 3 or (a >= 3 and a == b))
def points(r): return 0 if r < 3 else 3.0 ** (r - 2)

def move(b, d):
    """(board after the move, entry cells) like Projection; no entries = illegal."""
    out = list(b); entries = []
    for idx in LANES[d]:
        a = [b[i] for i in idx]; ch = False
        for j in range(1, 4):
            if a[j] and (not a[j - 1] or mergeable(a[j], a[j - 1])):
                x, y = a[j], a[j - 1]; a[j - 1] = x if not y else 3 if x + y == 3 else x + 1; a[j] = 0; ch = True
        if ch:
            for j in range(4): out[idx[j]] = a[j]
            entries.append(idx[3])
    return out, entries
def legal(b): return any(move(b, d)[1] for d in range(4))

class RNG:
    def __init__(self, seed): self.state = seed & 0xFFFFFFFF
    def next(self): self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF; return self.state / 4294967296.0
    def index(self, n): return int(self.next() * n)

def bonuses(max_rank):
    """[(cards, probabilities, window probability)] under bonus rule 1 (window, then card)."""
    n = max_rank - 6
    if n <= 0: return []
    windows, width = max(1, n - 2), min(3, n)
    return [([4 + w + j for j in range(width)], [1 / width] * width, 1 / windows) for w in range(windows)]

class Game:
    def __init__(self, seed):
        self.rng = RNG(seed); self.draws = 0; self.slot = -1; self.turns = 0; self.over = False; self.score_points = 0.0
        self.refill(); b = [self.normal() for _ in range(9)] + [0] * 7
        for i in range(15, 0, -1):
            j = self.rng.index(i + 1); b[i], b[j] = b[j], b[i]
        self.b = b; self.draw()
        # public state (card-memory.js / common.h observe, search5p.h Cycle)
        self.counts = [4, 4, 4]
        for r in b:
            if 1 <= r <= 3: self.counts[r - 1] -= 1
        self.counts = self.observe(self.counts); self.cycle = (0, 0, 0)
    def refill(self):
        self.deck = [i // 4 + 1 for i in range(12)]
        for i in range(11, 0, -1):
            j = self.rng.index(i + 1); self.deck[i], self.deck[j] = self.deck[j], self.deck[i]
        self.left = 12
    def normal(self):
        if not self.left: self.refill()
        self.left -= 1; return self.deck[self.left]
    def draw(self):
        d = self.draws; self.draws += 1
        if d >= 1:
            pos = (d - 1) % 21
            if pos == 0: self.slot = self.rng.index(21)
            lst = bonuses(max(self.b))
            if pos == self.slot and lst:
                r = self.rng.next(); s = 0; pick = lst[-1]
                for item in lst:
                    s += item[2]
                    if r < s: pick = item; break
                self.next = {'cards': pick[0], 'p': pick[1], 'bonus': True}; return
        self.next = {'cards': [self.normal()], 'p': [1.0], 'bonus': False}
    def observe(self, c):
        c = list(c)
        if not self.next['bonus']:
            if sum(c) == 0: c = [4, 4, 4]
            c[self.next['cards'][0] - 1] -= 1
        return c
    def step(self, d):
        a, entries = move(self.b, d)
        assert entries, 'illegal move'
        if max(a) >= FINAL_RANK:
            self.b = a; self.turns += 1; self.over = True; self.next = None; return
        pos = entries[self.rng.index(len(entries))]; card = self.next['cards'][0]
        if len(self.next['cards']) > 1:
            r = self.rng.next(); s = 0; card = self.next['cards'][-1]
            for c, p in zip(self.next['cards'], self.next['p']):
                s += p
                if r < s: card = c; break
        a[pos] = card; self.b = a; self.turns += 1; self.score_points += points(card)
        self.over = not legal(a)
        if self.over: self.next = None; return
        # cycle of the draw after this move: possible = a bonus could be shown (board before the draw)
        possible = bool(bonuses(max(a)))
        self.draw()
        pos_, used, excl = self.cycle; pos_ += 1
        if self.next['bonus']: used = 1
        elif possible: excl += 1
        self.cycle = (0, 0, 0) if pos_ == 21 else (pos_, used, excl)
        self.counts = self.observe(self.counts)
    def score(self): return sum(points(r) for r in self.b)

if __name__ == '__main__':
    import json, sys
    from pathlib import Path
    files = sorted(Path(__file__).resolve().parent.parent.joinpath('v46-margins', 'data').glob('m-*.jsonl'))[:int(sys.argv[1]) if len(sys.argv) > 1 else 6]
    games = checked = 0
    for f in files:
        cur = None
        for line in f.open():
            m = json.loads(line)
            if m['move'] == 0: cur = Game(m['seed']); games += 1
            st = (cur.b, cur.next['cards'], cur.next['bonus'], cur.counts, list(cur.cycle))
            exp = (m['board'], m['next']['cards'], m['next']['bonus'], m['counts'], m['cycle'])
            assert st == exp, (m['seed'], m['move'], st, exp)
            for d in range(4): assert bool(move(cur.b, d)[1]) == bool(m['dirs'][d]['legal'])
            cur.step(m['chosen']); checked += 1
    print(games, 'games,', checked, 'positions identical to the C++ records (board, preview, counts, cycle)')
