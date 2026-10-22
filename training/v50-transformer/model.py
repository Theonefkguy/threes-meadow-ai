"""V50: a small transformer that scores the four directions of a Threes position.
Tokens: 16 cells (card rank 0-15 + learned cell position), one preview token (multi-hot of the
preview's card ranks + bonus flag), one token for the remaining 1/2/3 counts, one for the bonus
cycle (position in the 21-draw block, shown yet, excluded positions), and a summary token.
Output from the summary token: 4 direction logits (illegal ones masked) and 4 predicted gaps.
Symmetries: SYM[s] = (cell map, direction map) for the 8 rotations / mirrors."""
import torch
import torch.nn as nn

def _symmetries():
    out = []
    for s in range(8):
        def f(y, x, s=s):
            if s >= 4: x = 3 - x
            for _ in range(s % 4): y, x = x, 3 - y
            return y, x
        cell = [0] * 16                         # new board[i] = old board[cell[i]]
        for y in range(4):
            for x in range(4):
                ny, nx = f(y, x); cell[ny * 4 + nx] = y * 4 + x
        vec = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}; dirs = [0] * 4   # new direction of old direction d
        for d, (dy, dx) in vec.items():
            a, b = f(1, 1), f(1 + dy, 1 + dx); dirs[d] = [k for k, v in vec.items() if v == (b[0] - a[0], b[1] - a[1])][0]
        out.append((cell, dirs))
    return out
SYM = _symmetries()

class Net(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=512):
        super().__init__()
        self.cell = nn.Embedding(16, d); self.pos = nn.Parameter(torch.randn(16, d) * 0.02)
        self.preview = nn.Linear(17, d); self.counts = nn.Linear(15, d); self.cycle = nn.Linear(44, d)
        self.summary = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 8)

    def forward(self, board, cards, bonus, counts, cycle, legal):
        """board [B,16] long ranks; cards [B,3] long (0 = none); bonus [B] bool; counts [B,3] long 0-4;
        cycle [B,3] long (pos 0-20, used 0/1, excl 0-21); legal [B,4] bool -> (logits [B,4], gaps [B,4])."""
        B = board.shape[0]
        x = self.cell(board.clamp(max=15)) + self.pos
        pv = torch.zeros(B, 17, device=board.device).scatter_(1, cards, 1.0); pv[:, 0] = 0; pv[:, 16] = bonus.float()
        cn = nn.functional.one_hot(counts.clamp(0, 4), 5).flatten(1).float()
        cy = torch.cat([nn.functional.one_hot(cycle[:, 0].clamp(0, 20), 21), cycle[:, 1:2].clamp(0, 1), nn.functional.one_hot(cycle[:, 2].clamp(0, 21), 22)], 1).float()
        t = torch.cat([self.summary.expand(B, -1, -1), x, self.preview(pv)[:, None], self.counts(cn)[:, None], self.cycle(cy)[:, None]], 1)
        h = self.head(self.norm(self.enc(t)[:, 0]))
        return h[:, :4].masked_fill(~legal, -1e9), h[:, 4:]
