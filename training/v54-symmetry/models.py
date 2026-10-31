"""V54: two ways to make the V50 transformer exactly symmetric under the 8 rotations / mirrors.
Both keep V50's forward signature (board, cards, bonus, counts, cycle, legal) -> (logits [B,4], gaps [B,4]),
so train.py's loss and evaluation work unchanged.
- Canon: the V50 Net on a canonical orientation: of the 8 transformed boards the lexicographically
  largest (row-major ranks); outputs are mapped back to the original directions. One forward pass.
- Shared: one network judges "move left" only. For each direction the board is rotated so that this
  direction becomes left, and the four rotated boards go through the same network (4 passes); each
  gives one logit and one gap. The remaining symmetry that keeps "left" (the up-down mirror) is built in:
  a cell's position vector depends on its column and on whether its row is an edge or inner row, and
  attention between cells gets a learned bias per layer and head from (|row difference|, column
  difference). So both models give the same choice in all 8 orientations by construction."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'v50-transformer'))
from model import Net, SYM

def _tokens(m, board, cards, bonus, counts, cycle, cells):
    """Token sequence as in V50 (summary, 16 cells, preview, counts, cycle); cells = the cell vectors."""
    B = board.shape[0]
    pv = torch.zeros(B, 17, device=board.device).scatter_(1, cards, 1.0); pv[:, 0] = 0; pv[:, 16] = bonus.float()
    cn = F.one_hot(counts.clamp(0, 4), 5).flatten(1).float()
    cy = torch.cat([F.one_hot(cycle[:, 0].clamp(0, 20), 21), cycle[:, 1:2].clamp(0, 1), F.one_hot(cycle[:, 2].clamp(0, 21), 22)], 1).float()
    return torch.cat([m.summary.expand(B, -1, -1), cells, m.preview(pv)[:, None], m.counts(cn)[:, None], m.cycle(cy)[:, None]], 1)

class Canon(nn.Module):
    def __init__(self, **kw):
        super().__init__(); self.net = Net(**kw)
        self.register_buffer('CELL', torch.tensor([s[0] for s in SYM])); self.register_buffer('DIRS', torch.tensor([s[1] for s in SYM]))
        self.register_buffer('W', 16 ** torch.arange(7, -1, -1))
    def frame(self, board):
        """index of the canonical symmetry per board: largest first 8 cells, then largest last 8."""
        T = board.clamp(max=15)[:, self.CELL]                          # [B,8,16]
        k1 = (T[..., :8] * self.W).sum(-1); k2 = (T[..., 8:] * self.W).sum(-1)
        k2 = k2.masked_fill(k1 < k1.max(1, keepdim=True).values, -1)
        return k2.argmax(1)
    def forward(self, board, cards, bonus, counts, cycle, legal):
        s = self.frame(board); cell = self.CELL[s]; di = self.DIRS[s]          # original direction d -> new di[:, d]
        lg = torch.zeros_like(legal).scatter(1, di, legal)
        logits, gaps = self.net(board.gather(1, cell), cards, bonus, counts, cycle, lg)
        return logits.gather(1, di), gaps.gather(1, di)
    def features(self, board, cards, bonus, counts, cycle):
        return self.net.features(board.gather(1, self.CELL[self.frame(board)]), cards, bonus, counts, cycle)

class Shared(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=512):
        super().__init__()
        self.cell = nn.Embedding(16, d); self.pos = nn.Parameter(torch.randn(8, d) * 0.02)      # (column, edge/inner row)
        self.preview = nn.Linear(17, d); self.counts = nn.Linear(15, d); self.cycle = nn.Linear(44, d)
        self.summary = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation='gelu', batch_first=True, norm_first=True) for _ in range(layers))
        self.bias = nn.Parameter(torch.zeros(layers, heads, 4, 7))            # [layer, head, |row diff|, column diff + 3]
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 2); self.H = heads
        r, c = torch.arange(16) // 4, torch.arange(16) % 4
        self.register_buffer('POS', c * 2 + torch.minimum(r, 3 - r))           # 0..7
        self.register_buffer('DR', (r[:, None] - r[None]).abs()); self.register_buffer('DC', c[:, None] - c[None] + 3)
        # for direction d: the rotation (s < 4) that turns d into left (0)
        rot = [next(s for s in range(4) if SYM[s][1][d] == 0) for d in range(4)]
        self.register_buffer('ROT', torch.tensor([SYM[s][0] for s in rot]))   # [4,16]
    def trunk(self, board, cards, bonus, counts, cycle):
        """summary token [B, d] of boards already turned so that the judged move is left."""
        B = board.shape[0]
        t = _tokens(self, board, cards, bonus, counts, cycle, self.cell(board.clamp(max=15)) + self.pos[self.POS])
        for li, layer in enumerate(self.layers):
            m = torch.zeros(self.H, 20, 20, device=board.device); m[:, 1:17, 1:17] = self.bias[li][:, self.DR, self.DC]
            t = layer(t, src_mask=m.repeat(B, 1, 1))
        return self.norm(t[:, 0])
    def forward(self, board, cards, bonus, counts, cycle, legal):
        B = board.shape[0]; rep = lambda x: x.repeat(4, *[1] * (x.dim() - 1))
        bd = board[:, self.ROT].transpose(0, 1).reshape(4 * B, 16)               # direction-major
        h = self.head(self.trunk(bd, rep(cards), rep(bonus), rep(counts), rep(cycle))).view(4, B, 2)
        return h[..., 0].T.masked_fill(~legal, -1e9), h[..., 1].T
