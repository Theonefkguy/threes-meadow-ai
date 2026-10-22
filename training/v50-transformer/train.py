"""V50: distil the website AI into the transformer of model.py.
Target per position: softmax(-gap / TAU) over the legal directions (gap = (best - value) / |best|,
the AI's 6-ply root values), plus a regression of min(gap, 0.1) * 10 per legal direction (weight 0.5).
Training data: every position except the validation games (V46 odd seeds), with one random symmetry
per sample. Positions with one legal direction or a best value of 0 are skipped.
Reports per epoch on the validation games and the human positions (V47 / V48): near (chosen move
within 0.25% of the AI's best), match, bad5 (>= 5% worse). Checkpoint every epoch; SIGTERM saves and exits 75.
DAgger rounds: --extra adds labelled positions from the student's own games (dagger/*.npz, repeated
--repeat times), --init starts from an earlier model. --late-extra files are added like --extra, and their positions whose top card is >= 1536 are repeated --late K more times.
  uv run python train.py NAME [--epochs N] [--small] [--init runs/X/model.pt] [--extra A.npz,B.npz] [--repeat K] [--late-extra C.npz,..] [--late K] [--lr X]"""
import argparse, json, math, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch
from model import Net, SYM
L = Path(__file__).resolve().parent; DATA = Path('/Volumes/Extreme SSD/Datasets/threes-v50')
TAU, BATCH, LR = 0.002, 2048, 1e-3
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def arrays(d, rows):
    """Compact numpy arrays (uint8 inputs, float32 gaps) of the selected rows."""
    v = d['values'][rows].astype(np.float64); legal = ~np.isnan(v)
    with np.errstate(invalid='ignore', divide='ignore'):
        top = np.nanmax(v, 1); gap = (top[:, None] - v) / np.abs(top)[:, None]
    gap = np.where(legal, np.nan_to_num(gap, nan=9.0), 9.0)
    return {'board': d['board'][rows].astype(np.uint8), 'cards': d['cards'][rows].astype(np.uint8), 'bonus': d['bonus'][rows].astype(bool),
            'counts': d['counts'][rows].astype(np.uint8), 'cycle': d['cycle'][rows].astype(np.uint8), 'legal': legal, 'gap': gap.astype(np.float32)}
def on_dev(a):
    """Arrays -> device tensors, kept compact (uint8); see batch() for the model's dtypes."""
    return {k: torch.as_tensor(np.ascontiguousarray(v)).to(dev) for k, v in a.items()}
def batch(b, idx=None):
    """A slice of compact tensors with the model's dtypes (long inputs)."""
    out = {k: (v if idx is None else v[idx]) for k, v in b.items()}
    for k in ('board', 'cards', 'counts', 'cycle'): out[k] = out[k].long()
    return out
def tensors(d, rows): return batch(on_dev(arrays(d, rows)))
def usable(d):
    v = d['values']; legal = ~np.isnan(v)
    return (legal.sum(1) >= 2) & (np.nanmax(np.where(legal, v, -np.inf), 1) != 0)

CELL = torch.tensor([s[0] for s in SYM], device=dev); DIRS = torch.tensor([s[1] for s in SYM], device=dev)
def augment(b):
    s = torch.randint(0, 8, (len(b['board']),), device=dev); out = dict(b)
    out['board'] = b['board'].gather(1, CELL[s])
    di = DIRS[s]
    out['legal'] = torch.zeros_like(b['legal']).scatter(1, di, b['legal'])
    out['gap'] = torch.zeros_like(b['gap']).scatter(1, di, b['gap'])
    return out

def loss_fn(net, b):
    logits, gp = net(b['board'], b['cards'], b['bonus'], b['counts'], b['cycle'], b['legal'])
    tgt = torch.softmax((-b['gap'] / TAU).masked_fill(~b['legal'], -1e9), 1)
    ce = -(tgt * torch.log_softmax(logits, 1)).sum(1).mean()
    reg = (((gp - b['gap'].clamp(max=0.1) * 10) ** 2) * b['legal']).sum() / b['legal'].sum()
    return ce + 0.5 * reg, ce, reg

@torch.no_grad()
def evaluate(net, b, human=None):
    net.eval(); picks = []
    for i in range(0, len(b['board']), 8192):
        x = batch(b, slice(i, i + 8192)); logits, _ = net(*(x[k] for k in ('board', 'cards', 'bonus', 'counts', 'cycle', 'legal'))); picks.append(logits.argmax(1))
    net.train(); p = torch.cat(picks); g = b['gap'].gather(1, p[:, None])[:, 0]
    best = b['gap'].argmin(1)
    out = {'n': len(p), 'near': (g < 0.0025).float().mean().item(), 'match': (p == best).float().mean().item(), 'bad5': (g >= 0.05).float().mean().item(),
           'meanGap': g.clamp(max=1).mean().item()}
    if human is not None: out['humanAgree'] = (p == human).float().mean().item()
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--small', action='store_true')
    ap.add_argument('--init'); ap.add_argument('--extra', default=''); ap.add_argument('--repeat', type=int, default=1); ap.add_argument('--lr', type=float, default=LR); ap.add_argument('--late', type=int, default=0); ap.add_argument('--late-extra', default='')
    args = ap.parse_args(); epochs = args.epochs; LR = args.lr
    run = L / 'runs' / args.name; run.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(DATA / 'positions.npz')); ok = usable(d)
    parts, va = [arrays(d, ok & ~d['val'])], on_dev(arrays(d, ok & d['val']))
    lf = set(filter(None, args.late_extra.split(',')))
    for f in [*filter(None, args.extra.split(',')), *lf]:
        e = dict(np.load(f)); parts += [arrays(e, usable(e))] * args.repeat
        if f in lf: parts += [arrays(e, usable(e) & (e['board'].max(1) >= 12))] * args.late
    tr = on_dev({k: np.concatenate([a[k] for a in parts]) for k in parts[0]}); del parts
    h = dict(np.load(DATA / 'human.npz')); hok = usable(h)
    hum = {s: (on_dev(arrays(h, hok & m)), torch.as_tensor(h['human'][hok & m].astype(np.int64), device=dev)) for s, m in (('v47', h['seed'] == 47), ('v48', h['seed'] != 47))}
    n = len(tr['board']); steps = epochs * (n // BATCH)
    print(f'train {n} val {len(va["board"])} human {[len(v[1]) for v in hum.values()]} steps {steps} on {dev}'
          + (f', GPU memory {torch.mps.driver_allocated_memory() / 2**30:.1f}G' if dev.type == 'mps' else ''), flush=True)
    torch.manual_seed(50)
    net = Net(**({'d': 64, 'layers': 2, 'heads': 4, 'ff': 256} if args.small else {})).to(dev)
    if args.init: net.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / (50 if args.init else 500)) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    start = 0; ck = run / 'ckpt.pt'
    if ck.exists():
        s = torch.load(ck, map_location=dev); net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); start = s['epoch']
        print('resumed at epoch', start, flush=True)
    def save(epoch): torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch}, ck)
    state = {'epoch': start}
    signal.signal(signal.SIGTERM, lambda *a: (save(state['epoch']), sys.exit(75)))
    log = open(run / 'log.jsonl', 'a')
    for epoch in range(start, epochs):
        t0 = time.time(); perm = torch.randperm(n, device=dev); tot = [0.0, 0.0, 0.0]; k = 0
        for i in range(0, n - BATCH + 1, BATCH):
            b = augment(batch(tr, perm[i:i + BATCH]))
            loss, ce, reg = loss_fn(net, b); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot[0] += loss.item() if k % 50 == 0 else 0; k += 1
        state['epoch'] = epoch + 1; save(epoch + 1)
        rec = {'epoch': epoch + 1, 'sec': round(time.time() - t0), 'val': evaluate(net, va), **{s: evaluate(net, b, hh) for s, (b, hh) in hum.items()}}
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(json.dumps({'epoch': rec['epoch'], 'sec': rec['sec'], **{k: {m: round(v, 4) for m, v in rec[k].items()} for k in ('val', 'v47', 'v48')}}), flush=True)
    torch.save(net.state_dict(), run / 'model.pt')
