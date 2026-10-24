"""V56: does a larger shared model learn more? V54's from-scratch recipe and data (V46 AI positions,
DAgger rounds 1-14 repeated 4 times, round 7-14 positions with a card >= 1536 2 more times; 2 epochs,
AdamW 1e-3, 500 warm-up steps, cosine, batch 2048, one random symmetry per sample, seed 54) for the
V54 shared model at a given width / depth. Run on an AutoDL GPU (CUDA: bf16 autocast, torch.compile,
the eval-mode attention fast path off, as V55 fit.py). Checkpoint every 1/4 epoch, resumable.
  python fit.py NAME --d D --layers N --heads H     (ff = 4 d)  ->  runs/NAME/{model.pt, log.jsonl}
V56_DATA: the data folder (default: the V50 one on the external SSD)."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v54-symmetry')); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Shared
from train import arrays, on_dev, batch, usable, augment, loss_fn, evaluate, DATA, BATCH, dev
DATA = Path(os.environ.get('V56_DATA') or DATA); cuda = dev.type == 'cuda'

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--epochs', type=int, default=2)
    args = ap.parse_args(); run = L / 'runs' / args.name; run.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(DATA / 'positions.npz')); ok = usable(d)
    parts, va = [arrays(d, ok & ~d['val'])], on_dev(arrays(d, ok & d['val']))
    for r in range(1, 15):
        e = dict(np.load(DATA / 'dagger' / f'round-{r}.npz')); parts += [arrays(e, usable(e))] * 4
        if r >= 7: parts += [arrays(e, usable(e) & (e['board'].max(1) >= 12))] * 2
    tr = on_dev({k: np.concatenate([a[k] for a in parts]) for k in parts[0]}); del parts
    h = dict(np.load(DATA / 'human.npz')); hok = usable(h)
    hum = {s: (on_dev(arrays(h, hok & m)), torch.as_tensor(h['human'][hok & m].astype(np.int64), device=dev)) for s, m in (('v47', h['seed'] == 47), ('v48', h['seed'] != 47))}
    n = len(tr['board']); per = n // BATCH; steps = args.epochs * per; quarter = per // 4
    torch.manual_seed(54); net = Shared(d=args.d, layers=args.layers, heads=args.heads, ff=4 * args.d).to(dev)
    (run / 'config.json').write_text(json.dumps({'d': args.d, 'layers': args.layers, 'heads': args.heads, 'ff': 4 * args.d}) + '\n')
    print(f'{args.name}: {sum(p.numel() for p in net.parameters())} parameters, train {n}, steps {steps} on {dev}', flush=True)
    if cuda: torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True; torch.backends.mha.set_fastpath_enabled(False)  # the fast path ignores the attention bias
    f = torch.compile(net) if cuda else net
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    step = 0; ck = run / 'ckpt.pt'; g = torch.Generator(device=dev)
    if ck.exists():
        s = torch.load(ck, map_location=dev); net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); step = s['step']
        print('resumed at step', step, flush=True)
    log = open(run / 'log.jsonl', 'a'); t0 = time.time(); perm_of = {}
    while step < steps:
        epoch = step // per
        if epoch not in perm_of: g.manual_seed(54 + epoch); perm_of.clear(); perm_of[epoch] = torch.randperm(n, device=dev, generator=g)
        i = step % per; b = augment(batch(tr, perm_of[epoch][i * BATCH:(i + 1) * BATCH]))
        with torch.autocast('cuda', torch.bfloat16, enabled=cuda): loss, ce, reg = loss_fn(f, b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 50 == 0: loss.item()
        if step % quarter == 0 or step % per == 0 or step == steps:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, ck)
            if step % per == 0 or step == steps:
                rec = {'epoch': step / per, 'sec': round(time.time() - t0), 'val': evaluate(net, va), **{s: evaluate(net, b, hh) for s, (b, hh) in hum.items()}}
                log.write(json.dumps(rec) + '\n'); log.flush()
                print(json.dumps({'epoch': rec['epoch'], 'sec': rec['sec'], **{k: {m: round(v, 4) for m, v in rec[k].items()} for k in ('val', 'v47', 'v48')}}), flush=True)
    torch.save(net.state_dict(), run / 'model.pt')
