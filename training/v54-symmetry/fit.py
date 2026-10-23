"""V54: train an architecture from scratch on the data of V50's last DAgger round (dag14): the V46 AI
positions, DAgger rounds 1-14 repeated 4 times, and the round 7-14 positions with a card >= 1536 repeated
2 more times (30.3M rows per epoch). Recipe as V50 train.py from scratch (AdamW 1e-3, 500 warm-up steps,
cosine, batch 2048, one random symmetry per sample), the same for every architecture.
  arch: base (V50 Net, the control), canon, shared (models.py).
Checkpoint every 1/4 epoch; SIGTERM saves and exits 75. Per epoch: validation games and V47 / V48 positions.
  uv run --project training/v50-transformer python training/v54-symmetry/fit.py ARCH [--epochs N] [--steps N]
--steps: stop after N steps (speed test, nothing saved)."""
import argparse, json, math, signal, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L)); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Net, Canon, Shared
from train import arrays, on_dev, batch, usable, augment, loss_fn, evaluate, DATA, BATCH, dev
ARCH = {'base': Net, 'canon': Canon, 'shared': Shared}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('arch', choices=ARCH); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--steps', type=int)
    args = ap.parse_args(); run = L / 'runs' / args.arch; run.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(DATA / 'positions.npz')); ok = usable(d)
    parts, va = [arrays(d, ok & ~d['val'])], on_dev(arrays(d, ok & d['val']))
    for r in range(1, 15):
        e = dict(np.load(DATA / 'dagger' / f'round-{r}.npz')); parts += [arrays(e, usable(e))] * 4
        if r >= 7: parts += [arrays(e, usable(e) & (e['board'].max(1) >= 12))] * 2
    tr = on_dev({k: np.concatenate([a[k] for a in parts]) for k in parts[0]}); del parts
    h = dict(np.load(DATA / 'human.npz')); hok = usable(h)
    hum = {s: (on_dev(arrays(h, hok & m)), torch.as_tensor(h['human'][hok & m].astype(np.int64), device=dev)) for s, m in (('v47', h['seed'] == 47), ('v48', h['seed'] != 47))}
    n = len(tr['board']); per = n // BATCH; steps = args.epochs * per; quarter = per // 4
    torch.manual_seed(54); net = ARCH[args.arch]().to(dev)
    print(f'{args.arch}: {sum(p.numel() for p in net.parameters())} parameters, train {n}, steps {steps} on {dev}', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    step = 0; ck = run / 'ckpt.pt'; g = torch.Generator(device=dev)
    if ck.exists() and not args.steps:
        s = torch.load(ck, map_location=dev); net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); step = s['step']
        print('resumed at step', step, flush=True)
    def save(): torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': state['step']}, ck)
    state = {'step': step}
    if not args.steps: signal.signal(signal.SIGTERM, lambda *a: (save(), sys.exit(75)))
    log = open(run / 'log.jsonl', 'a') if not args.steps else None; t0 = time.time(); perm_of = {}
    while step < (args.steps or steps):
        epoch = step // per
        if epoch not in perm_of: g.manual_seed(54 + epoch); perm_of.clear(); perm_of[epoch] = torch.randperm(n, device=dev, generator=g)   # same order after a resume
        perm = perm_of[epoch]
        i = step % per; b = augment(batch(tr, perm[i * BATCH:(i + 1) * BATCH]))
        loss, ce, reg = loss_fn(net, b); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1; state['step'] = step
        if step % 50 == 0: loss.item()                       # wait for the GPU now and then (MPS queues work without limit)
        if args.steps and step % 50 == 0: print(step, f'{(time.time() - t0) / step * 1000:.0f} ms/step, loss {loss.item():.4f}', flush=True)
        if not args.steps and (step % quarter == 0 or step == steps):
            save()
            if step % per == 0 or step == steps:
                rec = {'epoch': step / per, 'sec': round(time.time() - t0), 'val': evaluate(net, va), **{s: evaluate(net, b, hh) for s, (b, hh) in hum.items()}}
                log.write(json.dumps(rec) + '\n'); log.flush()
                print(json.dumps({'epoch': rec['epoch'], 'sec': rec['sec'], **{k: {m: round(v, 4) for m, v in rec[k].items()} for k in ('val', 'v47', 'v48')}}), flush=True)
    if not args.steps: torch.save(net.state_dict(), run / 'model.pt')
    else: print(f'{args.arch}: {(time.time() - t0) / step * 1000:.0f} ms/step -> {(time.time() - t0) / step * per / 60:.0f} min per epoch')
