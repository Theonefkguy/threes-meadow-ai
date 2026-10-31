"""V55: one DAgger fine-tune of the V54 shared model, as V50 pipeline2.sh does for the base model: start
from the previous round's model; train on the V46 AI positions, DAgger rounds 1..ROUND repeated 4 times,
and the round 7..ROUND positions with a card >= 1536 repeated 2 more times; 2 epochs, AdamW 3e-4,
50 warm-up steps, cosine, batch 2048, one random symmetry per sample.
Checkpoint every 1/4 epoch; SIGTERM saves and exits 75. Per epoch: validation games and V47 / V48 positions.
From round CLOUD_FROM on, on a Mac this hands the round to cloud.py (an AutoDL 4090); there (CUDA) the
training runs in bf16 autocast with torch.compile, about 12x faster than the Mac's GPU in fp32.
  uv run --project ../v50-transformer python fit.py ROUND INIT.pt   ->  runs/rROUND/model.pt
V55_DATA: the data folder (default: the V50 one on the external SSD)."""
import json, math, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v54-symmetry')); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Shared
from train import arrays, on_dev, batch, usable, augment, loss_fn, evaluate, DATA, BATCH, dev
EPOCHS, LR, CLOUD_FROM = 2, 3e-4, 16
DATA = Path(os.environ.get('V55_DATA') or DATA); cuda = dev.type == 'cuda'

if __name__ == '__main__':
    R, init = int(sys.argv[1]), sys.argv[2]
    if R >= CLOUD_FROM and not cuda:
        import cloud; cloud.train(R, init); sys.exit(0)
    run = L / 'runs' / f'r{R}'; run.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(DATA / 'positions.npz')); ok = usable(d)
    parts, va = [arrays(d, ok & ~d['val'])], on_dev(arrays(d, ok & d['val']))
    for r in range(1, R + 1):
        e = dict(np.load(DATA / 'dagger' / f'round-{r}.npz')); parts += [arrays(e, usable(e))] * 4
        if r >= 7: parts += [arrays(e, usable(e) & (e['board'].max(1) >= 12))] * 2
    tr = on_dev({k: np.concatenate([a[k] for a in parts]) for k in parts[0]}); del parts
    h = dict(np.load(DATA / 'human.npz')); hok = usable(h)
    hum = {s: (on_dev(arrays(h, hok & m)), torch.as_tensor(h['human'][hok & m].astype(np.int64), device=dev)) for s, m in (('v47', h['seed'] == 47), ('v48', h['seed'] != 47))}
    n = len(tr['board']); per = n // BATCH; steps = int(os.environ.get('V55_STEPS') or EPOCHS * per); quarter = per // 4   # V55_STEPS: smoke test
    net = Shared().to(dev); net.load_state_dict(torch.load(init, map_location=dev))
    print(f'round {R}: init {init}, train {n}, steps {steps} on {dev}', flush=True)
    if cuda: torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if cuda: torch.backends.mha.set_fastpath_enabled(False)   # CUDA's eval-mode fast path ignores the attention bias (validation near 57% instead of 81%)
    f = torch.compile(net) if cuda else net
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    step = 0; ck = run / 'ckpt.pt'; g = torch.Generator(device=dev)
    if ck.exists():
        s = torch.load(ck, map_location=dev); net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); step = s['step']
        print('resumed at step', step, flush=True)
    def save(): torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': state['step']}, ck)
    state = {'step': step}; signal.signal(signal.SIGTERM, lambda *a: (save(), sys.exit(75)))
    log = open(run / 'log.jsonl', 'a'); t0 = time.time(); perm_of = {}
    while step < steps:
        epoch = step // per
        if epoch not in perm_of: g.manual_seed(55 * 100 + R * 10 + epoch); perm_of.clear(); perm_of[epoch] = torch.randperm(n, device=dev, generator=g)   # same order after a resume
        i = step % per; b = augment(batch(tr, perm_of[epoch][i * BATCH:(i + 1) * BATCH]))
        with torch.autocast('cuda', torch.bfloat16, enabled=cuda): loss, ce, reg = loss_fn(f, b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1; state['step'] = step
        if step % 50 == 0: loss.item()                       # wait for the GPU now and then (MPS queues work without limit)
        if step % quarter == 0 or step % per == 0 or step == steps:     # per is not always a multiple of quarter
            save()
            if step % per == 0 or step == steps:
                rec = {'epoch': step / per, 'sec': round(time.time() - t0), 'val': evaluate(net, va), **{s: evaluate(net, b, hh) for s, (b, hh) in hum.items()}}
                log.write(json.dumps(rec) + '\n'); log.flush()
                print(json.dumps({'epoch': rec['epoch'], 'sec': rec['sec'], **{k: {m: round(v, 4) for m, v in rec[k].items()} for k in ('val', 'v47', 'v48')}}), flush=True)
    torch.save(net.state_dict(), run / 'model.pt')
