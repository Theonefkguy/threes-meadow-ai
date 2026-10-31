"""V53 steps 1-2: which confidence signal best picks the moves worth handing to the search?
All 176k validation positions (100 AI games, V46 odd seeds). Target: regret of the model's choice
(relative gap to the AI's best) and bad5 (regret >= 5%). Scores (larger = less sure):
  each signal of conf.SIG alone; logistic regression on all signals; the Risk network on
  [final summary token, sorted P, signals] trained (a) on bad5 (BCE) or (b) on min(regret, 0.2) (MSE);
  for comparison (a) on the summary token alone and without it.
Learned scores: 5-fold cross-validation by game. Metrics: AUC for bad5 (all, and within 3072+), share of
bad5 and of total regret in the least sure 10 / 20 / 30%, plus the oracle (sorted by true regret).
The score with the most regret at 20% is refit on all positions -> risk.pt (for play.py).  -> signals.json
  SMOKE=1 for a 20000-position test."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L))
from conf import signals, SIG, SIGN, Risk, inputs, dev
from common import load_net, sample, to_dev, gaps

def auc(s, y):
    o = np.argsort(s, kind='stable'); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def metrics(s, bad, c, late):
    s = s + np.random.default_rng(0).uniform(0, 1e-7, len(s)) * (np.abs(s).max() + 1)    # random tie-breaking
    o = np.argsort(-s); out = {'auc': auc(s, bad), 'aucLate': auc(s[late], bad[late])}
    for q in (0.1, 0.2, 0.3):
        f = o[:int(q * len(s))]; out[f'bad5@{q:.0%}'] = float(bad[f].sum() / bad.sum()); out[f'regret@{q:.0%}'] = float(c[f].sum() / c.sum())
    return out

def fit_logreg(Z, y, steps=500):
    w = torch.zeros(Z.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    Z = torch.as_tensor(Z, device=dev); y = torch.as_tensor(y, device=dev, dtype=torch.float32); opt = torch.optim.Adam([w, b], 0.05)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y) + 1e-4 * (w ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    w, b = w.detach().cpu().numpy(), b.detach().cpu().numpy()
    return (w, b), (lambda Zt: Zt @ w + b)

def fit_risk(Z, y, kind, epochs=20, seed=0):
    torch.manual_seed(seed); net = Risk(Z.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    Z = torch.as_tensor(Z, device=dev); y = torch.as_tensor(y, device=dev, dtype=torch.float32); n = len(y)
    for _ in range(epochs):
        for j in torch.randperm(n, device=dev).split(1024):
            out = net(Z[j])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y[j]) if kind == 'bad5' else ((out - y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net, (lambda Zt: net(torch.as_tensor(Zt, device=dev)).detach().cpu().numpy())

if __name__ == '__main__':
    t0 = time.time(); net = load_net(); a = sample(20000 if os.environ.get('SMOKE') else 10 ** 7, seed=53); n = len(a['board'])
    S, H, P = [], [], []
    for i in range(0, n, 4096):
        s, h, p = signals(net, to_dev(a, slice(i, i + 4096))); S.append(s); H.append(h); P.append(p)
    S, H, P = np.concatenate(S), np.concatenate(H), np.concatenate(P); print(f'{n} positions, signals {time.time() - t0:.0f} s', flush=True)
    g = gaps(a['values']); c = g[np.arange(n), P.argmax(1)]; assert np.isfinite(c).all(); c = np.maximum(c, 0); bad = c >= 0.05; late = a['phase'] >= 13
    out = {'positions': n, 'bad5': float(bad.mean()), 'late': float(late.mean()), 'bad5Late': float(bad[late].mean()), 'scores': {}}
    out['scores']['oracle (true regret)'] = metrics(c, bad, c, late)
    for k, name in enumerate(SIG): out['scores'][name] = metrics(SIGN[name] * S[:, k], bad, c, late)
    # cross-validated learned scores
    games = np.unique(a['seed']); np.random.default_rng(53).shuffle(games)
    fold = np.array([{g: i % 5 for i, g in enumerate(games)}[s] for s in a['seed']])
    cv = {k: np.zeros(n) for k in ('logistic (signals)', 'network, bad5', 'network, regret', 'network, bad5, summary token only', 'network, bad5, no summary token')}
    for f in range(5):
        tr, te = fold != f, fold == f; mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-6
        cv['logistic (signals)'][te] = fit_logreg(((S[tr] - mu) / sd).astype(np.float32), bad[tr])[1](((S[te] - mu) / sd).astype(np.float32))
        Ztr, Zte = inputs(S[tr], H[tr], P[tr], mu, sd), inputs(S[te], H[te], P[te], mu, sd)
        cv['network, bad5'][te] = fit_risk(Ztr, bad[tr], 'bad5')[1](Zte)
        cv['network, regret'][te] = fit_risk(Ztr, np.minimum(c[tr], 0.2), 'regret')[1](Zte)
        cv['network, bad5, summary token only'][te] = fit_risk(Ztr[:, :128], bad[tr], 'bad5')[1](Zte[:, :128])      # what the model 'knows'
        cv['network, bad5, no summary token'][te] = fit_risk(Ztr[:, 128:], bad[tr], 'bad5')[1](Zte[:, 128:])
        print('fold', f, f'{time.time() - t0:.0f} s', flush=True)
    for k, v in cv.items(): out['scores'][k] = metrics(v, bad, c, late)
    best = max((k for k in out['scores'] if not k.startswith('oracle') and 'summary token' not in k), key=lambda k: out['scores'][k]['regret@20%']); out['best'] = best
    # refit the best on everything; thresholds = validation quantiles of its score
    mu, sd = S.mean(0), S.std(0) + 1e-6; save = {'best': best, 'mu': mu, 'sd': sd}
    if best in SIG: score = SIGN[best] * S[:, SIG.index(best)]
    elif best.startswith('network'):
        rnet, fn = fit_risk(inputs(S, H, P, mu, sd), bad if best.endswith('bad5') else np.minimum(c, 0.2), best.split(', ')[1]); score = fn(inputs(S, H, P, mu, sd)); save['state'] = rnet.state_dict()
    else:
        Z = ((S - mu) / sd).astype(np.float32); save['logreg'], fn = fit_logreg(Z, bad); score = fn(Z)
    save['thresholds'] = {q: float(np.quantile(score, 1 - q)) for q in (0.1, 0.2, 0.3)}; out['thresholds'] = save['thresholds']
    torch.save(save, L / 'risk.pt')
    (L / ('signals-smoke.json' if os.environ.get('SMOKE') else 'signals.json')).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for k, v in out['scores'].items(): print(f'{k:22s}', ' '.join(f'{m} {x:.3f}' for m, x in v.items()))
    print('best', best, f'{time.time() - t0:.0f} s')
