"""Explicit commands only; no scheduler or detached processes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['collect', 'test', 'report', 'all'])
    ap.add_argument('--out', type=Path, default=HERE / 'runs/default')
    ap.add_argument('--cases', type=int, default=18)
    ap.add_argument('--back', type=int, choices=range(20, 31), default=25)
    ap.add_argument('--horizon', type=int, default=75)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=736210)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    assert min(args.cases, args.horizon, args.repeats, args.workers) > 0
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ['cases', 'back', 'horizon', 'repeats', 'seed']}
    config['modelSHA256'] = hashlib.sha256((ROOT/'training/v7/base.ntd').read_bytes()).hexdigest()
    config['sourceSHA256'] = hashlib.sha256((HERE/'experiment.cpp').read_bytes()).hexdigest()
    manifest = out/'config.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise SystemExit('Configuration changed: choose a new --out directory.')
    else:
        manifest.write_text(json.dumps(config, indent=2)+'\n')
    binary = out/'experiment'
    if args.command != 'report':
        subprocess.run(['g++', '-O3', '-std=c++17', str(HERE/'experiment.cpp'), '-o', str(binary)], cwd=ROOT, check=True)
    def collect(i):
        if (out/f'source-{i}.json').exists() and (out/f'snapshot-{i}.txt').exists():
            return
        subprocess.run([str(binary), 'collect', str(out), str(i), str(args.seed+i), str(args.back), '10000'], cwd=ROOT, check=True)
        print(f'collected {i+1}/{args.cases}', flush=True)
    def test(job):
        i, r = job
        if all((out/f'result-{i}-{r}-d{d}.json').exists() for d in [4, 5]):
            return
        subprocess.run([str(binary), 'test', str(out), str(i), str(r), str(args.seed+100000+i*args.repeats+r), str(args.horizon)], cwd=ROOT, check=True)
        print(f'tested case={i} repeat={r}', flush=True)
    if args.command in ['collect', 'all']:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(collect, range(args.cases)))
    if args.command in ['test', 'all']:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(test, [(i,r) for i in range(args.cases) for r in range(args.repeats)]))
    if args.command in ['report', 'test', 'all']:
        subprocess.run([sys.executable, str(HERE/'report.py'), str(out)], check=True)

if __name__ == '__main__':
    main()
