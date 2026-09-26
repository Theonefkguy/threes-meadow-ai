"""Build V21 pools under official-modern rules (run from repo root). Resumable by stage.
A eval-3072: openings 21110001+ (1600 games) -> first 1024 first-3072 snapshots by seed.
B eval-6144: openings 21310001+ (2400 games) -> every first-3072 extended with 4-ply/0.003
  (rollout seeds 21320001+index) to its first 6144 -> first 512 by source seed.
C train: openings 21510001+ (1200 games) -> first-1536 and first-3072 pools; the 3072s are
  extended the same way (rollout 21520001+index) -> train-6144 pool. No filtering anywhere."""
from pathlib import Path
import subprocess, concurrent.futures as cf
R = Path('training/v21-late-train/pools'); raw = R/'raw'; raw.mkdir(parents=True, exist_ok=True); X = '/tmp/v21-snap'
def sh(args): subprocess.run([str(a) for a in args], check=True)
def pools(tag, first, games, parts=4):
    if (R/f'{tag}-3072.txt').exists(): return
    per = games//parts
    with cf.ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda i: sh([X,'pool',first+i*per,per,raw/f'{tag}-1536.{i}',raw/f'{tag}-3072.{i}']), range(parts)))
    for m in ['1536','3072']:
        lines = sorted((l for i in range(parts) for l in (raw/f'{tag}-{m}.{i}').read_text().splitlines()), key=lambda l: int(l.split()[0]))
        (R/f'{tag}-{m}.txt').write_text(''.join(l+'\n' for l in lines))
def extend(tag, rollout, chunk=100):
    src = R/f'{tag}-3072.txt'; out = R/f'{tag}-6144.txt'
    if out.exists(): return
    n = len(src.read_text().splitlines())
    jobs = [(i, raw/f'{tag}-6144.{i:05d}') for i in range(0, n, chunk)]
    def run(j):
        i, f = j; tmp = f.parent/(f.name+'.tmp'); 
        if f.exists(): return
        tmp.unlink(missing_ok=True); tmp.touch()
        sh([X,'extend','training/v7/base.ntd',4,0.003,src,i,chunk,rollout,14,tmp]); tmp.rename(f)
    with cf.ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
    lines = sorted((l for _, f in jobs for l in f.read_text().splitlines()), key=lambda l: int(l.split()[0]))
    out.write_text(''.join(l+'\n' for l in lines))
pools('evalA', 21110001, 1600); pools('evalB', 21310001, 2400); pools('train', 21510001, 1200)
extend('evalB', 21320001); extend('train', 21520001)
def head(src, dst, k):
    lines = (R/src).read_text().splitlines(); assert len(lines) >= k, (src, len(lines)); (R/dst).write_text(''.join(l+'\n' for l in lines[:k]))
head('evalA-3072.txt', 'EVAL3072.txt', 1024); head('evalB-6144.txt', 'EVAL6144.txt', 512)
for f in ['train-1536.txt','train-3072.txt','train-6144.txt']: (R/f.replace('train','TRAIN')).write_text((R/f).read_text())
for f in sorted(R.glob('*.txt')): print(f.name, len(f.read_text().splitlines()), flush=True)
