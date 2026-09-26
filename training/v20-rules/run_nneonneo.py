"""Run nneonneo/threes-ai (commit c554820, unmodified, own RNG seeded from /dev/urandom) N games.
Records final score, highest rank (14 = 6144), move count and wall time per game."""
import subprocess, sys, time, json, re, concurrent.futures, os
N = int(sys.argv[1]); out = sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
def one(i):
    t = time.time()
    p = subprocess.run([os.environ.get('THREES_REFERENCE_BIN', 'threes')], capture_output=True, text=True, check=True)
    m = re.search(r'Game over\. Your score is (\d+)\. The highest rank you achieved was (\d+)\.', p.stdout)
    r = {'game': i, 'score': int(m.group(1)), 'rank': int(m.group(2)), 'moves': p.stdout.count('Move #'), 'wallSec': time.time() - t}
    with open(out, 'a') as f: f.write(json.dumps(r) + '\n')
    print('done', i, r['rank'], flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex: list(ex.map(one, range(N)))
