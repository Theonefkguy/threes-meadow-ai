import csv
import json
from pathlib import Path
import statistics
import sys

def report(out):
    cfg = json.loads((out/'config.json').read_text())
    rows=[]
    all_results={4:[],5:[]}
    for i in range(cfg['cases']):
        groups={4:[],5:[]}
        for r in range(cfg['repeats']):
            tape=[int(line.split()[0]) for line in (out/f'tape-{i}-{r}.txt').read_text().splitlines()]
            for d in [4,5]:
                x=json.loads((out/f'result-{i}-{r}-d{d}.json').read_text())
                assert (x['id'],x['rep'],x['depth'])==(i,r,d)
                replay=[json.loads(line) for line in (out/f'replay-{i}-{r}-d{d}.jsonl').read_text().splitlines()]
                assert len(replay)==x['steps']
                assert [step['spawnRank'] for step in replay]==tape[:x['steps']], 'Different cards: invalid comparison'
                groups[d].append(x)
                all_results[d].append(x)
            assert groups[4][-1]['seed']==groups[5][-1]['seed']
        row={'case':i}
        for d in [4,5]:
            a=groups[d]
            row[f'd{d}_steps']=round(statistics.mean(x['steps'] for x in a),2)
            row[f'd{d}_max_tiles']='/'.join(str(x['maxTile']) for x in a)
            row[f'd{d}_escape']=sum(x['escaped'] for x in a)/len(a)
        rows.append(row)
    with (out/'summary.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lines=['# 关键局面对照测试', '', '固定牌值、共享落点随机数；位置随合法入口映射，不保证绝对坐标一致。', '',
           '| 起点 | 四层平均存活步数 | 五层平均存活步数 | 四层最大牌（逐次） | 五层最大牌（逐次） | 四层脱困率 | 五层脱困率 |',
           '|---|---:|---:|---|---|---:|---:|']
    for x in rows:
        lines.append(f"| {x['case']} | {x['d4_steps']} | {x['d5_steps']} | {x['d4_max_tiles']} | {x['d5_max_tiles']} | {x['d4_escape']:.1%} | {x['d5_escape']:.1%} |")
    lines+=['', f"脱困：续玩 {cfg['horizon']} 步后仍未结束；步数为截断存活步数，不是完整寿命。", '']
    for d,a in all_results.items():
        lines.append(f"{d} 层：脱困率 {statistics.mean(x['escaped'] for x in a):.1%}；平均存活 {statistics.mean(x['steps'] for x in a):.2f} 步；搜索 CPU {sum(x['cpuSeconds'] for x in a):.2f} 秒。")
    lines+=['', '这是四层失败轨迹上的定向诊断，不能外推整体胜率；没有证明每个截取点必然可救。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    report(Path(sys.argv[1]))
