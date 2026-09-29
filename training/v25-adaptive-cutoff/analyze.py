"""Analyze only completed frozen paired test; no interim outcome inspection."""
from pathlib import Path
import json,math,statistics as st,hashlib
ROOT=Path(__file__).resolve().parent

def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n
 center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [center-half,center+half]

def paired_ci(d):
 mean=st.mean(d);se=st.stdev(d)/math.sqrt(len(d));return [mean-1.959963984540054*se,mean+1.959963984540054*se]

def mcnemar(b,c):
 n=b+c
 return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.

def main():
 protocol=json.loads((ROOT/'protocol.json').read_text())
 assert json.loads((ROOT/'progress.json').read_text())['status']=='complete'
 rows={arm:[json.loads(s) for s in (ROOT/f'games-{arm}.jsonl').read_text().splitlines()] for arm in ('baseline','adaptive')}
 expected=list(range(protocol['seeds'][0],protocol['seeds'][1]+1));n=len(expected)
 for arm,r in rows.items():
  assert [x['seed'] for x in r]==expected
  assert all(sum(x['depthHist'])==x['moves'] and sum(x['depthHist'][:5])==0 for x in r)
 b,a=rows['baseline'],rows['adaptive']
 assert all(x['r1536']==y['r1536'] and x['r3072']==y['r3072'] for x,y in zip(b,a))
 metrics={}
 for milestone in (1536,3072,6144,12288):
  key=f'r{milestone}';bv=[int(r[key]) for r in b];av=[int(r[key]) for r in a]
  only_a=sum(y and not x for x,y in zip(bv,av));only_b=sum(x and not y for x,y in zip(bv,av))
  metrics[key]={'baseline':{'count':sum(bv),'rate':st.mean(bv),'wilson95':wilson(sum(bv),n)},'adaptive':{'count':sum(av),'rate':st.mean(av),'wilson95':wilson(sum(av),n)},'difference':st.mean(av)-st.mean(bv),'pairedDifferenceApprox95':paired_ci([y-x for x,y in zip(bv,av)]),'adaptiveOnly':only_a,'baselineOnly':only_b,'mcnemarExactP':mcnemar(only_a,only_b)}
 order=sorted(('r6144','r12288'),key=lambda k:metrics[k]['mcnemarExactP']);prev=0
 for i,k in enumerate(order):
  prev=max(prev,min(1.,(2-i)*metrics[k]['mcnemarExactP']));metrics[k]['holmP']=prev
 secondary={}
 for key in ('score','moves','searchCPU','maxMoveCPU'):
  bv=[r[key] for r in b];av=[r[key] for r in a];d=[y-x for x,y in zip(bv,av)]
  secondary[key]={'baselineMean':st.mean(bv),'adaptiveMean':st.mean(av),'baselineMedian':st.median(bv),'adaptiveMedian':st.median(av),'pairedDifferenceMean':st.mean(d),'pairedDifferenceApprox95':paired_ci(d)}
 browser=json.loads((ROOT/'browser-results.json').read_text())
 native=[json.loads(s) for s in (ROOT/'browser-reference.jsonl').read_text().splitlines()]
 native={(r['index'],r['arm']):r for r in native}
 mapping={'left':0,'right':1,'up':2,'down':3}
 mismatch=[]
 for r in browser['parity']:
  ref=native[(r['index'],r['arm'])];direction=mapping.get(r['direction'],r['direction'])
  if direction!=ref['direction'] or r['depth']!=ref['depth']:mismatch.append({'js':r,'native':ref})
 assert len(browser['parity'])==len(native)==640
 performance_pass=all(r['completed5']/r['n']>=.99 and r['p95Ms']<160 for r in browser['summary'])
 significant_gain=any(metrics[k]['holmP']<.05 and metrics[k]['difference']>0 for k in ('r6144','r12288'))
 no_decline=all(metrics[k]['difference']>=0 for k in ('r6144','r12288'))
 result={'protocol':protocol,'nPairs':n,'milestones':metrics,'secondary':secondary,'browserSummary':browser['summary'],'browserUserAgent':browser['userAgent'],'parityComparisons':len(native),'parityMismatches':mismatch,'adoptionGate':{'significantPrimaryGain':significant_gain,'noObservedPrimaryDecline':no_decline,'browserPerformancePass':performance_pass,'parityPass':not mismatch,'recommend':significant_gain and no_decline and performance_pass and not mismatch},'progress':json.loads((ROOT/'progress.json').read_text()),'fileSHA256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('games-baseline.jsonl','games-adaptive.jsonl','browser-results.json','browser-reference.jsonl')}}
 (ROOT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 pct=lambda v:f'{v*100:.2f}%'
 lines=['# 3072 后降低剪枝阈值：配对评测','',f'完成 {n} 组配对、{2*n} 局新开局评测，8 进程。模型固定为 V23；搜索深度 5，叶节点估值下限 0。基线全程 0.01；候选在真实棋盘最大牌达到 3072 后采用 0.005（6144 阶段继续 0.005）。','',f"原生评测墙钟耗时：{result['progress']['elapsedSeconds']/60:.1f} 分钟。",'','## 达成率','','|最大牌|基线|候选|候选减基线|配对差值约 95% CI|Holm 校正 p|','|---|---:|---:|---:|---:|---:|']
 for rank in (1536,3072,6144,12288):
  r=metrics[f'r{rank}'];ci=r['pairedDifferenceApprox95'];hp=f"{r['holmP']:.4f}" if 'holmP'in r else '—'
  lines.append(f"|{rank}|{r['baseline']['count']}/{n} ({pct(r['baseline']['rate'])})|{r['adaptive']['count']}/{n} ({pct(r['adaptive']['rate'])})|{r['difference']*100:+.2f} 个百分点|[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] 个百分点|{hp}|")
 lines+=['','差值置信区间为配对样本正态近似；显著性结论用预先指定的精确 McNemar 检验，对两个主要终点（6144、12288）做 Holm 校正。各组 Wilson 区间、配对不一致数和原始 p 值见 results.json。1536、3072 为一致性检查，不是额外主要终点。','同一随机种子保证相同开局；策略分歧后后续随机事件不是一一匹配。固定样本量，中途没有查看胜率后择机停止。','','## 次要指标','','|指标|基线均值|候选均值|','|---|---:|---:|']
 for key,label in [('score','最终分数'),('moves','步数'),('searchCPU','每局搜索 CPU 秒')]:
  r=secondary[key];lines.append(f"|{label}|{r['baselineMean']:.3f}|{r['adaptiveMean']:.3f}|")
 lines+=['','分数、步数只做描述；CPU 时间是在 8 进程并发下测得，不能当作网页每步延迟。','','## 浏览器测试','','原生测试结束后运行，320 个固定历史局面，每策略每局面 3 次；预算 160ms、最多 400 万节点。局面选择和统计口径在结果分析前固定。1536/3072 局面来自 V18 历史池，6144 来自 V21 历史池；预览概率已统一为现代规则的窗口内均匀分布。历史池不是新开局胜率样本。','','|阶段|策略|次数|平均 ms|P95 ms|最大 ms|完成深度 5|','|---|---|---:|---:|---:|---:|---:|']
 for r in browser['summary']:lines.append(f"|{r['phase']}|{r['arm']}|{r['n']}|{r['meanMs']:.2f}|{r['p95Ms']:.2f}|{r['maxMs']:.2f}|{r['completed5']}/{r['n']}|")
 lines += ['',f"浏览器与原生的无时间限制决策一致性：{len(native)-len(mismatch)}/{len(native)}。",'',f"浏览器：{browser['userAgent']}",'','本机 Apple M2 Pro，8 性能核 + 4 能效核，16GB 内存。保留了用户后台应用。浏览器结论仅适用于此桌面环境，不代表手机性能。160ms 是软预算，不能保证每一步绝不超时。','','## 结论','']
 if result['adoptionGate']['recommend']:lines.append('候选通过预先指定的采用门槛：至少一个主要终点显著提高，另一个观察值未下降，且浏览器各阶段深度完成率 ≥99%、P95 <160ms，并通过原生/JS 决策一致性核对。建议采用，但本次未修改线上版本。')
 else:lines.append('候选未通过全部预先指定的采用门槛，建议保留线上 0.01 基线。未显著不等于两者等效；尤其 12288 事件较少，检测小幅改善的能力有限。')
 lines.extend([f"- 显著主要终点改善：{significant_gain}",f"- 两个主要终点观察值均未下降：{no_decline}",f"- 浏览器性能门槛：{performance_pass}",f"- 原生/JS 决策一致性：{not mismatch}",'','## 复现与文件','','- protocol.json：开跑前固定的参数、种子及模型/源码哈希。','- source/：评测源码及其本地依赖；model.bin：实际使用的固定模型。','- games-baseline.jsonl、games-adaptive.jsonl：全部原始对局。','- browser/：固定局面及浏览器基准页面。','- browser-protocol.json、browser-results.json、browser-reference.jsonl：浏览器协议与原始数据。','- analyze.py：统计代码；所有数据齐备后运行 `python3 analyze.py`。','- run.py：8 进程断点续跑脚本；在原环境核对二进制哈希后可重跑。','','跨平台可编译 `c++ -O3 -std=c++17 source/training/v24-acceptance/game.cpp -o game`，用 `game model.bin 5 0.01 0 1 FIRST_SEED COUNT OUT` 跑基线；候选将阈值参数换为 `0.005`。本评测中的源代码仅在真实根棋盘最大牌达到 3072 后才采用该参数；此前两组均为 0.01。其他平台二进制哈希会不同，应另存协议、记录编译器与新哈希，不要篡改本次冻结协议。',''])
 (ROOT/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({'milestones':metrics,'adoptionGate':result['adoptionGate'],'browserSummary':browser['summary']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
