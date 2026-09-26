import fs from 'node:fs';
const pilot=JSON.parse(fs.readFileSync('benchmarks/pilot-results.json'));
const final=JSON.parse(fs.readFileSync('benchmarks/validation-results.json'));
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const median=a=>{a=[...a].sort((a,b)=>a-b);return(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
const names={baseline:'原 AI',fast:'仅加速搜索',memory:'加速＋记牌',space:'记牌＋空位与解困',chain:'记牌＋合并路线',deep:'记牌＋空位与解困＋更深搜索'};
function table(rows){
 let s='| 方案 | 局数 | 平均分 | 中位数 | 达到 1536 | 达到 3072 | 每步平均计算时间 |\n|---|---:|---:|---:|---:|---:|---:|\n';
 for(const n of [...new Set(rows.map(r=>r.name))]){
  const rs=rows.filter(r=>r.name===n),scores=rs.map(r=>r.score),turns=rs.reduce((s,r)=>s+r.turns,0);
  s+=`| ${names[n]} | ${rs.length} | ${Math.round(mean(scores)).toLocaleString('en-US')} | ${median(scores).toLocaleString('en-US')} | ${rs.filter(r=>r.highest>=1536).length}/${rs.length} | ${rs.filter(r=>r.highest>=3072).length}/${rs.length} | ${(rs.reduce((s,r)=>s+r.meanMs*r.turns,0)/turns).toFixed(1)} ms |\n`;
 }
 return s;
}
let report=`# 合三 AI 策略对比\n\n## 实验条件\n\n- 游戏规则：当前网页的等概率奖励牌版本，所有 AI 使用同一引擎。\n- 原 AI：保留修改前源代码于 benchmarks/baseline.js，仅调整导入路径。\n- 初筛：六种方案，种子 101–106，每步最多 6,000 搜索节点、160 ms。\n- 复测：新种子 1001–1012，每步最多 24,000 搜索节点、160 ms；原 AI 与路线版最多三步，深搜版最多四步。\n- 每局最多 2,500 步；记录 terminal 字段以区分完整对局与截断。\n- 按种子交错运行各方案，不并行运行比赛；同一种子保证相同开局，但不同决策后的随机来牌序列可能分岔，并非逐步相同来牌。\n- 保留每局原始数据、节点数、搜索深度、耗时。耗时来自本次服务器运行，不能视为 iPhone 实测。\n- 加速版本缓存行移动、使用扁平棋盘；记牌只从公开开局和预告计算剩余牌数，不读取暗牌顺序。\n- 以初筛成绩的几何平均数挑出深搜版与路线版参加复测，避免只选单局最高分。\n\n## 初筛\n\n${table(pilot)}\n## 新种子复测\n\n${table(final)}\n`;
const base=final.filter(r=>r.name==='baseline');
for(const name of ['chain','deep']){
 const rows=final.filter(r=>r.name===name),pairs=rows.map(r=>[r,base.find(b=>b.seed===r.seed)]).filter(p=>p[1]);
 const ratios=pairs.map(([a,b])=>Math.log(a.score/b.score));
 let seed=2026;const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
 const boot=Array.from({length:10000},()=>mean(Array.from({length:ratios.length},()=>ratios[Math.floor(rand()*ratios.length)]))).sort((a,b)=>a-b);
 report+=`\n${names[name]}：${pairs.filter(([a,b])=>a.score>b.score).length}/${pairs.length} 个种子得分超过原 AI。配对得分比的几何平均为 ${Math.exp(mean(ratios)).toFixed(2)} 倍；探索性 95% bootstrap 区间为 ${Math.exp(boot[250]).toFixed(2)}–${Math.exp(boot[9750]).toFixed(2)} 倍。\n`;
}
report+='\n## 采用方案\n\n网页采用「记牌＋合并路线」版，保持原来的三步、24,000 节点、160 ms 限制。未采用复测较弱的深搜版。\n\n## 限制\n\n样本量小，候选方案也在比较后选择；结果是本次实验的观察，不是长期提升保证或原版游戏成绩。未进行 iPhone 实机验证。没有训练神经网络，也没有更改游戏抽牌概率来提高成绩。\n\n## 复现\n\n运行 `node benchmarks/compare.mjs pilot` 与 `node benchmarks/compare.mjs validation baseline,chain,deep 12`。脚本默认从已有结果续跑；追加 `--fresh` 可重新运行并覆盖对应结果文件。时间预算会受机器性能影响，分数不保证跨设备完全相同。\n';
fs.writeFileSync('BENCHMARKS.md',report);
console.log(table(final));
