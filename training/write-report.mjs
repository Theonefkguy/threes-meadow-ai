import fs from 'node:fs';
const meta=JSON.parse(fs.readFileSync('dist/models/ntuple-v1.json'));
const rows=JSON.parse(fs.readFileSync('training/evaluation-v1.json'));
const names={classic:'原 AI：经典搜索',single:'单阶段学习＋搜索',staged:'三阶段学习＋搜索'};
const fmt=n=>new Intl.NumberFormat('zh-CN',{maximumFractionDigits:1}).format(n);
const grouped=Object.keys(names).map(name=>{
 const r=rows.filter(x=>x.name===name),n=r.length,sorted=r.map(x=>x.score).sort((a,b)=>a-b);
 return {name,n,mean:r.reduce((a,b)=>a+b.score,0)/n,median:(sorted[Math.floor((n-1)/2)]+sorted[Math.floor(n/2)])/2,
 reached:[1536,3072,6144].map(t=>r.filter(x=>x.highest>=t).length),ms:r.reduce((a,b)=>a+b.meanMs*b.turns,0)/r.reduce((a,b)=>a+b.turns,0),terminal:r.filter(x=>x.terminal).length};
});
let report=`# 强化学习第一版：训练与测试\n\n## 交付模型\n\n- 方法：8 种四格 N-tuple 图案、8 种对称变换，三个阶段，afterstate TD(0)。\n- 预训练：正常开局 ${fmt(meta.pretraining.episodes)} 回合。\n- 后续两步策略训练：正常开局、中期、后期分别 ${meta.training.episodes.map(fmt).join('、')} 回合。\n- 最终模型使用的训练量：${fmt(meta.totalEpisodesUsed)} 回合，${fmt(meta.totalStepsUsed)} 次有效移动。中后期回合从真实局面续玩，不是这么多局全部从头玩。\n- 课程池：${meta.training.pool1536} 个可继续玩的 1536 局面、${meta.training.pool3072} 个可继续玩的 3072 局面；容量上限各 5,000。\n- 模型大小：${fmt(meta.bytes)} 字节（约 6 MiB）；SHA-256：\`${meta.sha256}\`。\n- 来牌规则保持不变。评分表只看棋盘，搜索看公开提示和公开历史推算的余牌数，不看暗牌顺序。\n\n## 完整对局测试\n\n所有测试从九牌开局开始，种子 810001–810012，与训练种子分开。每步上限均为 160 ms、24,000 节点和三步搜索，仅采用完整搜索深度。双方实际节点数和时间不相同。相同种子保证相同开局；行动分岔后随机来牌未必逐张相同。基准分组先运行，两个学习组随后有并行时段，每组内部逐局运行；采用相同预算上限，实际耗时不是严格隔离的性能基准，且非 iPhone 实测。\n\n| 策略 | 完整对局 | 平均分 | 中位数 | ≥1536 | ≥3072 | ≥6144 | 每步平均时间 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n`;
for(const g of grouped)report+=`| ${names[g.name]} | ${g.terminal}/${g.n} | ${fmt(g.mean)} | ${fmt(g.median)} | ${g.reached[0]}/${g.n} | ${g.reached[1]}/${g.n} | ${g.reached[2]}/${g.n} | ${fmt(g.ms)} ms |\n`;
report+='\n单阶段比较使用最终模型的前期参数处理所有阶段；三阶段比较按最大牌 1536、3072 切换参数。因此它是对“是否切换阶段模型”的消融，不是两套等训练成本的独立模型。前期参数同样经历过从正常开局玩到后期的训练。\n\n';
const base=grouped[0],rl=grouped[2];report+=`三阶段模型的平均分相对原 AI 提高约 ${fmt((rl.mean/base.mean-1)*100)}%。这仅是 ${rl.n} 个开局上的点估计，不足以保证长期优势，也不能等同于论文报告的成绩。\n\n`;
report+='## 文件与复现\n\n- 方法、命令及限制：[RL.md](RL.md)\n- 每局原始测试数据：[training/evaluation-v1.json](training/evaluation-v1.json)\n- 预训练摘要：[training/pretraining.json](training/pretraining.json)\n- 分阶段训练日志：[training/run-v1-refined/progress.jsonl](training/run-v1-refined/progress.jsonl)\n- 模型元数据：[dist/models/ntuple-v1.json](dist/models/ntuple-v1.json)\n\n训练器与网页引擎通过了随机棋盘移动/计分、固定种子轨迹、奖励牌分布对照。导出的模型通过了 C++ 与 JS 预测、对称性对照。自动玩操作通过事件测试与真实 Worker 入口测试；当前环境没有兼容的浏览器预览，未做 iPhone 实机验证。\n';
fs.writeFileSync('RL-RESULTS.md',report);console.log(report);
