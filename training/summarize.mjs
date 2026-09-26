import fs from 'node:fs';
const input=process.argv[2]||'training/evaluation-v1.json';
const all=JSON.parse(fs.readFileSync(input));
const result={};
for(const name of [...new Set(all.map(r=>r.name))]){
 const games=all.filter(r=>r.name===name),sorted=games.map(r=>r.score).sort((a,b)=>a-b),n=games.length;
 result[name]={games:n,complete:games.filter(r=>r.terminal).length,mean:games.reduce((s,r)=>s+r.score,0)/n,
 median:(sorted[Math.floor((n-1)/2)]+sorted[Math.floor(n/2)])/2,
 reached1536:games.filter(r=>r.highest>=1536).length,reached3072:games.filter(r=>r.highest>=3072).length,reached6144:games.filter(r=>r.highest>=6144).length,
 meanMs:games.reduce((s,r)=>s+r.meanMs*r.turns,0)/games.reduce((s,r)=>s+r.turns,0)};
}
console.log(JSON.stringify(result,null,2));
