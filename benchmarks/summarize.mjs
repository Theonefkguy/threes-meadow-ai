import fs from 'node:fs';
const rows=JSON.parse(fs.readFileSync(process.argv[2]));
const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
const median=a=>{a=[...a].sort((a,b)=>a-b);return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
for(const name of [...new Set(rows.map(r=>r.name))]){
 const group=rows.filter(r=>r.name===name),scores=group.map(r=>r.score);
 console.log(JSON.stringify({name,n:group.length,mean:Math.round(avg(scores)),median:median(scores),geomean:Math.round(Math.exp(avg(scores.map(Math.log)))),reach768:group.filter(r=>r.highest>=768).length,reach1536:group.filter(r=>r.highest>=1536).length,reach3072:group.filter(r=>r.highest>=3072).length,ms:avg(group.map(r=>r.meanMs)).toFixed(1),depth:avg(group.map(r=>r.meanDepth)).toFixed(2),censored:group.filter(r=>!r.terminal).length}));
}
