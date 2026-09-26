// node training/v17-fast-depth/js-parity.mjs STATES CPP_JSONL DEPTH THRESHOLD [OUT]
// Runs dist/fast-search.js (no time/node limit) on the C++ sampled states and
// compares chosen direction and root values with the native v17 search output.
import {readFileSync,writeFileSync} from 'node:fs';
import {decodeModel} from '../../dist/rl-model.js';
import {chooseFastMove} from '../../dist/fast-search.js';
const [statesPath,cppPath,depthArg,thrArg,outPath]=process.argv.slice(2);
const depth=+depthArg,threshold=+thrArg;
const buf=readFileSync(process.env.MODEL||'training/v7/base.ntd');const model=decodeModel(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));
const fromRank=r=>r<3?r:3*2**(r-3);
const states=readFileSync(statesPath,'utf8').trim().split('\n').map(line=>{
  const t=line.trim().split(/\s+/).map(Number);const ranks=t.slice(0,16),size=t[16];
  const cards=[],probs=[];for(let i=0;i<size;i++){cards.push(fromRank(t[18+2*i]));probs.push(t[19+2*i]);}
  return {board:[0,1,2,3].map(y=>ranks.slice(y*4,y*4+4).map(fromRank)),next:{candidates:cards,probabilities:probs,bonus:!!t[17]},remaining:t.slice(24,27)};
});
const cpp=readFileSync(cppPath,'utf8').trim().split('\n').map(JSON.parse);
const names=['left','right','up','down'];let same=0,maxRel=0;const times=[],rows=[];
for(const [i,s] of states.entries()){
  const stats={};const t0=performance.now(),c0=process.cpuUsage();
  const r=chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:depth,threshold,budgetMs:1e9,maxNodes:1e12,stats,clampLeaf:process.env.CLAMP==='1'});
  const ms=performance.now()-t0,cu=process.cpuUsage(c0);times.push((cu.user+cu.system)/1000);if(r.depth!==depth)throw Error('incomplete');
  const c=cpp[i];if(names[c.dir]===r.direction)same++;
  for(let d=0;d<4;d++){const a=stats.values[d]??-1,b=c.q[d];if(b!==-1)maxRel=Math.max(maxRel,Math.abs(a-b)/Math.max(1,Math.abs(b)));}
  rows.push({i,dir:names.indexOf(r.direction),ms,nodes:r.nodes});
}
times.sort((a,b)=>a-b);const mean=times.reduce((a,b)=>a+b,0)/times.length;
console.log(JSON.stringify({states:states.length,depth,threshold,sameDirection:same,maxRelValueDiff:maxRel,meanCpuMs:mean,p95CpuMs:times[Math.floor(.95*times.length)],maxCpuMs:times.at(-1)}));
if(outPath)writeFileSync(outPath,rows.map(r=>JSON.stringify(r)).join('\n')+'\n');
