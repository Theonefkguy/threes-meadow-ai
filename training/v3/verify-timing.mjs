import fs from 'node:fs';
import {decodeModel} from '../../dist/rl-model.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const decode=p=>{const b=fs.readFileSync(p);return decodeModel(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const models={v2:decode('dist/models/ntuple-v2.bin'),v3:decode('dist/models/ntuple-v3.bin')};
const all=fs.readFileSync('training/v3/parity-selected.jsonl','utf8').trim().split('\n').map(JSON.parse);
const rows=all.filter((_,i)=>i%Math.max(1,Math.floor(all.length/60))===0),out={};
const val=r=>r<3?r:3*2**(r-3);
for(const name of ['v2','v3'])out[name]={ms:[],depths:[0,0,0,0],differentFromUnlimited:0};
for(const [i,r] of rows.entries()){
 const board=Array.from({length:4},(_,j)=>r.ranks.slice(j*4,j*4+4).map(val)),preview={candidates:r.cards.map(val),probabilities:r.weights};
 for(const name of i%2?['v3','v2']:['v2','v3']){
  const ideal=chooseLearnedMove(board,preview,models[name],{remaining:r.counts,budgetMs:Infinity});
  const t=performance.now(),answer=chooseLearnedMove(board,preview,models[name],{remaining:r.counts,budgetMs:160});out[name].ms.push(performance.now()-t);out[name].depths[answer.depth]++;out[name].differentFromUnlimited+=answer.direction!==ideal.direction;
 }
}
for(const name of ['v2','v3']){const a=out[name],ms=a.ms.sort((x,y)=>x-y);out[name]={positions:ms.length,meanMs:ms.reduce((x,y)=>x+y,0)/ms.length,p95Ms:ms[Math.floor(.95*(ms.length-1))],maxMs:ms.at(-1),depths:a.depths,differentFromUnlimited:a.differentFromUnlimited};}
out.environment='Actual JS search in Node on this evaluation machine; not an iPhone benchmark. Alternating model order, unlimited reference preceding timed call; sampled on-policy positions, not complete games.';
fs.writeFileSync('training/v3/timing.json',JSON.stringify(out,null,2)+'\n');console.log(out);
