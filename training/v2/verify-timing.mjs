import fs from 'node:fs';
import {decodeModel} from '../../dist/rl-model.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const buffer=path=>{const b=fs.readFileSync(path);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const rows=fs.readFileSync('training/v2/parity.jsonl','utf8').trim().split('\n').map(JSON.parse).filter((_,i)=>i%7===0);
const value=r=>r<3?r:3*2**(r-3),result={};
for(const name of ['v1','v2']){
 const model=decodeModel(buffer(`dist/models/ntuple-${name}.bin`)),ms=[],depths=[0,0,0,0];
 for(const row of rows){
  const board=Array.from({length:4},(_,i)=>row.ranks.slice(i*4,i*4+4).map(value)),hint={candidates:row.cards.map(value),probabilities:row.weights};
  const start=performance.now(),r=chooseLearnedMove(board,hint,model,{remaining:row.counts});ms.push(performance.now()-start);depths[r.depth]++;
 }
 ms.sort((a,b)=>a-b);result[name]={positions:rows.length,meanMs:ms.reduce((a,b)=>a+b,0)/ms.length,p95Ms:ms[Math.floor(.95*(ms.length-1))],maxMs:ms.at(-1),completedDepthCounts:depths};
}
result.environment='Node.js on this execution machine under concurrent native evaluation load; not a phone or browser benchmark';
fs.writeFileSync('training/v2/timing.json',JSON.stringify(result,null,2)+'\n');console.log(result);
