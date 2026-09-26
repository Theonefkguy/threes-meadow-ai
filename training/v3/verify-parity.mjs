import fs from 'node:fs';
import assert from 'node:assert/strict';
import {decodeModel} from '../../dist/rl-model.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const [path,probe,goalPath]=process.argv.slice(2);
if(goalPath)throw Error('Only the selected score policy is shipped; candidate B remains an offline experiment.');
const decode=p=>{const r=fs.readFileSync(p);return decodeModel(r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength));};
const score=decode(path),head=goalPath?decode(goalPath):null,bonus=head?200000:0;
const model=head?{value(b,s){let h=Math.max(...b),z=h>=13&&h<14?head.value(b,2):0,p=h>=14?1:h<13?0:z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));return score.value(b,s)+bonus*p;}}:score;
const rows=fs.readFileSync(probe,'utf8').trim().split('\n').map(JSON.parse);
const selected=[];for(const stage of [0,1,2]){const group=rows.filter(r=>Math.max(...r.ranks)>= (stage===2?13:stage===1?12:0)&&Math.max(...r.ranks)<(stage===2?99:stage===1?13:12));for(let i=0;i<group.length;i+=Math.max(1,Math.floor(group.length/20)))selected.push(group[i]);}
const val=r=>r<3?r:3*2**(r-3),dirs=['left','right','up','down'];
for(const row of selected){const board=Array.from({length:4},(_,i)=>row.ranks.slice(i*4,i*4+4).map(val)),preview={candidates:row.cards.map(val),probabilities:row.weights};const result=chooseLearnedMove(board,preview,model,{remaining:row.counts,budgetMs:Infinity,terminalBonus:bonus});assert.equal(result.direction,dirs[row.direction]);assert.equal(result.depth,row.depth);assert.equal(result.nodes,row.nodes);}
console.log(JSON.stringify({model:path,head:goalPath||null,positions:selected.length,exact:['direction','depth','nodes']}));
