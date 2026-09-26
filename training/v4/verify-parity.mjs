import fs from 'node:fs';
import assert from 'node:assert/strict';
import {decodeModel} from '../../dist/rl-model.js';
import {decodeResidual} from '../../dist/v4-model.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const [path,probe,extra]=process.argv.slice(2),buffer=p=>{const b=fs.readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const base=decodeModel(buffer(path)),model=extra&&extra!=='-'?decodeResidual(buffer(extra),base):base;
const all=fs.readFileSync(probe,'utf8').trim().split('\n').map(JSON.parse),rows=[];
for(const stage of [0,1,2]){const group=all.filter(r=>{const h=Math.max(...r.ranks);return stage===0?h<12:stage===1?h===12:h>=13;});for(let i=0;i<group.length;i+=Math.max(1,Math.floor(group.length/25)))rows.push(group[i]);}
const value=r=>r<3?r:3*2**(r-3),dirs=['left','right','up','down'];
for(const r of rows){const board=Array.from({length:4},(_,i)=>r.ranks.slice(4*i,4*i+4).map(value)),hint={candidates:r.cards.map(value),probabilities:r.weights};const answer=chooseLearnedMove(board,hint,model,{remaining:r.counts,budgetMs:Infinity});assert.equal(answer.direction,dirs[r.direction]);assert.equal(answer.depth,r.depth);assert.equal(answer.nodes,r.nodes);}
console.log(JSON.stringify({model:path,residual:extra||null,positions:rows.length,exact:['direction','nodes','depth']}));
