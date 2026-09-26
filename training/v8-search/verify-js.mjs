import fs from 'node:fs';
import assert from 'node:assert/strict';
import {decodeModel} from '../../dist/rl-model.js';
import {chooseMCTSMove,SearchRandom} from '../../dist/mcts-search.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const buffer=fs.readFileSync('dist/models/ntuple-v8.bin'),model=decodeModel(buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength));
const rows=fs.readFileSync('training/v8-search/parity.jsonl','utf8').trim().split('\n').map(JSON.parse),value=r=>r<3?r:3*2**(r-3),dirs=['left','right','up','down'];
let count=0;const timings=[];
for(const x of rows){const board=Array.from({length:4},(_,r)=>x.ranks.slice(r*4,r*4+4).map(value)),next={candidates:x.cards.map(value),probabilities:x.weights};const a=chooseMCTSMove(board,next,model,{remaining:x.counts,budgetMs:1e9,maxSimulations:x.limit,seed:x.seed});assert.equal(a.direction,dirs[x.direction]);assert.deepEqual(a.visits,x.visits);assert.equal(a.depth,x.depth);assert.equal(a.simulations,x.limit);count++;
 if(x.limit===640){const start=performance.now(),b=chooseMCTSMove(board,next,model,{remaining:x.counts,seed:x.seed});timings.push({ms:performance.now()-start,simulations:b.simulations,changed:b.direction!==a.direction});}
}
// Established late-game code is used verbatim with the same frozen weights.
for(const x of rows.slice(0,8)){const ranks=x.ranks.slice();ranks[ranks.indexOf(Math.max(...ranks))]=12;const board=Array.from({length:4},(_,r)=>ranks.slice(r*4,r*4+4).map(value)),next={candidates:x.cards.map(value),probabilities:x.weights};const options={remaining:x.counts,budgetMs:1e9};const a=chooseMCTSMove(board,next,model,options),b=chooseLearnedMove(board,next,model,options);assert.equal(a.direction,b.direction);assert.equal(a.nodes,b.nodes);assert.equal(a.algorithm,'expectimax');}
const sorted=timings.map(x=>x.ms).sort((a,b)=>a-b);const output={nativeParityCases:count,lateParityCases:8,timing:{positions:timings.length,mean:sorted.reduce((a,b)=>a+b,0)/sorted.length,p95:sorted[Math.floor(sorted.length*.95)],max:sorted.at(-1),directionChangesVs640:timings.filter(x=>x.changed).length,budgetLimited:timings.filter(x=>x.simulations<640).length},note:'Browser JavaScript executed in Node; no iPhone performance or game-success claim'};
fs.writeFileSync('training/v8-search/js-validation.json',JSON.stringify(output,null,2)+'\n');console.log(output);
