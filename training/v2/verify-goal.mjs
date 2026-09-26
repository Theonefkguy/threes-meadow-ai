import fs from 'node:fs';
import assert from 'node:assert/strict';
import {decodeModel} from '../../dist/rl-model.js';
import {decodeGoalModel} from '../../dist/goal-model.js';
import {chooseGoalMove} from '../../dist/goal-search.js';
const buffer=path=>{const b=fs.readFileSync(path);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const score=decodeModel(buffer('dist/models/ntuple-v1.bin'));
const model=decodeGoalModel(buffer('training/v2/parity-model.goal'));
const value=r=>r<3?r:3*2**(r-3),dirs=['left','right','up','down'];
const rows=fs.readFileSync('training/v2/parity-goal-frozen.jsonl','utf8').trim().split('\n').map(JSON.parse);
let checked=0;
for(const row of rows){
 const board=Array.from({length:4},(_,i)=>row.ranks.slice(i*4,i*4+4).map(value));
 const preview={candidates:row.cards.map(value),probabilities:row.weights};
 const result=chooseGoalMove(board,preview,model,score,{remaining:row.counts,budgetMs:Infinity});
 assert.equal(result.direction,dirs[row.direction],`direction seed=${row.seed} turn=${row.turn}`);
 assert.equal(result.depth,row.depth,`depth seed=${row.seed} turn=${row.turn}`);
 assert.equal(result.nodes,row.nodes,`nodes seed=${row.seed} turn=${row.turn}`);checked++;
}
console.log(JSON.stringify({checked,match:'goal direction, completed depth and visited nodes',nodeLimit:24000,wallTimeLimit:false}));
