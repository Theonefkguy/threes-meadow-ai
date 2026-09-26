import fs from 'node:fs';
import assert from 'node:assert/strict';
import {decodeModel} from '../../dist/rl-model.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
const raw=fs.readFileSync('dist/models/ntuple-v1.bin');
const model=decodeModel(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
const value=r=>r<3?r:3*2**(r-3),dirs=['left','right','up','down'];
const rows=fs.readFileSync('training/v2/parity.jsonl','utf8').trim().split('\n').map(JSON.parse);
let checked=0;
for(const row of rows){
 const board=Array.from({length:4},(_,i)=>row.ranks.slice(i*4,i*4+4).map(value));
 const preview={candidates:row.cards.map(value),probabilities:row.weights};
 const result=chooseLearnedMove(board,preview,model,{remaining:row.counts,budgetMs:Infinity});
 assert.equal(result.direction,dirs[row.direction],`direction seed=${row.seed} turn=${row.turn}`);
 assert.equal(result.depth,row.depth,`depth seed=${row.seed} turn=${row.turn}`);
 assert.equal(result.nodes,row.nodes,`nodes seed=${row.seed} turn=${row.turn}`);checked++;
}
console.log(JSON.stringify({checked,match:'direction, completed depth and visited nodes',nodeLimit:24000,wallTimeLimit:false}));
