import assert from 'node:assert/strict';
import fs from 'node:fs';
import {decodeModel} from '../../dist/rl-model.js';
import {reachProbability} from '../../dist/v6-model.js';
import {chooseEarlyGoalMove} from '../../dist/v6-search.js';
import {chooseLearnedMove} from '../../dist/rl-search.js';
import {projectRanks} from '../../dist/ai-search.js';
const bytes=p=>fs.readFileSync(p),decode=b=>decodeModel(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
const base=bytes('dist/models/ntuple-v4.bin'),v4=decode(base);
const sigmoid={value:()=>0};assert.equal(reachProbability(sigmoid,Array(16).fill(0)),.5);
assert.equal(reachProbability({value:()=>1000},Array(16).fill(0)),1);assert.equal(reachProbability({value:()=>-1000},Array(16).fill(0)),0);
const value=r=>r<3?r:3*2**(r-3),board=b=>Array.from({length:4},(_,i)=>b.slice(i*4,i*4+4).map(value));
let cases=0;
for(const arm of ['control','hard','probability']){
 const raw=bytes(`training/v6/run-${arm}/checkpoint.ntd`),m=decode(raw);assert.deepEqual(raw.subarray(16+8*65536*4),base.subarray(16+8*65536*4));
 const b=[11,11,0,0,0,1,2,0,0,0,0,0,0,0,0,0],goal=[12,...b.slice(1)],next={candidates:[1],probabilities:[1]},options={remaining:[2,3,4],budgetMs:Infinity};
 if(arm==='probability'){
  assert(reachProbability(m,b)>=0&&reachProbability(m,b)<=1);assert.equal(reachProbability(m,goal),1);
  const a=chooseEarlyGoalMove(board(b),next,m,options),d=['left','right','up','down'].indexOf(a.direction);assert(Math.max(...projectRanks(b,d).board)>=12);
  assert.deepEqual(chooseEarlyGoalMove(board(goal),next,m,options),chooseLearnedMove(board(goal),next,v4,options));
  assert.equal(chooseEarlyGoalMove(Array.from({length:4},()=>[1,1,1,1]),next,m,options).direction,null);
 }
 cases++;
}
console.log(JSON.stringify({passed:true,arms:cases,checks:['finite weights','bitwise frozen V4 stages','bounded sigmoid','absorbing1536','terminal','V4 handoff']}));
