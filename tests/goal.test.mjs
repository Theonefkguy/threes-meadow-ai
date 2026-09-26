import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeGoalModel,contextIndices} from '../dist/goal-model.js';
import {chooseGoalMove} from '../dist/goal-search.js';
import {chooseLearnedMove} from '../dist/rl-search.js';
import {createGame,projectMove} from '../dist/engine.js';
import {initialCounts} from '../dist/card-memory.js';
const BASE=8*65536,EXTRA=2*131072;
const modelBuffer=context=>{const b=new ArrayBuffer(16+2*(BASE+(context?EXTRA:0))*4);new Uint32Array(b,0,4).set([0x324c4f47,2,BASE,context?EXTRA:0]);return b;};
test('goal model is bounded, treats achieved target as success, and validates files',()=>{
 for(const context of [false,true]){
  const b=modelBuffer(context),model=decodeGoalModel(b),board=Array(16).fill(0),hint={cards:[1]},counts=[3,4,4];
  assert.equal(model.value(board,hint,counts),.5);board[0]=13;assert.equal(model.value(board,hint,counts),1);
  assert.throws(()=>decodeGoalModel(b.slice(0,-4)));new Float32Array(b,16)[0]=NaN;assert.throws(()=>decodeGoalModel(b));
 }
 assert.throws(()=>decodeGoalModel(new ArrayBuffer(4)));
});
test('public hint/count features change the probability head without hidden deck access',()=>{
 const buffer=modelBuffer(true),weights=new Float32Array(buffer,16),board=[12,1,2,...Array(13).fill(0)],preview={cards:[1]},counts=[3,4,2];
 const [index]=contextIndices(board,preview,counts);weights[BASE+EXTRA+BASE+index]=2;
 const model=decodeGoalModel(buffer);
 assert(model.value(board,preview,counts)>.8);
 assert.equal(model.value(board,{cards:[2]},counts),.5);
 assert.equal(model.value(board,preview,[2,4,2]),.5);
});
test('guided goal search keeps legal moves under budget and returns to scoring after 3072',()=>{
 const g=createGame(()=>.5),counts=initialCounts(g.board,g.next),model=decodeGoalModel(modelBuffer(true)),score={value:b=>b.filter(r=>r===0).length*20};
 const before=JSON.stringify([g.board,g.next,counts]);
 const a=chooseGoalMove(g.board,g.next,model,score,{remaining:counts,maxNodes:1,budgetMs:0});
 assert.equal(a.depth,1);assert(projectMove(g.board,a.direction).changed);assert.equal(JSON.stringify([g.board,g.next,counts]),before);
 assert.throws(()=>chooseGoalMove(g.board,g.next,model,score));
 g.board[0][0]=3072;const opts={remaining:counts,maxDepth:2,budgetMs:Infinity};
 assert.deepEqual(chooseGoalMove(g.board,g.next,model,score,opts),chooseLearnedMove(g.board,g.next,score,opts));
});
