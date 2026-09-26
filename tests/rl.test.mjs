import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseLearnedMove } from '../dist/rl-search.js';
import { phaseForRanks,decodeModel } from '../dist/rl-model.js';
import { projectMove,entryCell,score,createGame } from '../dist/engine.js';
const dirs=['left','right','up','down'];
const rank=n=>n<3?n:Math.log2(n/3)+3;
const stub={value:b=>b.filter(n=>n===0).length*31 + b[0]*7-b[5]*3};

// Independent two-move oracle on the canonical number-board engine.
function oracle(board,next){
 let best=-Infinity,dir=null;
 for(const d of dirs){
  const m=projectMove(board,d);if(!m.changed)continue;let q=score(m.board)-score(board),ev=0;
  for(const lane of m.lanes)for(let i=0;i<next.candidates.length;i++){
   const b=m.board.map(r=>r.slice()),[y,x]=entryCell(d,lane);b[y][x]=next.candidates[i];
   let continuation=-Infinity;
   for(const d2 of dirs){const a=projectMove(b,d2);if(a.changed)continuation=Math.max(continuation,score(a.board)-score(b)+stub.value(a.board.flat().map(rank)));}
   if(continuation===-Infinity)continuation=0;
   ev+=(next.probabilities?.[i]??1/next.candidates.length)*(score(b)-score(m.board)+continuation);
  }
  q+=ev/m.lanes.length;if(q>best){best=q;dir=d;}
 }
 return dir;
}
test('learned search agrees with independent reward/terminal/bonus oracle',()=>{
 for(let seed=1;seed<=25;seed++){
  let s=seed;const rng=()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296),g=createGame(rng);
  g.board[1][1]=384;
  const next={candidates:[6,12,24],probabilities:[.5,.25,.25]};
  assert.equal(chooseLearnedMove(g.board,next,stub,{maxDepth:2,maxNodes:1e7,budgetMs:1e6}).direction,oracle(g.board,next));
 }
 const b=[[3,3,24,96],[12,48,192,6],[24,96,6,12],[48,192,12,24]],next={candidates:[1]};
 assert.equal(chooseLearnedMove(b,next,stub,{maxDepth:2}).direction,oracle(b,next));
});
test('learned search has legal fallback, preserves visible state and handles terminal/range',()=>{
 const g=createGame(()=>.5),next={candidates:[6,12,24],probabilities:[.5,.25,.25]},before=JSON.stringify([g.board,next]);
 const r=chooseLearnedMove(g.board,next,stub,{maxNodes:1,budgetMs:0});assert.equal(r.depth,1);assert(projectMove(g.board,r.direction).changed);
 assert.equal(JSON.stringify([g.board,next]),before);
 const terminal=[[3,6,12,24],[6,12,24,48],[12,24,48,96],[24,48,96,192]];
 assert.equal(chooseLearnedMove(terminal,{candidates:[1]},stub).direction,null);
 g.board[0][0]=24576;assert.equal(chooseLearnedMove(g.board,next,stub).unsupported,true);
});
test('phase thresholds and model format are explicit',()=>{
 assert.equal(phaseForRanks([11]),0);assert.equal(phaseForRanks([12]),1);assert.equal(phaseForRanks([13]),2);
 assert.throws(()=>decodeModel(new ArrayBuffer(20)));
});
