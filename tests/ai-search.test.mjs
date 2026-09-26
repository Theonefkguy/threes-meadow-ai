import test from 'node:test';
import assert from 'node:assert/strict';
import { projectMove, entryCell, createGame } from '../dist/engine.js';
import { projectFast, initialCounts, observePreview } from '../dist/ai-search.js';
test('fast search movement matches actual game in all directions on varied boards',()=>{
 let s=5;const rng=()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296);
 const vals=[0,0,0,1,2,3,6,12,24,48,96,192,384,768,1536,3072,6144,24576];
 for(let n=0;n<1500;n++){
  const b=Array.from({length:4},()=>Array.from({length:4},()=>vals[Math.floor(rng()*vals.length)]));
  for(const dir of ['left','right','up','down']){
   const reference=projectMove(b,dir),fast=projectFast(b,dir);
   assert.equal(!!fast,reference.changed);
   if(fast){assert.deepEqual(fast.board,reference.board);assert.deepEqual(fast.entries,reference.lanes.map(l=>{const[y,x]=entryCell(dir,l);return y*4+x;}));}
  }
 }
});
test('public counting tracks remaining composition across refills and bonus cards',()=>{
 let s=42;const rng=()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296);
 const game=createGame(rng);let counts=initialCounts(game.board,game.next);
 for(let i=0;i<500;i++){
  assert.deepEqual(counts,[1,2,3].map(n=>game.deck.filter(v=>v===n).length));
  game.board=[[0,1536,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  game.move('left');counts=observePreview(counts,game.next);
 }
});
