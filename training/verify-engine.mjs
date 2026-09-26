import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { createGame, projectMove, entryCell, score, bonusPreviews } from '../dist/engine.js';
const rank=n=>n<3?n:Math.log2(n/3)+3;
const value=r=>r<3?r:3*2**(r-3);
const dirs=['left','right','up','down'];
let state=984331;const rng=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
const boards=Array.from({length:500},()=>Array.from({length:4},()=>Array.from({length:4},()=>value(Math.floor(rng()*17)))));
const input=boards.flatMap(b=>dirs.map((d,i)=>`move ${i} ${b.flat().map(rank).join(' ')}`)).join('\n');
const output=execFileSync('training/bin/probe',{input,encoding:'utf8'}).trim().split('\n');let line=0;
for(const b of boards)for(const d of dirs){
 const m=projectMove(b,d),numbers=output[line++].split(' ').map(Number),[size,reward]=numbers;
 assert.equal(size,m.lanes.length);assert.equal(reward,score(m.board)-score(b));
 assert.deepEqual(numbers.slice(2,18).map(value),m.board.flat());
 assert.deepEqual(numbers.slice(18),m.lanes.map(l=>{const[y,x]=entryCell(d,l);return y*4+x;}));
}
for(let seed=100;seed<120;seed++){
 let state=seed;const rng=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296),g=createGame(rng);
 const lines=execFileSync('training/bin/probe',{input:`game ${seed}\n`,encoding:'utf8'}).trim().split('\n');
 let i=0;for(const line of lines){if(line==='END')break;const a=line.split(' ').map(Number);let p=0;
  assert.deepEqual(a.slice(p,p+=16).map(value),g.board.flat());const size=a[p++];assert.deepEqual(a.slice(p,p+=size).map(value),g.deck);
  assert.equal(a[p++],g.next.candidates.length);for(let j=0;j<g.next.candidates.length;j++){assert.equal(value(a[p++]),g.next.candidates[j]);assert.ok(Math.abs(a[p++]-(g.next.probabilities?.[j]??1))<1e-12);}
  for(let d=i%4,k=0;k<4;k++,d=(d+1)%4)if(g.move(dirs[d]).changed)break;i++;
 }
}
console.log('C++ trainer matches canonical engine: 2,000 moves/rewards and 20 seeded game trajectories.');
for(let r=6;r<=16;r++){
 const a=execFileSync('training/bin/probe',{input:`bonus ${r}\n`,encoding:'utf8'}).trim().split(' ').map(Number),previews=bonusPreviews(value(r));
 let p=0;assert.equal(a[p++],previews.length);
 for(const preview of previews){assert.ok(Math.abs(a[p++]-preview.probability)<1e-12);assert.equal(a[p++],preview.candidates.length);
  for(let i=0;i<preview.candidates.length;i++){assert.equal(value(a[p++]),preview.candidates[i]);assert.ok(Math.abs(a[p++]-preview.probabilities[i])<1e-12);}}
}
console.log('Bonus preview and conditional distributions match at every threshold through 24,576.');
