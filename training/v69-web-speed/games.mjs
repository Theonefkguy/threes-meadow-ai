// V69: whole games on the website engine (dist/engine.js, public counts and bonus cycle tracked as dist/game.js
// does) with the V31 web model and website settings, no time limit. Every move is searched by every arm (in a
// rotating order); the game follows arm 0, and each other arm's direction and four root values are compared
// with arm 0's.
//   node games.mjs ARMS FIRST_SEED COUNT OUT
// ARMS as in bench.mjs (ref, new, new:1.2.3.4.6, ...). OUT: one JSON line per game; OUT.diff: one line per differing move.
import { readFileSync, appendFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import * as ref from './ref-fast-search.js';
import * as cur from '../../dist/fast-search.js';
import { decodeModel } from '../../dist/rl-model.js';
import { createGame, score } from '../../dist/engine.js';
import { initialCounts, observePreview, initialCycle, observeCycle } from '../../dist/card-memory.js';

const [armList,firstS,countS,out]=process.argv.slice(2);
const Q=new URL('../web-release/models/',import.meta.url);
const buf=p=>{const b=readFileSync(new URL(p,Q));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const model=decodeModel(buf('ntuple-v31.bin'),['ntuple-v31-s1.f16','ntuple-v31-s2.f16','ntuple-v31-s3.f16'].map(buf));
const arms=armList.split(',').map(a=>{const [m,s]=a.split(':');return {name:a,fn:(m==='ref'?ref:cur).chooseFastMove,depths:s?s.split('.').map(Number):null};});
// Deterministic generator per seed (xorshift32 after a splitmix-style scramble).
const rngFor=seed=>{let s=Math.imul(seed^0x9e3779b9,0x85ebca6b)>>>0||1;return ()=>{s^=s<<13;s>>>=0;s^=s>>>17;s^=s<<5;s>>>=0;return s/4294967296;};};
for(let seed=+firstS;seed<+firstS+ +countS;seed++){
  const g=createGame(rngFor(seed));let counts=initialCounts(g.board,g.next),cycle=initialCycle();
  const ms=arms.map(()=>0),differ=arms.map(()=>0);let moves=0,unsupported=false;
  while(!g.over){
    const res=Array(arms.length);
    for(let r=0;r<arms.length;r++){const j=(r+moves)%arms.length,a=arms[j],stats={};
      const opt={remaining:counts,maxDepth:6,threshold:0.005,clampLeaf:true,budgetMs:1e9,maxNodes:1e12,stats,cycle};
      if(a.depths)opt.depths=a.depths;
      const t0=performance.now(),m=a.fn(g.board,g.next,model,opt);ms[j]+=performance.now()-t0;
      res[j]={m,values:stats.values};
    }
    if(res[0].m.unsupported){unsupported=true;break;}
    for(let j=1;j<arms.length;j++)if(res[j].m.direction!==res[0].m.direction||res[j].values.some((v,d)=>!Object.is(v,res[0].values[d]))){
      differ[j]++;
      // Details of each difference (with the full root state, to replay it) in OUT.diff.
      appendFileSync(out+'.diff',JSON.stringify({seed,move:moves,arm:arms[j].name,board:g.board,next:g.next,remaining:counts,cycle,
        directions:[res[0].m.direction,res[j].m.direction],values:[res[0].values,res[j].values]})+'\n');
    }
    g.move(res[0].m.direction);moves++;
    counts=observePreview(counts,g.next);cycle=observeCycle(cycle,g.next,g.board);
  }
  const high=Math.max(...g.board.flat());
  appendFileSync(out,JSON.stringify({seed,moves,score:score(g.board),high,unsupported,arms:arms.map(a=>a.name),ms:ms.map(x=>+x.toFixed(1)),differ})+'\n');
}
