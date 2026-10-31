// V69: the website search in JavaScript (dist/fast-search.js) against the frozen original (ref-fast-search.js)
// on the V62 positions, website settings (6 plies, cutoff 0.005, clamped leaves, cycle-aware), no time limit.
//   node bench.mjs ARMS POSITIONS FIRST COUNT OUT
// ARMS: comma list of module or module:depths, e.g. ref,new,new:1.4.6 (ref = ref-fast-search.js, new = dist/fast-search.js;
// depths, dot-separated, are passed as the new option `depths`; default 1-6). Arms run back to back in a rotating order.
// OUT: one line per position: id, then per arm: ms nodes depth dir v0 v1 v2 v3 (values with 17 digits).
import { readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import * as ref from './ref-fast-search.js';
import * as cur from '../../dist/fast-search.js';
import { decodeModel } from '../../dist/rl-model.js';

const [armList,posPath,firstS,countS,out]=process.argv.slice(2);
const Q=new URL('../web-release/models/',import.meta.url);
const buf=p=>{const b=readFileSync(new URL(p,Q));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const model=decodeModel(buf('ntuple-v31.bin'),['ntuple-v31-s1.f16','ntuple-v31-s2.f16','ntuple-v31-s3.f16'].map(buf));
const arms=armList.split(',').map(a=>{const [m,s]=a.split(':');return {fn:(m==='ref'?ref:cur).chooseFastMove,depths:s?s.split('.').map(Number):null};});
const fromRank=r=>r<3?r:3*2**(r-3);
const first=+firstS,count=+countS,lines=readFileSync(posPath,'utf8').trim().split('\n').slice(first,first+count);
const outLines=[];
for(const line of lines){
  const x=line.trim().split(/\s+/).map(Number);let i=0;const id=x[i++];
  const ranks=x.slice(i,i+16);i+=16;const n=x[i++];const cards=x.slice(i,i+n);i+=n;i++;// bonus flag (implied by the cards)
  const remaining=x.slice(i,i+3);i+=3;const cycle={pos:x[i],used:x[i+1],excl:x[i+2]};
  const board=[0,1,2,3].map(r=>ranks.slice(r*4,r*4+4).map(fromRank));
  const next={candidates:cards.map(fromRank),probabilities:cards.map(()=>1/n)};
  const res=Array(arms.length);
  for(let r=0;r<arms.length;r++){const j=(r+id)%arms.length,a=arms[j],stats={};
    const opt={remaining,maxDepth:6,threshold:0.005,clampLeaf:true,budgetMs:1e9,maxNodes:1e12,stats,cycle};
    if(a.depths)opt.depths=a.depths;
    const t0=performance.now(),m=a.fn(board,next,model,opt),ms=performance.now()-t0;
    res[j]=[ms.toFixed(4),m.nodes,m.depth,['left','right','up','down'].indexOf(m.direction),...stats.values.map(v=>v===null?'nan':v.toPrecision(17))].join(' ');
  }
  outLines.push(id+' '+res.join(' '));
}
writeFileSync(out,outLines.join('\n')+'\n');
