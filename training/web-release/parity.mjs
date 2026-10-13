// Web release parity: shipped dist/fast-search.js + dist/rl-model.js (website settings, no time
// limit) against training/web-release/reference.cpp on the 320 V25 browser states.
//   node training/web-release/parity.mjs BASE.bin REFERENCE.jsonl [RESIDUAL.f16 ...]
// Checks every direction and completed depth for equality and the four root values to a relative
// 1e-12 (JS and C++ accumulate some chance sums in a different order; V21 measured < 1e-15).
import { readFileSync } from 'node:fs';
import { chooseFastMove } from '../../dist/fast-search.js';
import { decodeModel } from '../../dist/rl-model.js';
const [base,refPath,...res]=process.argv.slice(2);
const buf=p=>{const b=readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const model=decodeModel(buf(base),res.map(buf));
const states=JSON.parse(readFileSync(new URL('../v25-adaptive-cutoff/browser/states.json',import.meta.url)));
const ref=readFileSync(refPath,'utf8').trim().split('\n').map(l=>JSON.parse(l));
if(states.length!==ref.length)throw Error('state count mismatch');
const dirs=['left','right','up','down'];let bad=0,maxRel=0;const byPhase={};
states.forEach((s,i)=>{
  const stats={};const r=chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:5,threshold:0.01,clampLeaf:true,budgetMs:1e9,maxNodes:1e12,stats});
  const values=stats.values.map(v=>v===null?null:v);
  const close=(a,b)=>a===null||b===null?a===b:Math.abs(a-b)<=1e-12*Math.max(1,Math.abs(b));
  for(let d=0;d<4;d++)if(values[d]!==null&&ref[i].rootValues[d]!==null)maxRel=Math.max(maxRel,Math.abs(values[d]-ref[i].rootValues[d])/Math.max(1,Math.abs(ref[i].rootValues[d])));
  const ok=r.direction===dirs[ref[i].direction]&&r.depth===ref[i].depth&&values.every((v,d)=>close(v,ref[i].rootValues[d]));
  byPhase[s.phase]=(byPhase[s.phase]||0)+1;if(!ok){bad++;if(bad<=5)console.log('mismatch',i,s.phase,r.direction,dirs[ref[i].direction],values,ref[i].rootValues);}
});
console.log(JSON.stringify({states:states.length,byPhase,mismatches:bad,maxRootRelativeDifference:maxRel,residualStages:model.residuals.map((r,s)=>r?s:null).filter(s=>s!==null)}));
if(bad)process.exit(1);
