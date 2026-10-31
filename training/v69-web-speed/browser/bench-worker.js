// V69 browser timing: the original website search (ref-fast-search.js, depths 1-6) against dist/fast-search.js
// with the website setting (webSearchOptions), V31 web model (float16 residuals), on the first
// N V62 positions, no time limit; arms back to back in a rotating order. Also checks that directions and the four
// root values are identical.
import * as ref from '../ref-fast-search.js';
import * as cur from '../../../dist/fast-search.js';
import {decodeModel} from '../../../dist/rl-model.js';
self.onmessage=async({data})=>{
 try{
  const get=async p=>{const r=await fetch(p);if(!r.ok)throw Error('fetch '+p);return r.arrayBuffer();};
  const Q='../../web-release/models/';
  const model=decodeModel(await get(Q+'ntuple-v31.bin'),await Promise.all(['ntuple-v31-s1.f16','ntuple-v31-s2.f16','ntuple-v31-s3.f16'].map(f=>get(Q+f))));
  const lines=(await (await fetch('../../v62-time-share/positions.txt')).text()).trim().split('\n').slice(0,data.count);
  const fromRank=r=>r<3?r:3*2**(r-3);
  const arms=[
   {name:'original 1-6',fn:ref.chooseFastMove,opt:{maxDepth:6,threshold:0.005,budgetMs:400}},
   {name:'V69 website setting',fn:cur.chooseFastMove,opt:cur.webSearchOptions()}];
  const ms=arms.map(()=>[]);let differ=0;
  lines.forEach((line,id)=>{
   const x=line.trim().split(/\s+/).map(Number);let i=1;
   const ranks=x.slice(i,i+16);i+=16;const n=x[i++];const cards=x.slice(i,i+n);i+=n+1;
   const remaining=x.slice(i,i+3);i+=3;const cycle={pos:x[i],used:x[i+1],excl:x[i+2]};
   const board=[0,1,2,3].map(r=>ranks.slice(r*4,r*4+4).map(fromRank)),next={candidates:cards.map(fromRank),probabilities:cards.map(()=>1/n)};
   const res=[];
   for(let r=0;r<arms.length;r++){const j=(r+id)%arms.length,stats={};
    const t0=performance.now(),m=arms[j].fn(board,next,model,{...arms[j].opt,budgetMs:1e9,maxNodes:1e12,remaining,cycle,clampLeaf:true,stats});
    ms[j].push(performance.now()-t0);res[j]={dir:m.direction,depth:m.depth,values:stats.values};}
   if(res[1].dir!==res[0].dir||res[1].depth!==res[0].depth||res[1].values.some((v,d)=>!Object.is(v,res[0].values[d])))differ++;
   if(id%50===49)self.postMessage({type:'progress',message:`${id+1}/${lines.length}`});
  });
  const sum=t=>{const s=t.slice().sort((a,b)=>a-b);return {meanMs:+(t.reduce((a,b)=>a+b,0)/t.length).toFixed(2),p95Ms:+s[Math.ceil(.95*s.length)-1].toFixed(1),maxMs:+s.at(-1).toFixed(1)};};
  self.postMessage({type:'result',userAgent:navigator.userAgent,positions:lines.length,differ,arms:arms.map((a,j)=>({name:a.name,...sum(ms[j])})),ratio:+(ms[1].reduce((a,b)=>a+b,0)/ms[0].reduce((a,b)=>a+b,0)).toFixed(3)});
 }catch(e){self.postMessage({type:'error',message:String(e&&e.stack||e)});}
};
