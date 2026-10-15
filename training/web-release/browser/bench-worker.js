// Web release timing: the shipped dist/fast-search.js and dist/rl-model.js with website settings
// (depth 5, cutoff 0.01, clamp, 160 ms soft budget) on the 320 V25 browser states (64 at 1536,
// 128 at 3072, 128 at 6144). Model files come from the page's query string:
//   ?base=PATH&res=PATH,PATH   (paths relative to this page; res optional)
//   &cycle=1: cycle-aware search (V35) with state i at cycle {pos:i%21, used:i%3==0, excl:pos>>1}
import {chooseFastMove} from '../../../dist/fast-search.js';
import {decodeModel} from '../../../dist/rl-model.js';
self.onmessage=async({data})=>{
 try{
  const get=async p=>{const r=await fetch(p);if(!r.ok)throw Error('fetch '+p);return r.arrayBuffer();};
  const t0=performance.now();
  const model=decodeModel(await get(data.base),await Promise.all(data.res.map(get)));
  const loadMs=performance.now()-t0;
  const states=await (await fetch('../../v25-adaptive-cutoff/browser/states.json')).json();
  if(states.length!==320)throw Error('expected 320 states');
  const cycleOf=i=>data.cycle?{pos:i%21,used:i%3===0?1:0,excl:(i%21)>>1}:null;
  const choose=(s,budgetMs,i)=>chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:5,threshold:0.01,clampLeaf:true,budgetMs,maxNodes:4e6,cycle:cycleOf(i)});
  states.slice(0,10).forEach((s,i)=>choose(s,1e9,i));
  const measurements=[];
  for(let rep=0;rep<3;rep++){
   for(let index=0;index<states.length;index++){
    const start=performance.now();const r=choose(states[index],160,index);const ms=performance.now()-start;
    measurements.push({rep,index,phase:states[index].phase,ms,depth:r.depth,direction:r.direction,nodes:r.nodes});
   }
   self.postMessage({type:'progress',message:`Timing pass ${rep+1}/3 completed`});
  }
  const summary=['1536','3072','6144','all'].map(phase=>{
   const r=measurements.filter(m=>phase==='all'||m.phase===phase),t=r.map(m=>m.ms).sort((a,b)=>a-b);
   return {phase,n:r.length,meanMs:t.reduce((a,b)=>a+b,0)/t.length,p95Ms:t[Math.ceil(.95*t.length)-1],maxMs:t.at(-1),completedFullDepth:r.filter(m=>m.depth===5).length};
  });
  self.postMessage({type:'result',base:data.base,res:data.res,cycle:!!data.cycle,loadMs,userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,summary,measurements});
 }catch(e){self.postMessage({type:'error',message:String(e&&e.message||e)});}
};
