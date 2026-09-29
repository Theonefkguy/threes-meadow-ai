// V26 browser timing: the shipped dist/fast-search.js and V23 model on the 128 archived
// 6144-phase states of V25 (V21 pool). Arms as in ../protocol.json.
import {chooseFastMove} from '../../../dist/fast-search.js?v=21';
import {decodeModel} from '../../../dist/rl-model.js?v=21';
const ARMS={B:{maxDepth:5,threshold:.01},A1:{maxDepth:5,threshold:.005},A2:{maxDepth:6,threshold:.005}};
self.onmessage=async()=>{
 try{
  const model=decodeModel(await (await fetch('../../../dist/models/ntuple-v23.bin')).arrayBuffer());
  const states=(await (await fetch('../../v25-adaptive-cutoff/browser/states.json')).json()).filter(s=>s.phase==='6144');
  if(states.length!==128)throw Error('expected 128 6144-phase states');
  const arms=Object.keys(ARMS);
  const choose=(s,arm,budgetMs)=>chooseFastMove(s.board,s.next,model,{remaining:s.remaining,...ARMS[arm],clampLeaf:true,budgetMs,maxNodes:4e6});
  for(const s of states.slice(0,10))for(const arm of arms)choose(s,arm,1e9);
  const measurements=[];
  for(let rep=0;rep<3;rep++){
   for(let index=0;index<states.length;index++){
    const k=(index+rep)%3,order=[...arms.slice(k),...arms.slice(0,k)];
    for(const arm of order){
     const start=performance.now();const result=choose(states[index],arm,160);const ms=performance.now()-start;
     measurements.push({rep,index,arm,ms,depth:result.depth,direction:result.direction,nodes:result.nodes});
    }
   }
   self.postMessage({type:'progress',message:`Timing pass ${rep+1}/3 completed`});
  }
  const parity=[];
  for(let index=0;index<states.length;index++)for(const arm of arms){const r=choose(states[index],arm,1e9);parity.push({index,arm,direction:r.direction,depth:r.depth});}
  const summary=arms.map(arm=>{
   const r=measurements.filter(r=>r.arm===arm),times=r.map(r=>r.ms).sort((a,b)=>a-b);
   return {arm,n:r.length,meanMs:times.reduce((a,b)=>a+b,0)/times.length,p95Ms:times[Math.ceil(.95*times.length)-1],maxMs:times.at(-1),completedFullDepth:r.filter(x=>x.depth===ARMS[arm].maxDepth).length};
  });
  self.postMessage({type:'result',summary,measurements,parity,userAgent:navigator.userAgent});
 }catch(e){self.postMessage({type:'error',message:String(e.stack||e)});}
};
