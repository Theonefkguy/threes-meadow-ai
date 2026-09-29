import {chooseFastMove} from './fast-search.js';
import {decodeModel} from './rl-model.js';
self.onmessage=async()=>{
 try{
  const model=decodeModel(await (await fetch('./model.bin')).arrayBuffer());
  const states=await (await fetch('./states.json')).json();
  const arms=['baseline','adaptive'];
  const threshold=(s,arm)=>arm==='adaptive'&&Math.max(...s.board.flat())>=3072?.005:.01;
  const choose=(s,arm,budgetMs)=>chooseFastMove(s.board,s.next,model,{remaining:s.remaining,maxDepth:5,threshold:threshold(s,arm),clampLeaf:true,budgetMs,maxNodes:4e6});
  for(const phase of ['1536','3072','6144']) for(const s of states.filter(s=>s.phase===phase).slice(0,10))for(const arm of arms)choose(s,arm,1e9);
  const measurements=[];
  for(let rep=0;rep<3;rep++){
   for(let index=0;index<states.length;index++){
    const s=states[index];
    for(const arm of (index+rep)%2?arms.toReversed():arms){
     const start=performance.now();const result=choose(s,arm,160);const ms=performance.now()-start;
     measurements.push({rep,index,phase:s.phase,arm,ms,...result});
    }
   }
   self.postMessage({type:'progress',message:`Timing pass ${rep+1}/3 completed`});
  }
  const parity=[];
  for(let index=0;index<states.length;index++)for(const arm of arms)parity.push({index,arm,...choose(states[index],arm,1e9)});
  const summary=[];
  for(const phase of ['1536','3072','6144'])for(const arm of arms){
   const r=measurements.filter(r=>r.phase===phase&&r.arm===arm),times=r.map(r=>r.ms).sort((a,b)=>a-b);
   summary.push({phase,arm,n:r.length,meanMs:times.reduce((a,b)=>a+b,0)/times.length,p95Ms:times[Math.ceil(.95*times.length)-1],maxMs:times.at(-1),completed5:r.filter(r=>r.depth===5).length});
  }
  self.postMessage({type:'result',summary,measurements,parity,userAgent:navigator.userAgent});
 }catch(e){self.postMessage({type:'error',message:String(e.stack||e)});}
};
