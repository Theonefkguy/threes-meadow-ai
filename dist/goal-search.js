import { bonusPreviews } from './engine.js?v=20';
import { projectRanks } from './ai-search.js?v=20';
import { phaseForRanks } from './rl-model.js?v=20';
import { chooseLearnedMove } from './rl-search.js?v=20';
const directions=['left','right','up','down'];
const rank=n=>n<3?n:Math.log2(n/3)+3;
const value=r=>r<3?r:3*2**(r-3);
const points=r=>r<3?0:3**(r-2);
const score=b=>b.reduce((s,r)=>s+points(r),0);
const convert=p=>({cards:p.candidates.map(rank),weights:p.probabilities||p.candidates.map(()=>1/p.candidates.length)});
const previewCache=new Map(),GOAL_BONUS=60000;
export function chooseGoalMove(board,next,model,scoreModel,{remaining=null,maxDepth=3,maxNodes=24000,budgetMs=160}={}) {
  const b=board.flat().map(rank);
  if(Math.max(...b)>=13)return chooseLearnedMove(board,next,scoreModel,{remaining,maxDepth,maxNodes,budgetMs});
  if(!next?.candidates?.length)return {direction:null,depth:0,nodes:0};
  if(!remaining||remaining.length!==3||remaining.some(n=>!Number.isInteger(n)||n<0||n>4))throw new Error('缺少公开牌组计数');
  const root=convert(next),cache=new Map(),deadline=performance.now()+budgetMs,timeout=Symbol();
  let nodes=0,completed=0,best=null;
  function tick(){nodes++;if(completed&&(nodes>maxNodes||(nodes%64===0&&performance.now()>=deadline)))throw timeout;}
  function future(b,counts,depth){
    const high=Math.max(...b);
    if(!directions.some((_,d)=>projectRanks(b,d)))return high>=13?GOAL_BONUS:0;
    let bonuses=previewCache.get(high);
    if(!bonuses){bonuses=bonusPreviews(value(high)).map(p=>({...convert(p),probability:p.probability}));previewCache.set(high,bonuses);}
    const chance=bonuses.length?1/21:0,deck=counts.every(n=>n===0)?[4,4,4]:counts,size=deck.reduce((a,b)=>a+b,0);
    let total=0;
    for(let n=1;n<=3;n++)if(deck[n-1])total+=(1-chance)*deck[n-1]/size*decision(b,{cards:[n],weights:[1]},deck.map((v,i)=>v-(i===n-1)),depth).value;
    for(const preview of bonuses)total+=chance*preview.probability*decision(b,preview,counts,depth).value;
    return total;
  }
  function decision(b,preview,counts,depth){
    tick();const key=depth+'|'+b+'|'+preview.cards+'|'+preview.weights+'|'+counts;
    if(cache.has(key))return cache.get(key);
    const before=score(b);let bestValue=-Infinity,direction=null;
    for(let d=0;d<4;d++){
      const projected=projectRanks(b,d);if(!projected)continue;let q=score(projected.board)-before;
      if(depth===1)q+=scoreModel.value(projected.board,phaseForRanks(projected.board))+GOAL_BONUS*model.value(projected.board,preview,counts);
      else{let expected=0;for(const pos of projected.entries)for(let i=0;i<preview.cards.length;i++){
        tick();const spawned=projected.board.slice();spawned[pos]=preview.cards[i];expected+=preview.weights[i]*(points(preview.cards[i])+future(spawned,counts,depth-1));
      }q+=expected/projected.entries.length;}
      if(q>bestValue){bestValue=q;direction=directions[d];}
    }
    const result={value:direction?bestValue:0,direction};cache.set(key,result);return result;
  }
  for(let depth=1;depth<=maxDepth;depth++){
    try{const result=decision(b,root,remaining,depth);best=result.direction;completed=depth;}
    catch(e){if(e!==timeout)throw e;break;}
  }
  return {direction:best,depth:completed,nodes,stage:phaseForRanks(b),policy:'rl-v2'};
}
