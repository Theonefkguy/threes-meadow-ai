import { bonusPreviews } from './engine.js?v=20';
import { projectRanks } from './ai-search.js?v=20';
import { phaseForRanks } from './rl-model.js?v=20';
import { chooseLearnedMove } from './rl-search.js?v=20';
import { reachProbability } from './v6-model.js?v=20';

const directions=['left','right','up','down'];
const rank=n=>n<3?n:Math.log2(n/3)+3;
const value=r=>r<3?r:3*2**(r-3);
const previewCache=new Map();
const convert=p=>({cards:p.candidates.map(rank),weights:p.probabilities||p.candidates.map(()=>1/p.candidates.length)});

// Before 1536 all backed-up values are probabilities: success=1, death=0.
// Once reached, use frozen V4 score tables and its regular search.
export function chooseEarlyGoalMove(board,next,model,{remaining=null,maxDepth=3,maxNodes=24000,budgetMs=160}={}) {
  const b=board.flat().map(rank);
  if(Math.max(...b)>=12)return chooseLearnedMove(board,next,model,{remaining,maxDepth,maxNodes,budgetMs});
  if(!next?.candidates?.length)return {direction:null,depth:0,nodes:0};
  if(b.some(r=>r>15))return {unsupported:true,direction:null,depth:0,nodes:0};
  const root=convert(next),cache=new Map(),deadline=performance.now()+budgetMs,timeout=Symbol();
  let nodes=0,completed=0,best=null;
  const evaluate=b=>reachProbability(model,b);
  function tick(){nodes++;if(completed&&(nodes>maxNodes||(nodes%64===0&&performance.now()>=deadline)))throw timeout;}
  function future(b,counts,depth){
    if(Math.max(...b)>=12)return 1;
    // No hints are drawn after a terminal spawn.
    if(!directions.some((_,d)=>projectRanks(b,d)))return 0;
    const high=Math.max(...b);let bonuses=previewCache.get(high);
    if(!bonuses){bonuses=bonusPreviews(value(high)).map(p=>({...convert(p),probability:p.probability}));previewCache.set(high,bonuses);}
    const chance=bonuses.length?1/21:0;
    let total=0,deck=counts;
    if(deck&&deck.every(n=>n===0))deck=[4,4,4];
    const size=deck?deck.reduce((a,b)=>a+b,0):3;
    for(let n=1;n<=3;n++){
      const p=deck?deck[n-1]/size:1/3;if(!p)continue;
      total+=(1-chance)*p*decision(b,{cards:[n],weights:[1]},deck?deck.map((v,i)=>v-(i===n-1)):null,depth).value;
    }
    for(const preview of bonuses)total+=chance*preview.probability*decision(b,preview,counts,depth).value;
    return total;
  }
  function decision(b,preview,counts,depth){
    // At depth 1, the afterstate model uses only the board. Sharing these
    // cache entries avoids repeated evaluation without changing node counting.
    tick();const key=depth===1?'1|'+b.join(','):depth+'|'+b.join(',')+'|'+preview.cards+'|'+preview.weights+'|'+counts;
    if(cache.has(key))return cache.get(key);
    let bestValue=-Infinity,direction=null;
    for(let d=0;d<4;d++){
      const projected=projectRanks(b,d);if(!projected)continue;
      let q=0;
      if(Math.max(...projected.board)>=12)q=1;
      else if(depth===1)q=evaluate(projected.board);
      else{
        let expected=0;
        for(const pos of projected.entries)for(let i=0;i<preview.cards.length;i++){
          tick();const state=projected.board.slice();state[pos]=preview.cards[i];
          expected+=preview.weights[i]*future(state,counts,depth-1);
        }
        q+=expected/projected.entries.length;
      }
      if(q>bestValue){bestValue=q;direction=directions[d];}
    }
    const result={value:direction?bestValue:0,direction};cache.set(key,result);return result;
  }
  for(let depth=1;depth<=maxDepth;depth++){
    try{const result=decision(b,root,remaining,depth);best=result.direction;completed=depth;}
    catch(e){if(e!==timeout)throw e;break;}
  }
  return {direction:best,depth:completed,nodes,stage:phaseForRanks(b),policy:'rl-v6'};
}
