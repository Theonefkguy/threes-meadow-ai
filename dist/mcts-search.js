import {bonusPreviews} from './engine.js?v=20';
import {projectRanks} from './ai-search.js?v=20';
import {phaseForRanks} from './rl-model.js?v=20';
import {chooseLearnedMove} from './rl-search.js?v=20';
const directions=['left','right','up','down'];
const rank=n=>n<3?n:Math.log2(n/3)+3;
const value=r=>r<3?r:3*2**(r-3);
const points=r=>r<3?0:3**(r-2);
const score=b=>b.reduce((n,r)=>n+points(r),0);
const bonuses=new Map();
const convert=p=>({cards:p.candidates.map(rank),weights:p.probabilities||p.candidates.map(()=>1/p.candidates.length)});
// Same MT19937 stream as native experiments; never consumes real-game RNG.
export class SearchRandom {
 constructor(seed){this.state=new Uint32Array(624);this.state[0]=seed>>>0;for(let i=1;i<624;i++){const x=this.state[i-1];this.state[i]=(Math.imul(1812433253,x^(x>>>30))+i)>>>0;}this.index=624;}
 next(){if(this.index===624){for(let i=0;i<624;i++){const y=(this.state[i]&0x80000000)|(this.state[(i+1)%624]&0x7fffffff);this.state[i]=this.state[(i+397)%624]^(y>>>1)^((y&1)?0x9908b0df:0);}this.index=0;}let y=this.state[this.index++];y^=y>>>11;y^=(y<<7)&0x9d2c5680;y^=(y<<15)&0xefc60000;y^=y>>>18;return ((y>>>0)+.5)/4294967296;}
}
function publicSeed(b,p,c){let h=2166136261;for(const v of [...b,...p.cards,...c])h=Math.imul(h^v,16777619);return h>>>0;}
export function chooseMCTSMove(board,next,model,{remaining=null,budgetMs=160,maxSimulations=640,horizon=8,seed=null,forceEarly=false}={}){
 const b=board.flat().map(rank),stage=phaseForRanks(b);
 if(b.some(r=>r>15))return {unsupported:true,direction:null,stage};
 if(!next?.candidates?.length)return {direction:null,stage};
 // Preserve established late-game search, and its public-count fallback.
 if((stage>0&&!forceEarly)||!remaining)return {...chooseLearnedMove(board,next,model,{remaining,budgetMs}),algorithm:'expectimax'};
 const deadline=performance.now()+budgetMs,rootPreview=convert(next),rng=new SearchRandom(seed??publicSeed(b,rootPreview,remaining));
 const table=new Map(),nodes=[];let simulations=0,maxReached=0,evaluations=0;
 const legal=b=>directions.some((_,d)=>projectRanks(b,d));
 function node(b,p,c,depth=0){
  const key=depth+'|'+b.join(',')+'|'+p.cards+'|'+p.weights+'|'+c;
  if(table.has(key))return table.get(key);
  const before=score(b);let leaf=-Infinity;
  const edges=directions.map((_,d)=>{const projected=projectRanks(b,d);if(!projected)return null;const reward=score(projected.board)-before,initial=reward+model.value(projected.board,phaseForRanks(projected.board));evaluations++;leaf=Math.max(leaf,initial);return {projected,reward,initial,prior:0,total:0,visits:0};});
  if(leaf===-Infinity)leaf=0;let sum=0;
  for(const e of edges)if(e){e.prior=Math.exp(Math.max(-20,(e.initial-leaf)/2000));sum+=e.prior;}
  for(const e of edges)if(e)e.prior/=sum;
  const id=nodes.length;nodes.push({b,p,c,edges,leaf,visits:0});table.set(key,id);return id;
 }
 function nextPreview(b,c){
  const high=Math.max(...b);let list=bonuses.get(high);
  if(!list){list=bonusPreviews(value(high)).map(p=>({...convert(p),probability:p.probability}));bonuses.set(high,list);}
  if(list.length&&rng.next()<1/21){const u=rng.next();let sum=0;for(const p of list){sum+=p.probability;if(u<sum)return p;}return list.at(-1);}
  if(c.every(n=>n===0))c.splice(0,3,4,4,4);
  let u=rng.next()*(c[0]+c[1]+c[2]),card=2;
  for(let k=0;k<3;k++){u-=c[k];if(u<0){card=k;break;}}
  c[card]--;return {cards:[card+1],weights:[1]};
 }
 function simulate(id,depth){
  const n=nodes[id];maxReached=Math.max(maxReached,depth);if(depth>=horizon)return n.leaf;
  let chosen=-1,best=-Infinity;
  for(let d=0;d<4;d++){const e=n.edges[d];if(!e)continue;if(!e.visits){chosen=d;break;}const q=e.total/e.visits+4000*e.prior*Math.sqrt(n.visits+1)/(1+e.visits);if(q>best){best=q;chosen=d;}}
  if(chosen<0)return 0;
  const e=n.edges[chosen],state=e.projected.board.slice(),entries=e.projected.entries,entry=Math.min(entries.length-1,Math.floor(rng.next()*entries.length));
  const u=rng.next();let sum=0,card=n.p.cards.at(-1);
  for(let i=0;i<n.p.cards.length;i++){sum+=n.p.weights[i];if(u<sum){card=n.p.cards[i];break;}}
  state[entries[entry]]=card;let result=e.reward+points(card);
  if(legal(state)){const counts=n.c.slice(),p=nextPreview(state,counts),before=nodes.length,child=node(state,p,counts,depth+1);maxReached=Math.max(maxReached,depth+1);result+=child>=before?nodes[child].leaf:simulate(child,depth+1);}
  e.visits++;e.total+=result;n.visits++;return result;
 }
 const root=node(b,rootPreview,remaining.slice());
 do{simulate(root,0);simulations++;}while(simulations<maxSimulations&&(simulations<4||performance.now()<deadline));
 let direction=null,most=-1;for(let d=0;d<4;d++){const e=nodes[root].edges[d];if(e&&e.visits>most){most=e.visits;direction=directions[d];}}
 return {direction,stage,depth:maxReached,nodes:nodes.length,simulations,evaluations,algorithm:'mcts',visits:nodes[root].edges.map(e=>e?.visits??0)};
}
