// V17 fast learned search: the same public-information expectimax as rl-search.js,
// with (1) a probability cutoff and (2) typed-array boards, row tables and an
// open-addressing transposition table. A decision node whose path probability
// (product of all chance weights from the root) is below `threshold` is scored
// with the one-move afterstate value instead of its remaining depth.
// threshold=0 is the complete search. Only fully completed depths are used.
import { bonusPreviews } from './engine.js?v=20';

const directions=['left','right','up','down'];
const toRank=n=>n<3?n:Math.log2(n/3)+3;
const fromRank=r=>r<3?r:3*2**(r-3);
const POINTS=Float64Array.from({length:32},(_,r)=>r<3?0:3**(r-2));
const LINES=[0,1,2,3].map(d=>[0,1,2,3].map(lane=>[0,1,2,3].map(j=>
  d===0?lane*4+j:d===1?lane*4+3-j:d===2?j*4+lane:(3-j)*4+lane)));
const LINE_IDX=Int8Array.from(LINES.flat(2));
// Row table: key = r0|r1<<4|r2<<8|r3<<12 (r0 at the leading edge).
// ROW_OUT = moved key, or -1 if unchanged; ROW_REWARD = merge points gained.
// A merge producing rank 16 cannot be packed and is flagged with -2.
const ROW_OUT=new Int32Array(65536),ROW_REWARD=new Float64Array(65536);
const mergeable=(a,b)=>a&&b&&(a+b===3||(a>=3&&a===b));
for(let key=0;key<65536;key++){
  const r=[key&15,key>>4&15,key>>8&15,key>>12&15];let changed=false,reward=0,overflow=false;
  for(let j=1;j<4;j++)if(r[j]&&(!r[j-1]||mergeable(r[j],r[j-1]))){
    const a=r[j],b=r[j-1],c=!b?a:a+b===3?3:a+1;
    if(c>15)overflow=true;reward+=POINTS[c]-POINTS[a]-POINTS[b];r[j-1]=c;r[j]=0;changed=true;
  }
  ROW_OUT[key]=overflow?-2:changed?r[0]|r[1]<<4|r[2]<<8|r[3]<<12:-1;ROW_REWARD[key]=reward;
}
// Transposition table shared across calls; entries of older calls are free (stamp).
const BITS=18,SIZE=1<<BITS,MASK=SIZE-1;
const kLo=new Int32Array(SIZE),kHi=new Int32Array(SIZE),kPar=new Int32Array(SIZE),used=new Int32Array(SIZE),val=new Float64Array(SIZE);
let stamp=0;
// Flat pattern cell table (64 features x 4 cells), same order as rl-model.js.
const PATTERNS=[[0,1,2,3],[4,5,6,7],[0,1,4,5],[1,2,5,6],[5,6,9,10],[0,1,2,4],[0,1,5,6],[0,1,4,8]];
const FEAT=Int8Array.from(PATTERNS.flatMap(pattern=>Array.from({length:8},(_,sym)=>pattern.map(pos=>{
  let y=pos>>2,x=pos&3;if(sym>=4)x=3-x;for(let k=0;k<sym%4;k++)[y,x]=[x,3-y];return y*4+x;})).flat()));
// Same summation order as decodeModel().value; ranks here never exceed 15.
function leafValue(W,b,stage){
  const off=stage*524288;let v=0;
  for(let f=0;f<64;f++){const o=f*4;v+=W[off+(f>>3)*65536+(b[FEAT[o]]|b[FEAT[o+1]]<<4|b[FEAT[o+2]]<<8|b[FEAT[o+3]]<<12)];}
  return v;
}
const previewCache=new Map();
function bonusList(high){
  let list=previewCache.get(high);
  if(!list){list=bonusPreviews(fromRank(high)).map(p=>({cards:p.candidates.map(toRank),weights:p.probabilities,probability:p.probability,bonus:true}));previewCache.set(high,list);}
  return list;
}
const NORMAL=[1,2,3].map(n=>({cards:[n],weights:[1],probability:1,bonus:false}));

// Projection of board `b` in direction d into `out` (16 bytes). Returns the
// number of moved lanes (entries written to `entries`), 0 if illegal, -1 on overflow.
let lastReward=0;
function project(b,d,out,entries){
  let n=0,reward=0;
  for(let lane=0;lane<4;lane++){
    const o=(d*4+lane)*4,i0=LINE_IDX[o],i1=LINE_IDX[o+1],i2=LINE_IDX[o+2],i3=LINE_IDX[o+3];
    const r0=b[i0],r1=b[i1],r2=b[i2],r3=b[i3];
    out[i0]=r0;out[i1]=r1;out[i2]=r2;out[i3]=r3;
    const key=r0|r1<<4|r2<<8|r3<<12,m=ROW_OUT[key];
    if(m===-1)continue;if(m===-2)return -1;
    out[i0]=m&15;out[i1]=m>>4&15;out[i2]=m>>8&15;out[i3]=m>>12;entries[n++]=i3;reward+=ROW_REWARD[key];
  }
  lastReward=reward;return n;
}

// clampLeaf: the true future score is never negative, so leaf estimates below 0 are
// raised to 0 (otherwise a certain-death move can outrank a survivable one).
export function chooseFastMove(board,next,model,{remaining=null,maxDepth=4,threshold=0.003,maxNodes=4e6,budgetMs=160,now=()=>performance.now(),stats=null,clampLeaf=false}={}){
  const root=Uint8Array.from(board.flat().map(toRank));
  if(!next?.candidates?.length)return {direction:null,depth:0,nodes:0};
  let rootHigh=0;for(const r of root)rootHigh=Math.max(rootHigh,r);
  if(rootHigh>14)return {unsupported:true,direction:null,depth:0,nodes:0};
  const rootPreview={cards:next.candidates.map(toRank),weights:next.probabilities||next.candidates.map(()=>1/next.candidates.length)};
  // Transposition table (per call). Key: packed board (2x32 bits) + params.
  const stages=model.stages===4?4:3;
  const W=model.weights instanceof Float32Array&&model.weights.length===stages*524288?model.weights:null;
  if(++stamp>2e9){stamp=1;used.fill(0);}
  const deadline=now()+budgetMs,timeout=Symbol();
  let nodes=0,completed=0,cutoffs=0,best=null,rootValues=null;
  // Scratch buffers per recursion level (a level = one decision + its chance layer).
  const LEVELS=16,moved=Array.from({length:LEVELS},()=>Array.from({length:4},()=>new Uint8Array(16))),
    ent=Array.from({length:LEVELS},()=>Array.from({length:4},()=>new Int8Array(4))),
    size=Array.from({length:LEVELS},()=>new Int8Array(4)),rew=Array.from({length:LEVELS},()=>new Float64Array(4)),
    child=Array.from({length:LEVELS},()=>new Uint8Array(16));
  function tick(){nodes++;if(completed&&(nodes>maxNodes||((nodes&255)===0&&now()>=deadline)))throw timeout;}
  const maxRank=b=>{let h=0;for(let i=0;i<16;i++)if(b[i]>h)h=b[i];return h;};
  const stageOf=b=>{let h=0;for(let i=0;i<16;i++)if(b[i]>h)h=b[i];return stages===4&&h>=14?3:h>=13?2:h>=12?1:0;};
  // Generate the four projections of b at level L; returns false if no legal move.
  function expand(b,L){
    let any=false;
    for(let d=0;d<4;d++){const n=project(b,d,moved[L][d],ent[L][d]);if(n<0)throw new Error('rank overflow');size[L][d]=n;rew[L][d]=lastReward;any||=n>0;}
    return any;
  }
  // counts are packed as c1|c2<<3|c3<<6 (remaining 1s, 2s, 3s); -1 = unknown (equal thirds).
  function future(b,counts,depth,prob,L){
    if(!expand(b,L))return 0;
    let high=0,lo=0,hi=0;for(let i=0;i<8;i++){const x=b[i],y=b[i+8];if(x>high)high=x;if(y>high)high=y;lo|=x<<(i*4);hi|=y<<(i*4);}
    const list=bonusList(high),chance=list.length?1/21:0;let total=0,deck=counts;
    if(deck===0)deck=4|4<<3|4<<6;
    const d1=deck&7,d2=deck>>3&7,d3=deck>>6&7,sz=d1+d2+d3;
    // One-move values ignore the preview and counts, so every branch that is at
    // depth 1 (or cut to it) shares one value; accumulation order is unchanged.
    let one=NaN;
    for(let n=1;n<=3;n++){
      const cnt=n===1?d1:n===2?d2:d3;
      if(deck>=0&&!cnt)continue;const w=deck>=0?(1-chance)*cnt/sz:(1-chance)/3;
      if(depth===1||prob*w<threshold){if(one!==one)one=decision(b,lo,hi,NORMAL[0],-1,1,0,L,false);total+=w*one;if(depth>1)cutoffs++;continue;}
      total+=w*decision(b,lo,hi,NORMAL[n-1],deck>=0?deck-(1<<(3*(n-1))):-1,depth,prob*w,L,false);
    }
    for(const preview of list){
      const w=chance*preview.probability,pb=prob*chance*preview.probability;
      if(depth===1||pb<threshold){if(one!==one)one=decision(b,lo,hi,NORMAL[0],-1,1,0,L,false);total+=w*one;if(depth>1)cutoffs++;continue;}
      total+=w*decision(b,lo,hi,preview,counts,depth,pb,L,false);
    }
    return total;
  }
  // Returns the value; at the root also records direction and per-move values.
  function decision(b,lo,hi,preview,counts,depth,prob,L,isRoot){
    if(depth>1&&!isRoot&&prob<threshold){depth=1;cutoffs++;}
    tick();
    const par=depth===1?1:depth|preview.cards.length<<3|preview.cards[0]<<5|(counts>=0?counts:511)<<10;
    let h=Math.imul(lo^0x9e3779b9,0x85ebca6b)^Math.imul(hi^0xc2b2ae35,0x27d4eb2f)^Math.imul(par,0x165667b1);h^=h>>>15;h=Math.imul(h,0x2c1b3c6d);h^=h>>>12;
    let slot=-1;
    if(!isRoot)for(let i=0;i<8;i++){const s=(h+i)&MASK;
      if(used[s]!==stamp){if(slot<0)slot=s;break;}
      if(kLo[s]===lo&&kHi[s]===hi&&kPar[s]===par)return val[s];}
    let bestValue=-Infinity,direction=-1;const values=isRoot?[null,null,null,null]:null;
    for(let d=0;d<4;d++){
      const n=size[L][d];if(!n)continue;const after=moved[L][d];let q=rew[L][d];
      if(maxRank(after)>=15){}           // 12288 ends the game: merge points only, no spawn/future.
      else if(depth===1){const v=W?leafValue(W,after,stageOf(after)):model.value(after,stageOf(after));q+=clampLeaf&&v<0?0:v;}
      else{
        let expected=0;const c=child[L],cards=preview.cards,weights=preview.weights,e=ent[L][d];
        for(let k=0;k<n;k++)for(let i=0;i<cards.length;i++){
          tick();c.set(after);c[e[k]]=cards[i];
          expected+=weights[i]*(POINTS[cards[i]]+future(c,counts,depth-1,prob*weights[i]/n,L+1));
        }
        q+=expected/n;
      }
      if(values)values[d]=q;if(q>bestValue){bestValue=q;direction=d;}
    }
    const value=direction<0?0:bestValue;
    if(slot>=0){used[slot]=stamp;kLo[slot]=lo;kHi[slot]=hi;kPar[slot]=par;val[slot]=value;}
    if(isRoot){best=direction;rootValues=values;}
    return value;
  }
  if(!expand(root,0))return {direction:null,depth:0,nodes:0,policy:'rl-fast'};
  let rootLo=0,rootHi=0;for(let i=0;i<8;i++){rootLo|=root[i]<<(i*4);rootHi|=root[i+8]<<(i*4);}
  const packed=Array.isArray(remaining)&&remaining.length===3&&remaining.every(v=>Number.isInteger(v)&&v>=0&&v<=4)?remaining[0]|remaining[1]<<3|remaining[2]<<6:-1;
  let answer=null,answerValues=null;
  for(let depth=1;depth<=maxDepth;depth++){
    try{decision(root,rootLo,rootHi,rootPreview,packed,depth,1,0,true);answer=best;answerValues=rootValues;completed=depth;}
    catch(e){if(e!==timeout)throw e;break;}
  }
  if(stats){stats.cutoffs=cutoffs;stats.values=answerValues;}
  return {direction:answer<0||answer===null?null:directions[answer],depth:completed,nodes,stage:stageOf(root),policy:'rl-fast'};
}
